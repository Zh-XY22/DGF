"""One-epoch preparation plugins; complete joint true-label NLL is always retained.

Neighbor labels supervise only the router (no unrecorded expert pseudo-label loss).
Selective auxiliary NLL is position-marginal, distinct from the trajectory-mixture anchor.
Distillation marginalizes both latent mode and native family over complete trajectories.
"""
from pathlib import Path
import numpy as np
import torch
from .policy import joints,rollout
from .probability import observation_logp,action_marginal_logp,common_scaler,nll
from .distillation import forward_kl_surrogate,reverse_kl_surrogate
from .data import structured,assert_complete_future,forecast_dimensions
from .runtime import load_policy
from .opd import OPD_ARMS,RollingPrefixPreparation


LABEL_ARMS=('E04','E05','E09','E10','E11')
TEACHER_ARMS=('E06','E07','E08')
WARMUP_ARMS=('S04','S05','S06','S07','S08','S10','S11')


def marginal_actions(experts,weights,samples=8):
    # Independent mode per complete trajectory; never a shared Dirichlet pi.
    with torch.no_grad():
        packed=rollout(experts,weights,None,1,samples)
    return {k:packed[k][:,0].permute(1,0,2).detach() for k in ['raw','forecast','component']}


class Preparation:
    def __init__(self,config,device):
        self.config=config;self.device=device;self.protocol=config['protocol'];self.arm=config.get('preparation_arm',config.get('arm'))
        self.h=self.protocol['horizon'];self.dim=len(self.protocol['variables']) if self.protocol['mode']=='M' else 1
        root=Path(config['artifact'])
        self.indices=np.load(root/'train_indices.npy',allow_pickle=False)
        self.training_values=np.load(root/'normalized.npy',allow_pickle=False)[:self.protocol['train_end']]
        self.neighbors=None;self.masks={};self.teacher=None;self.opd=None;self.checked_windows=0;self.maximum_target_difference=0.
        if self.arm == 'S08' and config.get('repair_version') == 'S08-TIME-v2':
            from .warmup_repairs import hard_event_statistics
            self.hard_weights, self.hard_event_evidence = hard_event_statistics(
                self.training_values, self.indices, self.h)
        if self.arm in ['E04','E05']:
            self.neighbors=dict(np.load(Path(config['neighbor_artifact'])/'neighbors.npz',allow_pickle=False))
            assert np.array_equal(self.neighbors['query_origins'],self.indices[:,0])
            expected=self.indices[:,1] if self.dim==1 else np.zeros(len(self.indices),dtype=np.int64)
            # M mode builder uses variable id -1 or 0 as the same complete seven-variable entity.
            if self.dim==1:assert np.array_equal(self.neighbors['variable_ids'],expected)
            else:assert len(np.unique(self.neighbors['variable_ids']))==1
        if self.arm in ['E09','E10','E11','E15','R15']:
            directory=Path(config['selective_artifact'])
            assert np.array_equal(np.load(directory/'indices.npy',allow_pickle=False),self.indices)
            for key,name in [('hard','selected.npy'),('soft','soft_weight.npy'),('random','random_selected.npy')]:
                self.masks[key]=np.load(directory/name,allow_pickle=False)
                assert self.masks[key].shape==(len(self.indices),self.h,self.dim)
            if self.arm=='R15':
                self.masks['random_weight']=np.load(directory/'random_weight.npy',allow_pickle=False)
                assert self.masks['random_weight'].shape==(len(self.indices),self.h,self.dim)
                assert np.isfinite(self.masks['random_weight']).all() and (self.masks['random_weight']>=0).all() and (self.masks['random_weight']<=1).all()
        if self.arm in TEACHER_ARMS:
            teacher_config=config['teacher_config']
            for key in ['context','patch','horizon','mode','train_end','val_end','test_end','ddof']:
                assert teacher_config['protocol'][key]==self.protocol[key], 'Teacher data protocol differs: '+key
            self.teacher=load_policy(teacher_config,device).eval().requires_grad_(False)
            self.teacher_versions={n:p._version for n,p in self.teacher.named_parameters()}
        if self.arm in OPD_ARMS:
            self.opd=RollingPrefixPreparation(config,device)

    def verify_alignment(self,b,target):
        ids=b['_index'].detach().cpu().numpy();pairs=self.indices[ids]
        truth=[]
        for origin,var in pairs:
            cols=slice(None) if self.protocol['mode']=='M' else slice(int(var),int(var)+1)
            assert origin>=self.protocol['context'] and origin+self.h<=self.protocol['train_end']
            truth.append(self.training_values[origin:origin+self.h,cols])
        expected=torch.as_tensor(np.asarray(truth),device=target.device,dtype=target.dtype)
        actual=structured(target,self.dim,self.h)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
        self.checked_windows+=len(ids);self.maximum_target_difference=max(self.maximum_target_difference,float((actual-expected).abs().max()))
        assert_complete_future(b,self.h*self.dim)
        return ids

    def reliability(self,indices):
        ids=indices.detach().cpu().numpy() if torch.is_tensor(indices) else np.asarray(indices)
        # The same frozen X-dependent weights are used for every J/G candidate.
        key='random_weight' if self.arm=='R15' else 'soft'
        selected=self.masks[key][ids]
        if not hasattr(self,'reliability_evidence'):
            self.reliability_evidence=dict(status='ACTUALLY_LOADED_IN_RL',arm=self.arm,
                weight_filename='random_weight.npy' if key=='random_weight' else 'soft_weight.npy',
                artifact=self.config['selective_artifact'],first_batch_inputs=len(ids),
                minimum=float(selected.min()),maximum=float(selected.max()),mean=float(selected.mean()),
                nonzero_fraction=float(np.count_nonzero(selected)/selected.size),
                full_training_nonuniform=bool(self.masks[key].min()!=self.masks[key].max()),
                frozen=True,same_weights_for_all_J_G_candidates=True,
                scale='Reliability weights, distinct from observed/padding masks',
                evidence_scope='Actual selected file loaded and first RL minibatch used; not a performance conclusion')
        return torch.as_tensor(selected,device=self.device).detach()

    def batch_context(self,indices):
        if hasattr(self, 'hard_weights'):
            return dict(total_inputs=len(indices), weighted_elements=float(self.hard_weights[indices].sum()))
        if self.arm not in ['E09','E10','E11']:return None
        key={'E09':'hard','E10':'soft','E11':'random'}[self.arm]
        weights=self.masks[key][indices]
        denominator=float(weights.sum(dtype=np.float64))
        fallback=denominator==0
        if fallback:denominator=float(len(indices)*self.h*self.dim)
        return dict(total_inputs=len(indices),weighted_elements=denominator,all_empty_fallback=fallback)

    def auxiliary(self,b,experts,target,weights,context=None,student_policy=None,policy_version=None,capture=False):
        ids=self.verify_alignment(b,target);arm=self.arm;diag={}
        if arm in OPD_ARMS:
            loss,diag=self.opd.auxiliary(student_policy,b,policy_version,capture=capture)
        elif arm in ['E04','E05']:
            key='q' if arm=='E04' else 'q_random'
            q=torch.as_tensor(self.neighbors[key][ids],device=target.device,dtype=weights.dtype)
            confidence=torch.as_tensor(self.neighbors['confidence'][ids],device=target.device,dtype=weights.dtype).detach()
            good=confidence>0
            if good.any():torch.testing.assert_close(q[good].sum(-1),torch.ones_like(confidence[good]),atol=1e-6,rtol=1e-6)
            loss=-(confidence*(q.detach()*weights.log()).sum(-1)).mean()
            diag=dict(confident_fraction=float(good.float().mean()),mean_confidence=float(confidence.mean()),
                router_target='past-fold neighbor histogram' if arm=='E04' else 'same-neighbor matched-stratum shuffled histogram')
        elif arm in ['E09','E10','E11']:
            _,marginal,_=observation_logp(experts,weights,target)
            point_loss=structured(-marginal,self.dim,self.h)
            key={'E09':'hard','E10':'soft','E11':'random'}[arm]
            selected=torch.as_tensor(self.masks[key][ids],device=target.device,dtype=point_loss.dtype).detach()
            if context is None:context=self.batch_context(ids)
            if context['all_empty_fallback']:selected=torch.ones_like(selected)
            # The trainer multiplies micro-loss by micro_inputs/global_inputs.
            # This compensates exactly, yielding sum_all(w*loss)/sum_all(w).
            loss=(selected*point_loss).sum()*context['total_inputs']/(len(ids)*context['weighted_elements'])
            diag=dict(auxiliary_measure='position-marginal NLL; not complete trajectory mixture likelihood',
                mean_weight=float(selected.mean()),all_empty_fallback=context['all_empty_fallback'])
        elif arm in TEACHER_ARMS:
            with torch.no_grad():teacher,truth,teacher_w,_,_=joints(self.teacher(b),b)
            assert torch.equal(truth,target)
            loc,scale=common_scaler(experts);tloc,tscale=common_scaler(teacher)
            assert torch.equal(loc,tloc) and torch.equal(scale,tscale), 'Teacher/student mixed-measure lattices differ'
            assert [type(c) for c in experts[0].mixture.components]==[type(c) for c in teacher[0].mixture.components]
            if arm=='E06':
                mean=(torch.stack([e.mean for e in experts],1)*weights[:,:,None]).sum(1)
                teacher_mean=(torch.stack([e.mean for e in teacher],1)*teacher_w[:,:,None]).sum(1).detach()
                if not torch.isfinite(mean).all() or not torch.isfinite(teacher_mean).all():raise FloatingPointError('Undefined distillation point mean')
                loss=(mean-teacher_mean).square().mean();diag=dict(target='teacher mixture analytic mean')
            else:
                samples=int(self.config.get('distillation_samples',8))
                action=marginal_actions(teacher,teacher_w,samples) if arm=='E07' else marginal_actions(experts,weights,samples)
                lp=action_marginal_logp(experts,weights,action)
                with torch.no_grad():lq=action_marginal_logp(teacher,teacher_w,action)
                if arm=='E07':loss=forward_kl_surrogate(lp)/target.shape[-1];estimate=(lq-lp.detach())/target.shape[-1]
                else:
                    difference=(lp-lq).detach()
                    # Other independent actions provide an action-independent LOO baseline.
                    baseline=(difference.sum(0,keepdim=True)-difference)/(samples-1) if samples>1 else None
                    loss=reverse_kl_surrogate(lp,lq,baseline)/target.shape[-1];estimate=difference/target.shape[-1]
                diag=dict(samples=samples,mc_KL_per_position=float(estimate.mean()),
                    mc_standard_error_across_independent_actions=float(estimate.std(0,correction=1).mean()/samples**.5) if samples>1 else None,
                    sampled_actions_detached=all(not x.requires_grad for x in action.values()),
                    marginalization='sum positions within each mode, then logsumexp modes; native families also marginalized',
                    estimator='teacher cross-entropy' if arm=='E07' else 'fixed-action score-function +1, leave-one-out independent-action baseline')
            assert all(not p.requires_grad and p.grad is None and p._version==self.teacher_versions[n] for n,p in self.teacher.named_parameters())
        elif arm in WARMUP_ARMS:
            dim=forecast_dimensions(self.protocol);h=self.h
            mean=torch.stack([e.mean for e in experts],1)
            mean=(mean*weights[:,:,None]).sum(1)
            pred=structured(mean,dim,h);truth=structured(target,dim,h)
            # Packed observations carry a patch axis; retain only the valid
            # forecast elements before converting back to horizon x variables.
            packed_valid=(b['prediction_mask'][...,None]&b['observed_mask']).flatten(1)
            valid=packed_valid[packed_valid].reshape_as(target)
            valid=structured(valid.float(),dim,h).bool()
            if arm == 'S07' and self.config.get('repair_version') == 'S07-CROSS-v2':
                from .warmup_repairs import coarse_consistency
                if student_policy is None: raise ValueError('Shared student backbone is required')
                loss, diag = coarse_consistency(student_policy, b, pred, valid,
                    self.protocol, self.training_values, self.indices)
            elif arm == 'S08' and self.config.get('repair_version') == 'S08-TIME-v2':
                from .warmup_repairs import weighted_hard_event_loss
                if context is None: context = self.batch_context(ids)
                hard = torch.as_tensor(self.hard_weights[ids], device=pred.device, dtype=pred.dtype)
                loss = weighted_hard_event_loss(pred, truth, valid, hard, context)
                diag = dict(self.hard_event_evidence, auxiliary_measure='training-frozen temporal hard-event point MSE',
                    global_batch_weight_normalization=True)
            elif arm=='S04':
                loss=(pred-truth).abs().masked_select(valid).mean()
                diag=dict(auxiliary_measure='mixture-mean point MAE',valid_fraction=float(valid.float().mean()))
            elif arm=='S05':
                loss=(pred-truth).square().masked_select(valid).mean()
                diag=dict(auxiliary_measure='mixture-mean point MSE',valid_fraction=float(valid.float().mean()))
            elif arm=='S06':
                d1p=pred[:,1:,:]-pred[:,:-1,:]; d1t=truth[:,1:,:]-truth[:,:-1,:]
                d2p=d1p[:,1:,:]-d1p[:,:-1,:]; d2t=d1t[:,1:,:]-d1t[:,:-1,:]
                v1=valid[:,1:,:]&valid[:,:-1,:]; v2=v1[:,1:,:]&v1[:,:-1,:]
                loss=((d1p-d1t).square().masked_select(v1).mean() +
                      (d2p-d2t).square().masked_select(v2).mean())
                diag=dict(auxiliary_measure='first_and_second_difference_MSE',valid_fraction=float(valid.float().mean()))
            elif arm=='S07':
                terms=[]
                for factor in (2,4):
                    n=h//factor
                    pf=pred[:,:n*factor,:].reshape(pred.shape[0],n,factor,dim).mean(2)
                    tf=truth[:,:n*factor,:].reshape(truth.shape[0],n,factor,dim).mean(2)
                    vf=valid[:,:n*factor,:].reshape(valid.shape[0],n,factor,dim).all(2)
                    terms.append((pf-tf).square().masked_select(vf).mean())
                loss=sum(terms)
                diag=dict(auxiliary_measure='coarse_scale_interval_mean_MSE',scales=[2,4],valid_fraction=float(valid.float().mean()))
            elif arm=='S08':
                score=truth.abs().mean((1,2))+(truth[:,:,1:]-truth[:,:,:-1]).abs().mean((1,2))
                cap=score.detach().quantile(.75).clamp_min(1e-6)
                hard=(1.+2.*(score>cap).float()).detach()
                per=((pred-truth).square()*valid).sum((1,2))/valid.sum((1,2)).clamp_min(1)
                loss=(per*hard).sum()/hard.sum()
                diag=dict(auxiliary_measure='hard_event_weighted_point_MSE',weight_cap=3.,hard_fraction=float((hard>1).float().mean()),valid_fraction=float(valid.float().mean()))
            elif arm=='S10':
                loss=nll(experts,weights,target)
                diag=dict(auxiliary_measure='SFT_NLL_with_EMA_shadow',ema_decay=float(self.config.get('ema_decay',.99)))
            else:  # S11
                joint,_,_=observation_logp(experts,weights,target)
                per=-joint/target.shape[-1]
                env=(b['_index'].detach().long()/(len(self.indices)/4.)).floor().clamp_max(3).long()
                means=torch.stack([per[env==i].mean() for i in range(4) if (env==i).any()])
                loss=means.var(unbiased=False) if len(means)>1 else per.new_zeros(())
                diag=dict(auxiliary_measure='FOIL_environment_loss_variance',environments=4,nonempty_environments=int(len(means)))
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite warmup auxiliary')
        else:raise ValueError('Unsupported preparation auxiliary '+str(arm))
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite preparation auxiliary')
        return loss,dict(diag,preparation_arm=arm,auxiliary=float(loss.detach()))
