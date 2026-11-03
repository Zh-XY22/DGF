"""Train-only, equal-input-batch LR probes; no validation/test score selection."""
from .io_utils import load_checkpoint
import json
from pathlib import Path
import time
import torch
from .data import datasets,batch,assert_complete_future
from .policy import joints,rollout,expert_log_prob,mixture_nll
from .probability import nll,observation_logp,legacy_clipped_kl,action_marginal_logp,common_scaler
from .rewards import relative,ppo_surrogate
from .runtime import seed_all,load_policy,optimizer_for,checked_step,save_state,restore_state,atomic_json


def frozen_rollout(policy,b,j=8,g=4):
    with torch.no_grad():
        experts,target,w,alpha,_=joints(policy(b),b)
        action=rollout(experts,w,alpha,j,g)
        old=expert_log_prob(experts,action).detach()
        error=-(action['forecast'].double()-target[:,None,None].double()).square()
        reward=error.mean(-1)
        router=None if alpha is None else torch.distributions.Dirichlet(alpha[:,None]).log_prob(action['pi']).detach()
        return dict(action=action,old=old,advantage=relative(error),dir_adv=relative(reward.mean(2),axis=1),
                    old_router=router,reference_alpha=None if alpha is None else alpha.detach(),
                    zero_variance_fraction=float((error.std(2,correction=0)==0).double().mean()))


def position_loss(policy,b,data,anchor_coefficient):
    experts,target,w,alpha,_=joints(policy(b),b)
    lp=expert_log_prob(experts,data['action'])
    sample=-ppo_surrogate(lp,data['old'],data['advantage']).mean()
    router=sample.new_zeros(())
    if alpha is not None:
        cur=torch.distributions.Dirichlet(alpha[:,None]).log_prob(data['action']['pi'])
        router=-ppo_surrogate(cur,data['old_router'],data['dir_adv']).mean()
    # Probe has one rollout from the initial policy: old and reference coincide.
    kl=legacy_clipped_kl(lp,data['old']).mean()
    if alpha is not None:
        kl=kl+torch.distributions.kl_divergence(torch.distributions.Dirichlet(alpha),
                  torch.distributions.Dirichlet(data['reference_alpha'])).mean()
    anchor=nll(experts,w,target)
    loss=sample+router+.01*kl+anchor_coefficient*anchor
    delta=(lp.double()-data['old'].double()).detach()
    diagnostic=dict(sample_loss=float(sample.detach()),router_loss=float(router.detach()),
        anchor_nll=float(anchor.detach()),reference_kl=float(kl.detach()),
        max_position_log_ratio=float(delta.max()),max_abs_position_log_ratio=float(delta.abs().max()),
        max_joint_log_ratio=float(delta.sum(-1).max()),
        clip_fraction=float(((delta.exp()<.8)|(delta.exp()>1.2)).double().mean()),
        zero_variance_fraction=data['zero_variance_fraction'])
    return loss,diagnostic,(sample+router),anchor


def to_cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:to_cpu(v) for k,v in value.items()}
    return value


def to_device(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,dict):return {k:to_device(v,device) for k,v in value.items()}
    return value


def relative_update(policy,initial):
    num=den=0.;frozen_max=0.
    for name,p in policy.named_parameters():
        difference=p.detach().double()-initial[name].to(p.device).double()
        if p.requires_grad:num+=float(difference.square().sum());den+=float(initial[name].double().square().sum())
        else:frozen_max=max(frozen_max,float(difference.abs().max()))
    return (num/max(den,1e-30))**.5,frozen_max


def gradient_norm(loss,parameters,retain_graph=True):
    gradients=torch.autograd.grad(loss,parameters,retain_graph=retain_graph,allow_unused=True)
    return sum(float(g.detach().double().square().sum()) for g in gradients if g is not None)**.5


def probe(config_path,output,device='cuda',seed=1):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(config_path).read_text());seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    policy=load_policy(config,device).eval();ds=datasets(policy,config['dataset'],config['protocol'])['train']
    initial={n:p.detach().cpu().clone() for n,p in policy.named_parameters()}
    initial_state={n:p.detach().cpu().clone() for n,p in policy.state_dict().items()}
    n=config.get('global_input_batch',512);micro=config['microbatch']
    indices=torch.randperm(len(ds),generator=torch.Generator().manual_seed(seed))[:n].tolist()
    batches=[indices[i:i+micro] for i in range(0,n,micro)]
    began=time.time();records=[];torch.cuda.reset_peak_memory_stats()
    optimizer=optimizer_for(policy,config['lr']);optimizer.zero_grad(set_to_none=True)
    # First trajectory: one complete effective input batch, legal marginal NLL.
    nll_total=0.;measure_record=None
    for ix in batches:
        p=config['protocol'];b=batch(ds,ix,device);assert_complete_future(b,p['horizon']*(len(p['variables']) if p['mode']=='M' else 1))
        output_p=policy(b);experts,target,w,_,_=joints(output_p,b)
        loss=nll(experts,w,target)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite mixed-measure NLL')
        if measure_record is None:
            with torch.no_grad():
                joint,marginal,atomic=observation_logp(experts,w,target)
                sampled=experts[0].sample(4)
                sample_lp=action_marginal_logp(experts,w,sampled)
                means=torch.stack([e.mean for e in experts]);variances=torch.stack([e.variance for e in experts])
                assert torch.isfinite(means).all() and torch.isfinite(variances).all()
                assert (variances>=0).all() and torch.isfinite(sample_lp).all()
                measure_record=dict(native_legacy_nll=float(mixture_nll(output_p,b)),
                    legal_joint_nll_per_position=float((-joint/target.shape[-1]).mean()),
                    legal_marginal_nll=float(-marginal.mean()),atomic_label_fraction=float(atomic.float().mean()),
                    sample_marginal_logp_range=[float(sample_lp.min()),float(sample_lp.max())],
                    means_finite=True,variances_finite=True,common_lattice=True,
                    probability_version='mixed-measure-marginal-v1')
        (loss*len(ix)/n).backward();nll_total+=float(loss.detach())*len(ix)/n
    grad=checked_step(policy,optimizer);change,frozen=relative_update(policy,initial)
    assert frozen==0
    save_state(out/'nll_resume.pt',policy,optimizer,config,dict(phase='nll_probe',global_step=1,input_count=n))
    saved=load_checkpoint(out/'nll_resume.pt',map_location='cpu',weights_only=False)
    cursor,_=restore_state(out/'nll_resume.pt',policy,optimizer,config)
    assert cursor['global_step']==1 and all(torch.equal(v.cpu(),saved['policy_state'][k]) for k,v in policy.state_dict().items())
    records.append(dict(phase='nll',global_step=1,loss=nll_total,relative_parameter_update=change,
                        frozen_max_change=frozen,complete_state_reload=True,**grad))
    atomic_json(out/'progress.json',dict(records=records,measure=measure_record))
    # Second independent optimizer trajectory: restore original weights AND RNG.
    policy.load_state_dict(initial_state,strict=True);seed_all(seed)
    optimizer=optimizer_for(policy,config['lr'])
    rollouts=[]
    for ix in batches:
        b=batch(ds,ix,device);rollouts.append(to_cpu(frozen_rollout(policy,b)))
    for update in range(2):
        optimizer.zero_grad(set_to_none=True);summary={};component_gradients=None
        for ix,stored in zip(batches,rollouts):
            b=batch(ds,ix,device);data=to_device(stored,device)
            loss,diag,rl_term,anchor=position_loss(policy,b,data,config['probe_anchor_coefficient'])
            if update==0 and component_gradients is None:
                pars=[p for p in policy.parameters() if p.requires_grad]
                component_gradients=dict(rl_gradient_norm=gradient_norm(rl_term,pars),
                                         nll_gradient_norm=gradient_norm(anchor,pars),
                                         calibration_scope='first fixed training microbatch, not validation performance')
                assert diag['max_abs_position_log_ratio']<1e-5 and abs(diag['reference_kl'])<1e-6
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite probe RL loss')
            (loss*len(ix)/n).backward()
            for k,v in diag.items():summary[k]=max(summary.get(k,float('-inf')),v) if k.startswith('max_') else summary.get(k,0)+v*len(ix)/n
        grad=checked_step(policy,optimizer);change,frozen=relative_update(policy,initial)
        assert frozen==0
        records.append(dict(phase='position_rl_anchored_probe',global_step=update+1,
                            relative_parameter_update=change,frozen_max_change=frozen,
                            component_gradients=component_gradients,**summary,**grad))
        save_state(out/'rl_resume.pt',policy,optimizer,config,
                   dict(phase='rl_probe',global_step=update+1,input_count=n,rollout_complete=update==1),
                   extra=dict(rollouts=rollouts,indices=indices))
        atomic_json(out/'progress.json',dict(records=records,measure=measure_record))
    usable=all(r['relative_parameter_update']<=1e-3 and r['gradient_norm']>0 and
               r.get('max_joint_log_ratio',0)<=80 and r.get('reference_kl',0)<=1 for r in records)
    result=dict(status='PASS',usable=usable,meaning='Completed train-only calibration; PASS is not a prediction improvement',
        config=config,seed=seed,optimizer_trajectories=2,optimizer_steps=3,independent_inputs=n,
        records=records,measure=measure_record,actual_gpu=torch.cuda.get_device_name(),
        allocated_peak_bytes=torch.cuda.max_memory_allocated(),reserved_peak_bytes=torch.cuda.max_memory_reserved(),
        seconds=time.time()-began,validation_or_test_used=False)
    atomic_json(out/'result.json',result);return result
