"""Training-future prototypes supervise preparation; the router sees history only."""
from pathlib import Path
import numpy as np
import torch
from .runtime import atomic_json
from .probability import observation_logp

FEATURES=['level','direction','volatility','difference_rms','curvature_rms','turn_fraction','high_frequency_fraction','spectral_centroid']

def build_labels(artifact,output,seed=198401):
    root=Path(artifact);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    protocol=__import__('json').loads((root/'data_protocol.json').read_text())
    values=np.load(root/'normalized.npy',allow_pickle=False)[:protocol['train_end']]
    indices=np.load(root/'train_indices.npy',allow_pickle=False);h=protocol['horizon'];features=[]
    for origin,var in indices:
        assert origin>=protocol['context'] and origin+h<=protocol['train_end']
        history=values[origin-512:origin];future=values[origin:origin+h]
        scale=np.maximum(history.std(axis=0,ddof=0),1e-3)
        z=(future-history.mean(axis=0))/scale;d1=np.diff(z,axis=0);d2=np.diff(d1,axis=0)
        power=np.abs(np.fft.rfft(z-z.mean(axis=0),axis=0))**2;freq=np.fft.rfftfreq(h)
        total=np.maximum(power.sum(axis=0),1e-12)
        features.append([z.mean(),(z[-1]-z[0]).mean(),z.std(axis=0).mean(),np.sqrt((d1*d1).mean()),
            np.sqrt((d2*d2).mean()),((d1[1:]*d1[:-1])<0).mean(),
            (power[freq>=.25].sum(axis=0)/total).mean(),((power*freq[:,None]).sum(axis=0)/total).mean()])
    raw=np.asarray(features,dtype=np.float64);mean=raw.mean(0);scale=np.maximum(raw.std(0,ddof=0),1e-8)
    x=torch.as_tensor((raw-mean)/scale,dtype=torch.float64,device='cuda')
    generator=torch.Generator(device='cuda').manual_seed(seed)
    centers=x[torch.randperm(len(x),generator=generator,device='cuda')[:8]].clone()
    for step in range(40):
        dist=torch.cdist(x,centers).square();assignment=dist.argmin(1)
        next_centers=torch.stack([x[assignment==k].mean(0) if (assignment==k).any() else centers[k] for k in range(8)])
        change=float((next_centers-centers).abs().max());centers=next_centers
        if change<1e-8:break
    # Canonical lexicographic centroid order, fitted only on this H's training labels.
    order=sorted(range(8),key=lambda k:tuple(centers[k].cpu().tolist()));centers=centers[order]
    q=(-torch.cdist(x,centers).square()/2.).softmax(-1)
    assert torch.isfinite(q).all() and bool((q.std(0)>1e-5).any())
    np.savez_compressed(out/'mode_labels.npz',indices=indices,q=q.cpu().numpy(),features=raw,
        centers=centers.cpu().numpy(),feature_mean=mean,feature_scale=scale)
    result=dict(status='PASS',horizon=h,seed=seed,windows=len(indices),features=FEATURES,K=8,temperature=1.,
        clustering_steps=step+1,alignment='Lexicographic standardized training centroid order; across seeds shared; cross-H names are not assumed identical',
        responsibility_entropy=float(-(q*q.clamp_min(1e-300).log()).sum(-1).mean()),
        mean_responsibility=q.mean(0).cpu().tolist(),training_only=True,maximum_label_end=int(indices[:,0].max()+h),
        train_end=protocol['train_end'],future_labels_never_router_inputs=True,
        interpretation='Descriptor-based soft training responsibility, not conditional ambiguity or independent counterfactual labels',optimizer_updates=0,optimizer_trajectories=0)
    atomic_json(out/'result.json',result);return result

class ModeSpecialization:
    def __init__(self,config,device):
        self.config=config;self.device=device;self.checked_windows=0;self.maximum_target_difference=0.
        root=Path(config['artifact']);self.protocol=config['protocol']
        self.indices=np.load(root/'train_indices.npy',allow_pickle=False)
        self.values=np.load(root/'normalized.npy',allow_pickle=False)[:self.protocol['train_end']]
        stored=np.load(Path(config['mode_labels_artifact'])/'mode_labels.npz',allow_pickle=False)
        assert np.array_equal(stored['indices'],self.indices)
        self.q=stored['q'];assert self.q.shape==(len(self.indices),8)
        assert np.isfinite(self.q).all() and np.allclose(self.q.sum(1),1.)
    def batch_context(self,indices):return None
    def auxiliary(self,b,experts,target,weights,context=None,**unused):
        from .data import structured,forecast_dimensions
        ids=b['_index'].detach().cpu().numpy();pairs=self.indices[ids];h=self.protocol['horizon']
        truth=np.stack([self.values[t:t+h] for t,v in pairs]);actual=structured(target,forecast_dimensions(self.protocol),h)
        expected=torch.as_tensor(truth,device=target.device,dtype=target.dtype)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
        self.checked_windows+=len(ids);self.maximum_target_difference=max(self.maximum_target_difference,float((actual-expected).abs().max()))
        q=torch.as_tensor(self.q[ids],device=target.device,dtype=weights.dtype).detach()
        scores=[]
        for k,expert in enumerate(experts):
            unit=weights.new_ones((len(ids),1));joint,_,_=observation_logp([expert],unit,target);scores.append(-joint/target.shape[-1])
        expert_loss=(q*torch.stack(scores,-1)).sum(-1).mean()
        router_loss=-(q*weights.log()).sum(-1).mean();loss=expert_loss+router_loss
        return loss,dict(preparation_arm='D01',auxiliary=float(loss.detach()),router_soft_CE=float(router_loss.detach()),
            responsibility_weighted_joint_NLL=float(expert_loss.detach()),router_input='History representations only',
            auxiliary_measure='Combined mode-specialization preparation; full trajectory mixture NLL retained separately',
            label_entropy=float(-(q*q.clamp_min(1e-30).log()).sum(-1).mean()),responsibility_variation=float(q.std(0,correction=0).mean()))
