"""One-epoch simultaneous preparation components; exactly one true-label anchor."""
import json,time
from pathlib import Path
import numpy as np
import torch
from .preparation import Preparation,LABEL_ARMS,TEACHER_ARMS


class CompositePreparation:
    def __init__(self,config,device):
        if float(config.get('auxiliary_coefficient',1.))!=1.:raise ValueError('Composite coefficients are already applied inside the bundle')
        components=config['preparation_components']
        if not 1<=len(components)<=2:raise ValueError('Finite recipes allow at most two preparation components')
        self.parts=[];seen=set()
        for component in components:
            arm=component['arm'];coefficient=float(component['coefficient'])
            if arm not in LABEL_ARMS+TEACHER_ARMS or arm in seen or not 0<coefficient<=.1:
                raise ValueError('Invalid or repeated qualified preparation component')
            seen.add(arm);local=dict(config,preparation_arm=arm)
            self.parts.append((arm,coefficient,Preparation(local,device)))
        if len(seen & set(TEACHER_ARMS))>1 or len(seen & {'E04','E05'})>1 or len(seen & {'E09','E10','E11'})>1:
            raise ValueError('Conflicting controls or duplicate mechanisms cannot form a recipe')

    @property
    def checked_windows(self):return max(p.checked_windows for _,_,p in self.parts)

    @property
    def maximum_target_difference(self):return max(p.maximum_target_difference for _,_,p in self.parts)

    def batch_context(self,indices):return {arm:part.batch_context(indices) for arm,_,part in self.parts}

    def auxiliary(self,b,experts,target,weights,context=None):
        if context is None:context=self.batch_context(b['_index'].detach().cpu().tolist())
        total=weights.sum()*0.;records=[]
        for arm,coefficient,part in self.parts:
            loss,diagnostic=part.auxiliary(b,experts,target,weights,context[arm])
            total=total+coefficient*loss
            records.append(dict(arm=arm,coefficient=coefficient,**diagnostic))
        return total,dict(composite_auxiliary=float(total.detach()),components=records,
            true_label_anchor='Not included here; trainer adds it once',preparation_epochs=1)


def audit(config_path,output,device='cuda',seed=1):
    from .runtime import seed_all,load_policy,atomic_json
    from .data import datasets,batch
    from .policy import joints
    from .probability import nll,observation_logp
    from .preparation_audit import gradient,norm
    config=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    seed_all(seed);began=time.time();policy=load_policy(config,device).eval()
    parameters=[p for p in policy.parameters() if p.requires_grad];versions={n:p._version for n,p in policy.named_parameters()}
    cache=np.load(Path(config['neighbor_artifact'])/'neighbors.npz',allow_pickle=False);eligible=np.flatnonzero(cache['confidence']>0)
    assert len(eligible)>=8;indices=eligible[np.linspace(0,len(eligible)-1,8,dtype=int)].tolist()
    data=datasets(policy,config['dataset'],config['protocol'])['train'];b=batch(data,indices,device)
    experts,target,w,_,_=joints(policy(b),b);anchor=nll(experts,w,target)
    composite=CompositePreparation(config,device);extra,diag=composite.auxiliary(b,experts,target,w,composite.batch_context(indices))
    combined=anchor+extra;actual=gradient(combined,parameters)
    expected_g=gradient(anchor,parameters);independent=[]
    # Analytic target: confidence-weighted router CE and globally weighted position
    # marginal NLL. The complete trajectory anchor appears only once.
    for arm,coefficient,plugin in composite.parts:
        if arm=='E04':
            q=torch.as_tensor(cache['q'][indices],device=device,dtype=w.dtype);confidence=torch.as_tensor(cache['confidence'][indices],device=device,dtype=w.dtype)
            component=-(confidence[:,None]*q*w.log()).sum()/len(indices)
        elif arm=='E10':
            from .data import structured
            _,lp,_=observation_logp(experts,w,target)
            points=structured(-lp,plugin.dim,plugin.h);weights=torch.as_tensor(plugin.masks['soft'][indices],device=device,dtype=points.dtype)
            component=(points*weights).sum()/weights.sum()
        else:raise ValueError('Qualification target is the preregistered weak-label plus soft-selection bundle')
        g=gradient(component,parameters);expected_g=[x+coefficient*y for x,y in zip(expected_g,g)]
        independent.append(dict(arm=arm,coefficient=coefficient,gradient_norm=norm(g)))
    difference=norm([a-e for a,e in zip(actual,expected_g)])/norm(expected_g)
    assert difference<1e-5 and norm(actual)>0
    assert all(p._version==versions[n] for n,p in policy.named_parameters())
    result=dict(status='PASS',config=config,optimizer_trajectories=0,optimizer_steps=0,
        actual_gradient_norm=norm(actual),relative_gradient_difference=difference,independent_components=independent,
        anchor_count=1,parameters_unchanged=True,training_only=True,test_used=False,diagnostic=diag,
        actual_gpu=torch.cuda.get_device_name(),seconds=time.time()-began,
        scope='Mathematical composition and gradient routing only; no recipe effectiveness selection')
    atomic_json(out/'result.json',result);return result
