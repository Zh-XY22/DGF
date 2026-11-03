"""Frozen training-window gradient directions; never validation or test choice."""
from pathlib import Path
import json,math,gc
import torch
from .runtime import load_policy,seed_all,atomic_json
from .data import datasets,batch,structured,forecast_dimensions
from .policy import joints,expert_log_prob
from .rl import make_rollout
from .probability import legacy_clipped_kl
from .rewards import ppo_surrogate

def gradients(loss,parameters,retain=True):
    values=torch.autograd.grad(loss,parameters,allow_unused=True,retain_graph=retain)
    if any(not torch.isfinite(v).all() for v in values if v is not None):raise FloatingPointError('Nonfinite diagnostic gradient')
    return values

def summary(a,b):
    aa=sum(float(v.double().square().sum()) for v in a if v is not None);bb=sum(float(v.double().square().sum()) for v in b if v is not None)
    ab=sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b) if x is not None and y is not None)
    return dict(first_norm=math.sqrt(aa),second_norm=math.sqrt(bb),cosine=None if aa==0 or bb==0 else ab/math.sqrt(aa*bb))

def diagnose(model_root,cases,output,frozen_scale):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);rows=[]
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    for case in cases:
        h=case['horizon'];seed=case['seed'];artifact=Path(case['data']);p=json.loads((artifact/'data_protocol.json').read_text());p.update(builder_root=str(artifact/'processed'),artifact=str(artifact))
        base=dict(dataset='ETTh1',protocol=p,artifact=str(artifact),weights=str(model_root),route='A0',checkpoint_kind='policy_state',
            local_files_only=True,force_fixed_kappa=True,arm='E16',algorithm_arm='E16',nll_anchor_coefficient=0.,J=8,G=4)
        seed_all(198463+seed+h);policy=load_policy(dict(base,checkpoint=case['final']),'cuda').eval();reference=load_policy(dict(base,checkpoint=case['parent']),'cuda').eval().requires_grad_(False)
        data=datasets(policy,'ETTh1',p)['train'];parameters=[v for v in policy.parameters() if v.requires_grad];d=forecast_dimensions(p)
        for number,indices in enumerate([[0,1],[997,998],[2040,2041],[2919,2920]]):
            b=batch(data,indices,'cuda');record=make_rollout(policy,reference,b,base);experts,target,_,_,_=joints(policy(b),b)
            lp=expert_log_prob(experts,record['actions']);current=structured(lp,d,h);old=structured(record['old'],d,h)
            local=-ppo_surrogate(current,old,record['advantage'],.2).mean()
            reward=-(structured(record['actions']['forecast'],d,h).double()-structured(target,d,h)[:,None,None].double()).square().mean((-2,-1))
            advantage=((reward-reward.mean(2,keepdim=True))/frozen_scale).detach()
            global_loss=-(advantage*lp.double().sum(-1)).mean()
            regularizer=legacy_clipped_kl(lp,record['reference_lp']).mean()
            local_grad=gradients(local,parameters);global_grad=gradients(global_loss,parameters);kl_grad=gradients(regularizer,parameters,retain=False)
            credit=summary(local_grad,global_grad);constraint=summary(local_grad,kl_grad)
            std=reward.std(2,correction=0);low=float((std<1e-8).double().mean())
            rows.append(dict(horizon=h,seed=seed,measurement=number,indices=indices,credit_direction=credit,KL_direction=constraint,
                KL_coefficient=.01,weighted_KL_norm=.01*constraint['second_norm'],low_std_fraction=low,
                KL_constraints=constraint['cosine'] is not None and constraint['cosine']<-.5 and .01*constraint['second_norm']>=constraint['first_norm'],
                probability_space='Augmented native-family actions; A0 conditional factors',test_used=False))
            atomic_json(out/'progress.json',dict(rows=rows))
        del policy,reference,local_grad,global_grad,kl_grad;gc.collect();torch.cuda.empty_cache()
    by_h={}
    for h in [96,192,336,720]:
        values=[v['credit_direction']['cosine'] for v in rows if v['horizon']==h and v['credit_direction']['cosine'] is not None]
        by_h[str(h)]=sum(values)/len(values) if values else None
    negative=sum(v is not None and v<0 for v in by_h.values());kl_fraction=sum(v['KL_constraints'] for v in rows)/len(rows)
    result=dict(status='PASS',rows=rows,credit_mean_cosine_by_horizon=by_h,
        Q03_Q04_trigger=negative>=3 and all(v['credit_direction']['first_norm']>0 and v['credit_direction']['second_norm']>0 for v in rows),
        Q05_trigger=len(rows)>=4 and kl_fraction>=.25,KL_constraining_measurement_fraction=kl_fraction,
        optimizer_updates=0,optimizer_trajectories=0,test_used=False,validation_used=False,
        definition='Compare executed position-normalized GRPO gradient against same-action global-reward frozen-scale score gradient in the same parameter space. This is objective-direction diagnosis, not proof that their normalized objectives are identical.',
        sampling='Four fixed common training-window pairs at each final B02 state, both exploration seeds, all H; reference remains each branch warmup parent',
        limitation='Post-training train-only gradient measurements, not a reconstruction of every historical optimizer update; no claim of a true analytic reward gradient or unbiased standardized GRPO')
    atomic_json(out/'result.json',result);return result
