"""Train-only auxiliary gradient scales and cache alignment; zero optimizer updates."""
import json,math,time
from pathlib import Path
import numpy as np
import torch
from .runtime import seed_all,load_policy,atomic_json
from .data import datasets,batch
from .policy import joints
from .probability import nll
from .preparation import Preparation,LABEL_ARMS,TEACHER_ARMS
from .rl import make_rollout,objective


def gradient(loss,parameters):
    g=torch.autograd.grad(loss,parameters,allow_unused=True,retain_graph=True)
    if any(not torch.isfinite(v).all() for v in g if v is not None):raise FloatingPointError('Nonfinite auxiliary gradient')
    return [torch.zeros_like(p) if v is None else v.detach() for p,v in zip(parameters,g)]

def norm(g):return sum(float(v.double().square().sum()) for v in g)**.5


def audit(config_path,output,device='cuda',seed=1):
    config=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    seed_all(seed);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;began=time.time()
    policy=load_policy(config,device).eval();params=[p for p in policy.parameters() if p.requires_grad]
    versions={k:p._version for k,p in policy.named_parameters()};data=datasets(policy,config['dataset'],config['protocol'])['train']
    arms=LABEL_ARMS if config['audit_family']=='labels' else TEACHER_ARMS
    if config['audit_family']=='labels':
        cache=np.load(Path(config['neighbor_artifact'])/'neighbors.npz',allow_pickle=False)
        available=np.flatnonzero(cache['confidence']>0)
        assert len(available)>=8
        ids=available[np.linspace(0,len(available)-1,8,dtype=int)].tolist()
    else:ids=[int(i) for i in np.linspace(len(data)//4,len(data)-1,8,dtype=int)]
    b=batch(data,ids,device);records=[]
    for arm in arms:
        local=dict(config,preparation_arm=arm);plugin=Preparation(local,device)
        experts,target,w,_,_=joints(policy(b),b);main=nll(experts,w,target)
        auxiliary,diagnostic=plugin.auxiliary(b,experts,target,w,plugin.batch_context(ids))
        base_g=gradient(main,params);aux_g=gradient(auxiliary,params)
        base_norm=norm(base_g);aux_norm=norm(aux_g)
        assert base_norm>0 and aux_norm>0
        candidates=[.001,.01,.1] if arm in LABEL_ARMS else [1e-4,1e-3,1e-2]
        measures=[]
        for coefficient in candidates:
            combined=[a+coefficient*b for a,b in zip(base_g,aux_g)]
            ratio=coefficient*aux_norm/base_norm
            measures.append(dict(coefficient=coefficient,aux_to_anchor_gradient_ratio=ratio,
                combined_gradient_norm=norm(combined),usable=math.isfinite(ratio) and 0<ratio<=10))
        row=dict(arm=arm,anchor_gradient_norm=base_norm,auxiliary_gradient_norm=aux_norm,
            anchor=float(main.detach()),candidates=measures,diagnostic=diagnostic,
            aligned_windows=plugin.checked_windows,maximum_target_difference=plugin.maximum_target_difference)
        if arm in ['E09','E10','E11']:
            # Exact global weighted reduction under splitting, independent of GPU GEMM roundoff.
            key={'E09':'hard','E10':'soft','E11':'random'}[arm]
            weights=torch.as_tensor(plugin.masks[key][ids],device=device,dtype=torch.float64)
            points=torch.arange(weights.numel(),device=device,dtype=torch.float64).reshape_as(weights).requires_grad_()
            direct=(weights*points).sum()/weights.sum()
            context=plugin.batch_context(ids);pieces=[]
            for start in range(0,len(ids),2):
                sub=points[start:start+2];sw=weights[start:start+2]
                pieces.append((sub*sw).sum()*context['total_inputs']/(len(sub)*context['weighted_elements'])*len(sub)/len(ids))
            split=sum(pieces);dg=torch.autograd.grad(direct,points,retain_graph=True)[0];sg=torch.autograd.grad(split,points)[0]
            torch.testing.assert_close(dg,sg,atol=1e-14,rtol=1e-14)
            row['global_weighted_microbatch_gradient_error']=float((dg-sg).abs().max())
        records.append(row);atomic_json(out/'progress.json',records)
        print(json.dumps(dict(event='auxiliary_gradient',arm=arm,anchor_gradient_norm=base_norm,auxiliary_gradient_norm=aux_norm)),flush=True)
        del plugin,base_g,aux_g,combined,experts,target,w,main,auxiliary
    local_records=[]
    if config['audit_family']=='labels':
        reference=load_policy(config,device).eval().requires_grad_(False)
        base_cfg=dict(config,arm='E13',nll_anchor_coefficient=config['nll_anchor_coefficient'])
        seed_all(seed+901);old=make_rollout(policy,reference,b,base_cfg);loss,_=objective(policy,b,old,base_cfg);base_g=gradient(loss,params);base_norm=norm(base_g)
        for coefficient in [.001,.01,.1]:
            c=dict(base_cfg,arm='E14',difference_coefficients={'1':coefficient,'2':coefficient})
            seed_all(seed+901);record=make_rollout(policy,reference,b,c);candidate,_=objective(policy,b,record,c);g=gradient(candidate,params)
            change=norm([x-y for x,y in zip(g,base_g)])/base_norm
            local_records.append(dict(coefficient=coefficient,increment_to_base_gradient_ratio=change,combined_gradient_norm=norm(g),usable=0<change<=10))
        plugin=Preparation(dict(config,arm='E15'),device);mask=plugin.reliability(ids)
        c=dict(config,arm='E15');seed_all(seed+902);r=make_rollout(policy,reference,b,c,reliability=mask)
        assert not r['weight'].requires_grad and r['weight'].shape==(len(ids),config['protocol']['horizon'],7 if config['protocol']['mode']=='M' else 1)
        loss,_=objective(policy,b,r,c);assert norm(gradient(loss,params))>0
    assert all(p._version==versions[k] for k,p in policy.named_parameters())
    result=dict(status='PASS',config=config,optimizer_trajectories=0,optimizer_steps=0,records=records,
        difference_candidates=local_records,parameters_unchanged=True,selection_basis='training gradient scale only; no validation or test',
        target_aux_to_anchor_gradient_ratio=.1,coefficient_sharing='E04/E05 share; E09/E10/E11 share; one value across both datasets per family; E06/07/08 separate',
        seed=seed,probe_indices=ids,actual_gpu=torch.cuda.get_device_name(),seconds=time.time()-began)
    atomic_json(out/'result.json',result);return result
