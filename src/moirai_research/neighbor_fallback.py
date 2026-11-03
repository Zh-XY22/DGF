"""Restore the specified real-label fallback when strict-window coverage fails.

This does not turn approximate past-neighbor labels into independent conditional
futures. Ranking and the positive/shuffled pair remain unchanged where admitted.
"""
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .runtime import atomic_json,load_policy,seed_all
from .data import datasets,batch
from .policy import joints
from .probability import nll
from .preparation import Preparation
from .preparation_audit import gradient,norm


def apply_fallback(config_path,output,device='cuda',seed=1):
    config=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    seed_all(seed);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;began=time.time()
    source=Path(config['neighbor_artifact']);cache=dict(np.load(source/'neighbors.npz',allow_pickle=False))
    p=config['protocol'];h=p['horizon'];context=p['context']
    times=cache['query_origins'];fold=cache['fold'];count=np.zeros_like(times)
    for value in np.unique(fold):
        if value<0:continue
        origin=source/f'fold_{int(value):03d}.npz'
        with np.load(origin,allow_pickle=False) as record:
            library=record['library_origins'];assert np.all(library+h<=p['train_end'])
            selected=fold==value
            # Earlier neighbor's whole future, a one-horizon gap, then query
            # history. Candidate-to-candidate spacing also covers both histories.
            last=np.minimum(times[selected]-context-2*h,library.max())
            count[selected]=np.where(last>=context,(last-context)//(context+2*h)+1,0)
        shutil.copy2(origin,out/origin.name)
    np.testing.assert_array_equal(count,cache['strict_nonoverlap_candidate_count'])
    original=cache['confidence'].copy();enough=count>=2
    cache['confidence']=np.where(enough,original,0.)
    assert np.all(cache['confidence'][~enough]==0) and np.array_equal(cache['confidence'][enough],original[enough])
    np.savez_compressed(out/'neighbors.npz',**cache)
    policy=load_policy(config,device).eval();params=[v for v in policy.parameters() if v.requires_grad]
    versions={name:v._version for name,v in policy.named_parameters()}
    data=datasets(policy,config['dataset'],p)['train'];assert len(data)==len(times)
    fallback=np.flatnonzero(~enough);assert len(fallback)>0
    ids=fallback[np.linspace(0,len(fallback)-1,min(4,len(fallback)),dtype=int)].tolist()
    b=batch(data,ids,device);experts,target,w,_,_=joints(policy(b),b)
    anchor=nll(experts,w,target);base=gradient(anchor,params);assert norm(base)>0
    proofs=[]
    for arm in ['E04','E05']:
        local=dict(config,preparation_arm=arm,neighbor_artifact=str(out))
        plugin=Preparation(local,device)
        auxiliary,diag=plugin.auxiliary(b,experts,target,w,plugin.batch_context(ids))
        extra=gradient(auxiliary,params);combined=gradient(anchor+config['auxiliary_coefficient']*auxiliary,params)
        assert float(auxiliary.detach())==0 and norm(extra)==0
        maximum=max(float((a-z).abs().max()) for a,z in zip(base,combined))
        assert maximum==0
        proofs.append(dict(arm=arm,auxiliary_value=0.,auxiliary_gradient_norm=0.,anchor_gradient_norm=norm(base),
            combined_gradient_maximum_difference=maximum,parameters_unchanged=True,diagnostic=diag))
    assert all(v._version==versions[name] for name,v in policy.named_parameters())
    coverage=float((cache['confidence']>0).mean());all_fallback=not bool((cache['confidence']>0).any())
    result=dict(status='PASS',config=config,artifact=str(out/'neighbors.npz'),version='past-neighbor-strict-availability-fallback-v2',
        optimizer_trajectories=0,optimizer_steps=0,windows=len(times),previous_positive_confidence_windows=int((original>0).sum()),
        admitted_positive_confidence_windows=int((cache['confidence']>0).sum()),confidence_coverage=coverage,
        strict_candidate_availability_coverage=float(enough.mean()),real_label_fallback_windows=int((~enough).sum()),
        positive_and_shuffled_share_identical_fallback=True,proofs=proofs,probe_indices=ids,
        all_windows_real_label_only=all_fallback,ordinary_NLL_semantic_alias_valid=all_fallback,
        alias_scope='When every confidence is zero the added auxiliary and its full parameter gradient vanish for all inputs; same data order, NLL and optimizer define the ordinary preparation',
        strict_nonoverlap_applies_to='Availability gate only; admitted retrieval still follows declared causal past-label rule with overlapping histories',
        independence_claim=False,validation_or_test_used=False,actual_gpu=torch.cuda.get_device_name(),seconds=time.time()-began)
    atomic_json(out/'result.json',result);return result
