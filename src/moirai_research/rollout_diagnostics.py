"""Fixed-training-window group diagnostics; never reconstruct missing training logs."""
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from .runtime import seed_all,load_policy,atomic_json
from .data import datasets,batch,structured
from .policy import joints
from .rewards import relative,difference_terms


def group_statistics(reward):
    """Empirical [N,J,G] population variances, not independent-sample uncertainty."""
    if reward.ndim!=3:raise ValueError('Group reward requires [N,J,G]')
    x=reward.detach().double()
    if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite diagnostic reward')
    within=x.var(2,correction=0);between=x.mean(2).var(1,correction=0)
    total=x.flatten(1).var(1,correction=0)
    return dict(mean=float(x.mean()),within_G_variance_mean=float(within.mean()),
        group_mean_variance_across_J=float(between.mean()),all_candidate_variance_mean=float(total.mean()),
        variance_decomposition_maximum_error=float((total-within.mean(1)-between).abs().max()),
        zero_std_subgroup_fraction=float((within==0).double().mean()),
        std_G_quantiles=torch.quantile(within.sqrt().flatten(),torch.tensor([0.,.25,.5,.95,1.],device=x.device,dtype=x.dtype)).cpu().tolist(),
        correction=0,layout='N,J,G')


def numerical_checks(device):
    x=torch.tensor([[[0.,2.],[4.,6.]]],device=device,dtype=torch.float64)
    r=group_statistics(x)
    assert r['within_G_variance_mean']==1 and r['group_mean_variance_across_J']==4
    assert r['all_candidate_variance_mean']==5 and r['variance_decomposition_maximum_error']==0
    constant=group_statistics(torch.zeros((2,8,4),device=device))
    assert constant['all_candidate_variance_mean']==0 and constant['zero_std_subgroup_fraction']==1
    count=torch.tensor([224,224,224],device=device)
    weight=count/count.sum();values=torch.tensor([1.,2.,4.],device=device,dtype=torch.float64)
    executed=(values*weight).sum();ideal=values.mean()
    assert weight.dtype==torch.float32 and executed!=ideal
    return dict(population_variance_decomposition='PASS',constant_groups='PASS',optimizer_steps=0,
        V16_reduction_dtype_regression='PASS: preserve executed weights; record algebraic ideal separately',
        block_weight_dtype=str(weight.dtype),ideal_vs_executed_example_difference=float((executed-ideal).abs()))


@torch.no_grad()
def diagnose(policy,config,output,device='cuda',seed=1):
    from .rl import make_rollout
    from .preparation import Preparation
    out=Path(output);out.mkdir(parents=True,exist_ok=True);began=time.time()
    p=config['protocol'];h=p['horizon'];variables=len(p['variables']);dim=variables if p['mode']=='M' else 1
    data=datasets(policy,config['dataset'],p)['train']
    if p['mode']=='S':
        origins=np.linspace(0,len(data)//variables-1,4,dtype=int)
        indices=[int(origin)*variables+variable for origin in origins for variable in range(variables)]
    else:indices=np.linspace(0,len(data)-1,min(int(p.get('diagnostic_input_limit',32)),len(data)),dtype=int).tolist()
    versions={name:param._version for name,param in policy.named_parameters()}
    was_training=policy.training;policy.eval()
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    try:
        seed_all(seed+131071);packed=batch(data,indices,device)
        objective=dict(config,arm=config.get('algorithm_arm',config.get('arm','E03')))
        reliability=Preparation(objective,device).reliability(indices) if objective['arm']=='E15' else None
        record=make_rollout(policy,policy,packed,objective,reliability)
        _,target,w,alpha,_=joints(policy(packed),packed)
        prediction=structured(record['actions']['forecast'],dim,h).double();truth=structured(target,dim,h).double()
        error=-(prediction-truth[:,None,None]).square();weight=record['weight']
        accuracy=(error*weight[:,None,None]).sum((-2,-1))/weight.sum((-2,-1))[:,None,None]
        torch.testing.assert_close(accuracy.mean(),accuracy.new_tensor(record['reward_mean']),rtol=1e-12,atol=1e-12)
        full=accuracy.clone()
        block_arithmetic_delta=None
        if 'blocks' in record:
            # Preserve the executed integer-count division dtype and reduction.
            # Comparing ideal global MSE against a float32 1/3 block weight is
            # a different arithmetic path, even though the algebra is equal.
            blocks=record['blocks']['blocks'];observed=record['observed']
            count=torch.stack([observed[:,lo:hi].sum((-2,-1)) for lo,hi in blocks],-1)
            weights=count/count.sum(-1,keepdim=True).clamp_min(1)
            reward=torch.stack([error[...,lo:hi,:].sum((-2,-1))/count[:,b,None,None].clamp_min(1)
                for b,(lo,hi) in enumerate(blocks)],-1)
            full=(reward*weights[:,None,None]).sum(-1)
            torch.testing.assert_close(full,record['blocks']['total_reward'],rtol=1e-12,atol=1e-12)
            block_arithmetic_delta=float((full-accuracy).abs().max())
        if objective['arm']=='E14':
            for order in [1,2]:
                local,valid=difference_terms(prediction,truth,record['observed'],order)
                full+=objective['difference_coefficients'][str(order)]*local.sum((-2,-1))/valid.sum((-2,-1)).clamp_min(1)[:,None,None]
        torch.testing.assert_close(relative(full),record['mode_advantage'],rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(relative(full.mean(2),axis=1),record['dir_advantage'],rtol=1e-10,atol=1e-10)
        assert all(param._version==versions[name] and param.grad is None for name,param in policy.named_parameters())
        result=dict(status='PASS',scope='Fixed training inputs at this checkpoint; not retrospective training-curve reconstruction or another formal evaluation RNG repeat',
            seed=seed,diagnostic_rng_rule='seed+131071, restored on exit; identical cohort and rule for every readout',
            independent_inputs=len(indices),indices=indices,original_indices=packed['_index'].cpu().tolist(),
            J=8,G=4,K=len(w[0]),accuracy_reward=group_statistics(accuracy),full_router_reward=group_statistics(full),
            block_total_vs_global_accuracy_maximum_arithmetic_difference=block_arithmetic_delta,
            position_accuracy_reward_variance_G=float(error.var(2,correction=0).mean()),
            sample_forecast_variance_G=float(prediction.var(2,correction=0).mean()),
            kappa_range=None if alpha is None else [float(alpha.sum(-1).min()),float(alpha.sum(-1).max())],
            optimizer_trajectories=0,optimizer_steps=0,parameters_unchanged=True,test_used=False,
            actual_gpu=torch.cuda.get_device_name(),seconds=time.time()-began,
            limitations=['Empirical small-cohort group variances are not predictive calibration or independent-sample confidence intervals',
                'Sample dispersion is distinct from distances between mode distributions',
                'NLL controls receive hypothetical same-budget rollout diagnostics, not claimed RL training rewards'])
        atomic_json(out/'result.json',result);return result
    finally:
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
        policy.train(was_training)


def audit(config_path,output,device='cuda',seed=1):
    config=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    checks=numerical_checks(device);cases=[];began=time.time()
    for case in config['cases']:
        policy=load_policy(case['config'],device).eval()
        value=diagnose(policy,case['config'],out/case['case_id'],device,seed)
        cases.append(dict(case_id=case['case_id'],status=value['status'],receipt=str(out/case['case_id']/'result.json')))
        del policy;torch.cuda.empty_cache()
    result=dict(status='PASS',numerical_checks=checks,cases=cases,optimizer_trajectories=0,optimizer_steps=0,
        actual_gpu=torch.cuda.get_device_name(),seconds=time.time()-began,
        scope='Group variance/constant-reward checks and actual fixed-input rollout consequences; no parameter update, validation selection or test')
    atomic_json(out/'result.json',result);return result
