"""Complete frozen training prepass, indexed by absolute label time and variable."""
from .io_utils import load_checkpoint
import json
from pathlib import Path
import time
import numpy as np
import torch
from .runtime import load_policy,seed_all,atomic_json
from .policy import FIELDS,joints
from .data import structured,assert_complete_future
from .estimator import DLinear
from .selective import absolute_time_statistics,finalize_uncertainty,dual_mask,retention_matched_random_mask


def save_array(path, array):
    # Shared worker filesystem may reject dirty mmap pages (SIGBUS). Bounded RAM
    # arrays plus ordinary atomic writes avoid depending on filesystem mmap support.
    temporary=path.with_name(path.name+'.partial')
    with temporary.open('wb') as stream:
        np.save(stream,array,allow_pickle=False)
        stream.flush()
    temporary.replace(path)
    loaded=np.load(path,allow_pickle=False)
    assert loaded.shape==array.shape and loaded.dtype==array.dtype
    assert np.array_equal(loaded,array,equal_nan=True), 'Selective cache write/reload mismatch'


def build(config_path,output,device='cuda',seed=1):
    from uni2ts.model.moirai import MoiraiForecast
    config=json.loads(Path(config_path).read_text());p=config['protocol'];out=Path(output);out.mkdir(parents=True,exist_ok=True)
    seed_all(seed);began=time.time();policy=load_policy(config,device).eval().requires_grad_(False)
    versions={name:param._version for name,param in policy.named_parameters()}
    estimator=DLinear(p['context'],p['horizon']).to(device)
    saved=load_checkpoint(config['estimator_checkpoint'],map_location='cpu',weights_only=False)
    assert saved['config']['protocol']['context']==p['context'] and saved['config']['protocol']['horizon']==p['horizon']
    estimator.load_state_dict(saved['model_state'],strict=True);estimator.eval().requires_grad_(False)
    root=Path(config['artifact']);values=torch.tensor(np.load(root/'normalized.npy',allow_pickle=False)[:p['train_end']],device=device,dtype=torch.float32)
    assert torch.isfinite(values).all() and values.shape==(p['train_end'],7)
    indices=np.load(root/'train_indices.npy',allow_pickle=False);total=len(indices);assert total==p['windows']['train']
    dim=7 if p['mode']=='M' else 1;h=p['horizon'];context=p['context']
    expected_origins=np.repeat(np.arange(context,p['train_end']-h+1),7 if dim==1 else 1)
    assert np.array_equal(indices[:,0],expected_origins)
    if dim==1:assert np.array_equal(indices[:,1],np.tile(np.arange(7),len(expected_origins)//7))
    forecast=MoiraiForecast(module=policy.module,context_length=context,patch_size=p['patch'],prediction_length=h,
        num_samples=100,target_dim=dim,feat_dynamic_real_dim=0,past_feat_dynamic_real_dim=0)
    assert total*h*dim*14 < 2*1024**3, 'Selective cache exceeds preregistered RAM cap'
    ref=np.empty((total,h,dim),dtype=np.float32)
    est=np.empty((total,h,dim),dtype=np.float32)
    stats={k:torch.zeros(p['train_end'],7,device=device,dtype=torch.float64) for k in ['count','sum','sum_square']}
    def tensors(begin,end):
        ids=torch.as_tensor(indices[begin:end],device=device)
        times=ids[:,0,None]+torch.arange(h,device=device)
        past_time=ids[:,0,None]+torch.arange(-context,0,device=device)
        if dim==7:
            x=values[past_time];y=values[times];variables=torch.arange(7,device=device)[None,None].expand(len(ids),h,7)
        else:
            x=values[past_time,ids[:,1,None]][...,None];y=values[times,ids[:,1,None]][...,None]
            variables=ids[:,1,None,None].expand(len(ids),h,1)
        return x,y,times[...,None].expand_as(y),variables
    with torch.no_grad():
        for begin in range(0,total,64):
            end=min(total,begin+64);past,target,times,variables=tensors(begin,end)
            packed=forecast._convert(p['patch'],past,torch.ones_like(past,dtype=torch.bool),torch.zeros(past.shape[:2],device=device,dtype=torch.bool),
                future_target=target,future_observed_target=torch.ones_like(target,dtype=torch.bool),future_is_pad=torch.zeros(target.shape[:2],device=device,dtype=torch.bool))
            b={k:v for k,v in zip(FIELDS[:-1],packed)};b['patch_size']=torch.ones_like(b['time_id'])*p['patch']
            assert_complete_future(b,h*dim)
            experts,truth,w,_,_=joints(policy(b),b)
            assert torch.equal(structured(truth,dim,h),target)
            prediction=(torch.stack([structured(e.mean,dim,h) for e in experts],1)*w[:,:,None,None]).sum(1)
            prediction_estimator=estimator(past)
            if not torch.isfinite(prediction).all() or not torch.isfinite(prediction_estimator).all():raise FloatingPointError('Nonfinite frozen predictor')
            ref[begin:end]=prediction.cpu().numpy();est[begin:end]=prediction_estimator.cpu().numpy()
            partial=absolute_time_statistics((prediction-target).abs(),times,variables,torch.ones_like(target,dtype=torch.bool),p['train_end'],7)
            for k in stats:stats[k]+=partial[k]
            if begin%4096==0:print(json.dumps(dict(event='selective_prepass',windows=end,total=total,seconds=time.time()-began)),flush=True)
    save_array(out/'reference_prediction.npy',ref);save_array(out/'estimator_prediction.npy',est)
    uncertainty,unc_diag=finalize_uncertainty(stats,rejection=.1,min_observations=2)
    selected=np.empty((total,h,dim),dtype=np.bool_)
    soft=np.empty((total,h,dim),dtype=np.float32)
    fallback_count=0;peak_keep=torch.zeros(7,device=device);peak_total=torch.zeros_like(peak_keep)
    change_keep=torch.zeros_like(peak_keep);change_total=torch.zeros_like(peak_keep)
    threshold=torch.quantile(values.abs(),.95,dim=0);change_threshold=torch.quantile(values.diff(dim=0).abs(),.95,dim=0)
    coverage_sum=torch.zeros(h,7,device=device);coverage_total=torch.zeros_like(coverage_sum)
    for begin in range(0,total,256):
        end=min(total,begin+256);_,target,times,variables=tensors(begin,end)
        pred=torch.tensor(np.asarray(ref[begin:end]),device=device);pred_est=torch.tensor(np.asarray(est[begin:end]),device=device)
        observed=torch.ones_like(target,dtype=torch.bool);u=uncertainty[times,variables]
        mask,diagnostic=dual_mask(pred,target,pred_est,observed,u,rejection=.1,min_coverage=.5)
        selected[begin:end]=mask.cpu().numpy();soft[begin:end]=(.2+.8*mask.float()).cpu().numpy();fallback_count+=int(diagnostic['fallback'].sum())
        peak=target.abs()>=threshold[variables]
        jump=target.diff(dim=1).abs()>=change_threshold[variables[:,1:]]
        for v in range(7):
            membership=variables==v
            coverage_sum[:,v]+=(mask&membership).sum((0,2));coverage_total[:,v]+=membership.sum((0,2))
            peak_total[v]+=(peak&membership).sum();peak_keep[v]+=(peak&membership&mask).sum()
            change_total[v]+=(jump&membership[:,1:]).sum();change_keep[v]+=(jump&membership[:,1:]&mask[:,1:]&mask[:,:-1]).sum()
    save_array(out/'selected.npy',selected);save_array(out/'soft_weight.npy',soft)
    random_mask=np.empty((total,h,dim),dtype=np.bool_)
    for v in range(7):
        rows=np.arange(total) if dim==7 else np.flatnonzero(indices[:,1]==v)
        mask=torch.tensor(np.asarray(selected[rows,:,v:v+1] if dim==7 else selected[rows]),device=device)
        generator=torch.Generator(device=device).manual_seed(seed+7919+v)
        randomized=retention_matched_random_mask(mask,torch.ones_like(mask),generator)
        assert torch.equal(mask.sum(0),randomized.sum(0))
        if dim==7:random_mask[rows,:,v:v+1]=randomized.cpu().numpy()
        else:random_mask[rows]=randomized.cpu().numpy()
    save_array(out/'random_selected.npy',random_mask);save_array(out/'indices.npy',indices)
    np.savez_compressed(out/'absolute_time_statistics.npz',**{k:v.cpu().numpy() for k,v in stats.items()},
         variance=unc_diag['variance'].cpu().numpy(),uncertainty_keep=uncertainty.cpu().numpy(),insufficient=unc_diag['insufficient'].cpu().numpy())
    assert all(not param.requires_grad and param.grad is None and param._version==versions[name] for name,param in policy.named_parameters())
    assert all(not param.requires_grad and param.grad is None for param in estimator.parameters())
    result=dict(status='PASS',optimizer_trajectories=0,config=config,artifact=str(out),windows=total,
       reference_statistic='frozen warmup mixture analytic mean',uncertainty='population variance of absolute residual at true timestamp/variable',
       uncertainty_thresholds=unc_diag['thresholds'],hard_definition='uncertainty AND anomaly; code-consistent lower .1 quantile anomaly rule',
       soft_definition='.2+.8*hard_mask; full joint true-label NLL remains unmasked',
       fallback_window_variables=fallback_count,minimum_window_variable_coverage=.5,
       retention_by_horizon_variable=(coverage_sum/coverage_total.clamp_min(1)).cpu().tolist(),
       peak_retention=(peak_keep/peak_total.clamp_min(1)).cpu().tolist(),peak_counts=peak_total.cpu().tolist(),
       jump_pair_retention=(change_keep/change_total.clamp_min(1)).cpu().tolist(),jump_pair_counts=change_total.cpu().tolist(),
       random_control_exact_horizon_variable_counts=True,reference_parameters_unchanged=True,
       cache_io='bounded RAM arrays; ordinary atomic writes and exact reload assertions; no output mmap',
       train_only_statistics=True,validation_test_mask='all observed labels, never these training masks',
       seconds=time.time()-began,actual_gpu=torch.cuda.get_device_name(),rare_mode_retention='pending join with frozen neighbor cluster labels; not used to choose this mask')
    atomic_json(out/'result.json',result);return result
