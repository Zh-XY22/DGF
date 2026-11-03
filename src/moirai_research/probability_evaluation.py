"""Streamed distribution evaluation, separate from all three point decisions."""
import json
from pathlib import Path
import numpy as np
import torch
from .runtime import atomic_json,load_policy,seed_all
from .data import structured,assert_complete_future,forecast_dimensions
from .policy import joints,rollout,FIELDS
from .crps_math import probability_metrics
from .probability import observation_logp


def evaluate_distribution(policy,protocol,split,artifact,output=None,samples=1000,evaluation_seed=198367,input_block=4,limit=None):
    if samples not in (100,1000):raise ValueError('Only the two frozen distribution budgets are supported')
    if input_block<1:raise ValueError('Invalid streaming input block')
    device=next(policy.parameters()).device;h=protocol['horizon'];d=forecast_dimensions(protocol)
    if protocol['mode']!='M':raise ValueError('Four-H probability entry currently qualifies M organization only')
    root=Path(artifact);mean=torch.as_tensor(np.load(root/'scaler_mean.npy'),device=device,dtype=torch.float64)
    std=torch.as_tensor(np.load(root/'scaler_std.npy'),device=device,dtype=torch.float64)
    indices=np.load(root/(split+'_indices.npy'),allow_pickle=False)
    values=np.load(root/'normalized.npy',allow_pickle=False,mmap_mode='r')
    total=len(indices) if limit is None else min(len(indices),limit)
    lower,upper=(protocol['train_end'],protocol['val_end']) if split=='val' else (protocol['val_end'],protocol['test_end'])
    assert len(indices)==protocol['windows'][split] and np.all(indices[:,0]>=lower) and np.all(indices[:,0]+h<=upper)
    from uni2ts.model.moirai import MoiraiForecast
    forecast=MoiraiForecast(module=policy.module,context_length=protocol['context'],patch_size=protocol['patch'],
        prediction_length=h,num_samples=samples,target_dim=d,feat_dynamic_real_dim=0,past_feat_dynamic_real_dim=0)
    totals={'standardized':{},'original':{}};nll_sum=0.;nll_valid=True;processed=0;was_training=policy.training;policy.eval()
    per_window=[];issues=[]
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.no_grad(),torch.random.fork_rng(devices=devices):
        for start in range(0,total,input_block):
            ids=list(range(start,min(start+input_block,total)));origins=indices[ids,0]
            past=torch.tensor(np.asarray([values[i-protocol['context']:i] for i in origins]),device=device,dtype=torch.float32)
            truth=torch.tensor(np.asarray([values[i:i+h] for i in origins]),device=device,dtype=torch.float32)
            packed=forecast._convert(protocol['patch'],past,torch.ones_like(past,dtype=torch.bool),
                torch.zeros(past.shape[:2],device=device,dtype=torch.bool),future_target=truth,
                future_observed_target=torch.ones_like(truth,dtype=torch.bool),future_is_pad=torch.zeros(truth.shape[:2],device=device,dtype=torch.bool))
            b={k:v for k,v in zip(FIELDS[:-1],packed)};b['patch_size']=torch.ones_like(b['time_id'])*protocol['patch'];assert_complete_future(b,h*d)
            experts,target,w,_,_=joints(policy(b),b)
            assert torch.equal(structured(target,d,h),truth)
            rng=evaluation_seed+(104729 if split=='test' else 0)+start//input_block;torch.manual_seed(rng)
            actions=rollout(experts,w,None,1,samples) # independent mode per candidate; no shared Dirichlet draw
            x=structured(actions['forecast'][:,0],d,h).movedim(1,0);y=structured(target,d,h).double()
            if not torch.isfinite(x).all():raise FloatingPointError('Invalid distribution sample; no sample removed or replaced')
            for scale,draws,truth in [('standardized',x,y),('original',x.double()*std+mean,y*std+mean)]:
                metric_values=probability_metrics(draws,truth)
                for key,value in metric_values.items():
                    if key.endswith('_sum') or key in ['effective_elements','wQL9_abs_target_denominator']:
                        totals[scale][key]=totals[scale].get(key,0)+value
            # Small per-window sufficient statistics; never all ensemble members.
            from .crps_math import empirical_crps
            window_crps=empirical_crps(x,y).mean((-2,-1))
            per_window.extend(dict(origin=int(indices[i,0]),fair_CRPS=float(v)) for i,v in zip(ids,window_crps))
            try:
                joint,_,_=observation_logp(experts,w,target)
                if torch.isfinite(joint).all():nll_sum+=float((-joint/(h*d)).sum())
                else:nll_valid=False;issues.append(dict(start=start,diagnostic='NLL',reason='NONFINITE'))
            except (FloatingPointError,ValueError,RuntimeError,NotImplementedError) as error:
                if isinstance(error,torch.cuda.OutOfMemoryError):raise
                nll_valid=False;issues.append(dict(start=start,diagnostic='NLL',reason=type(error).__name__))
            processed+=len(ids)
    results={}
    for scale,sums in totals.items():
        count=sums['effective_elements'];denominator=sums['wQL9_abs_target_denominator']
        metrics=dict(fair_empirical_CRPS=sums['fair_CRPS_sum']/count,
            biased_empirical_CRPS=sums['biased_empirical_CRPS_sum']/count,
            paper_style_wQL9=sums['wQL9_pinball_twice_sum']/(9*denominator) if denominator else None,
            wQL9_status='VALID' if denominator else 'NA_ZERO_ABSOLUTE_TARGET_DENOMINATOR',effective_elements=count)
        for level in ['80','90','95']:
            metrics['coverage_'+level]=sums['covered_'+level+'_sum']/count
            metrics['width_'+level]=sums['width_'+level+'_sum']/count
        results[scale]=dict(metrics=metrics,sufficient_statistics=sums)
    result=dict(status='PASS' if limit is None else 'DIAGNOSTIC_ONLY',split=split,windows=processed,
        complete=limit is None,samples=samples,evaluation_seed=evaluation_seed,
        rng_rule='evaluation_seed + test*104729 + stream_block_index',input_block=input_block,
        independent_marginal_draws=True,shared_Dirichlet=False,point_readout_protocol='NONE_DISTRIBUTION_SCORE',
        distribution_evidence_multiplicity=1,results=results,
        joint_NLL_per_position=nll_sum/processed if nll_valid else None,NLL_diagnostic_issues=issues,
        quantile_rule='torch.quantile linear interpolation; 0.1..0.9',
        scale_note='Original-scale inverse uses dataset train scaler once; model internal instance scaling is already inverted by the native distribution',
        input_coordinates='Explicit normalized.npy and MoiraiForecast conversion, same as historical point evaluation',
        actual_model_precision=str(next(policy.parameters()).dtype),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        paper_comparability='Nine-quantile formula matched; original paper target scale and remaining data/training details not fully established',
        per_window=per_window,probability_diagnostics_gate_point_results=False)
    if was_training:policy.train()
    if output:atomic_json(Path(output)/'metrics.json',result)
    return result


def evaluate_config(config_path,output,device='cuda',seed=1):
    from .configuration import resolve_checkpoint
    config=resolve_checkpoint(json.loads(Path(config_path).read_text()));seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    if config.get('seed',seed)!=seed:raise ValueError('CLI and training identity seed differ')
    policy=load_policy(config,device);results={}
    for split in config.get('splits',['val']):
        if split=='test':
            from .four_horizon_test_gate import check_release
            check_release(config)
        if split not in ('val','test'):raise ValueError('Distribution entry accepts validation or frozen test only')
        results[split]=evaluate_distribution(policy,config['protocol'],split,config['artifact'],Path(output)/split,
            samples=config.get('distribution_samples',1000),evaluation_seed=config.get('distribution_eval_seed',198367),
            input_block=config.get('distribution_input_block',4),limit=config.get('diagnostic_limit'))
    result=dict(status='PASS' if all(v['complete'] for v in results.values()) else 'DIAGNOSTIC_ONLY',
        config=config,results=results,optimizer_trajectories=0)
    atomic_json(Path(output)/'result.json',result);return result
