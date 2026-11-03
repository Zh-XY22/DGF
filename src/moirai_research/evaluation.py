"""Full-window, fixed-RNG evaluation with separate deployable decision protocols."""
import json
from .configuration import resolve_checkpoint
from pathlib import Path
import time
import numpy as np
import torch
from .policy import FIELDS,joints,rollout
from .probability import observation_logp
from .data import structured,assert_complete_future
from .runtime import atomic_json,load_policy,seed_all
from .io_utils import buffered_npz
from .point_selection import POINT_PROTOCOLS, finite_point
from .actions import NonfiniteForecastSample


def evaluation(policy,protocol,split,output=None,seed=1,artifact=None,limit=None,artifact_mode="full"):
    """`limit` is diagnostic-only and cannot produce a formal acceptance receipt."""
    if artifact_mode not in ['full','compact_statistics']:raise ValueError('Unknown evaluation artifact mode')
    from uni2ts.model.moirai import MoiraiForecast
    from gluonts.model.forecast import SampleForecast
    import pandas as pd
    p=protocol;device=next(policy.parameters()).device
    root=Path(artifact or p['artifact']);out=Path(output) if output else None
    if out:out.mkdir(parents=True,exist_ok=True)
    values=np.load(root/'normalized.npy',allow_pickle=False,mmap_mode='r')
    indices=np.load(root/(split+'_indices.npy'),allow_pickle=False)
    scaler_std=np.load(root/'scaler_std.npy',allow_pickle=False)
    assert len(indices)==p['windows'][split]
    lo,hi=(p['train_end'],p['val_end']) if split=='val' else (p['val_end'],p['test_end']) if split=='test' else (p['context'],p['train_end'])
    assert np.all(indices[:,0]>=lo) and np.all(indices[:,0]+p['horizon']<=hi)
    assert np.all(indices[:,0]>=p['context'])
    if limit is not None:indices=indices[:limit]
    variables=len(p['variables']);multivariate=p['mode']=='M'
    dimensions=variables if multivariate else 1;h=p['horizon'];total=len(indices)
    assert values.shape[1]==variables and len(set(p['variables']))==variables
    input_block=int(p.get('eval_input_block',64));assert input_block>0
    peak_threshold=torch.as_tensor(np.quantile(np.abs(values[:p['train_end']]),.95,axis=0),device=device,dtype=torch.float64)
    forecast=MoiraiForecast(module=policy.module,context_length=p['context'],patch_size=p['patch'],
         prediction_length=h,num_samples=100,target_dim=dimensions,feat_dynamic_real_dim=0,past_feat_dynamic_real_dim=0)
    sums={};counts=torch.zeros(variables,device=device,dtype=torch.float64)
    nll_sum=0.;mode_counts=torch.zeros(policy.k,device=device,dtype=torch.float64);weight_sum=torch.zeros_like(mode_counts)
    pair_sum=0.;kappas=[];tail=[];maximum=0.;joint_count=0;began=time.time();was_training=policy.training;policy.eval()
    invalid_metrics=set();diagnostic_issues={};joint_invalid=False;pair_invalid=False;kappa_invalid=False
    sampling_invalid=False
    def issue(name,origin,message):
        record=diagnostic_issues.setdefault(name,dict(occurrences=0,first_origin=int(origin),first_message=str(message)))
        record['occurrences']+=1
    def diagnostic(name,origin,call,fallback):
        try:return call()
        except (FloatingPointError,ValueError,RuntimeError,NotImplementedError) as error:
            if isinstance(error,torch.cuda.OutOfMemoryError):raise
            issue(name,origin,type(error).__name__+': '+str(error));return fallback

    diagnostic_indices={0,1,total//2,total-2,total-1}
    with torch.no_grad():
        for start in range(0,total,input_block):
            origins=indices[start:start+input_block];n=len(origins)
            past=[];truth=[]
            for origin,var in origins:
                cols=slice(None) if multivariate else slice(int(var),int(var)+1)
                past.append(values[origin-p['context']:origin,cols]);truth.append(values[origin:origin+h,cols])
            past=torch.tensor(np.asarray(past),device=device,dtype=torch.float32)
            target=torch.tensor(np.asarray(truth),device=device,dtype=torch.float32)
            devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
            with torch.random.fork_rng(devices=devices):
                rng_seed=seed+65537+(104729 if split=='test' else 0)+start//input_block
                torch.manual_seed(rng_seed)
                packed=forecast._convert(p['patch'],past,torch.ones_like(past,dtype=torch.bool),
                    torch.zeros(past.shape[:2],device=device,dtype=torch.bool),future_target=target,
                    future_observed_target=torch.ones_like(target,dtype=torch.bool),
                    future_is_pad=torch.zeros(target.shape[:2],device=device,dtype=torch.bool))
                b={k:v for k,v in zip(FIELDS[:-1],packed)};b['patch_size']=torch.ones_like(b['time_id'])*p['patch']
                valid=assert_complete_future(b,h*dimensions)
                prediction=policy(b);experts,packed_truth,w,alpha,kappa=joints(prediction,b)
                assert torch.equal(structured(packed_truth,dimensions,h),target)
                actions=None
                try:
                    actions=rollout(experts,w,None,1,100) # independent marginal draws; never shared pi
                    samples=structured(actions['forecast'][:,0],dimensions,h)
                    if not torch.isfinite(samples).all():raise NonfiniteForecastSample(samples)
                except NonfiniteForecastSample as error:
                    sampling_invalid=True
                    invalid=~torch.isfinite(error.forecast)
                    issue('P2_sampling',start,'Nonfinite sampled forecast: '+str(int(invalid.sum()))+
                        ' elements; shape='+str(tuple(error.forecast.shape))+
                        '; expert='+str(getattr(error,'expert_index','unknown'))+'; no sample removed or replaced')
                    if out:
                        buffered_npz(out/f'nonfinite_samples_{start//input_block:05d}.npz',
                            forecast=error.forecast.cpu().numpy(),invalid=invalid.cpu().numpy(),indices=origins)
                    # Explicitly unavailable P2, not a replacement prediction.
                    samples=torch.full((n,100,h,dimensions),float('nan'),device=device)
                    actions=None
                mode_means=[]
                for mode,e in enumerate(experts):
                    mode_means.append(diagnostic('mode_mean_'+str(mode),start,
                        lambda e=e:structured(e.mean,dimensions,h).double(),
                        torch.full((n,h,dimensions),float('nan'),device=device,dtype=torch.float64)))
                means=torch.stack(mode_means,1)
                mean=samples.mean(1).double();mixture=(means*w[:,:,None,None]).sum(1)
                map_prediction=means[torch.arange(n,device=device),w.argmax(-1)]
                if actions is not None:
                    sf=SampleForecast(samples.permute(1,0,2,3).cpu().numpy().reshape(100,-1,dimensions),
                        start_date=pd.Period(p['source_first_timestamp'],freq=str(p['source_frequency']).lower()))
                    def quantile(q):return torch.as_tensor(sf.quantile(q).reshape(n,h,dimensions),device=device,dtype=torch.float64)
                    median=quantile(.5);lower=quantile(.1);upper=quantile(.9)
                else:
                    median=torch.full_like(mean,float('nan'));lower=median.clone();upper=median.clone()
                y=target.double()
                fallback=(torch.full((n,),float('nan'),device=device),
                    torch.full_like(packed_truth,float('nan')),torch.full_like(packed_truth,float('nan')))
                joint,marginal,atomic=diagnostic('probability_scores',start,
                    lambda:observation_logp(experts,w,packed_truth),fallback)
                def legacy_score():
                    position=torch.stack([d.log_prob(b['target'])[valid].reshape(n,dimensions*h) for d in prediction[0]],1)
                    return -torch.logsumexp(w.double().log()+position.double().sum(-1),-1)/(dimensions*h)
                legacy_joint=diagnostic('native_legacy_joint_NLL',start,legacy_score,
                    torch.full((n,),float('nan'),device=device))
                if torch.isfinite(joint).all():nll_sum+=float((-joint/(dimensions*h)).sum())
                else:joint_invalid=True;issue('joint_NLL_per_position',start,'Nonfinite probability diagnostic; point metrics retained')
                joint_count+=n
                if actions is not None:mode_counts+=torch.bincount(actions['mode'].flatten(),minlength=policy.k).double()
                weight_sum+=w.double().sum(0)
                if kappa is not None:
                    if torch.isfinite(kappa).all():kappas.append(kappa.flatten().cpu().numpy())
                    else:kappa_invalid=True;issue('kappa_quantiles',start,'Nonfinite diagnostic')
                pair=(means[:,:,None]-means[:,None,:]).square().mean((-2,-1))
                pair_distance=pair.sum((1,2))/(policy.k*(policy.k-1)) if policy.k>1 else torch.zeros(n,device=device,dtype=torch.float64)
                if torch.isfinite(pair_distance).all():pair_sum+=float(pair_distance.sum())
                else:pair_invalid=True;issue('mean_mode_pair_squared_distance',start,'Undefined mode-mean diagnostic')
                scale=torch.as_tensor(scaler_std,device=device)[None,None] if multivariate else torch.as_tensor(scaler_std[origins[:,1]],device=device)[:,None,None]
                ordered=samples.double().sort(1).values
                coefficient=(2*torch.arange(1,101,device=device,dtype=torch.float64)-101)[None,:,None,None]
                crps=(samples.double()-y[:,None]).abs().mean(1)-(ordered*coefficient).sum(1)/(100*99)
                metrics=dict(MSE=(mean-y).square().mean(-2),MAE=(median-y).abs().mean(-2),median_MSE=(median-y).square().mean(-2),
                  raw_MSE=((mean-y)*scale).square().mean(-2),raw_MAE=((median-y)*scale).abs().mean(-2),
                  MAP_MSE=(map_prediction-y).square().mean(-2),MAP_MAE=(map_prediction-y).abs().mean(-2),
                  mixture_analytic_MSE=(mixture-y).square().mean(-2),mixture_analytic_MAE=(mixture-y).abs().mean(-2),
                  raw_MAP_MSE=((map_prediction-y)*scale).square().mean(-2),raw_MAP_MAE=((map_prediction-y)*scale).abs().mean(-2),
                  raw_mixture_analytic_MSE=((mixture-y)*scale).square().mean(-2),raw_mixture_analytic_MAE=((mixture-y)*scale).abs().mean(-2),
                  CRPS=crps.mean(-2),raw_CRPS=(crps*scale).mean(-2),
                  coverage_80=((y>=lower)&(y<=upper)).double().mean(-2) if actions is not None else torch.full((n,dimensions),float('nan'),device=device),width_80=(upper-lower).mean(-2),
                  marginal_NLL=structured(-marginal,dimensions,h).mean(-2),
                  atomic_label_fraction=structured(atomic.double(),dimensions,h).mean(-2),
                  native_legacy_joint_NLL=legacy_joint[:,None].expand(n,dimensions))
                for label,pred in [('mean',mean),('MAP',map_prediction),('mixture_analytic',mixture)]:
                    metrics[label+'_diff1_L1']=(pred.diff(dim=-2)-y.diff(dim=-2)).abs().mean(-2)
                    metrics[label+'_diff2_L1']=(pred.diff(n=2,dim=-2)-y.diff(n=2,dim=-2)).abs().mean(-2)
                    a=torch.fft.rfft(pred,dim=-2).abs();a=a/(a.sum(-2,keepdim=True)+1e-6)
                    z=torch.fft.rfft(y,dim=-2).abs();z=z/(z.sum(-2,keepdim=True)+1e-6)
                    metrics[label+'_fft_L1']=(a-z).abs().mean(-2)
                    metrics[label+'_boundary_change_L1']=(pred[:,0]-y[:,0]).abs()
                    metrics[label+'_legacy_boundary_L1']=(pred[:,0]-past[:,-1]).abs()
                    threshold=peak_threshold[None,None] if multivariate else peak_threshold[torch.as_tensor(origins[:,1],device=device)][:,None,None]
                    peak=y.abs()>=threshold;turn=(y.diff(dim=-2)[:,:-1]*y.diff(dim=-2)[:,1:]<0)
                    metrics[label+'_peak_squared_sum']=torch.where(peak,(pred-y).square(),torch.zeros_like(y)).sum(-2)
                    metrics[label+'_peak_count']=peak.sum(-2).double()
                    metrics[label+'_turn_squared_sum']=torch.where(turn,(pred[:,1:-1]-y[:,1:-1]).square(),torch.zeros_like(y[:,1:-1])).sum(-2)
                    metrics[label+'_turn_count']=turn.sum(-2).double()
                for name,v in metrics.items():
                    if name not in sums:sums[name]=torch.zeros(variables,device=device,dtype=torch.float64)
                    if not torch.isfinite(v).all():
                        invalid_metrics.add(name);issue(name,start,'Nonfinite value; entire diagnostic/readout marked unavailable, no window dropped')
                        continue
                    if multivariate:sums[name]+=v.sum(0)
                    else:sums[name].index_add_(0,torch.as_tensor(origins[:,1],device=device),v[:,0])
                if multivariate:counts+=n
                else:counts.index_add_(0,torch.as_tensor(origins[:,1],device=device),torch.ones(n,device=device,dtype=torch.float64))
                if actions is not None:
                    maximum=max(maximum,float(samples.abs().max()))
                    errors=(mean-y).square().mean((-2,-1))
                    if torch.isfinite(errors).all():
                        largest=errors.topk(min(3,n)).indices.tolist()
                        for i in largest:tail.append(dict(window=start+i,origin=int(origins[i,0]),variable=int(origins[i,1]),MSE=float(errors[i]),max_abs_sample=float(samples[i].abs().max())))
                tail=sorted(tail,key=lambda x:x['MSE'],reverse=True)[:20]
                if out:
                    arrays=dict(indices=origins,window_indices=np.arange(start,start+n),
                       joint_NLL_per_position=(-joint/(h*dimensions)).cpu().numpy(),weights=w.cpu().numpy(),
                       mode_pair_distance=pair_distance.cpu().numpy())
                    if artifact_mode=='full':
                        arrays.update(target=y.cpu().numpy(),observed=np.ones(y.shape,dtype=bool),sample_mean=mean.cpu().numpy(),
                            sample_median=median.cpu().numpy(),MAP=map_prediction.cpu().numpy(),mixture_analytic=mixture.cpu().numpy(),
                            lower80=lower.cpu().numpy(),upper80=upper.cpu().numpy())
                    arrays.update({'metric_'+k:v.cpu().numpy() for k,v in metrics.items()})
                    buffered_npz(out/f'block_{start//input_block:05d}.npz',**arrays)
                    for i in range(n):
                        if artifact_mode=='full' and start+i in diagnostic_indices:
                            buffered_npz(out/f'diagnostic_{start+i}.npz',samples=samples[i].cpu().numpy(),
                                modes=np.empty(0,dtype=np.int64) if actions is None else actions['mode'][i].cpu().numpy(),means=means[i].cpu().numpy(),weights=w[i].cpu().numpy(),
                                alpha=np.empty(0) if alpha is None else alpha[i].cpu().numpy())
                if start%4096==0:print(json.dumps(dict(event='evaluation_progress',split=split,windows=start+n,total=total,seconds=time.time()-began)),flush=True)
    per_variable={name:{} for name in p['variables']};overall={}
    for name,s in sums.items():
        if name.endswith('_squared_sum') or name.endswith('_count'):continue
        if name in invalid_metrics:
            overall[name]=None
            for var in p['variables']:per_variable[var][name]=None
        else:
            v=s/counts.clamp_min(1);overall[name]=float(s.sum()/counts.sum())
            for i,var in enumerate(p['variables']):per_variable[var][name]=float(v[i]) if counts[i]>0 else None
    for prefix in ('mean','MAP','mixture_analytic'):
        for event in ('peak','turn'):
            k=prefix+'_'+event;num=sums[k+'_squared_sum'];den=sums[k+'_count']
            event_valid=k+'_squared_sum' not in invalid_metrics and k+'_count' not in invalid_metrics
            overall[k+'_MSE']=float(num.sum()/den.sum()) if event_valid and den.sum()>0 else None
            overall[k+'_count']=int(den.sum())
            for i,var in enumerate(p['variables']):per_variable[var][k+'_MSE']=float(num[i]/den[i]) if event_valid and den[i]>0 else None
    overall.update(joint_NLL_per_position=None if joint_invalid else nll_sum/joint_count,per_variable=per_variable,windows=total,
        complete=limit is None,split=split,num_samples=100,seed=seed,artifact_mode=artifact_mode,eval_rng_rule=f'seed+65537+test*104729+global{input_block}block',
        independent_marginal_samples=True,mode_selection='argmax history weight, analytic selected mean',
        units='standardized target; NLL nats per position relative to Lebesgue+counting measure',
        maximum_absolute_prediction=None if sampling_invalid else maximum,worst_windows=tail,
        worst_windows_complete=not sampling_invalid and 'MSE' not in invalid_metrics,
        mode_occupancy=None if sampling_invalid else (mode_counts/mode_counts.sum()).cpu().tolist(),
        mean_router_weights=(weight_sum/total).cpu().tolist(),mean_mode_pair_squared_distance=None if pair_invalid else pair_sum/total,
        kappa_quantiles=np.quantile(np.concatenate(kappas),[0,.25,.5,.75,1]).tolist() if kappas and not kappa_invalid else None,
        seconds=time.time()-began,peak_definition='abs standardized label above training-only per-variable .95 quantile',
        turn_definition='adjacent true first differences have opposite signs; zero is not a turn')
    overall['point_protocol_status']={key:dict(name=fields['name'],
        status='VALID' if all(finite_point(overall.get(fields[x])) for x in ['MSE','MAE']) else 'INVALID',
        selection_uses_only='MSE/MAE; probability and dynamics diagnostics never gate')
        for key,fields in POINT_PROTOCOLS.items()}
    overall['diagnostic_issues']=diagnostic_issues
    overall['diagnostics_gate_point_results']=False
    if was_training:policy.train()
    if out:atomic_json(out/'metrics.json',overall)
    return overall


def evaluate_config(config_path,output,device='cuda',seed=1,resume_directory=None):
    config=resolve_checkpoint(json.loads(Path(config_path).read_text()));seed_all(seed)
    if config.get('seed',seed)!=seed:raise ValueError('CLI seed differs from frozen configuration')
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    out=Path(output);out.mkdir(parents=True,exist_ok=True);policy=None
    checkpoint_metadata=None
    if config.get('checkpoint_kind')=='policy_state':
        from .io_utils import load_checkpoint
        saved=load_checkpoint(config['checkpoint'],map_location='cpu',weights_only=False);saved_config=saved.get('config',{})
        checkpoint_metadata=dict(saved_progress=saved.get('progress'),saved_seed=saved_config.get('seed'),saved_route=saved.get('route'),
            saved_revision=saved_config.get('revision'),joint_training=saved_config.get('joint_training',False),
            training_horizons=list(saved_config['protocols']) if saved_config.get('protocols') else [saved_config.get('protocol',{}).get('horizon')],
            interpretation='Actual retained checkpoint header; absent legacy fields stay unavailable')
        del saved
    results={};provenance={}
    prior=Path(resume_directory) if resume_directory else None
    if prior is not None:
        if json.loads((prior/'config.json').read_text())!=config:
            raise ValueError('Readout resume config differs from original frozen case')
    for split in config.get('splits',['val']):
        if split=='test' and config.get('four_horizon_campaign'):
            from .four_horizon_test_gate import check_release
            check_release(config)
        if split=='test' and not(config.get('historical_protocol_replay') or config.get('test_release_receipt')):
            raise ValueError('New-stage formal test is locked until validation selection is frozen')
        new_point_campaign = str(config.get('scientific_path','')).startswith('point-') or config.get('campaign') == 'point_opd_confirmation_20261024_130931'
        if split=='test' and new_point_campaign and not config.get('historical_protocol_replay'):
            from .point_test_gate import test_release
            test_release(config.get('test_release_receipt'),config.get('scientific_path'))
        saved=prior/split/'metrics.json' if prior is not None else None
        if saved is not None and saved.is_file():
            metric=json.loads(saved.read_text())
            assert metric['complete'] and metric['split']==split and metric['seed']==seed
            assert metric['windows']==config['protocol']['windows'][split] and metric['num_samples']==100
            assert metric['independent_marginal_samples']
            previous=prior/'result.json';generation={}
            if previous.is_file():
                generation=json.loads(previous.read_text()).get('readout_provenance',{}).get(split,{}).get('generation',{})
            results[split]=metric;provenance[split]=dict(action='REUSED_COMPLETED_SPLIT',source=str(saved),outputs=str(saved.parent),
                generation=generation,generation_scope='Preserved original split record; absent fields remain unknown, never filled from the resuming worker')
            (out/split).mkdir(exist_ok=True);atomic_json(out/split/'metrics.json',metric)
        else:
            if policy is None:policy=load_policy(config,device)
            results[split]=evaluation(policy,config['protocol'],split,out/split,seed,config['artifact'],config.get('diagnostic_limit'),artifact_mode=config.get('evaluation_artifact_mode','full'))
            provenance[split]=dict(action='EXECUTED_THIS_ATTEMPT',source=str(out/split/'metrics.json'),outputs=str(out/split),
                generation=dict(actual_gpu=torch.cuda.get_device_name(),model_precision=str(next(policy.parameters()).dtype),
                    metric_precision='float64 accumulation',tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                    tf32_cudnn=torch.backends.cudnn.allow_tf32))
    status='PASS' if all(x['complete'] for x in results.values()) else 'DIAGNOSTIC_ONLY'
    reward_diagnostic_receipt=None
    if config.get('rollout_diagnostic_model_config'):
        from .rollout_diagnostics import diagnose
        saved=prior/'group_reward_diagnostics/result.json' if prior is not None else None
        if saved is not None and saved.is_file():
            diagnostic=json.loads(saved.read_text());assert diagnostic['status']=='PASS' and diagnostic['test_used'] is False
            directory=out/'group_reward_diagnostics';directory.mkdir(exist_ok=True);atomic_json(directory/'result.json',diagnostic)
        else:
            if policy is None:policy=load_policy(config,device)
            diagnostic=diagnose(policy,config['rollout_diagnostic_model_config'],out/'group_reward_diagnostics',device,seed)
        reward_diagnostic_receipt=str(out/'group_reward_diagnostics/result.json')
    result=dict(status=status,config=config,checkpoint_metadata=checkpoint_metadata,results=results,readout_provenance=provenance,
        group_reward_diagnostic_receipt=reward_diagnostic_receipt,actual_gpu=torch.cuda.get_device_name(),
        world_size=torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1,
        world_size_source='Actual torch.distributed process group, or the uninitialized single-process entry',optimizer_trajectories=0)
    atomic_json(out/'result.json',result);return result


def historical_replay(config_path,output,device='cuda',seed=1):
    config=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    results=[]
    for case in config['cases']:
        key=case['case_id'];directory=out/key;directory.mkdir(exist_ok=True)
        local=directory/'config.json';atomic_json(local,case)
        receipt=directory/'result.json'
        if receipt.exists():
            r=json.loads(receipt.read_text())
            if r.get('status')!='PASS' or r.get('config')!=case:raise ValueError('Existing readout conflicts with frozen case')
        else:
            prior=Path(config['resume_replay_result'])/key if config.get('resume_replay_result') else None
            if prior is not None and not (prior/'config.json').is_file():prior=None
            r=evaluate_config(str(local),directory,device,seed,prior)
        # Old engineering metric parity is checked, never silently replaced.
        previous=case.get('historical_test_metrics',{})
        comparison={}
        if 'test' in r['results']:
            for metric,old in previous.items():
                value=r['results']['test'].get(metric)
                if old is not None and value is not None:
                    comparison[metric]=dict(previous=old,reloaded=value,delta=value-old,
                       within_tolerance=abs(value-old)<=max(1e-5,abs(old)*1e-4))
        results.append(dict(case_id=key,roles=case.get('roles',[]),receipt=str(receipt),comparison=comparison))
        atomic_json(out/'progress.json',results)
        torch.cuda.empty_cache()
    result=dict(status='PASS',cases=results,optimizer_trajectories=0,
        limitation='Protocol diagnostics on previously selected checkpoint. MAP/analytic output was not reselected from unavailable alternative checkpoints.',
        numerical_parity_conflicts=[dict(case=x['case_id'],metric=k,**v) for x in results for k,v in x['comparison'].items() if not v['within_tolerance']])
    atomic_json(out/'result.json',result);return result
