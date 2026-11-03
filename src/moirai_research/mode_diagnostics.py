"""Train-only mode naming, followed by full stored-validation diagnostics.

The cluster taxonomy is an already fitted past-training fold. Mode names are
matched using fixed training windows only; validation never chooses a mapping.
No parameter updates and no forecast changes are made by this diagnostic.
"""
from .io_utils import load_checkpoint
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from .data import structured, assert_complete_future
from .neighbors import summaries
from .policy import FIELDS, joints
from .runtime import atomic_json, load_policy, seed_all
from .io_utils import buffered_npz


def mode_mapping(weights, labels, classes=8):
    from scipy.optimize import linear_sum_assignment
    target=torch.nn.functional.one_hot(labels.long(),classes).to(weights.dtype)
    table=weights.T@target
    if weights.shape[1]!=classes:return None,table
    rows,columns=linear_sum_assignment(-table.detach().cpu().numpy())
    assert np.array_equal(rows,np.arange(classes))
    return torch.as_tensor(columns,device=weights.device),table


def aligned_probabilities(weights,mapping):
    result=torch.empty_like(weights)
    result[:,mapping]=weights
    return result


def classification_scores(probabilities,labels):
    p=probabilities.double();y=torch.nn.functional.one_hot(labels.long(),p.shape[1]).double()
    torch.testing.assert_close(p.sum(-1),torch.ones(len(p),device=p.device,dtype=p.dtype),rtol=1e-5,atol=1e-6)
    assert torch.isfinite(p).all() and (p>=0).all()
    certainty,pred=p.max(-1);correct=pred==labels;ece=p.new_zeros(())
    bins=[]
    for b in range(10):
        use=(certainty>=b/10)&(certainty<(b+1)/10) if b<9 else certainty>=.9
        count=int(use.sum())
        accuracy=float(correct[use].double().mean()) if count else None
        confidence=float(certainty[use].mean()) if count else None
        if count:ece+=use.double().mean()*abs(accuracy-confidence)
        bins.append(dict(bin=b,count=count,accuracy=accuracy,confidence=confidence))
    return dict(windows=len(p),log_loss=float(-(y*p.clamp_min(1e-6).log()).sum(-1).mean()),
        Brier_sum_over_modes=float((p-y).square().sum(-1).mean()),top1_accuracy=float(correct.double().mean()),
        ECE_10_bins=float(ece),bins=bins,log_probability_floor=1e-6)


def numerical_checks(device):
    labels=torch.arange(8,device=device).repeat(3)
    p=torch.nn.functional.one_hot(labels,8).double()*.8+.2/8
    perm=torch.tensor([3,5,1,7,2,0,6,4],device=device)
    mapping,_=mode_mapping(p[:,perm],labels)
    aligned=aligned_probabilities(p[:,perm],mapping)
    torch.testing.assert_close(aligned,p,rtol=0,atol=0)
    assert classification_scores(aligned,labels)['top1_accuracy']==1.
    singleton,_=mode_mapping(torch.ones(len(labels),1,device=device),labels)
    assert singleton is None
    return dict(train_permutation_alignment='PASS',single_head_not_eight_class_calibration='PASS')


def diagnose(config_path,output,device='cuda',seed=1):
    config=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    seed_all(seed);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    if config.get('expected_gpu_contains'):
        assert config['expected_gpu_contains'] in torch.cuda.get_device_name(), 'Qualification device does not match its registered scope'
    began=time.time();checks=numerical_checks(device)
    model_config=config['model_config'];p=model_config['protocol'];h=p['horizon'];dim=len(p['variables']) if p['mode']=='M' else 1
    assert model_config['seed']==seed and not config.get('test_used',False)
    root=Path(model_config['artifact']);saved=Path(config['validation_directory'])
    metric=json.loads((saved/'metrics.json').read_text())
    assert metric['complete'] and metric['split']=='val' and metric['seed']==seed
    assert metric['windows']==p['windows']['val']
    values=np.load(root/'normalized.npy',mmap_mode='r',allow_pickle=False)
    train=np.load(root/'train_indices.npy',allow_pickle=False);validation=np.load(root/'val_indices.npy',allow_pickle=False)
    cluster_root=Path(config['cluster_artifact']);folds=sorted(cluster_root.glob('fold_*.npz'));assert folds
    with np.load(folds[-1],allow_pickle=False) as f:
        center=torch.as_tensor(f['centers'],device=device);mean=torch.as_tensor(f['descriptor_mean'],device=device);std=torch.as_tensor(f['descriptor_std'],device=device)
        library=f['library_origins'];assert int(library.max())+h<=p['train_end']
        taxonomy_last_label=int(library.max())+h
    assert len(center)==8 and (std>0).all()
    if dim==1:
        unique=np.unique(train[:,0]);positions=np.unique(np.linspace(0,len(unique)-1,min(144,len(unique)),dtype=int))
        chosen=np.flatnonzero(np.isin(train[:,0],unique[positions]))
        assert len(chosen)==len(positions)*len(p['variables'])
    else:chosen=np.unique(np.linspace(0,len(train)-1,min(1024,len(train)),dtype=int))
    assert np.all(train[chosen,0]>=p['context']) and np.all(train[chosen,0]+h<=p['train_end'])
    policy=load_policy(model_config,device).eval();versions={n:v._version for n,v in policy.named_parameters()}
    stored=load_checkpoint(model_config['checkpoint'],map_location='cpu',weights_only=False)['policy_state']
    loaded=policy.state_dict();assert loaded.keys()==stored.keys()
    assert all(torch.equal(loaded[key].detach().cpu(),value) for key,value in stored.items())
    del stored,loaded
    from uni2ts.model.moirai import MoiraiForecast
    forecast=MoiraiForecast(module=policy.module,context_length=p['context'],patch_size=p['patch'],prediction_length=h,
        num_samples=100,target_dim=dim,feat_dynamic_real_dim=0,past_feat_dynamic_real_dim=0)
    def inputs(origins):
        past=[];future=[]
        for origin,var in origins:
            cols=slice(None) if p['mode']=='M' else slice(int(var),int(var)+1)
            past.append(values[origin-p['context']:origin,cols]);future.append(values[origin:origin+h,cols])
        return torch.as_tensor(np.asarray(past),device=device,dtype=torch.float32),torch.as_tensor(np.asarray(future),device=device,dtype=torch.float32)
    def taxonomy(x,y):
        _,d,_,_=summaries(x.float(),y.float())
        d=d.flatten(1)
        assert d.shape[1]==center.shape[1]
        return torch.cdist((d-mean)/std,center).argmin(-1)
    def prediction(x,y):
        packed=forecast._convert(p['patch'],x,torch.ones_like(x,dtype=torch.bool),torch.zeros(x.shape[:2],device=device,dtype=torch.bool),
            future_target=y,future_observed_target=torch.ones_like(y,dtype=torch.bool),future_is_pad=torch.zeros(y.shape[:2],device=device,dtype=torch.bool))
        b={k:v for k,v in zip(FIELDS[:-1],packed)};b['patch_size']=torch.ones_like(b['time_id'])*p['patch']
        assert_complete_future(b,h*dim)
        experts,target,w,_,_=joints(policy(b),b)
        torch.testing.assert_close(structured(target,dim,h),y,rtol=0,atol=0)
        return experts,w
    train_weights=[];train_labels=[]
    diagnostic_microbatch=int(p.get('diagnostic_microbatch',32));assert diagnostic_microbatch>0
    with torch.no_grad():
        for start in range(0,len(chosen),diagnostic_microbatch):
            x,y=inputs(train[chosen[start:start+diagnostic_microbatch]])
            _,w=prediction(x,y);train_weights.append(w.double());train_labels.append(taxonomy(x,y))
    tw=torch.cat(train_weights);ty=torch.cat(train_labels);mapping,contingency=mode_mapping(tw,ty)
    label_counts=torch.bincount(ty,minlength=8).double();prior=(label_counts+1e-6)/(len(ty)+8e-6)
    weak=None
    with np.load(cluster_root/'neighbors.npz',allow_pickle=False) as f:
        q=torch.as_tensor(f['q'][chosen],device=device,dtype=torch.float64)
        confidence=torch.as_tensor(f['confidence'][chosen],device=device,dtype=torch.float64)
        good=confidence>0
        if policy.k==8 and good.any():
            weak=dict(training_windows=int(good.sum()),mean_confidence=float(confidence[good].mean()),
                index_convention_soft_label_cross_entropy=float(-(q[good]*tw[good].clamp_min(1e-6).log()).sum(-1).mean()),
                index_convention_soft_label_Brier=float((q[good]-tw[good]).square().sum(-1).mean()),
                scope='Fixed training diagnostic windows; weak labels do not name unanchored heads, and this is not held-out evidence')
        else:weak=dict(training_windows=int(good.sum()),reason='No confident weak labels or single-head model')
    buffered_npz(out/'training_mode_alignment.npz',training_indices=chosen,origins=train[chosen],weights=tw.cpu().numpy(),
        future_clusters=ty.cpu().numpy(),soft_contingency=contingency.cpu().numpy(),
        mode_to_cluster=np.empty(0,dtype=int) if mapping is None else mapping.cpu().numpy(),centers=center.cpu().numpy(),
        descriptor_mean=mean.cpu().numpy(),descriptor_std=std.cpu().numpy())
    all_weights=[];all_labels=[];all_indices=[];all_errors=[];fixed=[];offset=0;first_difference=None
    historical_parity={};entry_parity=None
    with torch.no_grad():
        for block in sorted(saved.glob('block_*.npz')):
            with np.load(block,allow_pickle=False) as f:
                origins=f['indices'];x,y=inputs(origins)
                np.testing.assert_array_equal(origins,validation[offset:offset+len(origins)])
                assert f['observed'].all()
                torch.testing.assert_close(torch.as_tensor(f['target'],device=device).float(),y,rtol=0,atol=0)
                w=torch.as_tensor(f['weights'],device=device,dtype=torch.float64)
                assert w.shape==(len(origins),policy.k)
                if offset==0:
                    experts,current=prediction(x,y)
                    first_difference=float((current.double()-w).abs().max())
                    def difference(a,b,rtol,atol):
                        delta=(a-b).abs();outside=delta>atol+rtol*b.abs()
                        return dict(maximum_absolute=float(delta.max()),root_mean_square=float(delta.square().mean().sqrt()),
                            compared_elements=a.numel(),outside_original_tolerance=int(outside.sum()),
                            within_original_tolerance=not bool(outside.any()),rtol=rtol,atol=atol)
                    historical_parity['router']=difference(current.double(),w,1e-3,1e-5)
                    current_mean=torch.stack([structured(e.mean,dim,h) for e in experts],1)[0].double()
                    if (saved/'diagnostic_0.npz').exists():
                        with np.load(saved/'diagnostic_0.npz',allow_pickle=False) as d:
                            historical_parity['mode_means']=difference(current_mean,torch.as_tensor(d['means'],device=device),1e-3,1e-4)
                    else:raise ValueError('Missing original fixed-window mode means')
                    # Qualify two actual interfaces on this device separately
                    # from replaying values recorded on another GPU. Keep every
                    # historical tolerance failure; do not widen that tolerance.
                    from .evaluation import evaluation
                    check_dir=out/'same_device_entry_check'
                    check=evaluation(policy,p,'val',check_dir,seed,root,len(origins))
                    assert not check['complete'] and check['windows']==len(origins)
                    with np.load(check_dir/'diagnostic_0.npz',allow_pickle=False) as d:
                        separate_mean=torch.as_tensor(d['means'],device=device,dtype=torch.float64)
                        separate_weights=torch.as_tensor(d['weights'],device=device,dtype=torch.float64)
                    torch.testing.assert_close(current_mean,separate_mean,rtol=1e-6,atol=1e-7)
                    torch.testing.assert_close(current[0].double(),separate_weights,rtol=1e-6,atol=1e-7)
                    entry_parity=dict(mode_means=difference(current_mean,separate_mean,1e-6,1e-7),
                        router=difference(current[0].double(),separate_weights,1e-6,1e-7),
                        reference='Current canonical evaluation entry, identical checkpoint/input block/device/precision')
                labels=taxonomy(x,y)
                errors=torch.stack([torch.as_tensor(f['metric_'+key],device=device).double() for key in ['MSE','MAE','CRPS','marginal_NLL']],-1)
                assert errors.shape==(len(origins),dim,4)
                all_weights.append(w);all_labels.append(labels);all_indices.append(origins);all_errors.append(errors)
                offset+=len(origins)
    assert offset==len(validation)==p['windows']['val']
    weights=torch.cat(all_weights);labels=torch.cat(all_labels);errors=torch.cat(all_errors);indices=np.concatenate(all_indices)
    aligned=aligned_probabilities(weights,mapping) if mapping is not None else None
    target=torch.nn.functional.one_hot(labels.long(),8).double()
    joint=weights.T@target/len(labels);mode_prior=joint.sum(-1);future_prior=joint.sum(0)
    mi=(joint*(joint.clamp_min(1e-12)/(mode_prior[:,None]*future_prior[None]).clamp_min(1e-12)).log()).sum()
    entropy_mode=-(mode_prior*mode_prior.clamp_min(1e-12).log()).sum();entropy_label=-(future_prior*future_prior.clamp_min(1e-12).log()).sum()
    normalizer=float((entropy_mode*entropy_label).clamp_min(0).sqrt())
    rare=prior<.05;rows=[]
    for var in range(len(p['variables'])):
        for cluster in range(8):
            selected=(labels==cluster)
            if dim==1:selected=selected&torch.as_tensor(indices[:,1]==var,device=device)
            count=int(selected.sum())
            row=dict(variable=int(var),future_cluster=cluster,windows=count,training_prior=float(prior[cluster]),
                rare_by_fixed_training_prior=bool(rare[cluster]))
            row.update({key:float(errors[selected,var if p['mode']=='M' else 0,j].mean()) if count else None for j,key in enumerate(['MSE','MAE','CRPS','marginal_NLL'])})
            rows.append(row)
    with (out/'validation_cluster_errors.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    buffered_npz(out/'validation_modes.npz',indices=indices,future_clusters=labels.cpu().numpy(),weights=weights.cpu().numpy(),
        aligned_weights=np.empty((0,8)) if aligned is None else aligned.cpu().numpy(),errors=errors.cpu().numpy())
    for f in sorted(saved.glob('diagnostic_*.npz')):
        with np.load(f,allow_pickle=False) as d:
            fixed.append(dict(source=str(f),weights=d['weights'].tolist(),mode_counts=np.bincount(d['modes'].reshape(-1),minlength=policy.k).tolist(),
                analytic_mean_range=[float(d['means'].min()),float(d['means'].max())]))
    assert all(v._version==versions[n] for n,v in policy.named_parameters())
    result=dict(status='PASS',scope='Training-frozen mode names and full-validation diagnostic; no method selection or test execution',
        qualification_version='same-device-entry-v2',loaded_parameters_equal_stored_checkpoint=True,
        same_device_entry_parity=entry_parity,historical_output_parity=historical_parity,
        historical_reproduction_status='WITHIN_ORIGINAL_TOLERANCE' if all(v['within_original_tolerance'] for v in historical_parity.values()) else 'MISMATCH_RETAINED',
        config=config,numerical_checks=checks,optimizer_trajectories=0,optimizer_steps=0,parameters_unchanged=True,
        training_alignment_windows=len(chosen),validation_windows=len(labels),taxonomy_source=str(folds[-1]),taxonomy_last_label_exclusive=taxonomy_last_label,
        mapping_rule='Hungarian maximum soft mode/future-cluster contingency on fixed training origins; frozen before validation',
        mode_to_cluster=None if mapping is None else mapping.cpu().tolist(),training_cluster_prior=prior.cpu().tolist(),
        router_calibration=None if aligned is None else classification_scores(aligned,labels),
        calibration_nonapplicability='Single mode is not an eight-class calibrated router' if mapping is None else None,
        training_prior_reference=classification_scores(prior[None].expand(len(labels),-1),labels),
        uniform_reference=classification_scores(torch.full((len(labels),8),1/8,device=device,dtype=torch.float64),labels),
        soft_mode_cluster_mutual_information=float(mi),permutation_invariant_normalized_MI=float(mi)/normalizer if normalizer>0 else 0.,
        weak_label_training_diagnostic=weak,mode_occupancy=metric['mode_occupancy'],mean_router_weights=metric['mean_router_weights'],
        mean_mode_pair_squared_distance=metric['mean_mode_pair_squared_distance'],kappa_quantiles=metric['kappa_quantiles'],
        maximum_absolute_prediction=metric['maximum_absolute_prediction'],worst_windows=metric['worst_windows'],
        fixed_validation_diagnostics=fixed,same_checkpoint_router_maximum_difference=first_difference,
        rare_cluster_rule='Training alignment label prior < .05; no validation threshold fitting',
        caveats=['Cluster labels are engineered training-future descriptors, not ground-truth latent modes',
            'Qualification PASS verifies stored parameter equality, same-device entry consistency and stored validation statistics; it does not waive or certify historical cross-device output reproduction',
            'Training-only naming can overfit its finite alignment subset; held-out validation reports the consequence',
            'Weak-label diagnostics use a fixed training subset and are not held-out generalization evidence',
            'Mode usage, dispersion, concentration or MI alone does not prove useful forecasting modes'],
        actual_gpu=torch.cuda.get_device_name() if str(device).startswith('cuda') else None,seconds=time.time()-began,test_used=False)
    atomic_json(out/'result.json',result);return result


def audit(config_path,output,device='cuda',seed=1):
    from .analysis import numerical_checks as statistics_checks,time_errors
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(config_path).read_text());proof=statistics_checks(device)
    diagnostic=diagnose(config_path,out/'mode_diagnostics',device,seed)
    metric=json.loads((Path(config['validation_directory'])/'metrics.json').read_text())
    rows=[]
    for protocol in ['engineering','MAP','mixture_analytic']:
        times,values=time_errors(config['validation_directory'],protocol,metric,device)
        rows.append(dict(protocol=protocol,origins=len(times),finite=bool(np.isfinite(values).all())))
    result=dict(status='PASS',optimizer_trajectories=0,optimizer_steps=0,config=config,
        qualification_version=diagnostic['qualification_version'],historical_reproduction_status=diagnostic['historical_reproduction_status'],
        mode_receipt=str(out/'mode_diagnostics/result.json'),statistics_checks=proof,full_validation_recomputation=rows,
        actual_gpu=diagnostic['actual_gpu'],seconds=diagnostic['seconds'],test_used=False,
        scope='Actual checkpoint, same-device entry parity, full stored-validation recomputation, calibration alignment and statistical qualification; historical output reproduction remains separately reported; not formal test')
    atomic_json(out/'result.json',result);return result


def probe_parity(config_path,output,device='cuda',seed=1):
    """Zero-update fixed-input attribution, preserving historical discrepancies."""
    from .evaluation import evaluation
    config=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    model=config['model_config'];p=model['protocol'];h=p['horizon'];dim=len(p['variables']) if p['mode']=='M' else 1
    if config.get('expected_gpu_contains'):
        assert config['expected_gpu_contains'] in torch.cuda.get_device_name(), 'Attribution probe requires its predeclared device family'
    seed_all(seed);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    policy=load_policy(model,device).eval();versions={n:v._version for n,v in policy.named_parameters()}
    saved=Path(config['validation_directory']);root=Path(model['artifact'])
    values=np.load(root/'normalized.npy',allow_pickle=False,mmap_mode='r');origins=np.load(root/'val_indices.npy',allow_pickle=False)[:64]
    past=[];target=[]
    for origin,var in origins:
        cols=slice(None) if p['mode']=='M' else slice(int(var),int(var)+1)
        past.append(values[origin-p['context']:origin,cols]);target.append(values[origin:origin+h,cols])
    x=torch.as_tensor(np.asarray(past),device=device,dtype=torch.float32);y=torch.as_tensor(np.asarray(target),device=device,dtype=torch.float32)
    from uni2ts.model.moirai import MoiraiForecast
    forecast=MoiraiForecast(module=policy.module,context_length=p['context'],patch_size=p['patch'],prediction_length=h,
        num_samples=100,target_dim=dim,feat_dynamic_real_dim=0,past_feat_dynamic_real_dim=0)
    def forward():
        packed=forecast._convert(p['patch'],x,torch.ones_like(x,dtype=torch.bool),torch.zeros(x.shape[:2],device=device,dtype=torch.bool),
            future_target=y,future_observed_target=torch.ones_like(y,dtype=torch.bool),future_is_pad=torch.zeros(y.shape[:2],device=device,dtype=torch.bool))
        b={k:v for k,v in zip(FIELDS[:-1],packed)};b['patch_size']=torch.ones_like(b['time_id'])*p['patch']
        assert_complete_future(b,h*dim);experts,truth,w,_,_=joints(policy(b),b)
        torch.testing.assert_close(structured(truth,dim,h),y,rtol=0,atol=0)
        return torch.stack([structured(e.mean,dim,h) for e in experts],1).double(),w.double()
    def comparison(a,b,rtol=1e-3,atol=1e-4):
        difference=(a-b).abs()
        return dict(maximum_absolute=float(difference.max()),root_mean_square=float(difference.square().mean().sqrt()),
            compared_elements=a.numel(),outside_original_tolerance=int((difference>atol+rtol*b.abs()).sum()),rtol=rtol,atol=atol)
    began=time.time()
    with torch.no_grad():
        direct,w=forward()
        metric=evaluation(policy,p,'val',out/'current_evaluation',seed,root,64)
        assert not metric['complete'] and metric['windows']==64
        with np.load(out/'current_evaluation/diagnostic_0.npz',allow_pickle=False) as f:
            same=torch.as_tensor(f['means'],device=device);same_w=torch.as_tensor(f['weights'],device=device,dtype=torch.float64)
        torch.testing.assert_close(direct[0],same,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(w[0],same_w,rtol=1e-6,atol=1e-7)
        with np.load(saved/'diagnostic_0.npz',allow_pickle=False) as f:
            historical=torch.as_tensor(f['means'],device=device);historical_w=torch.as_tensor(f['weights'],device=device,dtype=torch.float64)
        repeat,repeat_w=forward()
        torch.testing.assert_close(direct,repeat,rtol=0,atol=0);torch.testing.assert_close(w,repeat_w,rtol=0,atol=0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        try:strict,strict_w=forward()
        finally:torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    assert all(v._version==versions[n] for n,v in policy.named_parameters())
    buffered_npz(out/'fixed_input_parity.npz',origins=origins,direct_means=direct.cpu().numpy(),
        evaluation_means=same.cpu().numpy(),historical_means=historical.cpu().numpy(),
        tf32_disabled_means=strict.cpu().numpy(),weights=w.cpu().numpy(),historical_weights=historical_w.cpu().numpy())
    result=dict(status='PASS',scope='Fixed-input attribution probe; historical tolerance failures remain explicit, not waived',
        optimizer_trajectories=0,optimizer_steps=0,parameters_unchanged=True,config=config,
        same_device_same_batch_separate_entries=comparison(direct[0],same,1e-6,1e-7),
        historical_comparison=comparison(direct[0],historical),
        tf32_disabled_to_historical=comparison(strict[0],historical),
        tf32_on_vs_off=comparison(direct,strict),historical_router=comparison(w[0],historical_w),
        actual_gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,
        seconds=time.time()-began,validation_limit=64,test_used=False,
        interpretation='Same-device entry equality is checked independently of cross-device historical differences. This probe neither replaces stored predictions nor changes a scientific metric.')
    atomic_json(out/'result.json',result);return result
