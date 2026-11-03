"""Past-only, fold-frozen nearest-history weak labels; no validation/test fitting."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from .runtime import atomic_json


def summaries(history,future):
    """Per-variable history shape and history-scaled future dynamics."""
    mean=history.mean(1);scale=history.std(1,correction=0).clamp_min(1e-5)
    x=(history-mean[:,None])/scale[:,None];y=(future-mean[:,None])/scale[:,None]
    pooled=torch.nn.functional.adaptive_avg_pool1d(x.transpose(1,2),16)
    historical=torch.cat([pooled, x[:,-1,:,None],mean[...,None],scale.log()[...,None],
                           x.diff(dim=1).abs().mean(1)[...,None]],-1)
    amplitude=torch.fft.rfft(y,dim=1).abs();amplitude=amplitude/(amplitude.sum(1,keepdim=True)+1e-6)
    spectral=torch.nn.functional.adaptive_avg_pool1d(amplitude.transpose(1,2),8)
    dy=y.diff(dim=1);ddy=y.diff(n=2,dim=1)
    descriptive=torch.stack([y.mean(1),y.std(1,correction=0),y[:,-1]-y[:,0],
        y[:,0]-x[:,-1],dy.abs().mean(1),ddy.abs().mean(1),
        (dy[:,:-1]*dy[:,1:]<0).float().mean(1),y.abs().amax(1)],-1)
    return historical,torch.cat([descriptive,spectral],-1),mean,scale


def fit_modes(data,centers=None,iterations=20):
    """Fixed-seed farthest-point initialization; IDs aligned to previous fold."""
    from scipy.optimize import linear_sum_assignment
    if len(data)<8:raise ValueError('Insufficient past futures for eight clusters')
    if centers is None:
        indices=[0];distance=(data-data[0]).square().sum(-1)
        for _ in range(7):
            i=int(distance.argmax());indices.append(i);distance=torch.minimum(distance,(data-data[i]).square().sum(-1))
        centers=data[indices].clone()
        previous=None
    else:previous=centers.clone();centers=centers.clone()
    for _ in range(iterations):
        total=torch.zeros_like(centers);count=torch.zeros(8,device=data.device)
        for start in range(0,len(data),4096):
            x=data[start:start+4096];label=torch.cdist(x,centers).argmin(-1)
            total.index_add_(0,label,x);count.index_add_(0,label,torch.ones_like(label,dtype=data.dtype))
        centers=torch.where((count>0)[:,None],total/count[:,None].clamp_min(1),centers)
    if previous is not None:
        old,new=linear_sum_assignment(torch.cdist(previous,centers).cpu().numpy())
        assert np.array_equal(old,np.arange(8));centers=centers[new]
    return centers


def build_neighbors(values,protocol,output,seed=1,fold_width=512,query_chunk=64):
    """All calculations execute in the worker. `values` has only training rows.

    Both positive and shuffled-label controls use exactly the same past-fold
    library, neighbor IDs, distances, history scaling and weights. Negative labels
    are permuted in variable/time strata of that past library before retrieval;
    this preserves its per-stratum label frequency exactly, without query futures.
    """
    out=Path(output);out.mkdir(parents=True,exist_ok=True);device=values.device
    context=protocol['context'];h=protocol['horizon'];end=protocol['train_end'];mode=protocol['mode']
    assert len(values)==end
    origins=torch.arange(context,end-h+1,device=device);windows=len(origins);channels=values.shape[1]
    if 'variables' in protocol:assert channels==len(protocol['variables'])
    history=[];future=[];means=[];scales=[]
    for start in range(0,windows,128):
        o=origins[start:start+128]
        x=values[o[:,None]+torch.arange(-context,0,device=device)]
        y=values[o[:,None]+torch.arange(h,device=device)]
        a,b,c,d=summaries(x,y);history.append(a);future.append(b);means.append(c);scales.append(d)
    history=torch.cat(history);future=torch.cat(future);means=torch.cat(means);scales=torch.cat(scales)
    if mode=='M':
        hist=history.flatten(1)[:,None];desc=future.flatten(1)[:,None];groups=1
    else:hist=history;desc=future;groups=channels
    total=windows*groups
    assert total==protocol['windows']['train']
    # Dataset order is forecast-origin first, variable second, matching official S builder.
    query_origins=origins.repeat_interleave(groups);variable_ids=torch.arange(groups,device=device).repeat(windows) if mode=='S' else torch.full((windows,),-1,device=device)
    q=torch.zeros(total,8,device=device);q_random=torch.zeros_like(q);confidence=torch.zeros(total,device=device)
    neighbor_ids=torch.full((total,16),-1,device=device,dtype=torch.long);weights=torch.zeros(total,16,device=device)
    distances=torch.full((total,16),float('nan'),device=device);n_eff=torch.zeros(total,device=device)
    true_cluster=torch.full((total,),-1,device=device,dtype=torch.long);fold_ids=torch.full_like(true_cluster,-1)
    strict_count=torch.zeros_like(true_cluster);centers=None;desc_mean=None;desc_std=None;fold_records=[]
    began=time.time()
    for fold,lower_index in enumerate(range(0,windows,fold_width)):
        upper_index=min(windows,lower_index+fold_width);fold_origin=int(origins[lower_index])
        # Full horizon gap after labels; every item and shuffled label donor is known at fold start.
        past=(origins+h+h<=fold_origin).nonzero().flatten()
        qids=torch.arange(lower_index,upper_index,device=device)
        if len(past)<8:
            fold_records.append(dict(fold=fold,query_origin=fold_origin,past_windows=len(past),status='REAL_LABEL_ONLY_INSUFFICIENT_PAST'))
            continue
        fit=desc[past].reshape(-1,desc.shape[-1])
        if desc_mean is None:desc_mean=fit.mean(0);desc_std=fit.std(0,correction=0).clamp_min(1e-4)
        standardized=(fit-desc_mean)/desc_std
        centers=fit_modes(standardized,centers)
        labels=torch.cdist(standardized,centers).argmin(-1).reshape(len(past),groups)
        current_labels=torch.cdist(((desc[qids]-desc_mean)/desc_std).reshape(-1,desc.shape[-1]),centers).argmin(-1).reshape(len(qids),groups)
        permutation=labels.clone();strata_checked=0
        for v in range(groups):
            bands=origins[past]//fold_width
            for band in bands.unique():
                ix=(bands==band).nonzero().flatten();gen=torch.Generator(device=device).manual_seed(seed+fold*10007+v*701+int(band))
                order=torch.randperm(len(ix),device=device,generator=gen)
                permutation[ix,v]=labels[ix[order],v]
                assert torch.equal(torch.bincount(permutation[ix,v],minlength=8),torch.bincount(labels[ix,v],minlength=8))
                strata_checked+=1
        np.savez_compressed(out/f'fold_{fold:03d}.npz',centers=centers.cpu().numpy(),descriptor_mean=desc_mean.cpu().numpy(),
            descriptor_std=desc_std.cpu().numpy(),library_origins=origins[past].cpu().numpy(),labels=labels.cpu().numpy(),shuffled_labels=permutation.cpu().numpy())
        for v in range(groups):
            library=hist[past,v];feature_mean=library.mean(0);feature_std=library.std(0,correction=0).clamp_min(1e-4)
            library=(library-feature_mean)/feature_std
            for begin in range(lower_index,upper_index,query_chunk):
                ids=torch.arange(begin,min(upper_index,begin+query_chunk),device=device)
                queries=(hist[ids,v]-feature_mean)/feature_std
                distance=torch.cdist(queries,library)/library.shape[-1]**.5
                remaining=distance.clone();selected=[];chosen_distance=[]
                for _ in range(16):
                    d,ix=remaining.min(-1);okay=torch.isfinite(d);selected.append(torch.where(okay,ix,torch.full_like(ix,-1)));chosen_distance.append(d)
                    occupied=origins[past[ix]]
                    excluded=(origins[past][None]-occupied[:,None]).abs()<2*h
                    remaining=remaining.masked_fill(excluded & okay[:,None],float('inf'))
                selected=torch.stack(selected,-1);d=torch.stack(chosen_distance,-1);valid=selected>=0
                score=torch.where(valid,-d,torch.full_like(d,-torch.inf));a=score.softmax(-1)
                a=torch.where(valid,a,torch.zeros_like(a))
                assert torch.isfinite(a).all() and torch.allclose(a.sum(-1),torch.ones(len(ids),device=device))
                retrieved=past[selected.clamp_min(0)];retrieved_origins=origins[retrieved]
                assert ((retrieved_origins+h+h<=origins[ids,None])|~valid).all()
                pair_valid=valid[:,:,None]&valid[:,None,:]&~torch.eye(16,device=device,dtype=torch.bool)[None]
                assert (((retrieved_origins[:,:,None]-retrieved_origins[:,None,:]).abs()>=2*h)|~pair_valid).all()
                effective=1/a.square().sum(-1);mean_distance=(torch.where(valid,d,torch.zeros_like(d))*a).sum(-1)
                scale_query=scales[ids] if groups==1 else scales[ids,v,None]
                scale_neighbor=scales[retrieved] if groups==1 else scales[retrieved,v,None]
                ratio=torch.maximum(scale_neighbor/scale_query[:,None],scale_query[:,None]/scale_neighbor).amax(-1)
                abnormal=(torch.where(valid,ratio,torch.ones_like(ratio))*a).sum(-1)
                c=(effective/4).clamp(max=1)*torch.exp(-mean_distance)/abnormal.clamp_min(1)
                c=torch.where((valid.sum(-1)>=2)&(mean_distance<=3)&(abnormal<=10),c,torch.zeros_like(c))
                destination=ids*groups+v
                positive=torch.zeros(len(ids),8,device=device);negative=torch.zeros_like(positive)
                positive.scatter_add_(1,labels[selected.clamp_min(0),v],a);negative.scatter_add_(1,permutation[selected.clamp_min(0),v],a)
                q[destination]=positive;q_random[destination]=negative;confidence[destination]=c
                neighbor_ids[destination]=torch.where(valid,retrieved*groups+v,torch.full_like(retrieved,-1));weights[destination]=a
                distances[destination]=torch.where(valid,d,torch.full_like(d,float('nan')));n_eff[destination]=effective
                true_cluster[destination]=current_labels[ids-lower_index,v];fold_ids[destination]=fold
                strict_last=torch.minimum(origins[ids]-context-2*h,origins[past].max())
                strict_count[destination]=torch.where(strict_last>=context,(strict_last-context)//(context+2*h)+1,torch.zeros_like(strict_last))
                confidence[destination]=torch.where(strict_count[destination]>=2,c,torch.zeros_like(c))
        fold_records.append(dict(fold=fold,query_origin=fold_origin,past_windows=len(past),status='PAST_ONLY',
            negative_control_strata_checked=strata_checked,cluster_alignment='Hungarian to previous fold in frozen initial descriptor coordinates'))
        print(json.dumps(dict(event='neighbor_fold',**fold_records[-1],seconds=time.time()-began)),flush=True)
    entropy=-(q*torch.where(q>0,q.log(),torch.zeros_like(q))).sum(-1)
    np.savez_compressed(out/'neighbors.npz',query_origins=query_origins.cpu().numpy(),variable_ids=variable_ids.cpu().numpy(),
        neighbor_ids=neighbor_ids.cpu().numpy(),weights=weights.cpu().numpy(),distances=distances.cpu().numpy(),
        q=q.cpu().numpy(),q_random=q_random.cpu().numpy(),confidence=confidence.cpu().numpy(),
        n_eff=n_eff.cpu().numpy(),entropy=entropy.cpu().numpy(),true_cluster=true_cluster.cpu().numpy(),fold=fold_ids.cpu().numpy(),
        strict_nonoverlap_candidate_count=strict_count.cpu().numpy(),history_mean=means.cpu().numpy(),history_std=scales.cpu().numpy())
    result=dict(status='PASS',windows=total,folds=fold_records,coverage=float((confidence>0).float().mean()),
        version='past-neighbor-strict-availability-fallback-v2',
        confidence_rule='Real-label-only fallback when fewer than two strict full-window candidates exist; admitted ranking remains approximate causal past-neighbor retrieval',
        independence_claim=False,
        strict_full_history_future_nonoverlap_coverage=float((strict_count>=2).float().mean()),
        effective_neighbors_quantiles=torch.quantile(n_eff,torch.tensor([0.,.25,.5,.75,1.],device=device)).cpu().tolist(),
        control_equal_soft_label_fraction=float((q==q_random).all(-1).float().mean()),
        no_validation_test_future=True,query_future_used_only_for_expert_training_label=True,
        library_rule='origin+2H <= fold start; same variable/set; pair separation >=2H; at most16',
        negative_control='Same neighbors/weights/distances; future cluster labels permuted in past variable/time strata with exact library label-frequency conservation',
        seconds=time.time()-began,seed=seed)
    atomic_json(out/'audit.json',result);return result


def build_config(config_path,output,device='cuda',seed=1):
    config=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    p=config['protocol'];root=Path(config['artifact'])
    # Numerical checks remain inside the same HOPE worker and have no optimizer.
    time_axis=torch.arange(512,device=device,dtype=torch.float32)[:,None]
    phase=torch.arange(7,device=device,dtype=torch.float32)[None]
    toy=torch.sin(time_axis*.07+phase)+.2*torch.cos(time_axis*.19-phase)
    toy_p=dict(context=64,horizon=16,train_end=512,mode='S',windows=dict(train=(512-64-16+1)*7))
    build_neighbors(toy,toy_p,out/'causal_check_original',seed,fold_width=64)
    changed=toy.clone();changed[128:]+=100
    build_neighbors(changed,toy_p,out/'causal_check_changed_future',seed,fold_width=64)
    a=np.load(out/'causal_check_original/neighbors.npz');b=np.load(out/'causal_check_changed_future/neighbors.npz')
    query=(a['query_origins']==128)
    for key in ['q','q_random','neighbor_ids','confidence','weights']:
        assert np.array_equal(a[key][query],b[key][query]),'Query future leaked into '+key
    normalized=np.load(root/'normalized.npy',allow_pickle=False,mmap_mode='r')
    mean=np.load(root/'scaler_mean.npy',allow_pickle=False);std=np.load(root/'scaler_std.npy',allow_pickle=False)
    # Reconstruct raw units before local history scaling; no query-future normalization.
    values=torch.tensor(np.asarray(normalized[:p['train_end']])*std+mean,device=device,dtype=torch.float32)
    audit=build_neighbors(values,p,out/'training',seed,fold_width=config.get('fold_width',512))
    result=dict(status='PASS',optimizer_trajectories=0,config=config,audit=audit,
        causal_perturbation_check='Changing every value from query time onward leaves its weak labels, weights, neighbors and confidence exactly unchanged',
        actual_gpu=torch.cuda.get_device_name(),artifact=str(out/'training/neighbors.npz'))
    atomic_json(out/'result.json',result);return result
