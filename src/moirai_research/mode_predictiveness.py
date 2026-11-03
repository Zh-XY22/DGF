"""Train-fitted mode semantics and held-out validation predictiveness.

Descriptive analysis only: no parameter updates, test labels or candidate veto.
All calls to this numerical module belong in the model worker.
"""
from pathlib import Path
import json
import numpy as np
import torch
from .runtime import atomic_json,load_policy,seed_all
from .policy import joints,FIELDS
from .data import structured,assert_complete_future


def descriptor(future,history):
    """The exact eight training-prototype descriptors, using history scaling."""
    scale=np.maximum(history.std(axis=0,ddof=0),1e-3)
    z=(future-history.mean(axis=0))/scale;d1=np.diff(z,axis=0);d2=np.diff(d1,axis=0)
    power=np.abs(np.fft.rfft(z-z.mean(axis=0),axis=0))**2;freq=np.fft.rfftfreq(len(z))
    total=np.maximum(power.sum(axis=0),1e-12)
    return np.asarray([z.mean(),(z[-1]-z[0]).mean(),z.std(axis=0).mean(),np.sqrt((d1*d1).mean()),
        np.sqrt((d2*d2).mean()),((d1[1:]*d1[:-1])<0).mean(),
        (power[freq>=.25].sum(axis=0)/total).mean(),((power*freq[:,None]).sum(axis=0)/total).mean()])

def responsibilities(features,stored):
    z=(features-stored['feature_mean'])/stored['feature_scale']
    distance=((z[:,None,:]-stored['centers'][None,:,:])**2).sum(-1)
    logits=-distance/2.;logits-=logits.max(1,keepdims=True);q=np.exp(logits)
    return q/q.sum(1,keepdims=True)

def scores(probability,target):
    p=np.maximum(probability,1e-30);p=p/p.sum(1,keepdims=True)
    return dict(soft_log_loss=float(-(target*np.log(p)).sum(1).mean()),
        soft_Brier=float(((p-target)**2).sum(1).mean()),
        hard_accuracy=float((p.argmax(1)==target.argmax(1)).mean()))

def train_mapping(weights,responsibility):
    # Each row estimates p(training dynamic cluster | mode). Many-to-one maps
    # are deliberate; no validation labels choose permutations or names.
    counts=weights.T@responsibility+1e-12
    return counts/counts.sum(1,keepdims=True)

def association(left,right,k=8):
    table=np.zeros((k,k),dtype=np.float64);np.add.at(table,(left,right),1.)
    p=table/table.sum();a=p.sum(1);b=p.sum(0);denominator=a[:,None]*b[None,:]
    mask=p>0;mutual=float((p[mask]*np.log(p[mask]/denominator[mask])).sum())
    ha=float(-(a[a>0]*np.log(a[a>0])).sum());hb=float(-(b[b>0]*np.log(b[b>0])).sum())
    return dict(contingency=table.astype(int).tolist(),mutual_information=mutual,
        normalized_mutual_information=2*mutual/(ha+hb) if ha+hb else 0.,
        interpretation='Permutation-invariant descriptive window association; overlapping windows are not independent replicates')

def self_check():
    q=np.eye(8)[np.tile(np.arange(8),3)];w=q*.9+.1/8
    mapping=train_mapping(w,q);p=w@mapping
    permutation=np.asarray([7,2,4,0,6,1,3,5]);shuffled=w[:,permutation]
    np.testing.assert_allclose(p,shuffled@train_mapping(shuffled,q),rtol=1e-12,atol=1e-12)
    assert scores(p,q)['soft_log_loss']<scores(np.ones_like(q)/8,q)['soft_log_loss']
    return dict(status='PASS',training_only_mapping=True,permutation_invariant_predictions=True)

def diagnose(config_path,output,device='cuda'):
    settings=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    check=self_check();config=settings['model_configuration'];p=config['protocol'];h=p['horizon'];d=len(p['variables'])
    if p['dataset']!='ETTh1' or p['mode']!='M' or config['route']!='D-pre':raise ValueError('Only registered ETTh1 D-pre analysis is allowed')
    seed_all(config['seed']);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    artifact=Path(config['artifact']);values=np.load(artifact/'normalized.npy',allow_pickle=False,mmap_mode='r')
    stored=np.load(Path(settings['labels_artifact'])/'mode_labels.npz',allow_pickle=False)
    indices={split:np.load(artifact/(split+'_indices.npy'),allow_pickle=False) for split in ['train','val']}
    np.testing.assert_array_equal(indices['train'],stored['indices'])
    q={}
    for split in ['train','val']:
        lower,upper=(p['context'],p['train_end']) if split=='train' else (p['train_end'],p['val_end'])
        ids=indices[split];assert np.all(ids[:,0]>=lower) and np.all(ids[:,0]+h<=upper)
        features=np.stack([descriptor(values[t:t+h],values[t-512:t]) for t,var in ids])
        q[split]=responsibilities(features,stored)
        if split=='train':
            np.testing.assert_allclose(features,stored['features'],atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(q[split],stored['q'],atol=1e-10,rtol=1e-10)
    baseline=q['train'].mean(0);cache={};aliases=[]
    for case in settings['cases']:
        uri=case['checkpoint_source']
        if uri in cache:aliases.append(dict(label=case['label'],checkpoint_source=uri,canonical=cache[uri]['canonical']));continue
        actual=dict(config,checkpoint=case['checkpoint_local'],checkpoint_kind='policy_state')
        policy=load_policy(actual,device);policy.eval();target_device=next(policy.parameters()).device
        from uni2ts.model.moirai import MoiraiForecast
        forecast=MoiraiForecast(module=policy.module,context_length=p['context'],patch_size=p['patch'],prediction_length=h,
            num_samples=1,target_dim=d,feat_dynamic_real_dim=0,past_feat_dynamic_real_dim=0)
        weights={};mode_distance={};head_classes=[];input_isolation=[];kappa_values=[]
        with torch.no_grad():
            for split in ['train','val']:
                allw=[];distances=[]
                for start in range(0,len(indices[split]),32):
                    ids=indices[split][start:start+32];past=np.stack([values[t-p['context']:t] for t,var in ids]);truth=np.stack([values[t:t+h] for t,var in ids])
                    past=torch.as_tensor(past,device=target_device,dtype=torch.float32);truth=torch.as_tensor(truth,device=target_device,dtype=torch.float32)
                    def packed_batch(y):
                        packed=forecast._convert(p['patch'],past,torch.ones_like(past,dtype=torch.bool),torch.zeros(past.shape[:2],device=target_device,dtype=torch.bool),
                            future_target=y,future_observed_target=torch.ones_like(y,dtype=torch.bool),future_is_pad=torch.zeros(y.shape[:2],device=target_device,dtype=torch.bool))
                        b={k:v for k,v in zip(FIELDS[:-1],packed)};b['patch_size']=torch.ones_like(b['time_id'])*p['patch'];assert_complete_future(b,h*d)
                        return b
                    b=packed_batch(truth);experts,target,w,alpha,kappa=joints(policy(b),b)
                    means=torch.stack([structured(e.mean,d,h) for e in experts],1)
                    assert torch.equal(structured(target,d,h),truth)
                    if start==0:
                        changed=packed_batch(torch.zeros_like(truth));other,_,other_w,_,_=joints(policy(changed),changed)
                        other_means=torch.stack([structured(e.mean,d,h) for e in other],1)
                        torch.testing.assert_close(w,other_w,atol=0,rtol=0)
                        torch.testing.assert_close(means,other_means,atol=0,rtol=0)
                        input_isolation.append(dict(split=split,status='PASS',future_label_change_did_not_change_router_or_mode_means=True))
                    allw.append(w.double().cpu().numpy())
                    pair=(means[:,:,None]-means[:,None,:]).square().mean((-2,-1)).sum((1,2))/56
                    distances.extend(pair.double().cpu().tolist())
                    if kappa is not None:kappa_values.extend(kappa.flatten().double().cpu().tolist())
                    if split=='val':
                        array=means.cpu().numpy()
                        head_features=np.stack([descriptor(array[i,k],values[t-512:t]) for i,(t,var) in enumerate(ids) for k in range(8)])
                        head_classes.extend(responsibilities(head_features,stored).argmax(1).reshape(len(ids),8).tolist())
                weights[split]=np.concatenate(allw);mode_distance[split]=float(np.mean(distances))
        mapping=train_mapping(weights['train'],q['train']);mapped=weights['val']@mapping
        frequency=np.broadcast_to(baseline,q['val'].shape);measured=scores(mapped,q['val']);reference=scores(frequency,q['val'])
        aligned=scores(weights['val'],q['val']) if settings['method']=='D01' else None
        w=weights['val'];entropy=-(w*np.log(np.maximum(w,1e-30))).sum(1)
        canonical='checkpoint_'+str(len(cache));winner=w.argmax(1);hc=np.asarray(head_classes)
        result=dict(canonical=canonical,checkpoint_source=uri,train_windows=len(q['train']),validation_windows=len(q['val']),
            train_fitted_cluster_given_mode=mapping.tolist(),validation_mapped_router=measured,validation_training_frequency_baseline=reference,
            mapped_log_loss_delta=measured['soft_log_loss']-reference['soft_log_loss'],mapped_Brier_delta=measured['soft_Brier']-reference['soft_Brier'],
            D01_preparation_index_aligned_router=aligned,unmapped_mode_cluster_association=association(winner,q['val'].argmax(1)),
            router_entropy_mean=float(entropy.mean()),router_mean_probability=w.mean(0).tolist(),
            deterministic_MAP_mode_occupancy=(np.bincount(winner,minlength=8)/len(w)).tolist(),
            mode_mean_squared_distance_by_split=mode_distance,head_dynamic_class_distribution=[(np.bincount(hc[:,k],minlength=8)/len(hc)).tolist() for k in range(8)],
            MAP_head_dynamic_class_agreement=float((hc[np.arange(len(hc)),winner]==q['val'].argmax(1)).mean()),
            kappa_min=float(min(kappa_values)) if kappa_values else None,kappa_max=float(max(kappa_values)) if kappa_values else None,
            future_label_input_isolation_checks=input_isolation,source_receipt=settings['source_training_receipt'],
            interpretation='Mapping and baseline use training labels only; validation future labels only score held-out predictiveness. Mode distance/occupancy alone do not establish useful modes. This post-training analysis does not select candidates.')
        cache[uri]=result;aliases.append(dict(label=case['label'],checkpoint_source=uri,canonical=canonical))
        del policy
        if target_device.type=='cuda':torch.cuda.empty_cache()
        atomic_json(out/'partial_result.json',dict(cases=list(cache.values()),aliases=aliases))
    result=dict(status='PASS',path_id=settings['path_id'],method=settings['method'],horizon=h,seed=config['seed'],
        mathematical_self_check=check,cases=list(cache.values()),aliases=aliases,optimizer_updates=0,optimizer_trajectories=0,
        test_used=False,label_fit_split='train',mapping_fit_split='train',evaluation_split='val',
        interpretation='Exploratory two-seed held-out mode diagnosis, not five-seed confirmation or independent window significance')
    atomic_json(out/'result.json',result);return result
