"""HOPE-only public data construction and old/new numerical equivalence checks."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from .public_data import prepare_csv
from .runtime import atomic_json,load_policy,seed_all
from .data import datasets,batch
from .policy import joints


def audit(config_path,output,device='cuda',seed=1):
    config=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    seed_all(seed);began=time.time();case=config['case'];old=Path(config['artifact'])
    p=prepare_csv(config['source'],case['dataset'],out/'data',96,config.get('source_revision'))
    comparisons=[]
    for name in ['normalized.npy','scaler_mean.npy','scaler_std.npy','train_indices.npy','val_indices.npy','test_indices.npy']:
        a=np.load(old/name,allow_pickle=False);b=np.load(out/'data'/name,allow_pickle=False)
        assert a.shape==b.shape and a.dtype==b.dtype,name
        np.testing.assert_array_equal(a,b,err_msg=name)
        comparisons.append(dict(array=name,shape=list(a.shape),dtype=str(a.dtype),exact=True))
    # Compare actual public/native transform output, including absolute timing and masks.
    policy=load_policy(case,device).eval();a=datasets(policy,case['dataset'],case['protocol']);b=datasets(policy,case['dataset'],p)
    batches=[]
    with torch.no_grad():
        for split in ['train','val','test']:
            ix=sorted({0,1,len(a[split])//2,len(a[split])-1})
            x=batch(a[split],ix,device);y=batch(b[split],ix,device)
            assert x.keys()==y.keys()
            for key in x:torch.testing.assert_close(x[key],y[key],atol=0,rtol=0)
            ae,at,aw,_,_=joints(policy(x),x);be,bt,bw,_,_=joints(policy(y),y)
            torch.testing.assert_close(at,bt,atol=0,rtol=0);torch.testing.assert_close(aw,bw,atol=0,rtol=0)
            for u,v in zip(ae,be):torch.testing.assert_close(u.mean,v.mean,atol=0,rtol=0)
            batches.append(dict(split=split,indices=ix,fields=list(x),all_fields_exact=True,prediction_means_exact=True))
    result=dict(status='PASS',optimizer_trajectories=0,optimizer_steps=0,config=config,
        data_protocol=p,array_comparisons=comparisons,batch_comparisons=batches,
        actual_gpu=torch.cuda.get_device_name(),seconds=time.time()-began,
        scope='Data construction and selected packed windows; not a new test-metric result or clean-install acceptance')
    atomic_json(out/'result.json',result);return result
