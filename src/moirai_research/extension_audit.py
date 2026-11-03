"""Worker-only qualification for the declared all-variable Solar seam."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import datasets,batch,structured,assert_complete_future,forecast_dimensions
from .policy import joints
from .runtime import seed_all,load_policy,atomic_json


def audit(config_path,output,device='cuda',seed=1):
    config=json.loads(Path(config_path).read_text());p=config['protocol']
    assert p['dataset']=='SolarAL_hourly_mean' and p['horizon']==96 and p['mode']=='M'
    assert len(p['variables'])==137 and p['ddof']==1 and p['eval_input_block']==1
    assert (p['context'],p['patch'],p['train_end'],p['val_end'],p['test_end'])==(512,64,6132,7008,8760)
    out=Path(output);out.mkdir(parents=True,exist_ok=True);began=time.time();seed_all(seed)
    policy=load_policy(config,device).eval();versions={name:v._version for name,v in policy.named_parameters()}
    data=datasets(policy,config['dataset'],p);dim=forecast_dimensions(p);h=p['horizon']
    values=np.load(Path(config['artifact'])/'normalized.npy',allow_pickle=False,mmap_mode='r')
    records=[]
    with torch.no_grad():
        for split in ['train','val','test']:
            indices=np.load(Path(config['artifact'])/(split+'_indices.npy'),allow_pickle=False)
            for index in [0,len(indices)-1]:
                packed=batch(data[split],[index],device);valid=assert_complete_future(packed,h*dim)
                target=packed['target'][valid].reshape(1,h*dim)
                origin=int(indices[index,0]);expected=torch.as_tensor(np.asarray(values[origin:origin+h]),device=device,dtype=target.dtype)[None]
                torch.testing.assert_close(structured(target,dim,h),expected,rtol=1e-6,atol=1e-7)
                variates=packed['variate_id'][packed['prediction_mask']].unique()
                assert torch.equal(variates,torch.arange(dim,device=device,dtype=variates.dtype))
                # Actual legal zero labels must remain observed; they do not
                # become missing merely because the target value is zero.
                altered={k:v.clone() if torch.is_tensor(v) else v for k,v in packed.items()}
                altered['target'][valid]=0
                assert torch.equal(assert_complete_future(altered,h*dim),valid)
                original_output=policy(packed);changed_output=policy(altered)
                original_experts,_,w,alpha,_=joints(original_output,packed)
                changed_experts,_,changed_w,changed_alpha,_=joints(changed_output,altered)
                torch.testing.assert_close(w,changed_w,rtol=0,atol=0)
                torch.testing.assert_close(alpha,changed_alpha,rtol=0,atol=0)
                for original,changed in zip(original_experts,changed_experts):
                    assert torch.isfinite(original.mean).all()
                    torch.testing.assert_close(original.mean,changed.mean,rtol=0,atol=0)
                records.append(dict(split=split,index=index,origin=origin,variables=dim,
                    valid_future_positions=int(valid.sum()),future_labels_not_used_by_router_or_predictor=True,
                    target_matches_canonical_array=True,legitimate_zero_observed=True))
                atomic_json(out/'checks.json',records)
    assert all(v._version==versions[name] for name,v in policy.named_parameters())
    result=dict(status='PASS',config=config,checks=records,optimizer_trajectories=0,optimizer_steps=0,
        parameters_unchanged=True,actual_gpu=torch.cuda.get_device_name(),seconds=time.time()-began,
        scope='Protocol/packing/zero/future-label invariance only. No accuracy metric, training or test-based selection',
        calendar_boundary='Canonical model inputs use row-relative indices and no dynamic calendar covariates; real timestamp recovery is not claimed')
    atomic_json(out/'result.json',result);return result
