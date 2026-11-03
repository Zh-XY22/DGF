"""HOPE/GPU check of research base weights against the fixed public snapshot."""
import json
from pathlib import Path
import time
import torch
from uni2ts.model.moirai import MoiraiModule
from .runtime import atomic_json, seed_all


def audit(config_path, output, device='cuda', seed=1):
    config=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    began=time.time();seed_all(seed)
    assert config['model_revision'] and config['model_id']=='Salesforce/moirai-1.0-R-small'
    reference=MoiraiModule.from_pretrained(config['reference_weights'],local_files_only=True).to(device).eval()
    public=MoiraiModule.from_pretrained(config['public_weights'],revision=config['model_revision'],local_files_only=True).to(device).eval()
    a=reference.state_dict();b=public.state_dict();keys_equal=a.keys()==b.keys();records=[]
    with torch.no_grad():
        for name in sorted(a.keys()|b.keys()):
            x=a.get(name);y=b.get(name)
            compatible=x is not None and y is not None and x.shape==y.shape and x.dtype==y.dtype
            finite=compatible and (not x.is_floating_point() or bool(torch.isfinite(x).all() and torch.isfinite(y).all()))
            equal=bool(compatible and finite and torch.equal(x,y))
            records.append(dict(name=name,shape_reference=None if x is None else list(x.shape),
                shape_public=None if y is None else list(y.shape),equal=equal,finite=bool(finite),
                maximum_absolute_difference=0. if equal else float((x.double()-y.double()).abs().max()) if finite and x.numel() else None))
    equal=keys_equal and all(x['equal'] for x in records)
    result=dict(status='PASS' if equal else 'MISMATCH_RECORDED',config=config,
        complete_state_equal=equal,state_keys_equal=keys_equal,state_entries=len(records),records=records,
        reference_parameters=sum(p.numel() for p in reference.parameters()),
        public_parameters=sum(p.numel() for p in public.parameters()),
        optimizer_trajectories=0,optimizer_steps=0,actual_gpu=torch.cuda.get_device_name() if str(device).startswith('cuda') else None,
        seconds=time.time()-began,test_used=False,
        scope='Complete loaded pretrained state equality; no SFT, forecast metric or future-trajectory reproducibility claim')
    atomic_json(out/'result.json',result)
    if not equal:raise AssertionError('Research base and fixed public snapshot differ; mismatch retained for scientific interpretation')
    return result
