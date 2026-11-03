"""Frozen ETTh1 warmup checkpoint evaluation with explicit precision."""
import json
import math
from pathlib import Path
import time
import torch
from .evaluation import evaluation
from .point_test_gate import test_release
from .runtime import atomic_json,load_policy,seed_all


def finite_record(value,location='$',invalid=None):
    if invalid is None:invalid=[]
    if isinstance(value,float) and not math.isfinite(value):
        invalid.append(dict(location=location,raw_value=str(value),status='NONFINITE_RECORDED_NOT_POINT_VETO'))
        return None
    if isinstance(value,dict):return {k:finite_record(v,location+'.'+k,invalid) for k,v in value.items()}
    if isinstance(value,list):return [finite_record(v,location+'['+str(i)+']',invalid) for i,v in enumerate(value)]
    return value


def evaluate_frozen(config_path,output,device='cuda',seed=1):
    config=json.loads(Path(config_path).read_text())
    release=test_release(config['test_release_receipt'],config['scientific_path'])
    if release['campaign']!='sft_warmup_etth1_20261028_000001':raise ValueError('Not a warmup release')
    if config['seed']!=seed or config['dataset']!='ETTh1' or config['protocol']['horizon']!=96:
        raise ValueError('Warmup checkpoint scope differs')
    if config.get('splits')!=['val','test']:raise ValueError('Both full splits are required')
    if config.get('diagnostic_limit') is not None:raise ValueError('Formal windows cannot be truncated')
    seed_all(seed);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    began=time.time();policy=load_policy(config,device).eval();results={};comparisons=[]
    for split in ['val','test']:
        metric=evaluation(policy,config['protocol'],split,None,seed,config['artifact'],None)
        if not metric['complete'] or metric['windows']!=config['protocol']['windows'][split]:
            raise ValueError('Incomplete formal split')
        results[split]=metric
        atomic_json(out/(split+'.json'),finite_record(metric))
        if split=='val':
            for key in ['MAP_MSE','MAP_MAE','MSE','MAE','mixture_analytic_MSE','mixture_analytic_MAE']:
                actual=metric[key];expected=config['expected_validation'][key]
                tolerance=max(1e-7,abs(expected)*1e-6)
                passed=math.isfinite(actual) and abs(actual-expected)<=tolerance
                comparisons.append(dict(metric=key,actual=actual,expected=expected,tolerance=tolerance,passed=passed))
            atomic_json(out/'validation_parity.json',comparisons)
            if not all(row['passed'] for row in comparisons):raise ValueError('Full validation does not match frozen evidence')
    result=dict(status='PASS',config=config,results=results,validation_parity=comparisons,
        optimizer_updates=0,test_used=True,test_used_for_selection=False,seconds=time.time()-began,
        actual_gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,
        model_precision=str(next(policy.parameters()).dtype),metric_precision='float64',
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        native_horizon=96,world_size=1,raw_prediction_storage='Metrics only; no prediction arrays requested')
    invalid=[];serialized=finite_record(result,invalid=invalid);serialized['nonfinite_values']=invalid
    atomic_json(out/'result.json',serialized)
    return result
