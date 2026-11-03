"""HOPE-only, zero-update checks for independent point-readout failure isolation."""
import json
import math
import time
import types
from pathlib import Path
from unittest.mock import patch

import torch

from . import evaluation as current
from .actions import NonfiniteForecastSample
from .point_selection import finite_point, same_device_readout_check
from .runtime import atomic_json, load_policy, seed_all


def audit(config_path, output, device='cuda', seed=1):
    config=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    legacy=types.ModuleType('moirai_research._finite_readout_baseline')
    legacy.__package__='moirai_research'
    baseline=Path(config['baseline_evaluation_source'])
    legacy.__file__=str(baseline)
    exec(compile(baseline.read_text(),str(baseline),'exec'),legacy.__dict__)
    results=[];began=time.time()
    for index,case in enumerate(config['cases']):
        seed_all(seed);policy=load_policy(case,device).eval()
        kwargs=dict(protocol=case['protocol'],split='val',seed=seed,artifact=case['artifact'],limit=2)
        original=legacy.evaluation(policy,**kwargs)
        normal=current.evaluation(policy,**kwargs)
        checked=[]
        for name,value in original.items():
            if name=='seconds' or not isinstance(value,(int,float)) or isinstance(value,bool):continue
            if not math.isfinite(value):continue
            actual=normal.get(name)
            assert isinstance(actual,(int,float)) and math.isfinite(actual)
            assert abs(value-actual)<=max(1e-9,abs(value)*1e-7),(name,value,actual)
            checked.append(name)
        real_rollout=current.rollout
        def unusable_sample(*args,**kwargs):
            raise NonfiniteForecastSample(torch.tensor([float('inf'),float('nan')],device=device))
        with patch.object(current,'rollout',side_effect=unusable_sample):
            unavailable=current.evaluation(policy,output=out/f'case-{index}'/'sample-failure',**kwargs)
        assert unavailable['point_protocol_status']['P2']['status']=='INVALID'
        assert unavailable['MSE'] is None and unavailable['MAE'] is None
        assert unavailable['CRPS'] is None and unavailable['coverage_80'] is None
        assert unavailable['mode_occupancy'] is None and unavailable['maximum_absolute_prediction'] is None
        assert unavailable['worst_windows_complete'] is False
        assert list((out/f'case-{index}'/'sample-failure').glob('nonfinite_samples_*.npz'))
        for key in ['MAP_MSE','MAP_MAE','mixture_analytic_MSE','mixture_analytic_MAE']:
            assert finite_point(unavailable[key]) and unavailable[key]==normal[key]
        def overflowing_mean(*args,**kwargs):
            actions=real_rollout(*args,**kwargs)
            actions['forecast'].fill_(torch.finfo(actions['forecast'].dtype).max*.75)
            return actions
        with patch.object(current,'rollout',side_effect=overflowing_mean):
            overflow=current.evaluation(policy,output=out/f'case-{index}'/'mean-overflow',**kwargs)
        assert overflow['MSE'] is None and overflow['point_protocol_status']['P2']['status']=='INVALID'
        for key in ['MAP_MSE','MAP_MAE','mixture_analytic_MSE','mixture_analytic_MAE']:
            assert finite_point(overflow[key]) and overflow[key]==normal[key]
        def unrelated_failure(*args,**kwargs):raise RuntimeError('qualification: unrelated execution error')
        with patch.object(current,'rollout',side_effect=unrelated_failure):
            try:current.evaluation(policy,**kwargs)
            except RuntimeError as error:assert 'unrelated execution error' in str(error)
            else:raise AssertionError('Unrelated execution error must not be hidden')
        first=dict(MSE=1.,MAE=.8,MAP_MSE=1.2,MAP_MAE=.9,mixture_analytic_MSE=1.1)
        record=dict(validation_metric='mixture_analytic_MSE')
        scope=same_device_readout_check(record,first,dict(first))
        for bad in (1.5,None,float('nan')):
            try:same_device_readout_check(record,first,dict(first,mixture_analytic_MSE=bad))
            except RuntimeError:pass
            else:raise AssertionError('P3 readout mismatch was falsely verified')
        results.append(dict(dataset=case['dataset'],route=case['route'],finite_metric_parity=checked,
            sampled_failure_preserves_P1_P3=True,mean_overflow_preserves_P1_P3=True,
            unrelated_errors_propagate=True,raw_abnormal_forecasts_saved=True,reload_scope=scope,
            complete_validation=False,scope='Two fixed validation windows, not formal performance evaluation'))
        atomic_json(out/'checks.json',results)
        del policy;torch.cuda.empty_cache()
    assert len(results)==4 and {(r['dataset'],r['route']) for r in results}=={
        (d,a) for d in ('ETTh1','ETTm1') for a in ('A0','D-pre')}
    result=dict(status='PASS',cases=results,config=config,optimizer_trajectories=0,optimizer_steps=0,
        test_used=False,actual_gpu=torch.cuda.get_device_name(),seconds=time.time()-began,
        scope='Targeted real CUDA readout-failure isolation and finite-baseline parity; no training or full performance claim')
    atomic_json(out/'result.json',result)
    return result
