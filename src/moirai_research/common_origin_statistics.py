"""HOPE-side reduction of retained window scalars on predeclared common origins."""
from pathlib import Path
import numpy as np

METRICS=('MSE','MAE','MAP_MSE','MAP_MAE','mixture_analytic_MSE','mixture_analytic_MAE','CRPS',
         'raw_MSE','raw_MAE','raw_MAP_MSE','raw_MAP_MAE','raw_mixture_analytic_MSE','raw_mixture_analytic_MAE')

def reduce_blocks(directory,protocol,split,require_complete=True):
    if protocol['dataset']!='ETTh1' or protocol['mode']!='M':raise ValueError('Common-origin diagnostic is registered for ETTh1 M only')
    lower,upper=(protocol['train_end'],protocol['val_end']) if split=='val' else (protocol['val_end'],protocol['test_end'])
    maximum=upper-720;sums={k:0. for k in METRICS};counts={k:0 for k in METRICS};invalid=set();origins=[]
    for file in sorted(Path(directory).glob('block_*.npz')):
        with np.load(file,allow_pickle=False) as stored:
            index=stored['indices'];chosen=(index[:,0]>=lower)&(index[:,0]<=maximum);origins.extend(index[chosen,0].tolist())
            for key in METRICS:
                if 'metric_'+key not in stored:continue
                values=stored['metric_'+key][chosen]
                if not np.isfinite(values).all():invalid.add(key)
                else:sums[key]+=float(values.astype(np.float64).sum());counts[key]+=values.size
    expected=maximum-lower+1
    if len(set(origins))!=len(origins):raise ValueError('Duplicate multivariate prediction origins in common-window reduction')
    if require_complete and sorted(origins)!=list(range(lower,maximum+1)):raise ValueError('Incomplete common-origin diagnostic; no silent window deletion')
    if not origins:raise ValueError('No retained common-origin window statistics')
    return dict(status='PASS' if require_complete else 'DIAGNOSTIC_ONLY',split=split,common_origin_start=lower,common_origin_end_inclusive=maximum,
        common_windows=len(origins),expected_common_windows=expected,maximum_horizon=720,
        metrics={k:sums[k]/counts[k] if counts[k] and k not in invalid else None for k in METRICS},
        invalid_or_unavailable_metrics=[k for k in METRICS if not counts[k] or k in invalid],
        interpretation='Same prediction origins across four H; each stored per-variable error already averages only its own valid H positions. Diagnostic only; full standard-window metrics and frozen selections are unchanged.')

def boundary_self_check(output):
    """Small worker-only numerical boundary fixture, independent of a model."""
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    values=np.repeat(np.array([[1.],[3.],[100.]]),7,axis=1)
    np.savez_compressed(out/'block_00000.npz',indices=np.array([[8640,0],[10800,0],[10801,0]]),metric_MAP_MSE=values)
    result=reduce_blocks(out,dict(dataset='ETTh1',mode='M',train_end=8640,val_end=11520,test_end=14400),'val',require_complete=False)
    assert result['common_windows']==2 and result['metrics']['MAP_MSE']==2.
    return dict(status='PASS',lower_boundary_included=True,upper_boundary_included=True,later_origin_excluded=True)
