"""Independent NumPy reduction of retained point predictions; no model loading."""
import argparse,json
from pathlib import Path
import numpy as np


def reduce_blocks(directory):
    directory=Path(directory);totals={};count=0;windows=0;diagnostics=[]
    for file in sorted(directory.glob('block_*.npz')):
        with np.load(file,allow_pickle=False) as z:
            y=z['target'].astype(np.float64);mask=z['observed'].astype(bool)
            if mask.shape!=y.shape:raise ValueError('Observed mask differs from target shape')
            count+=int(mask.sum());windows+=len(y)
            for local_index,window in enumerate(z['window_indices']):
                file=directory/('diagnostic_'+str(int(window))+'.npz')
                if file.exists():
                    with np.load(file,allow_pickle=False) as diag:
                        alpha=diag['alpha'].astype(np.float64);w=diag['weights'].astype(np.float64)
                        expected=alpha/alpha.sum() if alpha.size else w/w.sum()
                        mode=int(np.argmax(expected));manual=diag['means'][mode].astype(np.float64)
                        if not np.array_equal(manual,z['MAP'][local_index]):raise ValueError('Manual expected-Dirichlet MAP differs from retained prediction')
                        diagnostics.append(dict(window=int(window),mode=mode,kappa=float(alpha.sum()) if alpha.size else None,positive_alpha=bool((alpha>0).all()) if alpha.size else None,scope='Retained representative diagnostic window, not full-window mode occupancy'))
            for protocol,mse_name,mae_name in [('P1','MAP','MAP'),('P2','sample_mean','sample_median'),('P3','mixture_analytic','mixture_analytic')]:
                x=z[mse_name].astype(np.float64);a=z[mae_name].astype(np.float64)
                if x.shape!=y.shape or a.shape!=y.shape or not np.isfinite(x[mask]).all() or not np.isfinite(a[mask]).all():raise ValueError('Invalid retained point forecast')
                entry=totals.setdefault(protocol,[0.,0.]);entry[0]+=float(np.square(x[mask]-y[mask]).sum(dtype=np.float64));entry[1]+=float(np.abs(a[mask]-y[mask]).sum(dtype=np.float64))
    if not count:raise ValueError('No complete prediction blocks')
    return dict(windows=windows,effective_elements=count,manual_MAP_diagnostics=diagnostics,protocols={k:dict(MSE=v[0]/count,MAE=v[1]/count) for k,v in totals.items()},scope='Independent reduction of saved predictions; no model prediction or optimizer update')

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--metrics',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=reduce_blocks(a.directory);expected=json.loads(Path(a.metrics).read_text())
    names={'P1':('MAP_MSE','MAP_MAE'),'P2':('MSE','MAE'),'P3':('mixture_analytic_MSE','mixture_analytic_MAE')}
    assert result['windows']==expected['windows']
    for protocol,keys in names.items():
        for metric,key in zip(['MSE','MAE'],keys):
            if abs(result['protocols'][protocol][metric]-expected[key])>1e-10*max(1.,abs(expected[key])):raise ValueError('Independent metric mismatch: '+protocol+' '+metric)
    result['status']='PASS';Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
