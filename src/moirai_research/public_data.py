"""Portable public CSV -> official LSF storage and explicit label-index protocol.

No cluster paths or default unpinned download. New content-digest computation is
intentionally absent; provenance remains explicit and is never called verified.
"""
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd
from uni2ts.data.builder.simple import SimpleFinetuneDatasetBuilder,SimpleEvalDatasetBuilder
from .runtime import atomic_json

VARIABLES=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
SOLAR_DATASET='SolarAL_hourly_mean'
SOLAR_VARIABLES=[f'plant_{i:03d}' for i in range(137)]


def prepare_csv(source,dataset,output,horizon=96,source_revision=None,context_override=None,train_origin_start=None):
    if dataset not in ['ETTh1','ETTm1',SOLAR_DATASET]:raise ValueError('Unsupported declared data protocol')
    if horizon not in [96,192,336,720]:raise ValueError('Unsupported declared horizon')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if source.startswith(('https://','http://')):
        if not source_revision:raise ValueError('Public URL requires an explicit source revision label')
        csv=out/(dataset+'.csv')
        with urllib.request.urlopen(source,timeout=120) as remote,csv.open('wb') as stream:
            while True:
                chunk=remote.read(1024*1024)
                if not chunk:break
                stream.write(chunk)
    else:csv=Path(source).expanduser().resolve()
    frame=pd.read_csv(csv,index_col=0,parse_dates=True)
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    variables=SOLAR_VARIABLES if dataset==SOLAR_DATASET else VARIABLES
    assert list(frame.columns)==variables and np.isfinite(frame.to_numpy()).all()
    h=int(horizon);is_hour=dataset=='ETTh1';context=5000 if is_hour else 4000
    train_end,val_end,test_end=(8640,11520,14400) if is_hour else (34560,46080,57600)
    mode='M' if is_hour else 'S';patch=64 if is_hour else 128;freq='h' if is_hour else '15min'
    if dataset==SOLAR_DATASET:
        assert len(frame)==8760 and h==96,'Only the declared hourly Solar H96 protocol is implemented'
        context=512;train_end,val_end,test_end=6132,7008,8760;mode='M';patch=64;freq='h'
        is_hour=True
    if context_override is not None:
        if dataset!='ETTh1' or context_override not in (512,2048):raise ValueError('Only the registered ETTh1 context contrasts are supported')
        context=int(context_override)
    origin_start=context if train_origin_start is None else int(train_origin_start)
    if origin_start<context or origin_start+h>train_end:raise ValueError('Invalid common training prediction origin')
    assert len(frame)>=test_end and frame.index.to_series().diff().dropna().eq(pd.Timedelta('1h' if is_hour else '15min')).all()
    mean=frame.iloc[:train_end].mean();std=frame.iloc[:train_end].std(ddof=1)
    assert (std>0).all();normalized=((frame-mean)/std).to_numpy(dtype=np.float64)
    for name,array in [('normalized.npy',normalized),('scaler_mean.npy',mean.to_numpy()),('scaler_std.npy',std.to_numpy())]:
        path=out/name;tmp=path.with_suffix('.partial')
        with tmp.open('wb') as stream:np.save(stream,array,allow_pickle=False)
        tmp.replace(path)
    storage=out/'processed';storage.mkdir(exist_ok=True);dataset_type='wide_multivariate' if mode=='M' else 'wide'
    train=SimpleFinetuneDatasetBuilder(dataset,None,None,None,None,None,mode=mode,storage_path=storage)
    train.build_dataset(file=csv,dataset_type=dataset_type,offset=train_end,freq=freq,normalize=True)
    SimpleEvalDatasetBuilder(dataset+'_eval',None,None,None,None,None,None,mode=mode,storage_path=storage).build_dataset(
        file=csv,dataset_type=dataset_type,freq=freq,mean=train.mean,std=train.std)
    np.testing.assert_allclose(train.mean.to_numpy(),mean.to_numpy(),atol=0,rtol=0)
    np.testing.assert_allclose(train.std.to_numpy(),std.to_numpy(),atol=0,rtol=0)
    windows={}
    for split,begin,end in [('train',origin_start,train_end),('val',train_end,val_end),('test',val_end,test_end)]:
        origins=np.arange(begin,end-h+1,dtype=np.int64);groups=1 if mode=='M' else len(variables)
        indices=np.column_stack([origins.repeat(groups),np.tile(np.arange(groups),len(origins))])
        assert (indices[:,0]>=begin).all() and (indices[:,0]+h<=end).all()
        np.save(out/(split+'_indices.npy'),indices,allow_pickle=False);windows[split]=len(indices)
    protocol=dict(dataset=dataset,mode=mode,context=context,patch=patch,horizon=h,variables=variables,
        train_end=train_end,val_end=val_end,test_end=test_end,ddof=1,train_distance=1,rl_val_distance=1,test_distance=1,
        eval_samples=100,windows=windows,builder_root=str(storage.resolve()),artifact=str(out.resolve()),
        scaler_mean=mean.to_dict(),scaler_std_ddof1=std.to_dict(),source_revision=source_revision,
        source_identity_status='UNVERIFIED_CONTENT; source file and declared revision preserved',
        source_first_timestamp=str(frame.index[0]),source_frequency=pd.infer_freq(frame.index),source_rows=len(frame),
         observed_zero_values_retained=True,missing_values='rejected for this complete-window protocol',
        builder_normalization='official train/eval Arrow builders use std+1e-10; explicit forecast arrays use std without epsilon')
    if train_origin_start is not None:protocol['train_origin_start']=origin_start
    if dataset==SOLAR_DATASET:
        protocol.update(eval_input_block=1,diagnostic_microbatch=1,diagnostic_input_limit=1,
            calendar_semantics='Synthetic hourly builder clock; canonical identity is source row index',
            protocol_origin='Predeclared engineering extension; not established equivalent to paper Solar')
    atomic_json(out/'data_protocol.json',protocol)
    atomic_json(out/'result.json',dict(status='PASS',kind='data',optimizer_trajectories=0,protocol=protocol,
       artifact=str(out.resolve()),source=str(csv),source_uri=source,source_bytes=csv.stat().st_size,
       full_label_intervals_verified=True,
       historical_equivalence='Same native builders/scaler/split formulas; numerical old/new parity requires independent HOPE evidence'))
    return protocol
