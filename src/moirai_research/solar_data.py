"""Declared Solar conversion. Run inside an experiment worker, never a coordinator."""
import gzip
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

from .public_data import prepare_csv,SOLAR_DATASET,SOLAR_VARIABLES
from .runtime import atomic_json


def prepare_solar(source,output,source_revision):
    if not source_revision:
        raise ValueError('An explicit source snapshot label is required')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'data_protocol.json').exists():
        raise ValueError('Completed data protocol exists; use its cache or a new output directory')
    if source.startswith(('http://','https://')):
        raw=out/'solar_AL.txt.gz';partial=raw.with_suffix('.partial')
        with urllib.request.urlopen(source,timeout=120) as remote,partial.open('wb') as stream:
            while True:
                chunk=remote.read(1024*1024)
                if not chunk:break
                stream.write(chunk)
        partial.replace(raw)
    else:
        raw=Path(source).expanduser().resolve()
    opener=gzip.open if raw.suffix=='.gz' else open
    with opener(raw,'rt') as stream:
        original=np.loadtxt(stream,delimiter=',',dtype=np.float64)
    if original.shape!=(52560,137) or not np.isfinite(original).all():
        raise ValueError('Source is not the complete declared 52560 by137 finite Solar matrix')
    hourly=original.reshape(8760,6,137).mean(1)
    frame=pd.DataFrame(hourly,columns=SOLAR_VARIABLES,
        index=pd.date_range('1970-01-01',periods=8760,freq='h',name='synthetic_builder_clock'))
    canonical=out/(SOLAR_DATASET+'.csv');frame.to_csv(canonical)
    # The official builder and explicit arrays both reparse this exact CSV.
    # No query future or nontraining value is used in scaler fitting.
    protocol=prepare_csv(str(canonical),SOLAR_DATASET,str(out),96,source_revision)
    protocol.update(source_raw_rows=52560,source_raw_variables=137,
        source_raw_frequency='10min as declared by original repository',
        aggregation='Arithmetic mean of successive disjoint six-row groups, no clipping or missing-value imputation',
        source_row_mapping='hour i corresponds to raw rows[6*i,6*i+6)',
        timestamp_warning='No original timestamp column; the builder clock is synthetic, not recovered calendar data')
    atomic_json(out/'data_protocol.json',protocol)
    result=dict(status='PASS',kind='data',optimizer_trajectories=0,protocol=protocol,
        artifact=str(out.resolve()),source=str(raw),source_uri=source,source_revision=source_revision,
        source_bytes=raw.stat().st_size,canonical_csv=str(canonical),
        full_label_intervals_verified=True,observed_zero_values_retained=True,
        raw_zero_positions=int((original==0).sum()),hourly_zero_positions=int((hourly==0).sum()),
        raw_window_mapping_saved=True,source_content_identity='UNVERIFIED; supplied source snapshot and raw file preserved',
        qualification_boundary='Data construction only. Model dimensions, router history and index equivalence require separate HOPE numerical checks')
    atomic_json(out/'raw_to_hourly_mapping.json',dict(groups=8760,group_width=6,raw_rows=52560,
        formula='hourly[i,var]=mean(raw[6*i:6*i+6,var])',original_timestamp_column=False))
    atomic_json(out/'result.json',result)
    return result
