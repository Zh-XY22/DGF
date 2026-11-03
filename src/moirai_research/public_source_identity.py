"""Worker-only public-version comparison; unavailable network stays UNKNOWN."""
from pathlib import Path
import json,urllib.request
from .runtime import atomic_json

def inspect(model_root,csv_file,output,source_versions,public_csv_file=None,skip_model_fetch=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);scratch=out.parent.parent/'public_source_identity';scratch.mkdir(exist_ok=True)
    data=dict(status='UNKNOWN_NOT_FETCHED');model=dict(status='UNKNOWN_NOT_FETCHED')
    url='https://raw.githubusercontent.com/zhouhaoyi/ETDataset/'+source_versions['data_revision']+'/ETT-small/ETTh1.csv'
    for attempt in range(2):
        try:
            if public_csv_file is not None:content=Path(public_csv_file).read_bytes()
            else:
                with urllib.request.urlopen(url,timeout=30) as response:content=response.read(8*1024*1024+1)
            if len(content)>8*1024*1024:raise ValueError('Unexpected public CSV size')
            file=scratch/'ETTh1.csv';file.write_bytes(content)
            import pandas as pd
            a=pd.read_csv(csv_file);b=pd.read_csv(file)
            columns=list(a.columns)==list(b.columns);shape=a.shape==b.shape
            equal=bool(columns and shape and a.equals(b))
            data=dict(status='EXACT_PUBLIC_DATA_MATCH' if equal else 'DATA_MISMATCH',same_columns=columns,same_shape=shape,
                cached_rows=len(a),public_rows=len(b),all_parsed_cells_equal=equal,
                raw_file_bytes_equal=Path(csv_file).read_bytes()==content,public_source='Fixed-version ETDataset ETTh1 CSV',fetch_attempt=attempt+1)
            break
        except Exception as error:
            data=dict(status='UNKNOWN_PUBLIC_FETCH_OR_PARSE_FAILED',error_type=type(error).__name__,fetch_attempt=attempt+1)
    if skip_model_fetch:model=dict(status='UNKNOWN_PUBLIC_SNAPSHOT_UNAVAILABLE',reason='Two bounded public network attempts already failed; no model equality asserted')
    for attempt in ([] if skip_model_fetch else range(2)):
        try:
            from huggingface_hub import snapshot_download
            public=snapshot_download(repo_id=source_versions['model_id'],revision=source_versions['model_revision'],
                cache_dir=str(scratch/'model_cache'),allow_patterns=['config.json','model.safetensors','pytorch_model.bin'],local_files_only=False)
        except Exception as error:
            model=dict(status='UNKNOWN_PUBLIC_SNAPSHOT_UNAVAILABLE',error_type=type(error).__name__,fetch_attempt=attempt+1)
            continue
        from .pretrained_audit import audit
        file=out/'model_comparison_configuration.json';atomic_json(file,dict(reference_weights=str(model_root),public_weights=public,
            model_id=source_versions['model_id'],model_revision=source_versions['model_revision']))
        try:comparison=audit(file,out/'model_comparison',device='cuda',seed=198631)
        except AssertionError:comparison=json.loads((out/'model_comparison/result.json').read_text())
        model=dict(status='EXACT_PUBLIC_MODEL_MATCH' if comparison['complete_state_equal'] else 'MODEL_MISMATCH',
            complete_loaded_state_equal=comparison['complete_state_equal'],state_entries=comparison['state_entries'],
            source='Fixed-version public snapshot and complete loaded parameter/buffer equality',comparison_receipt=str(out/'model_comparison/result.json'))
        break
    result=dict(status='PASS',operation='PUBLIC_SOURCE_INSPECTION_COMPLETED',data_comparison=data,model_comparison=model,
        optimizer_updates=0,optimizer_trajectories=0,test_prediction_used=False,test_used=False,
        interpretation='PASS describes a completed inspection, not a claim that unavailable public material matched. Explicit mismatches block formal release; network gaps stay UNKNOWN and are reported. No model weights or CSV are included in this receipt.')
    atomic_json(out/'result.json',result);return result
