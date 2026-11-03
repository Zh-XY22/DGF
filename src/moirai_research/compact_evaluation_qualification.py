"""Installed compact-output entry; unchanged numerical metrics, no optimizer."""
from pathlib import Path
import json,subprocess,sys,math
import numpy as np
import torch
from .runtime import atomic_json,seed_all,load_policy
from .public_data import prepare_csv
from .policy import Policy
from .train import weights_checkpoint
from .evaluation import evaluation

def qualify(model_root,csv_file,output,revision,check_common_origins=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);scratch=out.parent.parent/'compact_eval_work';scratch.mkdir(exist_ok=True)
    cases=[]
    common_boundary=None
    if check_common_origins:
        from .common_origin_statistics import boundary_self_check
        common_boundary=boundary_self_check(scratch/'common_boundary_fixture')
    for h in [96,192,336,720]:
        data=scratch/f'data_H{h}';protocol=prepare_csv(str(csv_file),'ETTh1',data,horizon=h,source_revision=revision)
        protocol=dict(protocol,eval_input_block=4)
        for route in ['A0','D-pre']:
            seed_all(198613);policy=Policy(str(model_root),route,local_files_only=True)
            parent=scratch/f'{route}_H{h}.pt';weights_checkpoint(parent,policy,dict(route=route),dict(global_step=0));del policy
            config=dict(dataset='ETTh1',weights=str(model_root),local_files_only=True,checkpoint=str(parent),checkpoint_kind='policy_state',
                route=route,force_fixed_kappa=True,protocol=protocol,artifact=str(data),seed=198613,splits=['val'],
                four_horizon_campaign=True,diagnostic_limit=6,evaluation_artifact_mode='compact_statistics')
            file=out/f'{route}_H{h}_config.json';directory=scratch/f'{route}_H{h}_compact';atomic_json(file,config)
            command=[sys.executable,'-m','moirai_research.cli','evaluate','--config',str(file),'--output',str(directory),'--device','cuda','--seed',str(config['seed'])]
            result=subprocess.run(command,capture_output=True,text=True,timeout=1800)
            if result.returncode:
                (out/f'{route}_H{h}_error.txt').write_text((result.stdout+'\n'+result.stderr)[-12000:]);raise RuntimeError('Compact public evaluation failed')
            compact=json.loads((directory/'result.json').read_text())['results']['val']
            torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
            policy=load_policy(config,'cuda');full_directory=scratch/f'{route}_H{h}_full'
            full=evaluation(policy,protocol,'val',full_directory,config['seed'],str(data),limit=6)
            differences={}
            for key,value in full.items():
                if key in ['seconds'] or isinstance(value,bool) or not isinstance(value,(int,float)):continue
                other=compact.get(key)
                if value is None:assert other is None;continue
                assert other is not None and math.isclose(value,other,rel_tol=1e-10,abs_tol=1e-10),(key,value,other)
                differences[key]=other-value
            blocks=sorted((directory/'val').glob('block_*.npz'));assert len(blocks)==2
            recomputed={};forbidden={'target','observed','sample_mean','sample_median','MAP','mixture_analytic','lower80','upper80'}
            for metric in ['MSE','MAE','MAP_MSE','MAP_MAE','mixture_analytic_MSE','mixture_analytic_MAE','CRPS']:
                pieces=[]
                for block in blocks:
                    with np.load(block,allow_pickle=False) as arrays:
                        assert not forbidden.intersection(arrays.files)
                        pieces.append(arrays['metric_'+metric])
                value=float(np.concatenate(pieces).mean());assert math.isclose(value,compact[metric],rel_tol=1e-12,abs_tol=1e-12)
                recomputed[metric]=value
            compact_bytes=sum(f.stat().st_size for f in (directory/'val').glob('*.npz'))
            full_bytes=sum(f.stat().st_size for f in full_directory.glob('*.npz'))
            assert 0<compact_bytes<full_bytes
            cases.append(dict(horizon=h,route=route,command=command,windows=6,stream_batches=2,
                compact_metrics=compact,full_metric_differences=differences,independently_reduced_window_metrics=recomputed,
                compact_npz_bytes=compact_bytes,full_npz_bytes=full_bytes,optimizer_updates=0,test_used=False))
            if check_common_origins:
                from .common_origin_statistics import reduce_blocks
                common=reduce_blocks(directory/'val',protocol,'val',require_complete=False)
                assert common['common_windows']==6 and math.isclose(common['metrics']['MAP_MSE'],compact['MAP_MSE'],rel_tol=1e-12,abs_tol=1e-12)
                cases[-1]['common_origin_statistics']=common
            del policy;torch.cuda.empty_cache();atomic_json(out/'progress.json',dict(cases=cases))
    result=dict(status='PASS',cases=cases,common_origin_boundary_check=common_boundary,optimizer_updates=0,optimizer_trajectories=0,test_used=False,
        scope='Installed public CLI; A0 and D-pre, four H, two validation batches, identical numerical metrics and independent stored-window reduction; interface proof, not full checkpoint replication')
    atomic_json(out/'result.json',result);return result
