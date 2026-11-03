"""Installed public CLI, finite updates and reload checks at all four H."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import torch
from .runtime import atomic_json,seed_all


def qualify_entry(model_root,csv_file,output,revision):
    from .public_data import prepare_csv
    from .policy import Policy
    from .train import weights_checkpoint
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    scratch=out.parent.parent/'entry_work';scratch.mkdir(exist_ok=True);cases=[]
    assert not scratch.is_relative_to(out.parent), 'Intermediate work must be outside the publication directory'
    methods={'P01':'fair_full_CRPS','P02':'fair_iid_score_CRPS','P06':'almost_fair_full_CRPS'}
    for h in [96,192,336,720]:
        protocol=prepare_csv(str(csv_file),'ETTh1',scratch/f'data_H{h}',horizon=h,source_revision=revision)
        seed_all(198341);policy=Policy(str(model_root),'A0',local_files_only=True)
        parent=scratch/f'public_initial_H{h}.pt';weights_checkpoint(parent,policy,dict(seed=198341,route='A0'),dict(global_step=0));del policy
        for method,objective in methods.items():
            config=dict(dataset='ETTh1',protocol=protocol,artifact=str(scratch/f'data_H{h}'),weights=str(model_root),route='A0',
                checkpoint=str(parent),checkpoint_kind='policy_state',local_files_only=True,force_fixed_kappa=True,
                phase='scientific',arm='E00',algorithm_arm='E00',recipe_id=method,direct_objective=objective,
                seed=198341,lr=5e-7,global_input_batch=2,microbatch=2,epochs=1,updates_per_input_batch=2,
                qualification_only=True,diagnostic_input_limit=2,diagnostic_validation_limit=2,warmup_campaign=True,
                four_horizon_campaign=True,record_probability_checkpoints=True,checkpoint_interval_batches=1,test_release=False)
            file=out/f'{method}_H{h}.json';atomic_json(file,config);directory=scratch/f'{method}_H{h}'
            command=[sys.executable,'-m','moirai_research.cli','train','--config',str(file),'--output',str(directory),'--device','cuda','--seed','198341']
            result=subprocess.run(command,capture_output=True,text=True,timeout=900)
            if result.returncode:
                (out/f'{method}_H{h}_error.txt').write_text((result.stdout+'\n'+result.stderr)[-8000:])
                raise RuntimeError('Public CRPS training entry failed: '+method+'/H'+str(h))
            receipt=json.loads((directory/'result.json').read_text());assert receipt['status']=='DIAGNOSTIC_ONLY'
            assert receipt['cursor']['global_step']==2 and receipt['first_step']['relative_parameter_change']>0
            final=receipt['readouts']['final'];assert final['reloaded'] and final['validation']['windows']==2
            retained=out/'diagnostic_weights'/f'{method}_H{h}_final.pt';retained.parent.mkdir(exist_ok=True)
            shutil.copy2(final['checkpoint'],retained)
            cases.append(dict(method=method,horizon=h,status='PASS',optimizer_updates=2,input_exposure=2,
                first_step=receipt['first_step'],validation=final['validation'],public_command=command,
                final_checkpoint=str(retained),public_initialization_reference='Same read-only cached public model; zero-update parent not redistributed',
                full_budget_training=False,test_used=False))
            atomic_json(out/'progress.json',dict(cases=cases,optimizer_updates=2*len(cases)))
    result=dict(status='PASS',cases=cases,optimizer_updates=24,optimizer_trajectories=12,test_used=False,
        qualified_recipes=['P01','P02','P06'],
        scope='Installed CLI P01/P02/P06; four H; two updates each; actual parameter change and independent selected/final reload. Diagnostic only, not complete training or formal test.')
    atomic_json(out/'result.json',result);return result
