"""Actual public template -> point/CRPS training -> reload -> metric reduction.

Small interface budget only. This never claims a complete training reproduction.
"""
from pathlib import Path
import json,math,subprocess,sys
import numpy as np
from .runtime import atomic_json
from .public_data import prepare_csv

def qualify(model_root,csv_file,templates,output,revision):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    scratch=out.parent.parent/'portable_four_horizon_acceptance';scratch.mkdir(exist_ok=True)
    cases=[];updates=0;trajectories=0;seed=198617
    def run(command,label):
        result=subprocess.run([str(v) for v in command],capture_output=True,text=True,timeout=1800)
        if result.returncode:
            (out/(label+'-error.txt')).write_text((result.stdout+'\n'+result.stderr)[-14000:])
            raise RuntimeError('Public portable entry failed: '+label)
        return [str(v) for v in command]
    for h in [96,192,336,720]:
        data=scratch/f'data_H{h}';protocol=prepare_csv(str(csv_file),'ETTh1',data,horizon=h,source_revision=revision)
        protocol['eval_input_block']=4;atomic_json(data/'data_protocol.json',protocol)
        receipts={};stage_commands=[];point_ordinary=None
        for recipe in ['X03','P01']:
            work=scratch/f'{recipe}_H{h}'
            command=[sys.executable,'-m','moirai_research.cli','make-four-horizon-configs','--template',Path(templates)/(recipe+'.json'),
                '--data-dir',data,'--work-dir',work,'--weights',model_root,'--seed',str(seed)]
            run(command,f'materialize_{recipe}_H{h}')
            manifest=json.loads((work/'stage_manifest.json').read_text())
            for stage in manifest['stages']:
                if recipe=='P01' and stage['name'].endswith(('SFT','ordinary')):continue
                file=Path(stage['configuration']);config=json.loads(file.read_text())
                config.update(qualification_only=True,diagnostic_input_limit=2,diagnostic_validation_limit=6,global_input_batch=2,microbatch=1)
                config.pop('declared_worker_effective_contract',None)
                if stage['name'].endswith('SFT'):config.update(fixed_epochs=1,max_epochs=1)
                else:config.update(epochs=1)
                if recipe=='P01' and stage['name'].endswith('warmup'):
                    config['checkpoint_from_receipt']['path']=point_ordinary
                atomic_json(file,config)
                actual_command=list(stage['command']);actual_command[0]=sys.executable
                executed=run(actual_command,f'{recipe}_H{h}_{stage["name"]}')
                receipt=json.loads((Path(stage['output'])/'result.json').read_text())
                # Preserve actual stage evidence before any acceptance assertion.
                atomic_json(out/f'{recipe}_H{h}_{stage["name"]}_training_receipt.json',receipt)
                count=receipt['cursor']['global_step'];updates+=count;trajectories+=1
                atomic_json(out/'executed_stage_accounting.json',dict(optimizer_updates=updates,optimizer_trajectories=trajectories,
                    last_completed_stage=stage['name'],recipe=recipe,horizon=h))
                assert receipt['status']=='DIAGNOSTIC_ONLY' and receipt['test_used'] is False
                update_key='relative_parameter_update' if stage['entrypoint']=='fit-supervised' else 'relative_parameter_change'
                assert receipt['first_step'][update_key]>0 and receipt['cursor']['global_step']>0
                assert receipt['readouts']['final']['reloaded'] and receipt['readouts']['final']['validation']['windows']==6
                stage_commands.append(dict(recipe=recipe,stage=stage['name'],command=executed,updates=count,
                    first_step=receipt['first_step'],scope='Explicit small interface budget; original full budget remains in the template'))
                receipts[recipe]=receipt
                if recipe=='X03' and stage['name'].endswith('ordinary'):point_ordinary=str(Path(stage['output'])/'result.json')
        evaluation=[]
        for recipe,receipt in receipts.items():
            config=dict(receipt['config'],checkpoint=receipt['readouts']['final']['checkpoint'],checkpoint_kind='policy_state',
                splits=['val'],diagnostic_limit=6,evaluation_artifact_mode='compact_statistics')
            config.pop('checkpoint_from_receipt',None)
            file=out/f'{recipe}_H{h}_evaluation.json';atomic_json(file,config)
            directory=scratch/f'{recipe}_H{h}_evaluation'
            command=run([sys.executable,'-m','moirai_research.cli','evaluate','--config',file,'--output',directory,'--device','cuda','--seed',str(seed)],f'evaluate_{recipe}_H{h}')
            raw=json.loads((directory/'result.json').read_text());metric=raw['results']['val']
            assert raw['status']=='DIAGNOSTIC_ONLY' and metric['windows']==6
            original=receipt['readouts']['final']['validation'];recomputed={}
            blocks=sorted((directory/'val').glob('block_*.npz'));assert len(blocks)==2
            for key in ['MSE','MAE','MAP_MSE','MAP_MAE','mixture_analytic_MSE','mixture_analytic_MAE','CRPS']:
                pieces=[]
                for block in blocks:
                    with np.load(block,allow_pickle=False) as arrays:pieces.append(arrays['metric_'+key])
                value=float(np.concatenate(pieces).mean());assert math.isclose(value,metric[key],rel_tol=1e-10,abs_tol=1e-10)
                assert math.isclose(original[key],metric[key],rel_tol=1e-6,abs_tol=1e-7)
                recomputed[key]=value
            evaluation.append(dict(recipe=recipe,command=command,metrics=metric,independently_reduced_window_statistics=recomputed,
                checkpoint_header=raw.get('checkpoint_metadata'),samples=100,windows=6,full_validation_or_test=False))
        cases.append(dict(horizon=h,stage_commands=stage_commands,evaluation=evaluation,
            shared_preparation='P01 reuses X03 public SFT and ordinary final; no duplicate shared updates',test_used=False))
        atomic_json(out/'progress.json',dict(cases=cases,optimizer_updates=updates,optimizer_trajectories=trajectories))
    result=dict(status='PASS',cases=cases,optimizer_updates=updates,optimizer_trajectories=trajectories,scientific_training_paths_added=0,
        test_used=False,full_budget_reproduction=False,interface_scope='Fresh installed public template CLI, public CSV construction, point X03 and proper-CRPS P01 updates, four H, save/reload, independent window-statistic metric reduction',
        qualification_limits='Two train inputs per stage and six validation windows; this is not complete scientific retraining or paper reproduction')
    atomic_json(out/'result.json',result);return result
