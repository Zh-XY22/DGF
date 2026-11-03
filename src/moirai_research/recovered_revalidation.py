"""Full retained-checkpoint validation reload, never optimizer or test work."""
from pathlib import Path
import copy,json,math,subprocess,sys
from .runtime import atomic_json

POINT_KEYS=['MSE','MAE','MAP_MSE','MAP_MAE','mixture_analytic_MSE','mixture_analytic_MAE']

def revalidate(config_path,output):
    config=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    source=config['source_configuration'];cases=config['cases'];done={};rows=[];point_errors=[];probability_errors=[]
    for case in cases:
        remote=case['checkpoint_source']
        if remote not in done:
            active=copy.deepcopy(source);active.update(weights=config['weights'],protocol=config['protocol'],artifact=config['artifact'],
                checkpoint=case['checkpoint_local'],checkpoint_kind='policy_state',seed=config['seed'],splits=['val'],test_release=False,
                four_horizon_campaign=True,evaluation_artifact_mode='compact_statistics')
            for key in ['diagnostic_limit','diagnostic_validation_limit','resume','completion_only','stop_after_epoch']:active.pop(key,None)
            file=out/f'checkpoint_{len(done)}.json';directory=out/f'checkpoint_{len(done)}';atomic_json(file,active)
            command=[sys.executable,'-m','moirai_research.cli','evaluate','--config',str(file),'--output',str(directory),'--device','cuda','--seed',str(config['seed'])]
            run=subprocess.run(command,capture_output=True,text=True,timeout=3600)
            if run.returncode:
                (out/'evaluation-error.txt').write_text((run.stdout+'\n'+run.stderr)[-16000:]);raise RuntimeError('Retained checkpoint validation reload failed')
            receipt=json.loads((directory/'result.json').read_text());metric=receipt['results']['val']
            assert receipt['status']=='PASS' and metric['complete'] and metric['windows']==config['protocol']['windows']['val']
            assert metric['split']=='val' and metric['num_samples']==100
            done[remote]=dict(metrics=metric,command=command,receipt=str(directory/'result.json'))
        actual=done[remote];expected=case['expected_validation'];comparisons={}
        for key in POINT_KEYS+['CRPS']:
            x=actual['metrics'].get(key);y=expected.get(key)
            if y is None:comparisons[key]=dict(status='ORIGINAL_METRIC_UNAVAILABLE',actual=x);continue
            match=x is not None and math.isfinite(x) and math.isclose(x,y,rel_tol=1e-6,abs_tol=1e-6)
            comparisons[key]=dict(status='MATCH_WITHIN_FROZEN_TOLERANCE' if match else 'MISMATCH',original=y,reloaded=x,delta=x-y if x is not None else None)
            if not match:(probability_errors if key=='CRPS' else point_errors).append(dict(label=case['label'],metric=key,comparison=comparisons[key]))
        rows.append(dict(label=case['label'],checkpoint_source=remote,metrics=actual['metrics'],comparisons=comparisons,
            public_command=actual['command'],reload_receipt=actual['receipt']))
        atomic_json(out/'progress.json',dict(completed_labels=rows,unique_checkpoints=len(done)))
    result=dict(status='PASS',path_id=config['path_id'],source_receipt=config['source_receipt'],cases=rows,
        expected_labels=sorted(v['label'] for v in cases),unique_checkpoints_reloaded=len(done),
        point_validation_status='MISMATCH_REQUIRES_DIAGNOSIS' if point_errors else 'PASS',
        probability_validation_status='MISMATCH_REQUIRES_DIAGNOSIS' if probability_errors else 'PASS',
        point_mismatches=point_errors,probability_mismatches=probability_errors,optimizer_trajectories=0,optimizer_updates=0,test_used=False,
        tolerance=dict(absolute=1e-6,relative=1e-6),scope='Full validation independent checkpoint reload and existing-scalar comparison. PASS denotes executed evidence collection, not automatic scientific agreement; mismatch statuses are separate. Original values and checkpoints are never changed.')
    atomic_json(out/'result.json',result);return result
