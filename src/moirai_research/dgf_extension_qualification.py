"""Actual intact-DGF extension versus uninterrupted execution on fixed train inputs."""
from pathlib import Path
import copy,json,shutil,torch
from .runtime import atomic_json
from .train import train
from .io_utils import load_checkpoint
from .resume_extension import extend_resume

def qualify(config,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    base=copy.deepcopy(config);base.update(qualification_only=True,diagnostic_input_limit=2,diagnostic_validation_limit=2,global_input_batch=2,microbatch=1,
        lr=9e-7,retain_all_validation_checkpoints=True,epochs=1,adaptive_dgf=dict(learning_rate={'kind':'constant'},component_gradient_steps=[0,1],calibrate_NLL_gradient=True))
    for key in ['resume','completion_only','registered_budget_extension']:base.pop(key,None)
    def execute(c,name):
        file=out/(name+'.json');atomic_json(file,c);train(file,out/name,'cuda',c['seed'])
        result=json.loads((out/name/'result.json').read_text());assert result['status']=='DIAGNOSTIC_ONLY';return result
    prefix=execute(base,'prefix')
    target=dict(base,epochs=2,registered_budget_extension=dict(prefix_epochs=1,target_epochs=2),recipe_id='QUALIFIED_DGF_DURATION',scientific_path='qualified-duration')
    mapping={str(f):str(f) for f in (out/'prefix/checkpoints').glob('*.pt')}
    def copy_checkpoint(source,destination):
        destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    resumed,proof=extend_resume(out/'prefix/resume.pt',target,mapping,copy_checkpoint,out/'extended')
    result=execute(dict(target,resume=resumed),'extended');golden=execute(dict(target,scientific_path='uninterrupted-duration'),'uninterrupted')
    a=load_checkpoint(out/'extended/checkpoints/final.pt',map_location='cpu',weights_only=False)['policy_state'];b=load_checkpoint(out/'uninterrupted/checkpoints/final.pt',map_location='cpu',weights_only=False)['policy_state']
    difference=max(float((a[k].double()-b[k].double()).abs().max()) for k in a)
    if difference>1e-6:raise ValueError('Complete DGF budget continuation differs from uninterrupted updates')
    assert result['optimizer_steps_this_attempt']==1 and result['cursor']['global_step']==2
    first=result.get('first_step_this_attempt') or result['first_step'];assert first['relative_parameter_change']>0
    assert (out/'prefix/checkpoints/initial.pt').read_bytes()==(out/'extended/checkpoints/initial.pt').read_bytes()
    for name in ['prefix','extended','uninterrupted']:
        records=[json.loads(line) for line in (out/name/'micro_diagnostics.jsonl').open()]
        d=records[0]['diagnostics'][0];assert d['gamma']>0 and d['lambda_dyn']>0 and d['lambda_div']>0
        assert d['sampling_evidence']['shared_pi'] and d['gradient_estimator'].startswith('detached_')
    receipt=dict(status='PASS',optimizer_updates=4,optimizer_trajectories=3,maximum_parameter_difference=difference,allowed_absolute_difference=1e-6,
        extension_provenance=proof,original_initial_preserved=True,complete_core_enabled=True,formal_test_used=False,
        scope='Two fixed train histories, one prefix update plus one full-state extended update versus two uninterrupted intact-DGF updates; not a scientific five-epoch result')
    atomic_json(out/'result.json',receipt);return receipt
