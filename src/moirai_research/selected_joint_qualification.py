"""Actual validation-selected joint recipe, small installed public CLI proof."""
from pathlib import Path
import json,subprocess,sys,math
from .runtime import atomic_json
from .joint_entry_qualification import make_joint_config
from .joint_curriculum import balanced_epoch_weights

def qualify(model_root,data_roots,parent,output,phase,overrides):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);scratch=out.parent.parent/'selected_joint_entry';scratch.mkdir(exist_ok=True)
    if phase not in ['scientific_RL','scientific_NLL','scientific_DIRECT']:raise ValueError('Unsupported joint scientific phase')
    if overrides.get('route','A0')!='A0' or overrides.get('transport_mode'):raise ValueError('This exact entry qualifies native A0 only')
    config=make_joint_config(model_root,data_roots,phase,198631,parent,qualification=True)
    protected={'weights','checkpoint','checkpoint_kind','seed','protocol','protocols','artifact','route','common_train_windows','qualification_only','diagnostic_input_limit','diagnostic_validation_limit','global_input_batch','microbatch','test_release'}
    if protected.intersection(overrides):raise ValueError('Selected entry must preserve qualification and scientific identity')
    config.update(overrides)
    # Only the update budget is shortened. Every actual scientific operator,
    # coefficient and selector remains exactly the registered selected recipe.
    config['epochs']=2 if config.get('curriculum_rule') else 1
    config.pop('updates_by_epoch',None)
    if config.get('curriculum_rule'):
        if config['curriculum_rule']!='balanced-short-to-long-v1':raise ValueError('Unknown selected curriculum')
        shape=balanced_epoch_weights([config['updates_per_input_batch']]*config['epochs'])
        config['horizon_weights_by_epoch']=shape['horizon_weights_by_epoch']
    file=out/'selected_configuration.json';atomic_json(file,config);directory=scratch/'training'
    command=[sys.executable,'-m','moirai_research.cli','train-joint','--config',str(file),'--output',str(directory),'--device','cuda','--seed',str(config['seed'])]
    run=subprocess.run(command,capture_output=True,text=True,timeout=2400)
    if run.returncode:
        (out/'selected-entry-error.txt').write_text((run.stdout+'\n'+run.stderr)[-16000:]);raise RuntimeError('Selected joint CLI failed')
    value=json.loads((directory/'result.json').read_text());expected=config['epochs']*config['updates_per_input_batch']
    assert value['status']=='DIAGNOSTIC_ONLY' and value['cursor']['global_step']==expected
    assert value['first_step']['relative_parameter_change']>0 and not value['test_used']
    assert all(len(v['validation_by_horizon'])==4 and v['reloaded'] for v in value['readouts'].values())
    assert value['horizon_input_exposures']=={str(h):2*config['epochs'] for h in [96,192,336,720]}
    if config.get('curriculum_rule'):
        for weight in value['horizon_loss_weighted_optimizer_updates'].values():assert math.isclose(weight,expected/4,abs_tol=1e-10)
    result=dict(status='PASS',qualified_recipes=[config['recipe_id']],selected_recipe_id=config['selected_recipe_id'],
        scientific_operator_overrides=overrides,actual_qualification_configuration=config,command=command,
        optimizer_updates=expected,optimizer_trajectories=1,first_step=value['first_step'],cursor=value['cursor'],
        one_common_checkpoint=value['readouts']['joint_selected'],horizon_weighted_updates=value['horizon_loss_weighted_optimizer_updates'],
        test_used=False,scope='Exact validation-selected native A0 operators and selector; shortened installed-CLI execution, all four H reload. Not complete-budget reproduction')
    atomic_json(out/'result.json',result);return result
