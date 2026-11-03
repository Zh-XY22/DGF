"""Fixed joint-operator interface fixtures; no scientific recipe selection."""
from pathlib import Path
import json,sys,subprocess,math
from .public_data import prepare_csv
from .runtime import atomic_json
from .joint_entry_qualification import make_joint_config

def qualify(model_root,csv_file,parent,output,revision,public_cli=False,only_fixture=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);scratch=out.parent.parent/'joint_operator_work';scratch.mkdir(exist_ok=True)
    roots={}
    for h in [96,192,336,720]:
        root=scratch/f'data_H{h}';prepare_csv(str(csv_file),'ETTh1',root,horizon=h,source_revision=revision);roots[str(h)]=root
    fixtures={
        'P01':dict(phase='scientific_DIRECT',direct_objective='fair_full_CRPS',crps_per_component=4),
        'P02':dict(phase='scientific_DIRECT',direct_objective='fair_iid_score_CRPS',crps_samples=32),
        'P03':dict(phase='scientific_DIRECT',direct_objective='fair_full_CRPS_NLL',probability_NLL_coefficient=.01,crps_per_component=4),
        'P06':dict(phase='scientific_DIRECT',direct_objective='almost_fair_full_CRPS',almost_fair_alpha=.95,crps_per_component=4),
        'R02_R03':dict(phase='scientific_RL',continuous_recipe='R02',clipping_recipe='R03',clip_low=.2,clip_high=.28),
        'R08':dict(phase='scientific_RL',continuous_recipe='R08'),
        'CURRICULUM_P01':dict(phase='scientific_DIRECT',direct_objective='fair_full_CRPS',crps_per_component=4,epochs=2,
            horizon_weights_by_epoch=[[.7,.1,.1,.1],[.1,.1,.1,.7]])}
    if only_fixture is not None:
        if only_fixture not in fixtures:raise ValueError('Unknown fixed joint entry fixture')
        fixtures={only_fixture:fixtures[only_fixture]}
    cases=[]
    for name,change in fixtures.items():
        config=make_joint_config(model_root,roots,change['phase'],198509,parent,qualification=True)
        config.update(change,recipe_id='joint-operator-fixture-'+name)
        if change['phase']=='scientific_DIRECT':config.update(joint_selection_objective='CRPS',joint_CRPS_reference={str(h):1. for h in [96,192,336,720]},joint_selection_rule='Minimum four-H mean CRPS/reference; worst ratio tie breaker; unit references for interface fixture only')
        file=out/(name+'.json');directory=scratch/name;atomic_json(file,config)
        command=[sys.executable,'-c','from moirai_research.joint_train import fit_joint;fit_joint('+repr(str(file))+','+repr(str(directory))+')']
        if public_cli:command=[sys.executable,'-m','moirai_research.cli','train-joint','--config',str(file),'--output',str(directory),'--device','cuda','--seed',str(config['seed'])]
        run=subprocess.run(command,capture_output=True,text=True,timeout=2400)
        if run.returncode:
            (out/(name+'-error.txt')).write_text((run.stdout+'\n'+run.stderr)[-16000:]);raise RuntimeError('Joint operator entry failed: '+name)
        value=json.loads((directory/'result.json').read_text());assert value['status']=='DIAGNOSTIC_ONLY'
        expected=config['epochs']*config['updates_per_input_batch'];assert value['cursor']['global_step']==expected
        assert value['first_step']['relative_parameter_change']>0 and value['test_used'] is False
        assert value['horizon_input_exposures']=={str(h):2*config['epochs'] for h in [96,192,336,720]}
        assert all(len(v['validation_by_horizon'])==4 and v['reloaded'] for v in value['readouts'].values())
        assert abs(sum(value['horizon_loss_weighted_optimizer_updates'].values())-expected)<1e-9
        curve=json.loads((directory/'validation_curve.json').read_text());training=json.loads((directory/'training_curve.json').read_text())
        if change['phase']=='scientific_DIRECT':
            for row in curve:
                crps=[row['metrics_by_horizon'][str(h)]['CRPS'] for h in [96,192,336,720]]
                assert math.isclose(row['score'][0],sum(crps)/4,rel_tol=1e-12) and math.isclose(row['score'][1],max(crps),rel_tol=1e-12)
        if name=='R02_R03':
            assert all(d['advantage_recipe']=='R02' and d['clipping_recipe']=='R03' and d['global_complete_batch_scale']>0 for row in training for d in row['diagnostics'])
        if name=='CURRICULUM_P01':assert training[0]['horizon_weights']!=training[-1]['horizon_weights']
        atomic_json(out/(name+'-observed.json'),value)
        cases.append(dict(fixture=name,phase=change['phase'],command=command,operator_configuration=change,optimizer_updates=expected,first_step=value['first_step'],
            cursor=value['cursor'],one_common_checkpoint=value['readouts']['joint_selected'],horizon_weights=[row['horizon_weights'] for row in training],
            weighted_optimizer_updates=value['horizon_loss_weighted_optimizer_updates'],test_used=False))
        atomic_json(out/'progress.json',dict(cases=cases))
    result=dict(status='PASS',qualified_recipes=[] if public_cli else ['J02','J03','J04'],entry_protocol='INSTALLED_PUBLIC_CLI' if public_cli else 'INSTALLED_PUBLIC_MODULE',operator_scope=['A0 joint R02+R03','A0 fresh R08','A0 direct P01/P02/P03/P06','positive four-H curriculum weights'],
        cases=cases,optimizer_trajectories=len(cases),optimizer_updates=sum(c['optimizer_updates'] for c in cases),
        formal_candidate_selected=False,full_budget_replication=False,test_used=False,
        limitations='Specific frozen candidate parameters still require matching entry evidence; this gate does not cover D-pre, functional transport calibrators, context changes, SAM, different coefficients or an automatically selected final recipe')
    atomic_json(out/'result.json',result);return result
