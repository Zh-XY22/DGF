"""Finite four-horizon joint entry with common-origin and reload evidence."""
from pathlib import Path
import json,sys,subprocess
import torch
from .runtime import atomic_json
from .public_data import prepare_csv
from .joint_train import fit_joint

def make_joint_config(model_root,data_roots,phase,seed,parent=None,qualification=False,revision=None):
    if phase not in {'supervised','ordinary','warmup_C02','warmup_DIRECT','scientific_RL','scientific_DIRECT','scientific_NLL'}:raise ValueError('Unknown joint phase')
    protocols={}
    for h,root in data_roots.items():
        root=Path(root);p=json.loads((root/'data_protocol.json').read_text());p.update(builder_root=str(root/'processed'),artifact=str(root));protocols[str(h)]=p
    if list(protocols)!=['96','192','336','720']:raise ValueError('Stable horizon order required')
    config=dict(dataset='ETTh1',weights=str(model_root),revision=revision,local_files_only=True,route='A0',force_fixed_kappa=True,
        protocol=protocols['96'],protocols=protocols,artifact=protocols['96']['artifact'],seed=seed,phase=phase,
        four_horizon_campaign=True,joint_training=True,test_release=False,global_input_batch=64 if phase=='supervised' else 512,
        microbatch=8,lr=1e-7 if phase=='supervised' else 5e-7,epochs=10 if phase=='supervised' else 1 if phase=='ordinary' else 2 if phase in ['warmup_C02','warmup_DIRECT'] else 4,
        updates_per_input_batch=2 if phase in ['scientific_RL','scientific_DIRECT','scientific_NLL'] else 1,common_train_windows=min(p['windows']['train'] for p in protocols.values()),
        joint_selection_objective='MSE',joint_selection_rule='Minimum worst four-H MAP validation error / frozen public point target; macro ratio tie breaker',
        K=1,J=8,G=4,nll_anchor_coefficient=0.,reference_beta=.01,arm='E16',checkpoint_kind='policy_state')
    if parent:config['checkpoint']=str(parent)
    if qualification:config.update(qualification_only=True,diagnostic_input_limit=2,diagnostic_validation_limit=2,global_input_batch=2,microbatch=1,epochs=1)
    return config

def execute_joint(model_root,data_roots,phase,seed,parent,output,revision=None,recipe_id=None,scientific_path=None,overrides=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);config=make_joint_config(model_root,data_roots,phase,seed,parent,revision=revision)
    config.update(recipe_id=recipe_id,scientific_path=scientific_path)
    if recipe_id in ['J02','J03','J04'] and not (overrides and overrides.get('selected_recipe_id')):raise ValueError('A joint candidate must bind the actual validation-selected common recipe')
    if overrides:
        if {'weights','revision','checkpoint','seed','protocol','protocols','artifact','test_release','route','common_train_windows'}.intersection(overrides):raise ValueError('Protected joint scientific identity override')
        config.update(overrides)
    if config.get('curriculum_rule'):
        if config['curriculum_rule']!='balanced-short-to-long-v1':raise ValueError('Unqualified curriculum rule')
        from .joint_curriculum import balanced_epoch_weights
        import math
        steps=math.ceil(config['common_train_windows']/config['global_input_batch'])*config['updates_per_input_batch']
        derivation=balanced_epoch_weights([steps]*config['epochs'])
        if config.get('horizon_weights_by_epoch') and config['horizon_weights_by_epoch']!=derivation['horizon_weights_by_epoch']:raise ValueError('Explicit curriculum weights disagree with the frozen derivation')
        config.update(horizon_weights_by_epoch=derivation['horizon_weights_by_epoch'],curriculum_derivation=derivation)
    file=out/'actual_configuration.json';atomic_json(file,config);result=fit_joint(file,out/'training')
    assert result['status']=='PASS';result['qualified_recipes']=[recipe_id] if recipe_id else [];atomic_json(out/'result.json',result);return result

def qualify_joint(model_root,csv_file,output,revision):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);scratch=out.parent.parent/'joint_entry_work';scratch.mkdir(exist_ok=True)
    roots={}
    for h in [96,192,336,720]:
        root=scratch/f'data_H{h}';prepare_csv(str(csv_file),'ETTh1',root,horizon=h,source_revision=revision);roots[str(h)]=root
    # A horizon contributes its position-normalized loss, never its length.
    x=torch.tensor(1.,device='cuda',requires_grad=True);loss=sum(((x-torch.zeros((h,7),device='cuda'))**2).mean()/4 for h in [96,192,336,720])
    gradient=torch.autograd.grad(loss,x)[0];torch.testing.assert_close(gradient,gradient.new_tensor(2.))
    stages=[];parent=None;ordinary=None
    for phase in ['supervised','ordinary','warmup_C02','scientific_RL']:
        config=make_joint_config(model_root,roots,phase,198443,parent,qualification=True)
        directory=scratch/phase;file=out/(phase+'.json');atomic_json(file,config)
        command=[sys.executable,'-c','from moirai_research.joint_train import fit_joint;fit_joint('+repr(str(file))+','+repr(str(directory))+')']
        run=subprocess.run(command,capture_output=True,text=True,timeout=2400)
        if run.returncode:
            (out/'joint_entry_error.txt').write_text((run.stdout+'\n'+run.stderr)[-16000:]);raise RuntimeError('Joint entry failed: '+phase)
        receipt=json.loads((directory/'result.json').read_text());assert receipt['status']=='DIAGNOSTIC_ONLY' and receipt['first_step']['relative_parameter_change']>0
        assert all(len(v['validation_by_horizon'])==4 and v['reloaded'] for v in receipt['readouts'].values())
        assert receipt['cursor']['horizon_input_exposures']=={str(h):2 for h in [96,192,336,720]}
        stages.append(dict(phase=phase,optimizer_updates=receipt['cursor']['global_step'],first_step=receipt['first_step'],
            command=command,one_common_checkpoint=receipt['readouts']['joint_selected'],cursor=receipt['cursor']))
        parent=receipt['readouts']['final']['checkpoint']
        if phase=='ordinary':ordinary=parent
    # Real complete-state continuation of the same two-epoch joint recipe.
    config=make_joint_config(model_root,roots,'scientific_RL',198443,ordinary,qualification=True);config['epochs']=2
    dest=scratch/'continuation';file=out/'continuation.json';atomic_json(file,dict(config,stop_after_epoch=1))
    fit_joint(file,dest);paused=json.loads((dest/'result.json').read_text());assert paused['status']=='INTERRUPTED_WITH_STATE'
    atomic_json(file,dict(config,resume=str(dest/'resume.pt')));fit_joint(file,dest);resumed=json.loads((dest/'result.json').read_text())
    assert resumed['cursor']['epoch']==2 and resumed['cursor']['global_step']==4
    result=dict(status='PASS',qualified_recipes=['J00','J01'],stages=stages,common_origin_only=True,per_H_normalized_gradient=float(gradient),
        continuation=dict(optimizer_rng_cursor_restored=True,cursor=resumed['cursor'],checkpoint=resumed['best_checkpoint']),
        optimizer_updates=sum(s['optimizer_updates'] for s in stages)+4,optimizer_trajectories=5,optimizer_process_starts=6,
        test_used=False,scope='One policy, four horizons, fixed full context, selected common state; finite public SFT/NLL/warmup/RL plus actual continuation/reload; not full-budget replication')
    atomic_json(out/'result.json',result);return result
