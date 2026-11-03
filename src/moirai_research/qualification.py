"""A real but explicitly small-budget public data -> SFT -> RL -> reload path."""
import copy,importlib.metadata,json,sys,time
from pathlib import Path
import torch
import numpy as np
from huggingface_hub import snapshot_download
from .runtime import atomic_json,seed_all,load_policy,optimizer_for,checked_step,save_state,restore_state
from .public_data import prepare_csv
from .configuration import make_configs,resolve_checkpoint
from .auxiliary_configuration import make_auxiliary_configs
from .supervised import fit
from .data import datasets,batch
from .policy import joints
from .rl import make_rollout,objective
from .evaluation import evaluation
from .probe import relative_update


def run(config_path,output,device='cuda',seed=1):
    config=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True);began=time.time()
    seed_all(seed)
    solar=config['dataset']=='SolarAL_hourly_mean'
    if solar:
        from .solar_data import prepare_solar
        p=prepare_solar(config.get('public_csv_cache') or config['public_csv_url'],out/'data',config['data_revision'])['protocol']
    else:p=prepare_csv(config.get('public_csv_cache') or config['public_csv_url'],config['dataset'],out/'data',96,config['data_revision'])
    public_data_matches_reference=None
    if config.get('reference_data_artifact'):
        for name in ['normalized.npy','scaler_mean.npy','scaler_std.npy','train_indices.npy','val_indices.npy','test_indices.npy']:
            np.testing.assert_array_equal(np.load(out/'data'/name,allow_pickle=False),np.load(Path(config['reference_data_artifact'])/name,allow_pickle=False),err_msg='Public data differs from research protocol: '+name)
        public_data_matches_reference=True
    weights=config.get('public_model_cache') or snapshot_download(repo_id=config['model_id'],revision=config['model_revision'],allow_patterns=['config.json','model.safetensors'])
    route=config['route']
    micro=config.get('microbatch',1 if solar else 8)
    make_configs(out/'data',out/'portable_configs','E01' if route=='A0' else 'E03',str(weights),config['model_revision'],seed,microbatch=micro)
    generated=json.loads((out/'portable_configs/supervised.json').read_text())
    assert generated['supervised_objective']==('official_PackedNLLLoss' if route=='A0' else 'legal_trajectory_mixture_NLL')
    neighbor_config=make_auxiliary_configs(out/'data',out/'portable_neighbor_configs','neighbors',seed,.1)
    assert neighbor_config['commands'][0][1]=='build-neighbors'
    if not solar:
        estimator_config=make_auxiliary_configs(out/'data',out/'portable_estimator_configs','estimator',seed)
        assert estimator_config['commands'][0][1]=='train-estimator'
    # Configuration-only check of the finite combined-recipe interface; this
    # does not pretend that the composite training trajectory was executed here.
    recipe=out/'recipe_interface.json';auxiliary=out/'recipe_auxiliary_interface.json'
    atomic_json(recipe,dict(components=['E14'],preparation_components=[],algorithm_arm='E14') if solar else
        dict(components=['E07','E13'],preparation_components=['E07'],algorithm_arm='E13'))
    atomic_json(auxiliary,dict(difference_coefficients={'1':.1,'2':.1}) if solar else
        dict(preparation_coefficients={'E07':.01},teacher_config={'qualification_placeholder_not_executed':True}))
    make_configs(out/'data',out/'portable_recipe_configs','NEW1',str(weights),config['model_revision'],seed,auxiliary,recipe)
    prepared_recipe=json.loads((out/'portable_recipe_configs/preparation.json').read_text())
    if solar:assert prepared_recipe['preparation']=='ordinary'
    else:assert prepared_recipe['preparation_components']==[dict(arm='E07',coefficient=.01)]
    base=dict(dataset=config['dataset'],route=route,weights=weights,revision=config['model_revision'],
        local_files_only=True,protocol=p,artifact=str(out/'data'),max_epochs=1,patience=3,global_input_batch=64,
        microbatch=micro,lr=config['supervised_lr'],supervised_objective='official_PackedNLLLoss' if route=='A0' else 'legal_trajectory_mixture_NLL',
        qualification_only=True,diagnostic_input_limit=128,diagnostic_validation_limit=64)
    file=out/'supervised.json';atomic_json(file,base);sft=fit(str(file),out/'supervised',device,seed)
    assert sft['status']=='DIAGNOSTIC_ONLY' and sft['optimizer_steps']==2
    rl=dict(base,checkpoint=sft['best_checkpoint'],checkpoint_kind='policy_state',force_fixed_kappa=True,arm='E01' if route=='A0' else 'E03',
            nll_anchor_coefficient=1.,lr=5e-7)
    binding=dict(rl);binding.pop('checkpoint');binding['checkpoint_from_receipt']=dict(path=str(out/'supervised/result.json'),key=['best_checkpoint'])
    rl=resolve_checkpoint(binding);assert rl['checkpoint']==sft['best_checkpoint']
    policy=load_policy(rl,device).eval();reference=load_policy(rl,device).eval().requires_grad_(False);optimizer=optimizer_for(policy,rl['lr'])
    rl_inputs=min(8,micro) if solar else 8
    data=datasets(policy,config['dataset'],p)['train'];b=batch(data,range(rl_inputs),device)
    initial={k:v.detach().cpu().clone() for k,v in policy.named_parameters()};actions=make_rollout(policy,reference,b,rl);updates=[]
    for i in range(2):
        optimizer.zero_grad(set_to_none=True);loss,diag=objective(policy,b,actions,rl);loss.backward();grad=checked_step(policy,optimizer)
        updates.append(dict(step=i+1,**diag,**grad))
    delta,frozen=relative_update(policy,initial);assert delta>0 and frozen==0
    state=out/'rl_resume.pt';save_state(state,policy,optimizer,rl,dict(epoch=0,next_input=rl_inputs,global_step=2))
    with torch.no_grad():
        previous=joints(policy(b),b);before=[e.mean.clone() for e in previous[0]]
    cursor,_=restore_state(state,policy,optimizer,rl);assert cursor['global_step']==2
    with torch.no_grad():
        after=joints(policy(b),b)
        for x,y in zip(before,after[0]):torch.testing.assert_close(x,y.mean,atol=0,rtol=0)
    val=evaluation(policy,p,'val',out/'rl_validation',seed,out/'data',64)
    assert not val['complete'] and val['windows']==64
    import moirai_research
    result=dict(status='PASS',scope='Two supervised updates and two fixed-rollout RL updates; diagnostic validation only',
        optimizer_trajectories=2,optimizer_steps=4,supervised_receipt=str(out/'supervised/result.json'),rl_updates=updates,
        public_configuration_assembly_executed=True,public_recipe_configuration_assembly_executed=True,
        public_auxiliary_configuration_assembly_executed=True,
        auxiliary_qualification_scope='Neighbor taxonomy configuration only; selective/teacher stages outside Solar scope' if solar else 'Neighbor/estimator configuration assembly only; no extra optimizer or numerical auxiliary trajectory',
        recipe_qualification_scope='Configuration assembly only; no composite optimizer trajectory in this interface check',
        public_checkpoint_receipt_resolution_executed=True,exact_reload=True,parameter_update=delta,frozen_max_change=frozen,validation=val,test_used=False,
        model_id=config['model_id'],model_revision=config['model_revision'],data_revision=config['data_revision'],
        public_transport_receipt=config.get('public_transport_receipt'),offline_cache_used=bool(config.get('public_csv_cache')),public_data_matches_reference=public_data_matches_reference,
        environment=dict(prefix=sys.prefix,python=sys.version,package_file=moirai_research.__file__,
          torch=torch.__version__,cuda=torch.version.cuda,dependencies={x:importlib.metadata.version(x) for x in ['numpy','pandas','datasets','gluonts','lightning','jax','safetensors','huggingface-hub']}),
        actual_gpu=torch.cuda.get_device_name(),seconds=time.time()-began)
    atomic_json(out/'result.json',result);return result
