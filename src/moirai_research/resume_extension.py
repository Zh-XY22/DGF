"""Explicit full-state budget extension; never edits the original prefix state."""
from pathlib import Path
from .io_utils import load_checkpoint,buffered_torch_save
from .runtime import atomic_json


def canonical(config):
    value={k:v for k,v in config.items() if k not in ['epochs','registered_budget_extension','scientific_path','recipe_id','fixed_epoch_checkpoints','weights','checkpoint','artifact']}
    value['protocol']={k:v for k,v in value['protocol'].items() if k not in ['artifact','builder_root']}
    return value


def extend_resume(source,new_config,checkpoint_map,download,output):
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    state=load_checkpoint(source,map_location='cpu',weights_only=False);old=state['config']
    if canonical(old)!=canonical(new_config):raise ValueError('Budget extension changes a scientific setting outside its registered allowance')
    epoch=state['cursor']['epoch']
    if state['cursor']['next_input']!=0 or not old['epochs']==epoch<new_config['epochs']:
        raise ValueError('Extension requires a completed prefix and a strictly greater target epoch')
    if old.get('dgf_complete'):
        from .adaptive_dgf import check_extension_clock
        check_extension_clock(old,new_config,state['cursor'])
    initial=[(path,uri) for path,uri in checkpoint_map.items() if Path(path).name=='initial.pt']
    if old.get('dgf_complete') and len(initial)!=1:raise ValueError('Full-method extension requires the original initial checkpoint')
    for path,uri in initial:
        local=directory/'checkpoints/initial.pt'
        if not local.exists():download(uri,local)
    replacements={}
    for label,path in state['extra']['best_paths'].items():
        if path not in checkpoint_map:raise ValueError('Selected prefix state lacks a persisted mapping')
        local=directory/'checkpoints'/Path(path).name
        if not local.is_file():download(checkpoint_map[path],local)
        replacements[label]=str(local)
    if epoch in new_config.get('fixed_epoch_checkpoints',[]):
        initial_budget=directory/'checkpoints'/f'budget-epoch{epoch}.pt'
        buffered_torch_save(dict(policy_state=state['policy_state'],route=state['route'],routing=state['routing'],
            fixed_kappa=state['fixed_kappa'],config=new_config,progress=state['cursor'],resumable=False,
            continuation='Reused complete prefix; full optimizer/RNG state is separate'),initial_budget)
        replacements['budget_epoch_'+str(epoch)]=str(initial_budget)
    state['extra']['best_paths']=replacements;state['config']=new_config
    destination=directory/'extension_resume.pt';buffered_torch_save(state,destination)
    evidence=dict(status='EXPLICIT_BUDGET_EXTENSION',source=str(source),prefix_epochs=epoch,target_epochs=new_config['epochs'],
        prefix_optimizer_updates=state['cursor']['global_step'],prefix_input_exposure=state['cursor']['independent_input_exposures'],
        optimizer_preserved=True,RNG_preserved=True,scheduler_preserved=True,data_cursor_preserved=True,
        scientific_settings_equal=True,selected_prefix_states_preserved=True,
        changed_metadata=['epochs','scientific_path','recipe_id','fixed_epoch_checkpoints','worker-local IO paths'],
        prefix_is_not_an_independent_second_training=True)
    atomic_json(directory/'extension_provenance.json',evidence);return str(destination),evidence
