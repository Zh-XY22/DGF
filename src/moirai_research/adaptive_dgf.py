"""Explicit extensions of the intact DGF objective and deterministic LR clock."""
import math

def qualification_settings(config):
    full=dict(config,qualification_only=True,diagnostic_input_limit=2,diagnostic_validation_limit=2,global_input_batch=2,microbatch=1,epochs=4,retain_all_validation_checkpoints=True,lr=1.5e-6)
    full['adaptive_dgf']=dict(learning_rate=dict(kind='piecewise',turn_step=2,final_lr=1.5e-7),component_gradient_steps=[0,1,2,3],calibrate_NLL_gradient=True)
    auxiliary=dict(full,epochs=1,adaptive_dgf=dict(full['adaptive_dgf'],learning_rate={'kind':'constant'},nll_coefficient=1.,nll_start_step=0))
    return full,auxiliary

def learning_rate(config,completed_updates,total_updates):
    setting=config.get('adaptive_dgf',{}).get('learning_rate',{'kind':'constant'})
    kind=setting['kind'];high=float(config['lr'])
    if not 0<=completed_updates<total_updates:raise ValueError('Invalid optimizer clock')
    if kind=='constant':return high
    low=float(setting['final_lr'])
    if not 0<low<=high:raise ValueError('Adaptive LR must stay positive and at most initial LR')
    turn=int(setting.get('turn_step',0))
    if not 0<=turn<total_updates:raise ValueError('Invalid schedule transition')
    if kind=='piecewise':return high if completed_updates<turn else low
    if kind=='cosine':
        phase=max(0.,(completed_updates-turn)/max(1,total_updates-1-turn))
        return low+(high-low)*.5*(1.+math.cos(math.pi*phase))
    raise ValueError('Unknown explicitly registered LR schedule')

def apply_learning_rate(optimizer,config,completed_updates,total_updates):
    value=learning_rate(config,completed_updates,total_updates)
    for group in optimizer.param_groups:group['lr']=value
    return value

def check_extension_clock(old,new,cursor):
    epoch=cursor['epoch'];step=cursor['global_step']
    if cursor['next_input']!=0 or not old['epochs']==epoch<new['epochs'] or not step or step%epoch:
        raise ValueError('Duration extension requires a complete equal-exposure epoch prefix')
    extension=new.get('registered_budget_extension',{})
    if extension.get('prefix_epochs')!=epoch or extension.get('target_epochs')!=new['epochs']:
        raise ValueError('Explicit duration registration differs from the saved cursor')
    target=step//epoch*new['epochs']
    for completed in range(step):
        if learning_rate(old,completed,step)!=learning_rate(new,completed,target):
            raise ValueError('Budget extension would retrospectively change the LR clock')
    return target

def diagnostic_requested(config):
    options=config.get('adaptive_dgf',{})
    return config.get('_microbatch_index')==0 and config.get('_optimizer_step') in options.get('component_gradient_steps',[])

def weighted_gradient_evidence(losses,policy):
    import torch
    parameters=[p for p in policy.parameters() if p.requires_grad]
    norms={};saved={}
    for name,loss in losses.items():
        gradients=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
        values=[g.detach().double() for g in gradients if g is not None]
        norms[name]=float(sum((g.square().sum() for g in values),torch.zeros((),device=parameters[0].device,dtype=torch.float64)).sqrt())
        saved[name]=[None if g is None else g.detach() for g in gradients]
    alignment={}
    if 'sample' in saved:
        for name in saved:
            if name=='sample':continue
            dot=sum(float((a.double()*b.double()).sum()) for a,b in zip(saved['sample'],saved[name]) if a is not None and b is not None)
            alignment['sample_vs_'+name]=None if not norms['sample']*norms[name] else dot/(norms['sample']*norms[name])
    return dict(weighted_parameter_L2_norms=norms,gradient_cosines=alignment,scope='Current checkpoint, pre-registered first training microbatch only; before global clipping; not all-input component gradient')

def regularize_and_diagnose(policy,experts,weights,target,loss,terms,config):
    from .probability import nll
    options=config.get('adaptive_dgf',{});coefficient=float(options.get('nll_coefficient',0.))
    if coefficient<0:raise ValueError('Negative NLL auxiliary is not authorized')
    if config.get('_optimizer_step',0)<int(options.get('nll_start_step',0)):coefficient=0.
    auxiliary=nll(experts,weights,target) if coefficient or options.get('calibrate_NLL_gradient') else None
    full=config['dgf_complete'];pieces={'sample':-terms['J_sample'],'dirichlet':-full['gamma']*terms['J_dir'],'diversity':-full['lambda_div']*terms['R_div']}
    if auxiliary is not None:
        pieces['NLL_unweighted']=auxiliary
        if coefficient:pieces['NLL_weighted']=coefficient*auxiliary;loss=loss+coefficient*auxiliary
    evidence=weighted_gradient_evidence(pieces,policy) if diagnostic_requested(config) else {}
    return loss,dict(auxiliary_NLL=None if auxiliary is None else float(auxiliary.detach()),auxiliary_NLL_coefficient=coefficient,adaptive_component_gradient_evidence=evidence)

def prefix_compatible(old,new,prefix_updates,total_updates):
    """Configuration-only check; the HOPE adapter separately checks state tensors."""
    ignored={'adaptive_dgf','scientific_path','recipe_id','source_role','weights','artifact','checkpoint','retain_all_validation_checkpoints','schedule','checkpoint_interval_batches'}
    def canonical(c):
        value={k:v for k,v in c.items() if k not in ignored}
        value['protocol']={k:v for k,v in value['protocol'].items() if k not in ['artifact','builder_root']}
        return value
    if canonical(old)!=canonical(new):raise ValueError('Shared prefix changes a scientific setting')
    options=new.get('adaptive_dgf',{})
    if options.get('nll_coefficient',0)>0 and options.get('nll_start_step',0)<prefix_updates:raise ValueError('NLL extension would alter the shared prefix')
    for step in range(prefix_updates):
        if learning_rate(old,step,total_updates)!=learning_rate(new,step,total_updates):raise ValueError('New schedule differs inside inherited prefix')
    return True

def adopt_prefix(source,new_config,checkpoint_map,download,output,expected_step):
    from pathlib import Path
    from .io_utils import load_checkpoint,buffered_torch_save
    from .runtime import atomic_json
    state=load_checkpoint(source,map_location='cpu',weights_only=False)
    old=state['config'];cursor=state['cursor'];total=28
    if cursor['global_step']!=expected_step or not state.get('optimizer_state',{}).get('state') or not state.get('rng'):raise ValueError('Incomplete or wrong full-state prefix')
    prefix_compatible(old,new_config,expected_step,total)
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True);paths={}
    for label,path in state['extra']['best_paths'].items():
        if path not in checkpoint_map:raise ValueError('Missing preserved prefix checkpoint')
        local=directory/'checkpoints'/Path(path).name
        if not local.is_file():download(checkpoint_map[path],local)
        paths[label]=str(local)
    state['extra']['best_paths']=paths
    state['config']=new_config
    destination=directory/'adaptive_prefix_resume.pt';buffered_torch_save(state,destination)
    proof=dict(status='EXPLICIT_NEW_SUFFIX_WITH_SHARED_FULL_STATE',inherited_source=str(source),prefix_optimizer_updates=expected_step,prefix_input_exposure=cursor['independent_input_exposures'],optimizer_preserved=True,RNG_preserved=True,data_cursor_preserved=True,scheduler_semantics='Old constant LR prefix verified identical; new stateless schedule determined by saved completed-update cursor and frozen config',old_source_remains_unchanged=True,full_old_configuration_continuation=False)
    atomic_json(directory/'adaptive_prefix_provenance.json',proof)
    return str(destination),proof
