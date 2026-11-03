"""Paper mixed-reward, fresh on-policy sample score plus separate Dirichlet score.

Native family labels are augmented actions, distinct from the outer trajectory
mode. History-only loc/scale keep the count-family support fixed during updates.
"""
import torch
from .policy import joints, rollout, expert_log_prob
from .data import structured, assert_complete_future, forecast_dimensions
from .dgf_reference_math import hierarchical_advantages, dynamic_terms, mode_diversity, complete_loss, masked_mean


def validate_contract(config):
    full = config.get('dgf_complete')
    if config['phase'] not in ['preparation', 'scientific']:
        raise ValueError('Unknown finite stage')
    if config.get('global_input_batch', 512) != 512 and not config.get('qualification_only'):
        raise ValueError('Formal independent input batch must be512')
    if config.get('updates_per_input_batch', 1) != 1:
        raise ValueError('Paper reference uses one update per fresh rollout')
    if config['phase']=='scientific' and int(config.get('epochs', 4))!=4 and not config.get('qualification_only'):
        extension=config.get('registered_budget_extension',{})
        if type(extension.get('prefix_epochs')) is not int or type(extension.get('target_epochs')) is not int or not 4<=extension['prefix_epochs']<extension['target_epochs']==config['epochs']:
            raise ValueError('Non-default science budget requires an explicit registered extension')
    if full:
        required=['gamma','lambda_dyn','lambda_div','dynamic_coefficients','diversity_cap','diversity_rho','fixed_total_concentration','enable_dirichlet_objective','enable_dynamic','enable_diversity']
        if any(k not in full for k in required):raise ValueError('Incomplete explicit DGF configuration')
        if config['route']!='D-post':raise ValueError('This full DGF reference uses the qualified D-post route')
        if config.get('method_kind')=='FULL':
            if full['fixed_total_concentration'] or not all(full[k] for k in ['enable_dirichlet_objective','enable_dynamic','enable_diversity']):raise ValueError('Disabled core feature cannot be called FULL')
            if min(full['gamma'],full['lambda_dyn'],full['lambda_div'],*full['dynamic_coefficients'])<=0:raise ValueError('All full-method core coefficients must be positive')
        if len(full['dynamic_coefficients'])!=4:raise ValueError('Four explicit dynamic descriptors are required')
        if config.get('PPO_clip') is not None or config.get('reference_beta',0)!=0 or config.get('nll_anchor_coefficient',0)!=0:
            raise ValueError('Engineering extensions are not enabled in this paper reference')
    return dict(phase=config['phase'],route=config['route'],J=config.get('J'),G=config.get('G'),K=config.get('K'),epochs=config.get('epochs'),updates_per_input_batch=1,schedule=config.get('adaptive_dgf',{}).get('learning_rate',{'kind':'constant'})['kind'],global_input_batch=config.get('global_input_batch',512),native_RL_sampling_active=config['phase']=='scientific' and config.get('arm') not in ['E00','E02'],objective='paper_mixed_on_policy' if full else 'trajectory_mixture_NLL' if config.get('arm') in ['E00','E02'] or config['phase']=='preparation' else 'matched_single_mode_position_RL',dgf_complete=bool(full))


def last_history(batch, dimensions):
    valid=batch['observed_mask'] & ~batch['prediction_mask'][...,None] & (batch['sample_id']>0)[...,None]
    variables=batch['variate_id'][...,None].expand_as(valid).flatten(1)
    values=batch['target'].flatten(1);valid=valid.flatten(1)
    positions=torch.arange(values.shape[-1],device=values.device)[None].expand_as(values)
    selected=[];observed=[]
    for variable in range(dimensions):
        index=torch.where(valid & (variables==variable),positions,torch.full_like(positions,-1)).max(-1).values
        selected.append(values.gather(-1,index.clamp_min(0)[:,None]).squeeze(-1));observed.append(index>=0)
    return torch.stack(selected,-1),torch.stack(observed,-1)


def feature_scales(config, batch):
    scales=config['protocol']['diversity_scales']
    result=torch.tensor([scales[k] for k in ['level','diff1','diff2','fft']],device=batch['target'].device,dtype=torch.float64)
    return result[None].expand(len(batch['target']),-1,-1)


def make_rollout(policy, reference, batch, config, reliability=None):
    if reliability is not None:raise ValueError('Selective weights are outside this finite reference')
    h=config['protocol']['horizon'];d=forecast_dimensions(config['protocol']);assert_complete_future(batch,h*d)
    settings=config['dgf_complete']
    with torch.no_grad():
        experts,target,w,alpha,kappa=joints(policy(batch),batch)
        if alpha is None or policy.k!=8:raise ValueError('Full reference requires eight outer modes and Dirichlet concentrations')
        action=rollout(experts,w,alpha,config['J'],config['G'])
        forecast=structured(action['forecast'],d,h).double();truth=structured(target,d,h).double()
        mask=torch.ones_like(truth,dtype=torch.bool)
        last,history_valid=last_history(batch,d)
        accuracy=-masked_mean((forecast-truth[:,None,None]).square(),mask[:,None,None],(-2,-1))
        terms=dynamic_terms(forecast,truth,mask,last.double(),history_valid)
        coefficients=torch.tensor(settings['dynamic_coefficients'],device=forecast.device,dtype=forecast.dtype)
        dynamic=-(terms*coefficients).sum(-1)
        reward=accuracy+(settings['lambda_dyn']*dynamic if settings['enable_dynamic'] else 0)
        a_sample,a_dir=hierarchical_advantages(reward)
        old_position=expert_log_prob(experts,action).detach()
        old_dir=torch.distributions.Dirichlet(alpha[:,None]).log_prob(action['pi']).detach()
        return dict(actions=action,old=old_position,old_router=old_dir,reward=reward.detach(),sample_advantage=a_sample,dir_advantage=a_dir,
            observed=mask,reward_mean=float(reward.mean()),reward_zero_std_fraction=float((reward.std(2,correction=0)==0).double().mean()),
            dynamic_terms=terms.mean((0,1,2)).cpu().tolist(),accuracy_mean=float(accuracy.mean()),
            sampling_evidence=dict(pi=action['pi'][:1,:2].cpu().tolist(),mode_id=action['mode'][:1,:2,:8].cpu().tolist(),
                expected_weights=w[:1].cpu().tolist(),kappa=kappa[:1].cpu().tolist(),layout='pi[N,B,K];mode[N,B,G];forecast[N,B,G,H,D]',shared_pi=True,mode_scope='complete_future_trajectory'))


def objective(policy,batch,record,config):
    settings=config['dgf_complete'];h=config['protocol']['horizon'];d=forecast_dimensions(config['protocol'])
    output=policy(batch);experts,target,w,alpha,kappa=joints(output,batch)
    position=expert_log_prob(experts,record['actions'])
    categorical=record['actions']['pi'].log().gather(-1,record['actions']['mode'])
    trajectory=position.double().sum(-1)+categorical.double()
    direction=torch.distributions.Dirichlet(alpha[:,None]).log_prob(record['actions']['pi']).double()
    if settings['enable_diversity']:
        means=torch.stack([structured(e.mean,d,h) for e in experts],1).double()
        diversity,distance=mode_diversity(means,record['observed'],feature_scales(config,batch),settings['diversity_cap'],settings['diversity_rho'])
    else:diversity=trajectory.new_zeros(());distance=None
    gamma=settings['gamma'] if settings['enable_dirichlet_objective'] else 0.
    coefficient=settings['lambda_div'] if settings['enable_diversity'] else 0.
    loss,terms=complete_loss(trajectory,direction,record['reward'],diversity,gamma,coefficient)
    adaptive={}
    if config.get('adaptive_dgf'):
        from .adaptive_dgf import regularize_and_diagnose
        loss,adaptive=regularize_and_diagnose(policy,experts,w,target,loss,terms,config)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full DGF objective')
    dependency={}
    if config.get('_optimizer_step',0)==0 and config.get('_paired_rollout_key','0/0/0').endswith('/0'):
        concentration=[p for p in policy.router_concentration.parameters() if p.requires_grad]
        if concentration and gamma:
            grads=torch.autograd.grad(-gamma*terms['J_dir'],concentration,retain_graph=True,allow_unused=True)
            dependency['weighted_J_dir_concentration_gradient_norm']=sum(float(g.detach().norm()) for g in grads if g is not None)
        if settings['enable_diversity']:
            head=next(policy.heads[0].parameters())
            grad=torch.autograd.grad(-coefficient*diversity,head,retain_graph=True,allow_unused=True)[0]
            dependency['weighted_diversity_head0_gradient_norm']=0. if grad is None else float(grad.detach().norm())
        dependency['scope']='First optimizer update and first microbatch only; targeted dependencies may legitimately be zero at saturation'
    delta=position.double()-record['old'].double()
    if delta.detach().abs().max()>1e-5:raise RuntimeError('On-policy actions were reused after a policy update')
    diagnostic=dict(loss=float(loss.detach()),**{k:float(v.detach()) for k,v in terms.items()},gamma=gamma,lambda_dyn=settings['lambda_dyn'],lambda_div=coefficient,
        reward_mean=record['reward_mean'],reward_zero_std_fraction=record['reward_zero_std_fraction'],accuracy_mean=record['accuracy_mean'],dynamic_terms=record['dynamic_terms'],
        sampling_evidence=record['sampling_evidence'],kappa=[float(kappa.detach().min()),float(kappa.detach().max())],
        max_fixed_action_log_probability_change=float(delta.detach().abs().max()),
        diversity_cap_fraction=None if distance is None else float((distance.detach()>=settings['diversity_cap']).double().mean()),
        mode_distance_mean=None if distance is None else float(distance.detach().mean()),
        router_entropy_mean=float((-(w.detach()*w.detach().log()).sum(-1)).mean()),
        component_gradient_evidence=dependency,gradient_estimator='detached_augmented_native_family_and_trajectory_score; separate detached group credit',objective_reduction='mean_NBG J_sample;mean_NB J_dir;mean_N R_div')
    diagnostic.update(adaptive)
    return loss,diagnostic
