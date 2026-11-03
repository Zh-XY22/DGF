"""Explicit A0 continuous-action adaptations for the finite four-H campaign.

Behavior-family actions are retained. These objectives do not turn the native
four distribution families into forecasting modes. All score inputs are fixed.
"""
import math
import torch
from .data import structured, forecast_dimensions
from .policy import joints, expert_log_prob
from .probability import nll, legacy_clipped_kl
from .rewards import relative, ppo_surrogate
from . import continuous_estimators as estimators


RECIPES={'R01','R02','R03','R04','R05','R06','R07','R08','R09','R10','R11','R12','R13'}


def clipping_operator(config):
    """Only position-advantage/position-clipping pairs are currently qualified.

    A trajectory geometric surrogate changes score granularity and cannot use
    this position-advantage compositor. No fallback silently discards a field.
    """
    base=config['continuous_recipe'];override=config.get('clipping_recipe')
    if override is not None and (base not in {'R01','R02','R07'} or override not in {'R03','R04'}):
        raise ValueError('Unqualified advantage/clipping combination')
    return override or base


def make_rollout(policy,reference,batch,config,reliability=None):
    from .rl import make_rollout as native_rollout
    recipe=config['continuous_recipe'];clipping_operator(config)
    if recipe not in RECIPES or policy.route!='A0':raise ValueError('Only qualified A0 continuous adaptations are supported')
    if recipe in {'R12','R13'} and (config.get('paired_action_cache') or config.get('paired_action_role')=='REGENERATE'):
        from .paired_credit import common_record
        return common_record(reference,batch,config)
    # Matched-credit pair uses exactly the same frozen behavior and RNG stream
    # throughout training. Both arms declare this off-policy design explicitly.
    behavior=reference if recipe in {'R12','R13'} else policy
    record=native_rollout(behavior,reference,batch,config,reliability)
    h=config['protocol']['horizon'];d=forecast_dimensions(config['protocol'])
    with torch.no_grad():
        experts,target,weights,_,_=joints(behavior(batch),batch)
        prediction=structured(record['actions']['forecast'],d,h);truth=structured(target,d,h)
        record['position_reward']=-(prediction.double()-truth[:,None,None].double()).square()
        if recipe in {'R01','R07'}:
            scale=config['frozen_reward_scale']
            if recipe=='R01':record['advantage']=estimators.fixed_scale_advantage(record['position_reward'],scale)
            else:
                mean=structured(experts[0].mean,d,h)
                baseline=-(mean.double()-truth.double()).square()
                record['advantage']=estimators.map_baseline_advantage(record['position_reward'],baseline,scale)
        if recipe in {'R12','R13'}:
            record['structured']=estimators.structured_credits(prediction,truth,
                config.get('difference_reward_first',.1),config.get('difference_reward_second',.1))
            record['behavior_scope']='Frozen shared RL parent; identical action streams for R12/R13 at matched input/seed'
    return record


def prepare_complete_batch(records,config):
    if config['continuous_recipe']!='R02':return
    scale=estimators.global_batch_scale([r['position_reward'] for r in records]).clamp_min(1e-6)
    for record in records:
        record['advantage']=estimators.fixed_scale_advantage(record['position_reward'],scale)
        record['global_complete_batch_scale']=float(scale)


def objective(policy,batch,record,config):
    recipe=config['continuous_recipe'];h=config['protocol']['horizon'];d=forecast_dimensions(config['protocol'])
    experts,target,weights,alpha,_=joints(policy(batch),batch)
    if alpha is not None or len(experts)!=1:raise ValueError('A0 qualification cannot be applied to shared-mode distributions')
    lp=expert_log_prob(experts,record['actions']);old=record['old'].detach()
    current=structured(lp,d,h);behavior=structured(old,d,h);adv=record['advantage'].detach()
    delta=current.double()-behavior.double()
    clipping=clipping_operator(config)
    if clipping=='R03':
        score=estimators.ppo_asymmetric(current,behavior,adv,config['clip_low'],config['clip_high'])
        policy_loss=-score.mean()
    elif clipping=='R04':
        score=estimators.cispo(current,behavior,adv,upper=config['cispo_upper'],lower=0.)
        policy_loss=-score.mean()
    elif recipe in {'R05','R06'}:
        policy_loss=-estimators.geometric_surrogate(current,behavior,record['mode_advantage'],
            config['geometric_clip'],block_length=None if recipe=='R05' else config['geometric_block_length'])
    elif recipe in {'R12','R13'}:
        term=record['structured'];joint_delta=delta.sum((-2,-1))
        # Clamping in log space is exactly the declared clipped IS weight and
        # prevents overflow without silently changing the registered bound.
        importance=joint_delta.clamp(math.log(.8),math.log(1.2)).exp().detach()
        if recipe=='R12':
            surrogate=importance*term['trajectory_advantage']*current.double().sum((-2,-1))
            policy_loss=-surrogate.mean()
        else:
            total=current.new_zeros((),dtype=torch.float64)
            for part in term['local']:
                dependency=estimators.local_dependency_score(current,part['positions'])
                local=(part['advantage']*dependency.double()).sum((-2,-1))/part['denominator']
                total=total+part['coefficient']*(importance*local).mean()
            policy_loss=-total
    else:
        policy_loss=-ppo_surrogate(current,behavior,adv,.2).mean()
    legacy=legacy_clipped_kl(lp,record['reference_lp']).mean()
    loss=policy_loss+.01*legacy
    diagnostics=dict(continuous_recipe=recipe,clipping_recipe=clipping,advantage_recipe=recipe,policy_loss=float(policy_loss.detach()),
        legacy_reference_regularizer=float(legacy.detach()),reference_beta=.01,
        behavior_is_fixed=recipe in {'R12','R13'},NLL_in_objective=False,
        max_position_log_ratio=float(delta.detach().max()),min_position_log_ratio=float(delta.detach().min()),
        position_clip_fraction=float(((delta.detach().exp()<.8)|(delta.detach().exp()>1.2)).double().mean()),
        reward_mean=record['reward_mean'],reward_zero_std_fraction=record['reward_zero_std_fraction'],
        reward_statistics=record.get('reward_statistics'),global_complete_batch_scale=record.get('global_complete_batch_scale'),
        paired_action_key=record.get('paired_action_key'),paired_action_role=record.get('paired_action_role'),paired_action_seed=record.get('paired_action_seed'))
    if recipe=='R10':
        kl_gradient,kl_estimate=estimators.exact_reference_kl_surrogate(lp,old,record['reference_lp'])
        coefficient=config['explicit_reference_KL_coefficient']
        loss=loss+coefficient*kl_gradient
        diagnostics.update(explicit_reference_KL_coefficient=coefficient,explicit_KL_estimate=float(kl_estimate.detach()),KL_gradient_surrogate=float(kl_gradient.detach()))
    if recipe=='R11':
        fraction=config['_optimizer_step']/max(1,config['_total_optimizer_steps']-1)
        coefficient=config['annealed_NLL_initial']*max(0.,1.-fraction)
        if coefficient:
            anchor=nll(experts,weights,target);loss=loss+coefficient*anchor
            diagnostics.update(NLL_in_objective=True,training_NLL=float(anchor.detach()))
        diagnostics['NLL_coefficient']=coefficient
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite registered continuous-policy objective')
    return loss,dict(diagnostics,loss=float(loss.detach()))
