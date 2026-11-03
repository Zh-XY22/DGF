"""Fixed-action hierarchical PPO; separate old rollout and frozen branch reference.

Expert factors include native family actions. This is the declared augmented
hierarchical surrogate, not clipped PPO on a single marginal trajectory density.
"""
import torch
from .policy import joints,rollout,expert_log_prob
from .probability import nll,legacy_clipped_kl
from .data import structured,assert_complete_future,forecast_dimensions
from .rewards import relative,ppo_surrogate,block_advantages,block_policy_loss,difference_terms,dependency_log_prob


def history_average(value):return value.flatten(1).mean(-1).mean()


def make_rollout(policy,reference,batch,config,reliability=None):
    h=config['protocol']['horizon'];dim=forecast_dimensions(config['protocol'])
    assert_complete_future(batch,h*dim)
    with torch.no_grad():
        experts,target,w,alpha,kappa=joints(policy(batch),batch)
        ref_experts,_,ref_w,ref_alpha,_=joints(reference(batch),batch)
        actions=rollout(experts,w,alpha,config.get('J',8),config.get('G',4))
        old=expert_log_prob(experts,actions);ref_lp=expert_log_prob(ref_experts,actions)
        forecast=structured(actions['forecast'],dim,h);truth=structured(target,dim,h)
        observed=torch.ones_like(truth,dtype=torch.bool)
        error=-(forecast.double()-truth[:,None,None].double()).square()
        weight=torch.ones_like(truth,dtype=torch.float64) if reliability is None else reliability.detach().double()
        if weight.shape!=truth.shape or (weight<0).any():raise ValueError('Reliability shape/range mismatch')
        if (weight.sum((-2,-1))==0).any():raise ValueError('Empty reliability requires explicit fallback before rollout')
        reward=(error*weight[:,None,None]).sum((-2,-1))/weight.sum((-2,-1))[:,None,None]
        router_reward=reward
        record=dict(actions=actions,old=old.detach(),reference_lp=ref_lp.detach(),advantage=relative(error),
            observed=observed,weight=weight,mode_advantage=relative(reward),
            dir_advantage=relative(reward.mean(2),axis=1),reference_w=ref_w.detach(),
            reference_alpha=None if ref_alpha is None else ref_alpha.detach(),
            old_router=None if alpha is None else torch.distributions.Dirichlet(alpha[:,None]).log_prob(actions['pi']).detach(),
            reward_mean=float(reward.mean()),reward_zero_std_fraction=float((reward.std(2,correction=0)==0).double().mean()))
        arm=config['arm']
        if arm in ['E12','E13','E14']:
            record['blocks']=block_advantages(forecast.double(),truth.double(),observed)
            # Block reward is global MSE for complete equal-size blocks. Router consequences stay global.
            total=record['blocks']['total_reward']
            router_reward=total
            record['mode_advantage']=relative(total);record['dir_advantage']=relative(total.mean(2),axis=1)
        if arm=='E14':
            record['differences']=[]
            common=record['blocks']['common_std'][...,None,None]
            full_reward=record['blocks']['total_reward'].clone()
            for order in [1,2]:
                local,valid=difference_terms(forecast.double(),truth.double(),observed,order)
                centered=local-local.mean(2,keepdim=True)
                adv=torch.where(common>0,centered/(common+1e-6),torch.zeros_like(centered))
                adv=torch.where(valid[:,None,None],adv,torch.zeros_like(adv)).detach()
                coefficient=config['difference_coefficients'][str(order)]
                full_reward=full_reward+coefficient*local.sum((-2,-1))/valid.sum((-2,-1)).clamp_min(1)[:,None,None]
                record['differences'].append(dict(order=order,advantage=adv,valid=valid,coefficient=coefficient))
            record['mode_advantage']=relative(full_reward);record['dir_advantage']=relative(full_reward.mean(2),axis=1)
            router_reward=full_reward
        # Reporting only: detached reductions, no random draw, in-place change
        # or use in the policy loss. Existing frozen workers retain old logs.
        from .rollout_diagnostics import group_statistics
        record['reward_statistics']=dict(version='fixed-actions-v1',accuracy=group_statistics(reward),
            full_router=group_statistics(router_reward),position_accuracy_variance_G=float(error.var(2,correction=0).mean()))
        return record


def objective(policy,batch,record,config):
    h=config['protocol']['horizon'];dim=forecast_dimensions(config['protocol'])
    output=policy(batch);experts,target,w,alpha,kappa=joints(output,batch)
    lp=expert_log_prob(experts,record['actions'])
    current=structured(lp,dim,h);old=structured(record['old'],dim,h)
    arm=config['arm']
    if arm in ['E12','E13','E14']:
        expert_loss=block_policy_loss(current,old,record['observed'],record['blocks'],local=arm!='E12',clip=.2)
    else:
        surrogate=ppo_surrogate(current,old,record['advantage'],.2)
        weight=record['weight'][:,None,None]
        expert_loss=-((surrogate*weight).sum((-2,-1))/weight.sum((-2,-1))).mean()
    local_loss=expert_loss.new_zeros(())
    for term in record.get('differences',[]):
        # Conditional on mode, these exact two/three-coordinate native family actions factorize.
        cur=dependency_log_prob(current,term['order']);prev=dependency_log_prob(old,term['order'])
        value=ppo_surrogate(cur,prev,term['advantage'],.2)
        valid=term['valid'][:,None,None]
        local_loss-=term['coefficient']*(torch.where(valid,value,torch.zeros_like(value)).sum((-2,-1))/valid.sum((-2,-1)).clamp_min(1)).mean()
    router_loss=expert_loss.new_zeros(())
    if alpha is not None:
        cur=torch.distributions.Dirichlet(alpha[:,None]).log_prob(record['actions']['pi'])
        router_loss=-ppo_surrogate(cur,record['old_router'],record['dir_advantage'],.2).mean()
    elif len(experts)>1:
        raise ValueError('This finite research rollout expects D-pre Dirichlet or A0, not a new softmax arm')
    kl=legacy_clipped_kl(lp,record['reference_lp']).mean()
    if alpha is not None:
        kl+=torch.distributions.kl_divergence(torch.distributions.Dirichlet(alpha),torch.distributions.Dirichlet(record['reference_alpha'])).mean()
    anchor_coefficient=0. if arm in ['E16','E17'] else config['nll_anchor_coefficient']
    # A disabled likelihood is not part of this objective, even if its diagnostic
    # would be undefined. In particular, zero times an infinite NLL is not zero.
    anchor=nll(experts,w,target) if anchor_coefficient!=0 else None
    loss=expert_loss+local_loss+router_loss+.01*kl
    if anchor is not None:loss=loss+anchor_coefficient*anchor
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite hierarchical objective')
    delta=(lp.double()-record['old'].double()).detach();joint_delta=delta.sum(-1)
    diagnostic=dict(loss=float(loss.detach()),expert_loss=float(expert_loss.detach()),local_loss=float(local_loss.detach()),
        router_loss=float(router_loss.detach()),reference_regularizer=float(kl.detach()),
        nll=None if anchor is None else float(anchor.detach()),
        nll_status='NOT_COMPUTED_ZERO_COEFFICIENT' if anchor is None else 'COMPUTED_TRAINING_ANCHOR',
        nll_anchor_coefficient=anchor_coefficient,
        max_position_log_ratio=float(delta.max()),min_position_log_ratio=float(delta.min()),
        joint_log_ratio_quantiles=torch.quantile(joint_delta.flatten(),torch.tensor([0.,.5,.95,1.],device=lp.device,dtype=torch.float64)).cpu().tolist(),
        position_clip_fraction=float(((delta.exp()<.8)|(delta.exp()>1.2)).double().mean()),
        reward_mean=record['reward_mean'],reward_zero_std_fraction=record['reward_zero_std_fraction'],
        reward_statistics=record.get('reward_statistics'),
        kappa=None if kappa is None else [float(kappa.min()),float(kappa.max())])
    return loss,diagnostic
