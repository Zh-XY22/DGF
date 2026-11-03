"""Continuous-action adaptations; augmented-position factors are explicit.

These are finite experiment mechanisms, not complete language-model algorithms.
Inputs are fixed sampled actions with stored behavior probabilities. Neither a
geometric ratio nor a clipped surrogate is called an exact joint IS estimator.
"""
import torch


def centered(reward,axis=2):return reward-reward.mean(axis,keepdim=True)


def fixed_scale_advantage(reward,scale,axis=2):
    if not torch.is_tensor(scale):scale=reward.new_tensor(scale)
    if not torch.isfinite(scale).all() or not (scale>0).all():raise ValueError('Frozen training scale must be finite positive')
    return (centered(reward,axis)/scale).detach()


def global_batch_scale(reward_chunks):
    """Complete independent-input batch, not a mean of microbatch std values."""
    count=sum(x.numel() for x in reward_chunks)
    total=sum(x.double().sum() for x in reward_chunks)
    square=sum(x.double().square().sum() for x in reward_chunks)
    variance=(square/count-(total/count).square()).clamp_min(0)
    return variance.sqrt().detach()


def ppo_asymmetric(current,behavior,advantage,low=.2,high=.28):
    ratio=(current.double()-behavior.detach().double()).exp();advantage=advantage.detach().double()
    return torch.minimum(ratio*advantage,ratio.clamp(1-low,1+high)*advantage)


def cispo(current,behavior,advantage,upper=1.2,lower=0.):
    ratio=(current.double()-behavior.detach().double()).exp()
    weight=ratio.clamp(lower,upper).detach()
    return weight*advantage.detach().double()*current.double()


def geometric_surrogate(current,behavior,trajectory_advantage,clip,block_length=None):
    """[N,J,G,H,D] -> scalar, H×D length normalization, fixed blocks if set."""
    if current.shape!=behavior.shape or current.ndim!=5:raise ValueError('Expected explicit N,J,G,H,D layout')
    h,d=current.shape[-2:];blocks=[(0,h)] if block_length is None else [(i,min(i+block_length,h)) for i in range(0,h,block_length)]
    result=current.new_zeros((),dtype=torch.float64)
    advantage=trajectory_advantage.detach().double()
    for begin,end in blocks:
        log_ratio=(current[...,begin:end,:].double()-behavior[...,begin:end,:].detach().double()).mean((-2,-1))
        ratio=log_ratio.exp()
        value=torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage)
        result=result+(end-begin)/h*value.mean()
    return result


def map_baseline_advantage(reward,map_reward,frozen_scale):
    if map_reward.shape!=reward.shape[:1]+reward.shape[-2:]:raise ValueError('MAP baseline must use the same history and positions')
    if frozen_scale<=0:raise ValueError('Invalid frozen training scale')
    return ((reward-map_reward.detach()[:,None,None])/frozen_scale).detach()


def exact_reference_kl_surrogate(current,behavior,reference):
    """Correct augmented-factor KL gradient surrogate under stored behavior.

    The returned loss has the desired score gradient, not the KL value. The
    diagnostic estimate uses rho*(log_current-log_reference). No clipping here.
    Conditional factorization is required; this entry is qualified only for A0.
    """
    delta=current.double()-behavior.detach().double();weight=delta.exp().detach()
    difference=current.double()-reference.detach().double()
    surrogate=(weight*(difference+1).detach()*current.double()).mean()
    estimate=(weight*difference.detach()).mean()
    return surrogate,estimate


def structured_credits(forecast,target,first_coefficient=.1,second_coefficient=.1):
    """One shared reward and common G normalizer, plus explicit local factors."""
    if forecast.ndim!=5 or target.shape!=forecast.shape[:1]+forecast.shape[-2:]:raise ValueError('Invalid forecast/target layout')
    prediction=forecast.double();truth=target[:,None,None].double();h,d=prediction.shape[-2:]
    accuracy=-(prediction-truth).square()
    total=accuracy.mean((-2,-1));local=[dict(positions=1,credit_reward=accuracy,coefficient=1.,denominator=h*d)]
    for order,coefficient in [(1,first_coefficient),(2,second_coefficient)]:
        error=-(prediction.diff(n=order,dim=-2)-truth.diff(n=order,dim=-2)).square()
        total=total+coefficient*error.mean((-2,-1))
        local.append(dict(positions=order+1,credit_reward=error,coefficient=coefficient,denominator=(h-order)*d))
    std=total.std(2,correction=0,keepdim=True)
    trajectory=torch.where(std>0,centered(total)/(std+1e-6),torch.zeros_like(total)).detach()
    for term in local:
        scale=std[...,None,None]
        term['advantage']=torch.where(scale>0,centered(term['credit_reward'])/(scale+1e-6),torch.zeros_like(term['credit_reward'])).detach()
    return dict(total_reward=total.detach(),trajectory_advantage=trajectory,local=local,common_std=std.detach())


def local_dependency_score(logp,positions):
    result=logp[...,:logp.shape[-2]-positions+1,:]
    for offset in range(1,positions):result=result+logp[...,offset:logp.shape[-2]-positions+1+offset,:]
    return result
