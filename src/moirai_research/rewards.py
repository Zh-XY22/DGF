"""Mask-preserving credit. Layout is [history, subgroup, candidate, time, variable]."""
import torch

EPS = 1e-6

def relative(reward, axis=2):
    if not torch.isfinite(reward).all():
        raise FloatingPointError('Nonfinite reward')
    center = reward - reward.mean(axis, keepdim=True)
    std = center.square().mean(axis, keepdim=True).sqrt()
    return torch.where(std > 0, center / (std + EPS), torch.zeros_like(center)).detach()

def masked_mean(value, mask, dims):
    mask = mask.expand_as(value)
    return torch.where(mask, value, torch.zeros_like(value)).sum(dims) / mask.sum(dims).clamp_min(1)

def accuracy(prediction, target, observed):
    """Zero-valued labels remain valid; padding must already be excluded from observed."""
    valid = observed[:,None,None].expand_as(prediction)
    return torch.where(valid, -(prediction-target[:,None,None]).square(), torch.zeros_like(prediction))

def block_advantages(prediction, target, observed, blocks=((0,32),(32,64),(64,96))):
    """Both E12 and E13 receive exactly this object; common normalizer is total reward std."""
    if prediction.ndim != 5 or target.shape != observed.shape:
        raise ValueError('Expected [N,J,G,H,D] and [N,H,D]')
    h = prediction.shape[-2]
    if blocks[0][0] != 0 or blocks[-1][1] != h or any(a[1] != b[0] for a,b in zip(blocks,blocks[1:])):
        raise ValueError('Blocks must be a disjoint complete horizon partition')
    error = accuracy(prediction, target, observed)
    count = torch.stack([observed[:,lo:hi].sum((-2,-1)) for lo,hi in blocks],-1)
    weights = count / count.sum(-1,keepdim=True).clamp_min(1)
    reward = torch.stack([error[...,lo:hi,:].sum((-2,-1)) / count[:,b,None,None].clamp_min(1)
                          for b,(lo,hi) in enumerate(blocks)],-1)
    total = (reward*weights[:,None,None]).sum(-1)
    center = total - total.mean(2,keepdim=True)
    std = center.square().mean(2,keepdim=True).sqrt()
    advantage = (reward-reward.mean(2,keepdim=True))/(std[...,None]+EPS)
    advantage = torch.where((std[...,None]>0)&(count[:,None,None]>0),advantage,torch.zeros_like(advantage))
    return dict(reward=reward.detach(),weights=weights.detach(),total_reward=total.detach(),
                advantage=advantage.detach(),common_std=std.detach(),blocks=blocks)

def block_log_prob(position_logp, observed, blocks):
    safe = torch.where(observed[:,None,None],position_logp,torch.zeros_like(position_logp))
    return torch.stack([safe[...,lo:hi,:].sum((-2,-1)) for lo,hi in blocks],-1)

def ppo_surrogate(current, old, advantage, clip=.2):
    if old.requires_grad or advantage.requires_grad:
        raise ValueError('Rollout actions/old probabilities and advantages must be detached')
    delta=current.double()-old.double();ratio=delta.exp()
    if not torch.isfinite(ratio).all():
        raise FloatingPointError('Importance ratio overflow; no silent truncation')
    term=ratio*advantage
    return term if clip is None else torch.minimum(term,ratio.clamp(1-clip,1+clip)*advantage)

def block_policy_loss(current, old, observed, block_record, local, clip=.2):
    cur=block_log_prob(current,observed,block_record['blocks'])
    prev=block_log_prob(old,observed,block_record['blocks'])
    weighted=block_record['advantage']*block_record['weights'][:,None,None]
    if local:
        score=ppo_surrogate(cur,prev,block_record['advantage'],clip)
        return -(score*block_record['weights'][:,None,None]).sum(-1).mean()
    return -ppo_surrogate(cur.sum(-1),prev.sum(-1),weighted.sum(-1),clip).mean()

def difference_terms(prediction,target,observed,order):
    """Each reward retains its actual two/three-position dependency set, including holes."""
    if order not in (1,2):raise ValueError('Only registered first/second differences')
    valid=observed[:,:-order].clone()
    for offset in range(1,order+1):valid &= observed[:,offset:observed.shape[1]-order+offset]
    delta=prediction.diff(n=order,dim=-2)-target.diff(n=order,dim=-2)[:,None,None]
    reward=torch.where(valid[:,None,None],-delta.abs(),torch.zeros_like(delta))
    return reward,valid

def dependency_log_prob(position_logp,order):
    """Legal only conditional on all shared latents when coordinates factorize."""
    h=position_logp.shape[-2]
    return sum(position_logp[...,offset:h-order+offset,:] for offset in range(order+1))

def boundary_error(prediction,target,last_history,valid):
    return masked_mean(((prediction[...,0,:]-last_history[:,None,None])-
                        (target[:,0,:]-last_history)[:,None,None]).abs(),valid[:,None,None],-1)
