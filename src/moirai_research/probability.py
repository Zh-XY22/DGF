"""Explicit measure-correct marginal prediction probabilities.

Historical native Mixture.log_prob remains available as a separately named legacy
score. This module does not change the four component families or their sampler.
"""
import torch
from uni2ts.distribution.negative_binomial import NegativeBinomial
from .distillation import mixed_measure_log_prob

VERSION='mixed-measure-marginal-v1'

def common_scaler(experts):
    first=experts[0]
    for expert in experts[1:]:
        if not torch.equal(first.loc,expert.loc) or not torch.equal(first.scale,expert.scale):
            raise ValueError('Experts do not share a history-only atomic lattice')
    return first.loc,first.scale

def position_logp(expert,raw,atomic):
    return mixed_measure_log_prob(
        [(c,isinstance(c,NegativeBinomial)) for c in expert.mixture.components],
        expert.mixture.weights.logits,raw,expert.scale,atomic)

def observation_logp(experts,weights,target):
    """[N,P] truth -> joint [N], marginal positions [N,P], atomic mask [N,P].

    Exact floating point lattice membership, without a tolerance band. Continuous
    labels do not acquire count mass merely because a real-extended NB formula
    happens to be finite. Values on the lattice use counting mass only.
    """
    if any(getattr(expert,'moving_atomic_support',False) for expert in experts):
        raise NotImplementedError('NLL is unavailable for cross-model comparison with moving atomic supports; point and CRPS results remain valid')
    loc,scale=common_scaler(experts);raw=(target-loc)/scale
    atomic=(raw>=0)&(raw==raw.round())
    lp=torch.stack([position_logp(e,raw,atomic) for e in experts],-1)
    joint=torch.logsumexp(lp.double().sum(-2)+weights.double().log(),-1)
    marginal=torch.logsumexp(lp.double()+weights.double().log()[:,None],-1)
    return joint,marginal,atomic

def nll(experts,weights,target):
    joint,_,_=observation_logp(experts,weights,target)
    return -(joint/target.shape[-1]).mean()

def action_marginal_logp(experts,weights,actions):
    """Marginalize mode and family for detached samples [S,N,P].

    All policies being compared must share the scaler/lattice. Action component
    labels identify whether the sample is atomic, but continuous family IDs and
    mode IDs are marginalized out, unlike augmented-action PPO.
    """
    common_scaler(experts)
    kinds=[isinstance(c,NegativeBinomial) for c in experts[0].mixture.components]
    atomic=torch.tensor(kinds,device=weights.device)[actions['component']]
    lp=torch.stack([position_logp(e,actions['raw'],atomic).double().sum(-1) for e in experts],-1)
    return torch.logsumexp(lp+weights.double().log()[None],-1)

def legacy_clipped_kl(logp,reference_logp):
    """Fixed historical verl low_var_kl formula, not an unbiased full KL claim.

    delta is bounded only outside the region where exp(delta)-delta-1 has already
    exceeded the historical cap 10; this preserves capped values and gradients.
    Reference: verl 070ed6ac, core_algos.kl_penalty(low_var_kl).
    """
    d=(reference_logp.double()-logp.double()).clamp(-12.,4.)
    return (d.exp()-d-1).clamp(-10.,10.).to(logp.dtype)
