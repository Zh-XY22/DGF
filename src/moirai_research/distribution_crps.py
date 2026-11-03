"""Full-family stratified CRPS gradients for a single Moirai trajectory mode.

Continuous components use legitimate pathwise derivatives. Discrete components
use detached samples plus their component score. Family probabilities are summed
exactly with gradients. Two independent banks make the pair expectation fair;
this is not a claim that a fixed-stratum ensemble is iid. No cross-mode alignment.
"""
import torch


def full_distribution_crps(components,probs,loc,scale,target,per_component=4,almost_fair_alpha=None):
    if len(components)!=probs.shape[-1] or per_component<1:raise ValueError('Invalid component/sample layout')
    banks=[];samples_used=2*per_component*len(components)
    for component in components:
        reparameterized=bool(component.has_rsample)
        draws=[]
        for bank in range(2):
            raw=component.rsample((per_component,)) if reparameterized else component.sample((per_component,)).detach()
            value=loc+scale*raw
            if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite distribution sample')
            score=None if reparameterized else component.log_prob(raw.detach())
            draws.append((value,score))
        banks.append(draws)
    def corrected(value,score):
        return value if score is None else value+value.detach()*(score-score.detach())
    absolute=target.new_zeros(target.shape,dtype=torch.float64)
    pair=absolute.clone()
    for c,draws in enumerate(banks):
        for value,score in draws:
            error=(value.double()-target.double()).abs()
            absolute=absolute+.5*probs[...,c].double()*corrected(error,score).mean(0)
        left,ls=draws[0]
        for d,right_draws in enumerate(banks):
            right,rs=right_draws[1]
            distance=(left[:,None].double()-right[None,:].double()).abs()
            score=None
            if ls is not None:score=ls[:,None].double()
            if rs is not None:score=rs[None,:].double() if score is None else score+rs[None,:].double()
            pair=pair+probs[...,c].double()*probs[...,d].double()*corrected(distance,score).mean((0,1))
    multiplier=1.
    if almost_fair_alpha is not None:
        if not 0<almost_fair_alpha<=1:raise ValueError('Almost-fair alpha must be in(0,1]')
        multiplier=1-(1-almost_fair_alpha)/samples_used
    loss=(absolute-.5*multiplier*pair).mean()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-distribution CRPS objective')
    return loss,dict(family_count=len(components),samples_per_component_per_bank=per_component,
        total_component_draws=samples_used,independent_banks=2,family_weights_exactly_marginalized=True,
        continuous_gradient='pathwise',discrete_gradient='score function; sampled values detached',
        almost_fair_multiplier=multiplier,objective_scope='Marginal per-position CRPS, then equal valid-position mean; not trajectory log likelihood')


def moirai_full_crps(experts,weights,target,per_component=4,almost_fair_alpha=None):
    if len(experts)!=1 or weights.shape[-1]!=1:
        raise ValueError('This qualified estimator currently requires A0; shared-mode multi-mode is a separate estimator')
    expert=experts[0]
    return full_distribution_crps(expert.mixture.components,expert.mixture.weights.probs,
        expert.loc,expert.scale,target,per_component=per_component,almost_fair_alpha=almost_fair_alpha)


def fresh_iid_crps_score(experts,weights,target,samples=32):
    from .crps_math import iid_crps_credit
    from .probability import position_logp
    from uni2ts.distribution.negative_binomial import NegativeBinomial
    if len(experts)!=1:raise ValueError('Fresh iid CRPS score is not valid for a shared-Dirichlet correlated ensemble')
    expert=experts[0]
    with torch.no_grad():actions=expert.sample(samples)
    credit=iid_crps_credit(actions['forecast'],target,iid=True,behavior_is_current=True)
    kinds=torch.tensor([isinstance(c,NegativeBinomial) for c in expert.mixture.components],device=target.device)
    atomic=kinds[actions['component']]
    logp=position_logp(expert,actions['raw'].detach(),atomic)
    loss=-(credit.detach()*logp.double()).mean()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fresh iid CRPS score objective')
    return loss,dict(samples=samples,behavior_is_current=True,iid=True,credit_pair_coefficient=1.,
        normalized_or_group_centered=False,clipped_or_reused=False,family_and_mode_marginalized=True,
        probability_space='Single mode, mixture-marginal position density relative to common continuous-plus-atomic measure',
        CRPS_estimator='Fresh iid score function before optimizer gradient-norm clipping')
