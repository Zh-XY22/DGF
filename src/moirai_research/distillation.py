"""Explicit fixed-action score-function estimators; no claim of autoregressive OPD."""
import torch

def forward_kl_surrogate(student_log_probability):
    """Teacher actions are fixed; teacher entropy is constant w.r.t. student parameters."""
    return -student_log_probability.mean()

def reverse_kl_surrogate(student_log_probability,teacher_log_probability,baseline=None):
    """Sample from the student, detach actions, and recompute current log p.

    This yields E_p[(log p-log q+1) score_p], not the incorrect fixed-action
    derivative of mean(log p-log q). A baseline must be independent of the action.
    """
    weight=(student_log_probability-teacher_log_probability+1).detach()
    if baseline is not None:weight=weight-baseline.detach()
    if not torch.isfinite(weight).all():raise FloatingPointError('Undefined or nonintegrable KL sample')
    return (weight*student_log_probability).mean()

def trajectory_mixture_log_prob(mode_position_logp,weights,observed):
    """[S,N,K,H,D] -> [S,N]; mode is fixed for the whole trajectory, not each point."""
    if mode_position_logp.ndim != 5:raise ValueError('Expected [S,N,K,H,D]')
    valid=observed[None,:,None].expand_as(mode_position_logp)
    joint=torch.where(valid,mode_position_logp,torch.zeros_like(mode_position_logp)).sum((-2,-1))
    return torch.logsumexp(joint+weights.log()[None],-1)

def mixed_measure_log_prob(components,log_weights,raw,scale,atomic):
    """Density w.r.t. Lebesgue + counting measure on the history-scaled NB lattice.

    `atomic` is supplied by the sampled family, never inferred using a loose
    tolerance. The final component order is explicitly provided as (dist,is_atomic).
    Teacher/student history scalers and atomic support must agree before use.
    Continuous density must not be added to atomic probability mass.
    """
    values=[]
    for i,(distribution,is_atomic) in enumerate(components):
        valid = atomic if is_atomic else ~atomic
        valid = valid & distribution.support.check(raw)
        if is_atomic:valid &= (raw==raw.round()) & (raw>=0)
        fallback=torch.zeros_like(raw) if is_atomic else distribution.mean.detach().expand_as(raw)
        safe=torch.where(valid,raw,fallback)
        lp=distribution.log_prob(safe)
        if not is_atomic:lp=lp-scale.log()
        values.append(torch.where(valid,lp+log_weights[...,i],torch.full_like(lp,-torch.inf)))
    result=torch.logsumexp(torch.stack(values,-1),-1)
    if not torch.isfinite(result).all():raise FloatingPointError('Action outside common mixed-measure support')
    return result
