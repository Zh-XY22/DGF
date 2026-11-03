"""Explicit direct-distribution and direct-MAP objectives for the new campaign."""
import torch
from .probability import nll


def direct_objective(experts,weights,target,config):
    objective=config['direct_objective']
    if objective=='frozen_residual_location_MSE':
        if len(experts)!=1 or getattr(experts[0],'transport_mode',None)!='P04_LOCATION':raise ValueError('P04 requires its functional frozen-residual distribution')
        variance=experts[0].variance
        if not torch.isfinite(variance).all() or (variance<0).any():raise FloatingPointError('P04 expected squared-error objective has no finite nonnegative second moment')
        point=(experts[0].mean.double()-target.double()).square().mean()
        loss=point+variance.detach().mean()
        diagnostic=dict(direct_objective=objective,point_MSE=float(point.detach()),frozen_residual_variance=float(variance.mean()),
            gradient_estimator='Exact finite-second-moment gradient of expected sample MSE with a fixed residual law',
            native_PPO=False,reference_KL=False,NLL_in_objective=False,scope='Location-risk adaptation; not a pure native-PPO scale-freezing ablation')
    elif objective=='frozen_mean_scale_CRPS':
        if len(experts)!=1 or getattr(experts[0],'transport_mode',None)!='P05_SCALE':raise ValueError('P05 requires its mean-preserving positive scale calibrator')
        from .distribution_crps import moirai_full_crps
        loss,diagnostic=moirai_full_crps(experts,weights,target,per_component=config.get('crps_per_component',4))
        diagnostic.update(direct_objective=objective,reference_mean_fixed=True,NLL_in_objective=False,
            gradient_estimator='Pathwise positive affine transport of a completely frozen mixed-family reference')
    elif objective in ['MAP_MAE','MAP_MSE']:
        means=torch.stack([e.mean for e in experts],1)
        chosen=weights.detach().argmax(-1)
        prediction=means[torch.arange(len(target),device=target.device),chosen]
        if not torch.isfinite(prediction).all():raise FloatingPointError('MAP component mean does not exist or is invalid')
        error=prediction.double()-target.double()
        loss=error.abs().mean() if objective=='MAP_MAE' else error.square().mean()
        diagnostic=dict(direct_objective=objective,NLL_in_objective=False,output='Legal history-MAP mode mean; native family mixture remains marginalized',valid_elements=target.numel())
    elif objective in ['fair_full_CRPS','fair_iid_score_CRPS','fair_full_CRPS_NLL','almost_fair_full_CRPS']:
        from .distribution_crps import moirai_full_crps,fresh_iid_crps_score
        if objective=='fair_iid_score_CRPS':loss,diagnostic=fresh_iid_crps_score(experts,weights,target,config.get('crps_samples',32))
        else:loss,diagnostic=moirai_full_crps(experts,weights,target,per_component=config.get('crps_per_component',4),
            almost_fair_alpha=config.get('almost_fair_alpha',.95) if objective=='almost_fair_full_CRPS' else None)
        if objective=='fair_full_CRPS_NLL':
            coefficient=config['probability_NLL_coefficient']
            if coefficient<=0:raise ValueError('P03 requires its frozen positive coefficient')
            anchor=nll(experts,weights,target);loss=loss+coefficient*anchor
            diagnostic.update(NLL_coefficient=coefficient,training_NLL=float(anchor.detach()))
        diagnostic['direct_objective']=objective
    else:raise ValueError('Unqualified direct objective: '+str(objective))
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite direct training objective')
    return loss,dict(diagnostic,loss=float(loss.detach()))
