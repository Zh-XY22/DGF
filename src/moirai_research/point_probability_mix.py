"""Explicit point-objective plus fresh CRPS-gradient tradeoff for A0.

The point branch retains its original rollout/old/reference semantics. The
probability term is freshly constructed under the current policy per update.
This weighted objective is not claimed to be a strictly proper score.
"""
import math
import torch
from .policy import joints
from .four_horizon_losses import direct_objective

PROBABILITY_OBJECTIVES={'fair_full_CRPS','fair_iid_score_CRPS','almost_fair_full_CRPS'}

def add_probability_term(policy,batch,base_loss,base_diagnostics,config):
    mix=config['point_probability_mix']
    if config.get('phase') not in ['scientific','scientific_RL','scientific_DIRECT']:raise ValueError('Point/CRPS mix is scientific-phase only; preparation remains unchanged')
    if policy.route!='A0' or config.get('transport_mode'):raise ValueError('This mixing entry only qualifies native A0 probability space')
    if config.get('direct_objective') not in [None,'MAP_MAE','MAP_MSE']:raise ValueError('The base objective is not the declared point component')
    if config.get('algorithm_arm',config.get('arm')) in ['E00','E02'] and config.get('direct_objective') is None:raise ValueError('A pure NLL control is not a point-loss mixing branch')
    objective=mix['probability_objective'];coefficient=float(mix['coefficient'])
    if objective not in PROBABILITY_OBJECTIVES or not math.isfinite(coefficient) or coefficient<=0:raise ValueError('Unqualified probability component or nonpositive coefficient')
    if not config.get('qualification_only') and not mix.get('calibration_evidence'):raise ValueError('Formal mixing coefficient lacks frozen training-only calibration evidence')
    probability_config=dict(config,direct_objective=objective,crps_per_component=mix.get('crps_per_component',4),crps_samples=mix.get('crps_samples',32),almost_fair_alpha=mix.get('almost_fair_alpha',.95))
    experts,target,weights,_,_=joints(policy(batch),batch)
    probability_loss,probability_diagnostics=direct_objective(experts,weights,target,probability_config)
    combined=base_loss+coefficient*probability_loss
    if not torch.isfinite(combined):raise FloatingPointError('Nonfinite point/CRPS training objective; not a missing evaluation diagnostic')
    detail=dict(base_loss=float(base_loss.detach()),probability_loss=float(probability_loss.detach()),coefficient=coefficient,
        probability_objective=objective,combined_loss=float(combined.detach()),fresh_current_policy_probability_term=True,
        preparation_changed=False,strictly_proper_objective=False,probability_diagnostics=probability_diagnostics)
    if mix.get('measure_component_gradients'):
        parameters=[p for p in policy.parameters() if p.requires_grad]
        def norm(value):
            gradients=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
            total=sum(g.detach().double().square().sum() for g in gradients if g is not None)
            return float(total.sqrt()) if torch.is_tensor(total) else 0.
        detail.update(base_gradient_norm=norm(base_loss),probability_gradient_norm=norm(probability_loss))
    return combined,dict(base_diagnostics,point_probability_mix=detail)
