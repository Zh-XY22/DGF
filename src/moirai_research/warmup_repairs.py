"""Finite, versioned repairs for the registered ETTh1 warmup methods."""
import numpy as np
import torch
from .policy import FIELDS, joints
from .probability import nll
from .data import structured


def coarse_consistency(policy, batch, fine_mean, valid, protocol, training_values, indices):
    from uni2ts.model.moirai import MoiraiForecast
    if protocol['mode'] != 'M' or protocol['horizon'] != 96:
        raise ValueError('Only the registered ETTh1 multivariate H96 protocol is qualified')
    ids = batch['_index'].detach().cpu().numpy()
    past = []
    for origin, _ in indices[ids]:
        if origin < protocol['context'] or origin + 96 > protocol['train_end']:
            raise ValueError('Auxiliary views must use training windows')
        past.append(training_values[origin-protocol['context']:origin])
    past = torch.as_tensor(np.asarray(past), device=fine_mean.device, dtype=torch.float32)
    dimension = fine_mean.shape[-1]
    terms = []
    for scale in (2, 4):
        if past.shape[1] % scale:
            raise ValueError('Coarse history must contain complete averaging blocks')
        coarse_past = past.reshape(len(past), -1, scale, dimension).mean(2)
        horizon = 96 // scale
        forecast = MoiraiForecast(module=policy.module, context_length=coarse_past.shape[1],
            patch_size=protocol['patch'], prediction_length=horizon, num_samples=100,
            target_dim=dimension, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        future_placeholder = coarse_past.new_zeros((len(past), horizon, dimension))
        packed = forecast._convert(protocol['patch'], coarse_past,
            torch.ones_like(coarse_past, dtype=torch.bool),
            torch.zeros(coarse_past.shape[:2], device=past.device, dtype=torch.bool),
            future_target=future_placeholder,
            future_observed_target=torch.ones_like(future_placeholder, dtype=torch.bool),
            future_is_pad=torch.zeros(future_placeholder.shape[:2], device=past.device, dtype=torch.bool))
        coarse_batch = dict(zip(FIELDS[:-1], packed))
        coarse_batch['patch_size'] = torch.ones_like(coarse_batch['time_id']) * protocol['patch']
        if not torch.equal(coarse_batch['target'][coarse_batch['prediction_mask']],
                           torch.zeros_like(coarse_batch['target'][coarse_batch['prediction_mask']])):
            raise AssertionError('Auxiliary future input is not a zero placeholder')
        experts, _, weights, _, _ = joints(policy(coarse_batch), coarse_batch)
        coarse_mean = (torch.stack([e.mean for e in experts], 1) * weights[:, :, None]).sum(1)
        coarse_mean = structured(coarse_mean, dimension, horizon)
        fine_pool = fine_mean.reshape(len(past), horizon, scale, dimension).mean(2)
        valid_pool = valid.reshape(len(past), horizon, scale, dimension).all(2)
        if not torch.isfinite(coarse_mean).all():
            raise FloatingPointError('Auxiliary forecast mean does not exist')
        terms.append((coarse_mean-fine_pool).square().masked_select(valid_pool).mean())
    return sum(terms), dict(auxiliary_measure='same-backbone coarse/fine mean consistency',
        scales=[2, 4], auxiliary_context=[2500, 1250], auxiliary_horizon=[48, 24],
        independent_coarse_forward_calls=2, future_labels_used_as_input=False,
        component_losses=[float(x.detach()) for x in terms])


def hard_event_statistics(values, indices, horizon):
    scores = []
    for origin, _ in indices:
        if origin < 0 or origin + horizon > len(values):
            raise ValueError('Hard-event fitting crosses the training boundary')
        y = values[origin:origin+horizon].astype(np.float64)
        scores.append(float(np.abs(y).mean() + np.abs(np.diff(y, axis=0)).mean()))
    scores = np.asarray(scores)
    threshold = float(np.quantile(scores, .75))
    weights = 1. + 2. * (scores > threshold)
    return weights, dict(fit_split='train', frozen=True, quantile=.75, threshold=threshold,
                         minimum=float(weights.min()), maximum=float(weights.max()),
                         hard_fraction=float((weights > 1).mean()),
                         temporal_difference_axis=0, fitting_windows=len(indices))


def weighted_hard_event_loss(prediction, target, valid, weights, context):
    if context is None or context['weighted_elements'] <= 0:
        raise ValueError('Global input-batch weight normalization is required')
    per_input = ((prediction-target).square() * valid).sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)
    return (per_input * weights).sum() * context['total_inputs'] / (len(weights) * context['weighted_elements'])


def masked_history_loss(policy, batch, seed, policy_version):
    axis = torch.arange(batch['observed_mask'].shape[-1], device=batch['observed_mask'].device)
    active = axis[None, None, :] < batch['patch_size'][..., None]
    if (batch['observed_mask'] & ~active).any():
        raise ValueError('Carrier padding cannot contain observed values')
    complete_patch = (batch['observed_mask'] | ~active).all(-1)
    eligible = (~batch['prediction_mask']) & complete_patch & (batch['sample_id'] > 0)
    counts = eligible.sum(-1)
    if not (counts == counts[0]).all() or int(counts[0]) < 4:
        raise ValueError('A fixed complete-history-patch population is required')
    number = max(1, round(.25 * int(counts[0])))
    selected = torch.zeros_like(eligible)
    for row, index in enumerate(batch['_index'].tolist()):
        available = eligible[row].nonzero().flatten()
        generator = torch.Generator().manual_seed(int(seed) + 1009 * int(index) + 1000003 * int(policy_version))
        choice = torch.randperm(len(available), generator=generator)[:number].to(available.device)
        selected[row, available[choice]] = True
        for var in batch['variate_id'][row, eligible[row]].unique():
            if not ((batch['variate_id'][row] == var) & eligible[row] & ~selected[row]).any():
                raise ValueError('A variable lost all unmasked historical scaling information')
    masked = dict(batch)
    masked['target'] = batch['target'].clone()
    masked['target'][batch['prediction_mask']] = 0
    masked['prediction_mask'] = batch['prediction_mask'] | selected
    # The module excludes masked patches from its scaler and replaces their embedding.
    output = policy(masked)
    score_batch = dict(masked, prediction_mask=selected)
    experts, truth, weights, _, _ = joints(output, score_batch)
    loss = nll(experts, weights, truth)
    return loss, dict(objective='masked complete-history-patch mixture NLL',
        requested_mask_ratio=.25, actual_mask_ratio=number/int(counts[0]),
        masked_patches_per_input=number, reconstruction_elements_per_input=truth.shape[-1],
        carrier_width=int(batch['observed_mask'].shape[-1]), active_patch_widths=batch['patch_size'].unique().tolist(),
        eligible_history_patches_per_input=int(counts[0]),
        future_labels_used_as_input=False, future_labels_used_in_loss=False,
        policy_version=int(policy_version), mask_varies_by_input_and_update=True,
        nll=float(loss.detach()))
