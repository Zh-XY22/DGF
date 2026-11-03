"""Three native H96 calls with detached, generated 32-step prefix states.

This is a forecast-specific rolling adapter, not an autoregressive Moirai rewrite.
Each block has one latent mode; native distribution families remain position-wise.
All returned forecasts are in the dataset's train-standardized coordinates, after
undoing the current call's internal history scaler. Only those forecasts, never
internal ``raw`` samples, are appended to the next history.
"""
from copy import deepcopy

import torch
from torch.distributions import Categorical, LogNormal, Normal, StudentT

from uni2ts.distribution.mixture import Mixture
from uni2ts.distribution.negative_binomial import NegativeBinomial

from .actions import JointForecast
from .data import assert_complete_future, forecast_dimensions
from .distillation import forward_kl_surrogate, reverse_kl_surrogate
from .policy import FIELDS, joints, rollout
from .probability import action_marginal_logp, common_scaler
from .runtime import load_policy


OPD_ARMS = ('O01', 'O02', 'O03')
VERSION = 'rolling-prefix-3x32-detached-state-block-marginal-v1'
BLOCK_LENGTH = 32
BLOCKS = 3


def packed_layout(batch, protocol):
    """Reject layouts this finite ETT adapter cannot faithfully reconstruct."""
    context, horizon, patch = (int(protocol[k]) for k in ('context', 'horizon', 'patch'))
    dimensions = forecast_dimensions(protocol)
    if horizon != BLOCK_LENGTH * BLOCKS:
        raise ValueError('Rolling OPD is preregistered only for H96 / three 32-step blocks')
    extra = {key for key in batch if key not in FIELDS and not key.startswith('_')}
    if extra:
        raise ValueError('Rolling covariates need an explicit adapter: ' + repr(sorted(extra)))
    target = batch['target']
    history_tokens = (context + patch - 1) // patch
    future_tokens = (horizon + patch - 1) // patch
    tokens = history_tokens + future_tokens
    if target.ndim != 3 or target.shape[1] != dimensions * tokens or target.shape[2] < patch:
        raise ValueError('Unsupported packed target layout')
    n = target.shape[0]
    expected_var = torch.arange(dimensions, device=target.device).repeat_interleave(tokens)
    expected_time = torch.arange(tokens, device=target.device).repeat(dimensions)
    expected_prediction = (expected_time >= history_tokens).expand(n, -1)
    for key, expected in [('variate_id', expected_var), ('time_id', expected_time)]:
        if not torch.equal(batch[key], expected.expand(n, -1)):
            raise ValueError('Noncanonical native token ordering: ' + key)
    if not torch.equal(batch['prediction_mask'], expected_prediction):
        raise ValueError('Prediction mask does not mark the complete native future')
    if not (batch['sample_id'] == 1).all() or not (batch['patch_size'] == patch).all():
        raise ValueError('Rolling adapter requires one unpacked sample and one fixed patch size')
    if batch['observed_mask'][..., patch:].any():
        raise ValueError('Padded feature slots must not be observed')
    assert_complete_future(batch, horizon * dimensions)
    return dict(n=n, dimensions=dimensions, context=context, horizon=horizon,
                patch=patch, history_tokens=history_tokens, future_tokens=future_tokens,
                tokens=tokens, feature_slots=target.shape[-1])


def unpack_history(batch, protocol):
    layout = packed_layout(batch, protocol)
    n, d, t, p = (layout[k] for k in ('n', 'dimensions', 'tokens', 'patch'))
    history_width = layout['history_tokens'] * p
    def unpack(value):
        flat = value.reshape(n, d, t, -1)[..., :p].reshape(n, d, t * p)
        return flat[..., :history_width][..., -layout['context']:].transpose(1, 2).clone()
    history = unpack(batch['target']).detach()
    observed = unpack(batch['observed_mask']).bool().detach()
    generated = unpack(batch.get('_opd_generated_mask', torch.zeros_like(batch['observed_mask']))).bool().detach()
    if not torch.isfinite(history).all():
        raise FloatingPointError('Nonfinite observed history')
    return history, observed, generated


def pack_history_like(template, protocol, history, observed, generated=None, advance=0):
    """Repatch after a 32-step shift; shifting whole tokens would be incorrect."""
    layout = packed_layout(template, protocol)
    n, d, t, p, c = (layout[k] for k in ('n', 'dimensions', 'tokens', 'patch', 'context'))
    if history.shape != (n, c, d) or observed.shape != history.shape:
        raise ValueError('Rolling history shape differs from the frozen context')
    if generated is None:
        generated = torch.zeros_like(observed, dtype=torch.bool)
    if generated.shape != history.shape or (generated & ~observed).any():
        raise ValueError('Generated values must be finite model-visible history')
    if not torch.isfinite(history).all():
        raise FloatingPointError('Nonfinite generated history; no clipping or substitution is allowed')
    result = {key: value.detach().clone() if torch.is_tensor(value) else deepcopy(value)
              for key, value in template.items() if key != '_opd_generated_mask'}
    target = torch.zeros_like(template['target']).reshape(n, d, t, -1)
    obs = torch.zeros_like(template['observed_mask']).reshape(n, d, t, -1)
    gen = torch.zeros_like(template['observed_mask']).reshape(n, d, t, -1)
    width = layout['history_tokens'] * p
    left = width - c
    values = history.transpose(1, 2).detach()
    hist_mask = observed.transpose(1, 2).bool().detach()
    generated_mask = generated.transpose(1, 2).bool().detach()
    padded = history.new_zeros((n, d, width))
    padded_observed = torch.zeros((n, d, width), device=history.device, dtype=torch.bool)
    padded_generated = torch.zeros_like(padded_observed)
    padded[..., left:] = torch.where(hist_mask, values, torch.zeros_like(values))
    padded_observed[..., left:] = hist_mask
    padded_generated[..., left:] = generated_mask
    target[:, :, :layout['history_tokens'], :p] = padded.reshape(n, d, -1, p)
    obs[:, :, :layout['history_tokens'], :p] = padded_observed.reshape(n, d, -1, p)
    gen[:, :, :layout['history_tokens'], :p] = padded_generated.reshape(n, d, -1, p)
    future = torch.arange(layout['future_tokens'] * p, device=history.device) < layout['horizon']
    obs[:, :, layout['history_tokens']:, :p] = future.reshape(1, 1, -1, p)
    result['target'] = target.reshape_as(template['target'])
    result['observed_mask'] = obs.reshape_as(template['observed_mask'])
    result['_opd_generated_mask'] = gen.reshape_as(template['observed_mask'])
    result['_opd_advance'] = int(advance)
    # Relative native patch times restart for the cropped window, as in official transforms.
    packed_layout(result, protocol)
    if result['target'][result['prediction_mask']].count_nonzero():
        raise RuntimeError('Real future labels entered the generated-prefix state')
    return result


def block_indices(protocol, device):
    d = forecast_dimensions(protocol)
    return (torch.arange(d, device=device)[:, None] * int(protocol['horizon']) +
            torch.arange(BLOCK_LENGTH, device=device)[None]).flatten()


def slice_experts(experts, indices):
    """Exact block marginal conditional on a trajectory mode, with family mixing intact."""
    result = []
    for expert in experts:
        components = []
        for component in expert.mixture.components:
            if isinstance(component, StudentT):
                parameters = {key: getattr(component, key).index_select(-1, indices) for key in ('df', 'loc', 'scale')}
            elif isinstance(component, (Normal, LogNormal)):
                parameters = {key: getattr(component, key).index_select(-1, indices) for key in ('loc', 'scale')}
            elif isinstance(component, NegativeBinomial):
                parameters = {key: getattr(component, key).index_select(-1, indices) for key in ('total_count', 'logits')}
            else:
                raise TypeError('Unqualified native OPD distribution family: ' + type(component).__name__)
            components.append(type(component)(**parameters, validate_args=False))
        mixture = Mixture(Categorical(logits=expert.mixture.weights.logits.index_select(-2, indices)), components)
        result.append(JointForecast(mixture, expert.loc.index_select(-1, indices), expert.scale.index_select(-1, indices)))
    return result


def marginal_samples(experts, weights, samples):
    with torch.no_grad():
        actions = rollout(experts, weights, None, 1, samples)
    return {key: actions[key][:, 0].permute(1, 0, 2).detach() for key in ('raw', 'forecast', 'component')}


class RollingPrefixPreparation:
    def __init__(self, config, device, teacher=None):
        self.config = config
        self.protocol = config['protocol']
        self.arm = config.get('preparation_arm', config.get('arm'))
        if self.arm not in OPD_ARMS:
            raise ValueError('Unknown rolling prefix treatment')
        self.samples = int(config.get('distillation_samples', 8))
        if self.samples != 8 or int(config.get('opd_block_length', 32)) != 32:
            raise ValueError('This finite OPD version freezes eight MC samples and 32-step blocks')
        self.teacher_config = config['teacher_config']
        for key in ('context', 'patch', 'horizon', 'mode', 'variables', 'train_end', 'val_end', 'test_end', 'ddof', 'scaler_mean', 'scaler_std_ddof1'):
            if self.teacher_config['protocol'].get(key) != self.protocol.get(key):
                raise ValueError('Teacher protocol differs: ' + key)
        self.teacher = (teacher if teacher is not None else load_policy(self.teacher_config, device)).eval().requires_grad_(False)
        self.teacher_versions = {name: parameter._version for name, parameter in self.teacher.named_parameters()}
        self.calls = 0

    def auxiliary(self, student_policy, original_batch, policy_version, capture=False):
        if student_policy is None or policy_version is None:
            raise ValueError('Rolling OPD requires the current student and explicit optimizer policy version')
        self.calls += 1
        history, observed, generated = unpack_history(original_batch, self.protocol)
        state = pack_history_like(original_batch, self.protocol, history, observed, generated, 0)
        dimensions = forecast_dimensions(self.protocol)
        indices = block_indices(self.protocol, history.device)
        losses, events, traces = [], [], []
        student_versions = {name: parameter._version for name, parameter in student_policy.named_parameters()}
        for block in range(BLOCKS):
            teacher_input = {key: state[key].detach().clone() for key in FIELDS}
            student_full, _, student_w, _, _ = joints(student_policy(state), state)
            with torch.no_grad():
                teacher_full, _, teacher_w, _, _ = joints(self.teacher(state), state)
            if not all(torch.equal(state[key], teacher_input[key]) for key in FIELDS):
                raise RuntimeError('Teacher/student did not consume the same immutable rolling state')
            student = slice_experts(student_full, indices)
            teacher = slice_experts(teacher_full, indices)
            loc, scale = common_scaler(student)
            teacher_loc, teacher_scale = common_scaler(teacher)
            if not torch.equal(loc, teacher_loc) or not torch.equal(scale, teacher_scale):
                raise ValueError('Teacher/student atomic lattices or scale coordinates differ at rolling state')
            if [type(c) for c in student[0].mixture.components] != [type(c) for c in teacher[0].mixture.components]:
                raise ValueError('Teacher/student native family supports differ')
            if not all(torch.isfinite(expert.mean).all() for expert in student + teacher):
                raise FloatingPointError('Nonfinite legal block mean; do not substitute location or truncate')
            actions = marginal_samples(student, student_w, self.samples) if self.arm == 'O02' else marginal_samples(teacher, teacher_w, self.samples)
            lp = action_marginal_logp(student, student_w, actions)
            with torch.no_grad():
                lq = action_marginal_logp(teacher, teacher_w, actions)
            if self.arm == 'O02':
                difference = (lp - lq).detach()
                baseline = (difference.sum(0, keepdim=True) - difference) / (self.samples - 1)
                loss = reverse_kl_surrogate(lp, lq, baseline) / (BLOCK_LENGTH * dimensions)
                estimate = difference / (BLOCK_LENGTH * dimensions)
            else:
                loss = forward_kl_surrogate(lp) / (BLOCK_LENGTH * dimensions)
                estimate = (lq - lp.detach()) / (BLOCK_LENGTH * dimensions)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite rolling-prefix distillation training loss')
            losses.append(loss)
            prefix_experts, prefix_weights = (teacher, teacher_w) if self.arm == 'O03' else (student, student_w)
            prefix = marginal_samples(prefix_experts, prefix_weights, 1)['forecast'][0]
            prefix = prefix.reshape(len(history), dimensions, BLOCK_LENGTH).transpose(1, 2).detach()
            if prefix.requires_grad or not torch.isfinite(prefix).all():
                raise FloatingPointError('Generated prefix must be detached and finite')
            history = torch.cat((history, prefix), 1)[:, -int(self.protocol['context']):].detach()
            observed = torch.cat((observed, torch.ones_like(prefix, dtype=torch.bool)), 1)[:, -int(self.protocol['context']):]
            generated = torch.cat((generated, torch.ones_like(prefix, dtype=torch.bool)), 1)[:, -int(self.protocol['context']):]
            next_state = pack_history_like(original_batch, self.protocol, history, observed, generated, (block + 1) * BLOCK_LENGTH)
            reconstructed, reconstructed_observed, reconstructed_generated = unpack_history(next_state, self.protocol)
            if not torch.equal(reconstructed[:, -BLOCK_LENGTH:], prefix):
                raise RuntimeError('Generated prefix was scaled or reordered during history append')
            if not reconstructed_observed[:, -BLOCK_LENGTH:].all() or not reconstructed_generated[:, -BLOCK_LENGTH:].all():
                raise RuntimeError('Generated prefix was ignored as padding or missing history')
            events.append(dict(block=block, native_horizon=96, used_block_length=BLOCK_LENGTH,
                forecast_origin_advance=block * BLOCK_LENGTH, next_history_advance=(block + 1) * BLOCK_LENGTH,
                policy_version=int(policy_version), prefix_refresh_call=self.calls,
                prefix_source='frozen_teacher' if self.arm == 'O03' else 'current_student',
                teacher_student_same_state=True, real_future_input_nonzero=0,
                block_positions=int(indices.numel()), mc_samples=self.samples,
                mc_KL_per_position=float(estimate.mean()),
                mc_standard_error_across_actions=float(estimate.std(0, correction=1).mean() / self.samples**.5),
                max_abs_log_density_difference=float((lp.detach() - lq).abs().max()),
                prefix_max_abs=float(prefix.abs().max()), prefix_mean=float(prefix.mean()),
                mean_internal_loc=float(loc.mean()), mean_internal_scale=float(scale.mean()),
                generated_history_positions_per_input=int(generated[0].sum()),
                generated_prefix_visible=True, prefix_coordinate='training-scaler dataset coordinate; native loc/scale already undone',
                probability='sum block positions conditional on mode, then logsumexp modes; native families marginalized',
                prefix_state_gradient='detached; no gradient through state distribution'))
            if capture:
                traces.append(prefix.detach().cpu().clone())
            state = next_state
        if not all(parameter._version == student_versions[name] for name, parameter in student_policy.named_parameters()):
            raise RuntimeError('Student parameters changed inside a single OPD objective evaluation')
        if not all(not parameter.requires_grad and parameter.grad is None and parameter._version == self.teacher_versions[name]
                   for name, parameter in self.teacher.named_parameters()):
            raise RuntimeError('Frozen teacher parameters or gradients changed')
        loss = torch.stack(losses).mean()
        diagnostic = dict(adapter_version=VERSION, preparation_arm=self.arm, native_calls=BLOCKS,
            block_length=BLOCK_LENGTH, policy_version=int(policy_version), prefix_refresh_call=self.calls,
            prefix_source='frozen_teacher' if self.arm == 'O03' else 'current_student',
            distillation_direction='RKL_student_to_teacher' if self.arm == 'O02' else 'FKL_teacher_to_student',
            training_loss=float(loss.detach()), block_reduction='mean of three block losses; each normalized by 32*variables',
            estimator='detached student action, score-function +1 and action-independent LOO baseline' if self.arm == 'O02' else 'teacher-action cross entropy',
            teacher_checkpoint=self.teacher_config.get('checkpoint'), teacher_frozen=True,
            rollout_reused_across_optimizer_updates=False, events=events)
        if capture:
            diagnostic['_prefix_traces'] = traces
        return loss, diagnostic
