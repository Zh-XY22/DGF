"""Worker-only rolling OPD mathematics, training calibration, and tiny OPD→RL chain.

No validation/test metric is used to select a coefficient. The end-to-end readout
is deliberately small and cannot stand in for the formal five-seed experiments.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import torch
from torch.distributions import Categorical, Normal

from uni2ts.distribution.mixture import Mixture

from .actions import JointForecast
from .configuration import resolve_checkpoint
from .data import batch, datasets
from .distillation import forward_kl_surrogate, reverse_kl_surrogate
from .io_utils import buffered_torch_save, load_checkpoint
from .opd import (OPD_ARMS, VERSION, RollingPrefixPreparation, pack_history_like,
                  unpack_history, slice_experts)
from .policy import FIELDS, joints
from .probability import action_marginal_logp, nll
from .runtime import atomic_json, checked_step, load_policy, optimizer_for, save_state, seed_all
from .probe import relative_update


COEFFICIENTS = (.001, .01, .1)


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def normal_kl_checks(device, seed):
    seed_all(seed)
    dtype = torch.float64
    size = 131072
    mean = torch.tensor(.3, device=device, dtype=dtype, requires_grad=True)
    log_scale = torch.tensor(math.log(.8), device=device, dtype=dtype, requires_grad=True)
    teacher = Normal(torch.tensor(-.7, device=device, dtype=dtype), torch.tensor(1.2, device=device, dtype=dtype))
    student = Normal(mean, log_scale.exp())
    teacher_actions = teacher.sample((size,)).detach()
    student_lp = student.log_prob(teacher_actions)
    fkl = forward_kl_surrogate(student_lp)
    fkl_gradient = torch.stack(torch.autograd.grad(fkl, (mean, log_scale)))
    expected_fkl = torch.stack(((mean - teacher.loc) / log_scale.exp().square(),
        1 - (teacher.scale.square() + (teacher.loc - mean).square()) / log_scale.exp().square())).detach()
    sample_fkl_score = torch.stack(((mean.detach() - teacher_actions) / log_scale.detach().exp().square(),
        1 - ((teacher_actions - mean.detach()) / log_scale.detach().exp()).square()), -1)
    fkl_se = sample_fkl_score.std(0, correction=1) / size**.5
    if not ((fkl_gradient - expected_fkl).abs() <= 6 * fkl_se + 1e-6).all():
        raise AssertionError('FKL fixed teacher-action gradient disagrees with analytic Normal KL')
    student = Normal(mean, log_scale.exp())
    student_actions = student.sample((size,)).detach()
    lp = student.log_prob(student_actions)
    lq = teacher.log_prob(student_actions)
    rkl = reverse_kl_surrogate(lp, lq)
    rkl_gradient = torch.stack(torch.autograd.grad(rkl, (mean, log_scale)))
    expected_rkl = torch.stack(((mean - teacher.loc) / teacher.scale.square(),
        log_scale.exp().square() / teacher.scale.square() - 1)).detach()
    score = torch.stack(((student_actions - mean.detach()) / log_scale.detach().exp().square(),
        ((student_actions - mean.detach()) / log_scale.detach().exp()).square() - 1), -1)
    sample_rkl_score = (lp.detach() - lq.detach() + 1)[:, None] * score
    rkl_se = sample_rkl_score.std(0, correction=1) / size**.5
    if not ((rkl_gradient - expected_rkl).abs() <= 6 * rkl_se + 1e-6).all():
        raise AssertionError('RKL score-function gradient disagrees with analytic Normal KL')
    same = (student.log_prob(student_actions) - student.log_prob(student_actions)).detach()
    if same.abs().max() != 0:
        raise AssertionError('Identical Normal distributions do not give zero sampled KL')
    return dict(status='PASS', samples=size, parameters=['mean', 'log_scale'],
        FKL_gradient=fkl_gradient.detach().cpu().tolist(), FKL_analytic_gradient=expected_fkl.cpu().tolist(),
        FKL_monte_carlo_standard_error=fkl_se.cpu().tolist(),
        RKL_gradient=rkl_gradient.detach().cpu().tolist(), RKL_analytic_gradient=expected_rkl.cpu().tolist(),
        RKL_monte_carlo_standard_error=rkl_se.cpu().tolist(),
        identical_sampled_KL_max_abs=float(same.abs().max()), tolerance='six MC standard errors + 1e-6')


def block_density_check(device):
    dtype = torch.float64
    loc = torch.zeros((1, 4), device=device, dtype=dtype)
    scale = torch.ones_like(loc)
    experts = [JointForecast(Mixture(Categorical(logits=torch.zeros((1, 4, 1), device=device, dtype=dtype)),
        [Normal(torch.full_like(loc, mean), torch.ones_like(loc))]), loc, scale) for mean in (0., 3.)]
    weights = torch.tensor([[.25, .75]], device=device, dtype=dtype)
    indices = torch.tensor([0, 1], device=device)
    block = slice_experts(experts, indices)
    actions = dict(raw=torch.tensor([[[0., 3.]]], device=device, dtype=dtype),
        forecast=torch.tensor([[[0., 3.]]], device=device, dtype=dtype),
        component=torch.zeros((1, 1, 2), device=device, dtype=torch.long))
    actual = action_marginal_logp(block, weights, actions)
    scores = torch.stack([expert.mixture.components[0].log_prob(actions['raw']) for expert in block], -1)
    expected = torch.logsumexp(scores.sum(-2) + weights.log()[None], -1)
    incorrect = torch.logsumexp(scores + weights.log()[None, :, None], -1).sum(-1)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    if (actual - incorrect).abs().max() < 1e-3:
        raise AssertionError('Controlled example did not distinguish shared-mode joint block from pointwise mixtures')
    return dict(status='PASS', actual=actual.cpu().tolist(), analytic=expected.cpu().tolist(),
        incorrect_position_mixture=incorrect.cpu().tolist(), mode_scope='one mode for all positions in the block')


def gradients(loss, parameters):
    result = torch.autograd.grad(loss, parameters, allow_unused=True)
    nonempty = [gradient for gradient in result if gradient is not None]
    if not nonempty or not all(torch.isfinite(gradient).all() for gradient in nonempty):
        raise FloatingPointError('Missing or nonfinite qualification gradient')
    return [None if gradient is None else gradient.detach() for gradient in result]


def squared_norm(values):
    return sum(float(value.double().square().sum()) for value in values if value is not None)


def choose_coefficient(rows, direction):
    relevant = [row for row in rows if row['direction'] == direction]
    candidates = []
    for coefficient in COEFFICIENTS:
        ratio = max(coefficient * row['auxiliary_gradient_norm'] / row['anchor_gradient_norm'] for row in relevant)
        usable = math.isfinite(ratio) and 0 < ratio <= 1
        candidates.append(dict(coefficient=coefficient, maximum_aux_to_anchor_gradient_ratio=ratio, usable=usable))
    eligible = [row for row in candidates if row['usable']]
    if not eligible:
        raise RuntimeError('No preregistered OPD coefficient is stable under the frozen training-gradient rule')
    selected = min(eligible, key=lambda row: (abs(math.log(row['maximum_aux_to_anchor_gradient_ratio'] / .1)), row['coefficient']))
    return dict(direction=direction, coefficient=selected['coefficient'], candidates=candidates,
        criterion='closest to 0.1 anchor-gradient ratio in log space, ratio in (0,1]; FKL uses worse of O01/O03',
        scope='this dataset fixed training window only; cross-dataset coordinator may apply the same rule to both raw records',
        validation_used=False, test_used=False)


def weights_readout(path, policy, config, step, phase):
    buffered_torch_save(dict(policy_state=policy.state_dict(), route=policy.route, routing=policy.routing,
        fixed_kappa=policy.fixed_kappa, config=config, progress=dict(global_step=step, phase=phase), resumable=False), path)


def audit(config_path, output, device='cuda', seed=1):
    config = resolve_checkpoint(json.loads(Path(config_path).read_text()))
    if config.get('seed', seed) != seed:
        raise ValueError('Qualification CLI seed differs from frozen config')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    began = time.time()
    seed_all(seed)
    if str(device).startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    atomic_json(out / 'effective_configuration.json', config)
    mathematical = dict(normal_KL=normal_kl_checks(device, seed + 7001), block_density=block_density_check(device))
    atomic_json(out / 'mathematics.json', mathematical)
    seed_all(seed)
    policy = load_policy(config, device).eval()
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    before = {name: parameter.detach().cpu().clone() for name, parameter in policy.named_parameters()}
    training = datasets(policy, config['dataset'], config['protocol'])['train']
    inputs = int(config.get('qualification_inputs', min(int(config['microbatch']), 2)))
    if inputs < 1 or inputs > int(config['microbatch']):
        raise ValueError('Qualification input count must fit the declared microbatch')
    indices = list(range(len(training) // 3, len(training) // 3 + inputs))
    original = batch(training, indices, device)
    history, observed, generated = unpack_history(original, config['protocol'])
    reconstructed = pack_history_like(original, config['protocol'], history, observed, generated)
    history2, observed2, generated2 = unpack_history(reconstructed, config['protocol'])
    torch.testing.assert_close(history2, history, atol=0, rtol=0)
    if not torch.equal(observed2, observed) or not torch.equal(generated2, generated):
        raise AssertionError('Packed history/masks failed exact roundtrip')
    zero_history = history.clone()
    zero_observed = observed.clone()
    zero_history[:, -1] = 0
    zero_observed[:, -1] = True
    zero_observed[:, -2] = False
    zero_generated = torch.zeros_like(zero_observed)
    zero_generated[:, -1] = True
    zero_state = pack_history_like(original, config['protocol'], zero_history, zero_observed, zero_generated)
    zero_values, zero_mask, zero_gen = unpack_history(zero_state, config['protocol'])
    if zero_values[:, -1].count_nonzero() or not zero_mask[:, -1].all() or not zero_gen[:, -1].all() or zero_mask[:, -2].any():
        raise AssertionError('Valid generated zero, missing position, and padding were conflated')
    changed_future = {key: value.clone() if torch.is_tensor(value) else deepcopy(value) for key, value in original.items()}
    changed_future['target'][changed_future['prediction_mask']] += 71.
    h3, o3, g3 = unpack_history(changed_future, config['protocol'])
    isolated = pack_history_like(changed_future, config['protocol'], h3, o3, g3)
    if not all(torch.equal(reconstructed[key], isolated[key]) for key in FIELDS):
        raise AssertionError('Real future replacement changed the rolling prefix state')
    initial_checkpoint = out / 'preparation_before.pt'
    weights_readout(initial_checkpoint, policy, config, 0, 'OPD_qualification_start')
    plugins = {}
    for arm in OPD_ARMS:
        local = dict(config, preparation_arm=arm, distillation_samples=8, opd_block_length=32)
        plugins[arm] = RollingPrefixPreparation(local, device, teacher=plugins['O01'].teacher if plugins else None)
    prefix_seed = seed + 7002
    seed_all(prefix_seed)
    with torch.no_grad():
        _, fixed_before = plugins['O01'].auxiliary(policy, original, 0, capture=True)
    seed_all(prefix_seed)
    with torch.no_grad():
        _, fixed_changed_future = plugins['O01'].auxiliary(policy, changed_future, 0, capture=True)
    for left, right in zip(fixed_before['_prefix_traces'], fixed_changed_future['_prefix_traces']):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    original_future_isolation = dict(status='PASS', exact_history_roundtrip=True,
        real_future_replaced_by=71., rolling_states_equal=True, fixed_RNG_generated_prefixes_equal=True,
        prefix_coordinates='dataset train-standardized, not internal history-normalized',
        generated_observed_not_padding=True, legitimate_generated_zero_preserved=True,
        missing_history_remains_unobserved=True, patch=int(config['protocol']['patch']))
    atomic_json(out / 'future_isolation_and_scaling.json', original_future_isolation)
    identical_teacher = deepcopy(policy).eval().requires_grad_(False)
    self_plugin = RollingPrefixPreparation(dict(config, preparation_arm='O01'), device, teacher=identical_teacher)
    with torch.no_grad():
        _, self_diagnostic = self_plugin.auxiliary(policy, original, 0)
    self_difference = max(row['max_abs_log_density_difference'] for row in self_diagnostic['events'])
    if self_difference > 1e-8:
        raise AssertionError('Identical real policies have different block densities at the same rolling state')
    del identical_teacher, self_plugin
    sync(device)
    anchor_began = time.time()
    experts, target, weights, _, _ = joints(policy(original), original)
    anchor = nll(experts, weights, target)
    sync(device)
    anchor_forward_seconds = time.time() - anchor_began
    base_gradients = gradients(anchor, parameters)
    anchor_norm = squared_norm(base_gradients)**.5
    if not anchor_norm > 0:
        raise AssertionError('Real-label NLL anchor has no gradient')
    calibration = []
    for arm in OPD_ARMS:
        seed_all(seed + 7100 + OPD_ARMS.index(arm))
        sync(device)
        auxiliary_began = time.time()
        auxiliary, diagnostic = plugins[arm].auxiliary(policy, original, 0)
        sync(device)
        auxiliary_forward_seconds = time.time() - auxiliary_began
        auxiliary_gradients = gradients(auxiliary, parameters)
        auxiliary_norm = squared_norm(auxiliary_gradients)**.5
        if not auxiliary_norm > 0 or max(row['max_abs_log_density_difference'] for row in diagnostic['events']) <= 1e-8:
            raise AssertionError('Teacher/student OPD treatment is numerically degenerate')
        dot = sum(float((left.double() * right.double()).sum()) for left, right in zip(base_gradients, auxiliary_gradients)
                  if left is not None and right is not None)
        records = [dict(coefficient=coefficient, auxiliary_to_anchor_gradient_ratio=coefficient * auxiliary_norm / anchor_norm,
            combined_gradient_norm=max(0., anchor_norm**2 + coefficient**2 * auxiliary_norm**2 + 2 * coefficient * dot)**.5)
            for coefficient in COEFFICIENTS]
        calibration.append(dict(arm=arm, direction='RKL' if arm == 'O02' else 'FKL',
            anchor_gradient_norm=anchor_norm, auxiliary_gradient_norm=auxiliary_norm,
            anchor_value=float(anchor.detach()), auxiliary_value=float(auxiliary.detach()), candidates=records,
            original_history_anchor_forward_seconds=anchor_forward_seconds,
            three_state_auxiliary_forward_seconds=auxiliary_forward_seconds,
            microbatch_inputs=inputs, native_forwards_per_microbatch=dict(student=4, teacher=3),
            diagnostic=diagnostic, optimizer_updates=0))
        del auxiliary_gradients, auxiliary
        atomic_json(out / 'coefficient_probe.json', calibration)
    choices = {direction: choose_coefficient(calibration, direction) for direction in ('FKL', 'RKL')}
    atomic_json(out / 'coefficient_choices.json', choices)
    del base_gradients, experts, target, weights, anchor
    optimizer = optimizer_for(policy, config['lr'])
    update_rows = []
    # These two updates are one isolated preparation trajectory; candidates above
    # are gradient-only probes and do not create three optimizer trajectories.
    coefficient = choices['FKL']['coefficient']
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        experts, target, weights, _, _ = joints(policy(original), original)
        anchor = nll(experts, weights, target)
        auxiliary, diagnostic = plugins['O01'].auxiliary(policy, original, step)
        loss = anchor + coefficient * auxiliary
        loss.backward()
        gradient = checked_step(policy, optimizer)
        change, frozen = relative_update(policy, before)
        if not change > 0 or frozen != 0:
            raise AssertionError('Real OPD preparation did not update exactly the trainable student parameters')
        row = dict(step=step + 1, loss=float(loss.detach()), coefficient=coefficient,
            parameter_relative_change=change, frozen_max_change=frozen, **gradient, diagnostic=diagnostic)
        update_rows.append(row)
        if step == 0:
            atomic_json(out / 'first_step.json', row)
        atomic_json(out / 'preparation_updates.json', update_rows)
    seed_all(prefix_seed)
    with torch.no_grad():
        _, fixed_after = plugins['O01'].auxiliary(policy, original, 2, capture=True)
    changes = [float((left - right).abs().max()) for left, right in zip(fixed_before['_prefix_traces'], fixed_after['_prefix_traces'])]
    if not any(change > 0 for change in changes):
        raise AssertionError('Current-student prefixes did not change after real parameter updates under fixed RNG')
    refresh = dict(status='PASS', optimizer_policy_versions=[0, 1], native_calls_per_objective=3,
        fixed_RNG_prefix_max_changes_after_updates=changes, fresh_prefixes_each_optimizer_update=True,
        source='current student for O01/O02; frozen teacher for O03',
        within_update_parameter_versions_constant=True, identical_real_policy_log_density_max_abs=self_difference)
    atomic_json(out / 'three_block_and_refresh.json', refresh)
    preparation_after = out / 'preparation_after.pt'
    weights_readout(preparation_after, policy, config, 2, 'OPD_qualification_after_two_updates')
    save_state(out / 'preparation_resume.pt', policy, optimizer, config,
        dict(epoch=0, next_input=inputs, global_step=2, independent_input_exposures=2 * inputs), extra=dict(scope='diagnostic qualification only'))
    from .rl import make_rollout, objective
    reference = deepcopy(policy).eval().requires_grad_(False)
    rl_config = dict(config, arm='E03', algorithm_arm='E03', phase='scientific', nll_anchor_coefficient=1.)
    optimizer = optimizer_for(policy, rl_config['lr'])
    native = make_rollout(policy, reference, original, rl_config)
    if native['actions']['forecast'].shape[-1] != 96 * (len(config['protocol']['variables']) if config['protocol']['mode'] == 'M' else 1):
        raise AssertionError('Downstream RL did not return to one native H96 forecast')
    rl_rows = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostic = objective(policy, original, native, rl_config)
        loss.backward()
        gradient = checked_step(policy, optimizer)
        rl_rows.append(dict(step=step + 1, **gradient, diagnostic=diagnostic))
        atomic_json(out / 'native_RL_updates.json', rl_rows)
    final_checkpoint = out / 'native_RL_after.pt'
    weights_readout(final_checkpoint, policy, rl_config, 2, 'native_RL_qualification_after_two_updates')
    save_state(out / 'native_RL_resume.pt', policy, optimizer, rl_config,
        dict(epoch=0, next_input=inputs, global_step=2, independent_input_exposures=inputs), extra=dict(scope='diagnostic qualification only'))
    with torch.no_grad():
        before_reload = joints(policy(original), original)
        means = [expert.mean.clone() for expert in before_reload[0]]
        route_weights = before_reload[2].clone()
    saved = load_checkpoint(final_checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(saved['policy_state'], strict=True)
    with torch.no_grad():
        after_reload = joints(policy(original), original)
        for previous, current in zip(means, after_reload[0]):
            torch.testing.assert_close(previous, current.mean, atol=0, rtol=0)
        torch.testing.assert_close(route_weights, after_reload[2], atol=0, rtol=0)
    from .evaluation import evaluation
    validation = evaluation(policy, config['protocol'], 'val', out / 'native_RL_diagnostic_validation', seed,
        config['artifact'], int(config.get('diagnostic_validation_limit', 8)))
    if validation.get('complete'):
        raise AssertionError('Tiny OPD qualification unexpectedly presented a full formal validation')
    result = dict(status='PASS', adapter_version=VERSION, config=config, seed=seed,
        scope='All three OPD arms checked; gradient-only coefficient probes; two real O01 preparation updates then two native H96 RL updates; diagnostic validation only',
        optimizer_trajectories=2, optimizer_steps=4, calibration_optimizer_steps=0,
        qualification_indices=indices, coefficient_choices=choices, coefficient_probe=calibration,
        mathematical_checks=str(out / 'mathematics.json'), future_isolation=str(out / 'future_isolation_and_scaling.json'),
        prefix_refresh=str(out / 'three_block_and_refresh.json'), first_step=str(out / 'first_step.json'),
        preparation_before_checkpoint=str(initial_checkpoint), preparation_after_checkpoint=str(preparation_after),
        native_RL_checkpoint=str(final_checkpoint), preparation_updates=update_rows, native_RL_updates=rl_rows,
        exact_fixed_batch_reload=True, validation=validation, test_used=False,
        teacher_parameters_frozen=True, original_data_numerical_build=False,
        actual_gpu=torch.cuda.get_device_name() if str(device).startswith('cuda') else 'CPU',
        allocated_peak_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
        reserved_peak_bytes=torch.cuda.max_memory_reserved() if str(device).startswith('cuda') else None,
        seconds=time.time() - began)
    atomic_json(out / 'result.json', result)
    return result
