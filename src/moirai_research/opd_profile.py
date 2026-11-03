"""Worker-only, fixed-coefficient full-global-batch rolling OPD performance profile.

Each arm restarts the same complete policy and a fresh optimizer/RNG stream.
There is no validation/test execution or coefficient selection in this entry.
The first of two real updates includes warmup; synchronization overhead is kept
in the reported wall time rather than extrapolating tiny-input qualification.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch

from .configuration import resolve_checkpoint
from .data import batch, datasets
from .opd import OPD_ARMS, VERSION, RollingPrefixPreparation
from .policy import joints
from .probability import nll
from .probe import relative_update
from .runtime import atomic_json, checked_step, load_policy, optimizer_for, save_state, seed_all


GLOBAL_BATCH = 512
UPDATES = 2


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())


def restore_rng(value):
    random.setstate(value[0]); np.random.set_state(value[1])
    torch.set_rng_state(value[2]); torch.cuda.set_rng_state_all(value[3])


def percentile(values, quantile):
    ordered = sorted(values)
    where = (len(ordered)-1)*quantile
    lower = math.floor(where); upper = math.ceil(where)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(where-lower)


def summary(values):
    return dict(count=len(values), total=sum(values), mean=statistics.mean(values),
        minimum=min(values), p50=percentile(values, .5), p95=percentile(values, .95), maximum=max(values),
        quantile_rule='linear interpolation of observed points; only two full batches per arm')


class ForwardTiming:
    """Top-level Policy forward wall times; no tensor or probability modification."""
    def __init__(self, policy, teacher, device):
        self.device = device
        self.rows = []
        self.started = {}
        self.handles = []
        for role, module in [('student', policy), ('teacher', teacher)]:
            def before(_module, _arguments, name=role):
                synchronize(self.device)
                self.started[name] = time.perf_counter()
            def after(_module, _arguments, _output, name=role):
                synchronize(self.device)
                self.rows.append(dict(role=name, seconds=time.perf_counter()-self.started.pop(name)))
            self.handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])

    def clear(self):
        self.rows = []; self.started = {}

    def close(self):
        for handle in self.handles:
            handle.remove()


def micro_update(policy, plugin, training, indices, config, coefficient, version, timing, device):
    timing.clear()
    synchronize(device); began = time.perf_counter()
    packed = batch(training, indices, device)
    synchronize(device); loaded = time.perf_counter()
    experts, target, weights, _, _ = joints(policy(packed), packed)
    anchor = nll(experts, weights, target)
    auxiliary, diagnostic = plugin.auxiliary(policy, packed, version)
    loss = anchor+coefficient*auxiliary
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite OPD profile training loss, not a performance-only diagnostic failure')
    synchronize(device); forward_end = time.perf_counter()
    (loss*len(indices)/GLOBAL_BATCH).backward()
    synchronize(device); backward_end = time.perf_counter()
    student = [row['seconds'] for row in timing.rows if row['role'] == 'student']
    teacher = [row['seconds'] for row in timing.rows if row['role'] == 'teacher']
    if len(student) != 4 or len(teacher) != 3 or diagnostic['native_calls'] != 3:
        raise AssertionError('OPD profile did not execute one anchor plus three student/teacher rolling calls')
    if len(diagnostic['events']) != 3 or [row['forecast_origin_advance'] for row in diagnostic['events']] != [0, 32, 64]:
        raise AssertionError('OPD profile lost a rolling state')
    return dict(inputs=len(indices), policy_version=version, loss=float(loss.detach()),
        anchor_nll=float(anchor.detach()), auxiliary_surrogate=float(auxiliary.detach()),
        load_seconds=loaded-began, forward_seconds=forward_end-loaded,
        backward_seconds=backward_end-forward_end, micro_wall_seconds=backward_end-began,
        student_forward_seconds=sum(student), teacher_forward_seconds=sum(teacher),
        other_forward_seconds=(forward_end-loaded)-sum(student)-sum(teacher),
        maximum_block_log_density_difference=max(row['max_abs_log_density_difference'] for row in diagnostic['events']),
        native_forward_calls=dict(student=len(student), teacher=len(teacher)),
        role_forward_intervals=deepcopy(timing.rows), diagnostic=diagnostic)


def fixed_coefficients(config):
    receipt = json.loads(Path(config['coefficient_freeze_receipt']).read_text())
    if receipt.get('status') != 'PASS' or receipt.get('test_used') is not False:
        raise ValueError('A successful training-only OPD qualification receipt must freeze coefficients')
    if receipt.get('config', {}).get('dataset') != config['dataset']:
        raise ValueError('Coefficient qualification dataset differs')
    supplied = config['frozen_coefficients']
    if set(supplied) != {'FKL', 'RKL'}:
        raise ValueError('Both frozen KL directions are required')
    for direction in supplied:
        recorded = receipt['coefficient_choices'][direction]['coefficient']
        if supplied[direction] != recorded or supplied[direction] not in (.001, .01, .1):
            raise ValueError('The performance profile cannot recalibrate or replace a qualified coefficient')
    return supplied


def profile(config_path, output, device='cuda', seed=1):
    config = resolve_checkpoint(json.loads(Path(config_path).read_text()))
    if config.get('performance_profile_only') is not True or seed != 1 or config.get('seed', seed) != seed:
        raise ValueError('This finite profile requires explicit performance-only scope and fixed seed1')
    if config.get('route') != 'D-pre' or config.get('global_input_batch') != GLOBAL_BATCH:
        raise ValueError('OPD performance uses the D-pre carrier and global independent input batch512')
    if config.get('profile_optimizer_updates_per_arm') != UPDATES or config.get('distillation_samples') != 8:
        raise ValueError('The registered profile has exactly two updates per arm and eight MC block samples')
    if config.get('nll_anchor_coefficient') != 1.:
        raise ValueError('OPD preparation keeps the same unit true-label NLL anchor in every arm')
    declared_micro = 16 if config['dataset'] == 'ETTh1' else 64 if config['dataset'] == 'ETTm1' else None
    if config.get('microbatch') != declared_micro or declared_micro is None or config['protocol']['horizon'] != 96:
        raise ValueError('Registered profiles start at H1 micro16 or M1 micro64 without changing H96')
    coefficients = fixed_coefficients(config)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'effective_configuration.json', config)
    began = time.perf_counter(); seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    policy = load_policy(config, device).eval()
    teacher = load_policy(config['teacher_config'], device).eval().requires_grad_(False)
    teacher_versions = {name: parameter._version for name, parameter in teacher.named_parameters()}
    initial = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
    initial_parameters = {name: initial[name] for name, _ in policy.named_parameters()}
    training = datasets(policy, config['dataset'], config['protocol'])['train']
    if len(training) < UPDATES*GLOBAL_BATCH:
        raise ValueError('Training set cannot provide the two distinct complete profiling batches')
    indices = torch.randperm(len(training), generator=torch.Generator().manual_seed(seed))[:UPDATES*GLOBAL_BATCH].tolist()
    atomic_json(out/'training_input_sequence.json', dict(seed=seed, indices=indices,
        split='train', rule='First1024 entries of fixed seed1 epoch0 native training order', unique_inputs=len(set(indices))))
    results = []
    completed_updates = 0
    for arm in OPD_ARMS:
        policy.load_state_dict(initial, strict=True)
        current_state=policy.state_dict()
        if not all(torch.equal(current_state[name].detach().cpu(), value) for name, value in initial.items()):
            raise AssertionError('A profile arm did not restart the exact shared policy state')
        del current_state
        seed_all(seed)
        optimizer = optimizer_for(policy, config['lr'])
        if optimizer.state:
            raise AssertionError('Profile arm reused an optimizer state from another arm')
        local = dict(config, arm=arm, preparation_arm=arm, phase='preparation',
            preparation=arm, qualification_only=True, diagnostic_input_limit=UPDATES*GLOBAL_BATCH)
        plugin = RollingPrefixPreparation(local, device, teacher=teacher)
        coefficient = coefficients['RKL' if arm == 'O02' else 'FKL']
        timing = ForwardTiming(policy, teacher, device)
        directory = out/arm; directory.mkdir(exist_ok=True)
        micro = declared_micro; step_rows = []; micro_rows = []; oom_rows = []
        arm_began = time.perf_counter()
        try:
            for update in range(UPDATES):
                chosen = indices[update*GLOBAL_BATCH:(update+1)*GLOBAL_BATCH]
                batch_rng = rng_state()
                while True:
                    restore_rng(batch_rng); optimizer.zero_grad(set_to_none=True)
                    torch.cuda.reset_peak_memory_stats()
                    synchronize(device); step_began = time.perf_counter()
                    current = []
                    oom = None
                    try:
                        for start in range(0, GLOBAL_BATCH, micro):
                            row = micro_update(policy, plugin, training, chosen[start:start+micro],
                                local, coefficient, update, timing, device)
                            row.update(arm=arm, update=update+1, microbatch=micro, input_offset=start)
                            current.append(row)
                    except torch.cuda.OutOfMemoryError as error:
                        oom = str(error).splitlines()[0][:600]
                    if oom is None:
                        break
                    oom_rows.append(dict(arm=arm, update=update+1, microbatch=micro,
                        global_input_batch=GLOBAL_BATCH, completed_microbatches=len(current),
                        elapsed_seconds=time.perf_counter()-step_began, reason=oom,
                        allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                        reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                        optimizer_step_performed=False, action='clear partial gradients, restore batch RNG, halve microbatch'))
                    atomic_json(directory/'oom_recovery.json', oom_rows)
                    optimizer.zero_grad(set_to_none=True); current.clear(); timing.clear()
                    torch.cuda.empty_cache()
                    if micro == 1:
                        raise RuntimeError('OPD profile cannot fit one complete input without changing the scientific protocol')
                    micro = max(1, micro//2)
                synchronize(device); step_update_began = time.perf_counter()
                # Optimizer-step OOM is not silently retried: a partial step cannot
                # be assumed untouched without loading a complete saved state.
                gradient = checked_step(policy, optimizer)
                synchronize(device); step_end = time.perf_counter()
                completed_updates += 1
                diagnostic_began = time.perf_counter()
                change, frozen = relative_update(policy, initial_parameters)
                if not change > 0 or frozen != 0:
                    raise AssertionError('Profile student update missing or frozen parameter changed')
                if not all(not parameter.requires_grad and parameter.grad is None and parameter._version == teacher_versions[name]
                           for name, parameter in teacher.named_parameters()):
                    raise AssertionError('Profile changed the frozen teacher')
                synchronize(device); diagnostic_end = time.perf_counter()
                forward = sum(row['forward_seconds'] for row in current)
                backward = sum(row['backward_seconds'] for row in current)
                teacher_seconds = sum(row['teacher_forward_seconds'] for row in current)
                student_seconds = sum(row['student_forward_seconds'] for row in current)
                row = dict(arm=arm, update=update+1, policy_version_before=update, policy_version_after=update+1,
                    independent_inputs=GLOBAL_BATCH, microbatch=micro, accumulation_microbatches=len(current),
                    weighted_loss=sum(value['loss']*value['inputs']/GLOBAL_BATCH for value in current),
                    load_seconds=sum(value['load_seconds'] for value in current),
                    forward_seconds=forward, backward_seconds=backward,
                    student_forward_seconds=student_seconds, teacher_forward_seconds=teacher_seconds,
                    other_forward_seconds=forward-student_seconds-teacher_seconds,
                    optimizer_step_seconds=step_end-step_update_began,
                    full_batch_seconds=step_end-step_began,
                    diagnostic_seconds=diagnostic_end-diagnostic_began,
                    independent_input_throughput=GLOBAL_BATCH/(step_end-step_began),
                    native_forward_calls=dict(student=sum(value['native_forward_calls']['student'] for value in current),
                        teacher=sum(value['native_forward_calls']['teacher'] for value in current)),
                    allocated_peak_bytes=torch.cuda.max_memory_allocated(), reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                    first_step_includes_warmup=update == 0, parameter_relative_change=change, frozen_max_change=frozen,
                    teacher_frozen=True, **gradient)
                storage_began = time.perf_counter()
                save_state(directory/'resume.pt', policy, optimizer, local,
                    dict(epoch=0, next_input=(update+1)*GLOBAL_BATCH, global_step=update+1,
                        independent_input_exposures=(update+1)*GLOBAL_BATCH),
                    extra=dict(performance_only=True, nonformal=True, arm=arm, declared_microbatch=declared_micro,
                        actual_microbatch=micro, completed_steps=step_rows+[row], oom_attempts=oom_rows))
                row['complete_state_save_seconds'] = time.perf_counter()-storage_began
                step_rows.append(row); micro_rows.extend(current)
                with (directory/'micro_timings.jsonl').open('a') as stream:
                    for value in current:
                        stream.write(json.dumps(value, allow_nan=False)+'\n')
                atomic_json(directory/'step_timings.json', step_rows)
                print(json.dumps(dict(event='OPD_performance_update', arm=arm, update=update+1,
                    inputs=GLOBAL_BATCH, microbatch=micro, seconds=row['full_batch_seconds'],
                    coefficient=coefficient, gradient_norm=row['gradient_norm'])), flush=True)
        finally:
            timing.close()
        arm_wall = time.perf_counter()-arm_began
        intervals = {name: summary([row[name] for row in step_rows]) for name in
            ('full_batch_seconds', 'load_seconds', 'forward_seconds', 'backward_seconds',
             'student_forward_seconds', 'teacher_forward_seconds', 'other_forward_seconds',
             'optimizer_step_seconds', 'diagnostic_seconds', 'complete_state_save_seconds')}
        result = dict(arm=arm, status='PASS', optimizer_trajectories=1, optimizer_steps=len(step_rows),
            original_start=config['checkpoint'], coefficient=coefficient, input_sequence=str(out/'training_input_sequence.json'),
            independent_input_exposures=UPDATES*GLOBAL_BATCH, global_input_batch=GLOBAL_BATCH,
            declared_microbatch=declared_micro, actual_microbatch=micro, optimizer_initially_empty=True,
            identical_shared_initial_policy=True, teacher_frozen=True, rolling_blocks=[0, 32, 64],
            policy_versions=[0, 1], prefix_refresh_each_optimizer_update=True,
            nonzero_teacher_student_treatment_observed=any(row['maximum_block_log_density_difference']>1e-8 for row in micro_rows),
            timings=intervals, complete_steps=step_rows,
            micro_intervals=summary([row['micro_wall_seconds'] for row in micro_rows]),
            compute_input_throughput=UPDATES*GLOBAL_BATCH/sum(row['full_batch_seconds'] for row in step_rows),
            end_to_end_input_throughput=UPDATES*GLOBAL_BATCH/arm_wall,
            wall_seconds=arm_wall, oom_recoveries=oom_rows, continuation_checkpoint=str(directory/'resume.pt'),
            allocated_peak_bytes=max(row['allocated_peak_bytes'] for row in step_rows),
            reserved_peak_bytes=max(row['reserved_peak_bytes'] for row in step_rows),
            precision='float32 model, TF32 enabled', actual_gpu=torch.cuda.get_device_name(), performance_only=True)
        results.append(result); atomic_json(directory/'result.json', result)
        atomic_json(out/'progress.json', dict(status='RUNNING', completed_arms=results, optimizer_steps=completed_updates))
        del optimizer, plugin
        torch.cuda.empty_cache()
    result = dict(status='PASS', config=config, adapter_version=VERSION, performance_only=True,
        optimizer_trajectories=3, optimizer_steps=completed_updates, arms=results,
        global_input_batch=GLOBAL_BATCH, training_windows_per_arm=UPDATES*GLOBAL_BATCH,
        minimum_safe_microbatch=min(row['actual_microbatch'] for row in results),
        formal_validation_executed=False, test_used=False, coefficient_selection_performed=False,
        source_checkpoint=config['checkpoint'], teacher_checkpoint=config['teacher_config']['checkpoint'],
        timing_scope='Per-native-forward and microbatch CUDA synchronization; first complete step includes warmup; storage and diagnostics separately timed',
        interval_limit='Two complete optimizer batches per arm; p95 is descriptive, not a stabilized latency estimate',
        actual_gpu=torch.cuda.get_device_name(), world_size=1, seconds=time.perf_counter()-began,
        allocation_accounting='Native job allocation and idle time are accounted separately by the campaign controller')
    atomic_json(out/'result.json', result)
    return result
