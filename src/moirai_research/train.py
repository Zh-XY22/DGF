"""Fixed-budget preparation/RL with point-only validation and resumable state."""
from .io_utils import checkpoint_exists, load_checkpoint, buffered_torch_save
import json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from .configuration import resolve_checkpoint
from .runtime import seed_all, load_policy, optimizer_for, checked_step, save_state, restore_state, atomic_json, StopRequested
from .data import datasets, batch
from .policy import joints
from .probability import nll
from .preparation import Preparation, LABEL_ARMS, TEACHER_ARMS, OPD_ARMS, WARMUP_ARMS
from .composite import CompositePreparation
from .rl import make_rollout, objective
from .probe import to_cpu, to_device, relative_update
from .evaluation import evaluation
from .point_selection import point_improvements, selection_records, finite_point, selected_checkpoint_path, same_device_readout_check
from .probability_selection import probability_improvements, probability_records
from .training_contract import validate_frozen_training_config
from .adaptive_dgf import apply_learning_rate


def weights_checkpoint(path, policy, config, progress):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    buffered_torch_save(dict(policy_state=policy.state_dict(), route=policy.route,
        transport_mode=getattr(policy,'transport_mode',None),
        routing=policy.routing, fixed_kappa=policy.fixed_kappa, config=config,
        progress=progress, resumable=False,
        continuation='Use resume.pt with optimizer/RNG/cursor, not this evaluation readout'), path)


def train(config_path, output, device='cuda', seed=1):
    full = resolve_checkpoint(json.loads(Path(config_path).read_text()))
    resume = full.get('resume'); completion_only = full.get('completion_only', False); stop_after_epoch=full.get('stop_after_epoch')
    config = {k: v for k, v in full.items() if k not in ['resume', 'completion_only','stop_after_epoch']}
    execution_contract = validate_frozen_training_config(config)
    if config.get('seed', seed) != seed:
        raise ValueError('CLI seed differs from frozen configuration')
    phase = config['phase']; arm = config.get('algorithm_arm', config.get('arm'))
    objective_config = dict(config, arm=arm)
    rollout_function, objective_function = make_rollout, objective
    if config.get('continuous_recipe'):
        from .four_horizon_rl import make_rollout as rollout_function, objective as objective_function
    if config.get('dgf_complete') and phase == 'scientific':
        from .dgf_reference_rl import make_rollout as rollout_function, objective as objective_function
    preparation_phase = phase == 'preparation'
    ordinary = preparation_phase and config['preparation'] == 'ordinary'
    qualified_preparation = preparation_phase and (config.get('preparation_arm') in
        LABEL_ARMS+TEACHER_ARMS+OPD_ARMS+WARMUP_ARMS or bool(config.get('preparation_components')) or config.get('preparation_arm')=='D01')
    if not ordinary and not qualified_preparation and arm not in [f'E{i:02d}' for i in range(18)]+['R15']:
        raise NotImplementedError('Unqualified method; no silent ordinary-preparation fallback')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This finite entry qualifies single-GPU execution only')
    if config.get('diagnostic_input_limit') is not None and not config.get('qualification_only'):
        raise ValueError('Formal training cannot truncate the registered input epochs')
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    stop = StopRequested(); seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    policy = load_policy(config, device).eval(); optimizer = optimizer_for(policy, config['lr'])
    reference = None if preparation_phase or arm in ['E00', 'E02'] else load_policy(config, device).eval().requires_grad_(False)
    data = datasets(policy, config['dataset'], config['protocol'])['train']; total = len(data)
    if config.get('diagnostic_input_limit') is not None:
        total = min(total, config['diagnostic_input_limit'])
    if config.get('dgf_complete') and config.get('method_kind')=='FULL' and not preparation_phase and not resume:
        from .dgf_core_checks import check
        check(policy,batch(data,list(range(min(2,total))),device),config,out/'full_feature_pretraining_checks.json')
    epochs = int(config.get('epochs', 1 if preparation_phase else 4))
    global_batch = config.get('global_input_batch', 512); micro = config['microbatch']
    if global_batch != 512 and not config.get('qualification_only'):
        raise ValueError('Formal independent input batch is frozen to 512')
    updates = int(config.get('updates_per_input_batch', 1 if preparation_phase else 2))
    updates_schedule = config.get('updates_by_epoch', [updates]*epochs)
    if len(updates_schedule)!=epochs or any(type(v) is not int or v<1 for v in updates_schedule):
        raise ValueError('Invalid frozen per-epoch update schedule')
    checkpoint_interval = int(config.get('checkpoint_interval_batches', 64))
    if checkpoint_interval < 1: raise ValueError('Checkpoint interval must be positive')
    plugin = CompositePreparation(config, device) if preparation_phase and config.get('preparation_components') else Preparation(dict(config, arm=config.get('preparation_arm', arm)), device) if (qualified_preparation and config.get('preparation_arm')!='D01') or arm in ['E15', 'R15'] else None
    if preparation_phase and config.get('preparation_arm')=='D01':
        from .mode_specialization import ModeSpecialization
        plugin=ModeSpecialization(config,device)
    ema_state = {n: p.detach().clone() for n, p in policy.named_parameters()} if preparation_phase and config.get('warmup_arm') == 'S10' else None
    nll_only = reference is None
    initial = {k: p.detach().cpu().clone() for k, p in policy.named_parameters()}
    cursor = dict(epoch=0, next_input=0, global_step=0, independent_input_exposures=0)
    curve = []; training_curve = []; point_values = {}; best_paths = {}; cache = {}; first = None
    seed_all(seed); began = time.time(); torch.cuda.reset_peak_memory_stats()
    if resume:
        if not checkpoint_exists(out/'checkpoints/initial.pt'):
            weights_checkpoint(out/'checkpoints/initial.pt', policy, config,
                dict(global_step=0, source='Original branch start reconstructed before continuation'))
        cursor, extra = restore_state(resume, policy, optimizer, config)
        curve = extra['curve']; training_curve = extra['training_curve']
        point_values = extra['point_values']; best_paths = extra['best_paths']; first = extra['first_step']
        if ema_state is not None:
            if extra.get('ema_state') is None: raise ValueError('EMA continuation state is missing')
            ema_state = {name: value.to(device) for name, value in extra['ema_state'].items()}
        if any(not checkpoint_exists(path) for path in best_paths.values()):
            raise RuntimeError('A selected prior point checkpoint is unavailable; selection cannot be silently changed')
        atomic_json(out/'resumption.json', dict(source=resume, cursor=cursor,
            configuration_equal=True, optimizer_state_entries=len(optimizer.state),
            optimizer_rng_cursor_restored=True, actual_gpu=torch.cuda.get_device_name(),
            bitwise_future_trajectory_guaranteed=False))
        if completion_only:
            assert cursor['epoch'] >= epochs and cursor['next_input'] == 0
    elif completion_only:
        raise ValueError('Completion-only requires a complete prior state')
    starting_global_step = cursor['global_step']
    starting_exposures = cursor['independent_input_exposures']
    attempt_initial = {name:p.detach().cpu().clone() for name,p in policy.named_parameters()}
    attempt_first = None
    def continuation():
        save_state(out/'resume.pt', policy, optimizer, config, cursor,
            extra=dict(curve=curve, training_curve=training_curve, point_values=point_values,
                best_paths=best_paths, first_step=first, ema_state=ema_state))
    def validate(epoch, fraction, actual_inputs):
        step = cursor['global_step']; cached = step in cache
        if not cached:
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
            try:
                cache[step] = evaluation(policy, config['protocol'], 'val', None, seed,
                    config['artifact'], config.get('diagnostic_validation_limit'))
            finally:
                random.setstate(rng[0]); np.random.set_state(rng[1])
                torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
        metrics = cache[step]; improved = point_improvements(metrics, point_values)
        if config.get('dgf_selection_only'):
            improved = [r for r in improved if r['label']=='P1_val_MSE_best']
        improved += probability_improvements(metrics, point_values, config.get('record_probability_checkpoints', False))
        progress = dict(epoch=epoch, scheduled_fraction=fraction, actual_inputs=actual_inputs,
            global_step=step, independent_input_exposures=cursor['independent_input_exposures'])
        if config.get('retain_all_validation_checkpoints') and not cached:
            retained=out/f'checkpoints/validation-step{step}.pt'
            if not retained.exists():weights_checkpoint(retained,policy,config,progress)
        if improved:
            path = out/'checkpoints/initial.pt' if step == 0 else out/f'checkpoints/point-best-step{step}.pt'
            if not path.exists(): weights_checkpoint(path, policy, config, progress)
            for record in improved: best_paths[record['label']] = str(path)
        row = dict(**progress, full_validation=metrics['complete'],
            reused_identical_parameter_state=cached, metrics=metrics, selected=improved,
            MSE=metrics['MSE'], MAE=metrics['MAE'], CRPS=metrics.get('CRPS'),
            joint_NLL_per_position=metrics.get('joint_NLL_per_position'))
        curve.append(row); atomic_json(out/'validation_curve.json', curve)
        print(json.dumps(dict(event='validation', **progress,
            selected=[x['label'] for x in improved], cached=cached)), flush=True)
    def retain_budget_checkpoint(completed_epoch):
        if completed_epoch not in config.get('fixed_epoch_checkpoints',[]):return
        file=out/'checkpoints'/('final.pt' if completed_epoch==epochs else f'budget-epoch{completed_epoch}.pt')
        weights_checkpoint(file,policy,config,cursor)
        best_paths['budget_epoch_'+str(completed_epoch)]=str(file)
    if not resume:
        weights_checkpoint(out/'checkpoints/initial.pt', policy, config, cursor)
        validate(-1, 0., 0); continuation()
    if resume and cursor.get('pending_epoch_validation') is not None:
        completed_epoch=cursor.pop('pending_epoch_validation')
        validate(completed_epoch, 1., total);retain_budget_checkpoint(completed_epoch+1);continuation()
    for epoch in range(cursor['epoch'], epochs):
        updates = updates_schedule[epoch]
        order = torch.randperm(total, generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        start = cursor['next_input'] if epoch == cursor['epoch'] else 0
        denominator = 16 if epoch == 0 else 4
        thresholds = [k/denominator for k in range(1, denominator) if math.ceil(k/denominator*total) > start]
        while start < total:
            ids = order[start:start+global_batch]
            chunks = [ids[i:i+micro] for i in range(0, len(ids), micro)]
            rollouts = []; packed_batches = []; block_began = time.time()
            context = plugin.batch_context(ids) if plugin is not None and preparation_phase else None
            for chunk_index,ix in enumerate(chunks):
                objective_config.update(_paired_epoch=epoch,_paired_rollout_key=f'{epoch}/{start}/{chunk_index}')
                b = batch(data, ix, 'cpu' if nll_only else device); packed_batches.append(to_cpu(b))
                if not nll_only:
                    reliability = plugin.reliability(ix) if arm in ['E15', 'R15'] else None
                    if arm in ['E15', 'R15']:
                        valid = b['prediction_mask'][..., None] & b['observed_mask']
                        plugin.verify_alignment(b, b['target'][valid].reshape(len(ix), -1))
                    rollouts.append(to_cpu(rollout_function(policy, reference, b, objective_config, reliability=reliability)))
            if config.get('continuous_recipe'):
                from .four_horizon_rl import prepare_complete_batch
                prepare_complete_batch(rollouts, config)
            for update in range(updates):
                if config.get('continuous_recipe')=='R08' and update>0:
                    rollouts=[to_cpu(rollout_function(policy,reference,to_device(b,device),objective_config)) for b in packed_batches]
                objective_config.update(_optimizer_step=cursor['global_step'],
                    _total_optimizer_steps=math.ceil(total/global_batch)*sum(updates_schedule))
                actual_lr=apply_learning_rate(optimizer,config,cursor['global_step'],objective_config['_total_optimizer_steps']) if config.get('adaptive_dgf') else config['lr']
                optimizer.zero_grad(set_to_none=True); loss_sum = 0.; diagnostics = []
                for ci, ix in enumerate(chunks):
                    b = to_device(packed_batches[ci], device)
                    objective_config.update(_microbatch_index=ci,_paired_rollout_key=f'{epoch}/{start}/{ci}')
                    if nll_only:
                        if config.get('warmup_reconstruction') == 'history_patch_v2':
                            from .warmup_repairs import masked_history_loss
                            loss, diag = masked_history_loss(policy, b, seed, cursor['global_step'])
                        else:
                            experts, target, w, _, _ = joints(policy(b), b)
                            if config.get('direct_objective'):
                                from .four_horizon_losses import direct_objective
                                loss, diag = direct_objective(experts, w, target, config)
                            else:
                                loss = nll(experts, w, target); diag = dict(nll=float(loss.detach()))
                            if qualified_preparation:
                                if config.get('preparation_components'):
                                    auxiliary, extra = plugin.auxiliary(b, experts, target, w, context)
                                else:
                                    auxiliary, extra = plugin.auxiliary(b, experts, target, w, context,
                                        student_policy=policy, policy_version=cursor['global_step'])
                                loss = loss+config['auxiliary_coefficient']*auxiliary
                                diag.update(extra, auxiliary_coefficient=config['auxiliary_coefficient'])
                    else:
                        loss, diag = objective_function(policy, b, to_device(rollouts[ci], device), objective_config)
                    if config.get('point_probability_mix'):
                        from .point_probability_mix import add_probability_term
                        loss,diag=add_probability_term(policy,b,loss,diag,config)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite formal training loss; not a missing diagnostic')
                    (loss*len(ix)/len(ids)).backward(); loss_sum += float(loss.detach())*len(ix)/len(ids)
                    diagnostics.append(dict(inputs=len(ix), **diag))
                sam_diagnostic={}
                if config.get('warmup_optimizer')=='SAM':
                    from .sam_warmup import second_gradient
                    sam_diagnostic=second_gradient(policy,optimizer,packed_batches,chunks,len(ids),plugin,context,config,to_device,device)
                component_gradient={}
                if config.get('dgf_complete'):
                    for name,module in [('concentration',policy.router_concentration),('router_mean',policy.router_mean),('mode_head0',policy.heads[0])]:
                        norms=[parameter.grad.detach().double().square().sum() for parameter in module.parameters() if parameter.grad is not None]
                        component_gradient[name+'_gradient_norm']=float(torch.stack(norms).sum().sqrt()) if norms else None
                gradient = checked_step(policy, optimizer,allow_zero_gradient=bool(config.get('dgf_complete'))); cursor['global_step'] += 1
                gradient.update(component_gradient)

                gradient.update(sam_diagnostic)
                if attempt_first is None:
                    change,frozen=relative_update(policy,attempt_initial)
                    if frozen!=0 or change<=0:raise RuntimeError('No effective parameter update in this attempt')
                    attempt_first=dict(**gradient,relative_parameter_change=change,frozen_max_change=frozen,global_step=cursor['global_step'])
                if ema_state is not None:
                    decay=float(config.get('ema_decay', .99))
                    with torch.no_grad():
                        for name, parameter in policy.named_parameters():
                            ema_state[name].mul_(decay).add_(parameter.detach(), alpha=1.-decay)
                if first is None:
                    change, frozen = relative_update(policy, initial); assert frozen == 0 and change > 0
                    first = dict(**gradient, relative_parameter_change=change,
                        frozen_max_change=frozen, global_step=cursor['global_step'],
                        parameters=sum(p.numel() for p in policy.parameters()),
                        trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad),
                        selective_treatment=getattr(plugin,'reliability_evidence',None))
                    atomic_json(out/'first_step.json', first)
                with (out/'micro_diagnostics.jsonl').open('a') as log:
                    log.write(json.dumps(dict(global_step=cursor['global_step'], diagnostics=diagnostics), allow_nan=False)+'\n')
                training_curve.append(dict(epoch=epoch, input_start=start, inputs=len(ids),
                    rollout_update=update, global_step=cursor['global_step'], loss=loss_sum,
                    lr=actual_lr, **gradient, elapsed_seconds=time.time()-began))
            start += len(ids)
            cursor.update(epoch=epoch, next_input=start, independent_input_exposures=epoch*total+start)
            if start >= total: cursor.update(epoch=epoch+1, next_input=0, pending_epoch_validation=epoch)
            while thresholds and start >= math.ceil(thresholds[0]*total):
                validate(epoch, thresholds.pop(0), start)
            if start == len(ids) or (start//global_batch) % checkpoint_interval == 0 or stop.requested or start == total:
                continuation()
            atomic_json(out/'training_curve.json', training_curve)
            print(json.dumps(dict(event='training_batch', epoch=epoch, input_end=start,
                total=total, global_step=cursor['global_step'], seconds=time.time()-block_began)), flush=True)
            if stop_after_epoch is not None and cursor['epoch']>=int(stop_after_epoch):stop.requested=True
            if stop.requested:
                continuation(); atomic_json(out/'result.json', dict(status='INTERRUPTED_WITH_STATE',
                    checkpoint=str(out/'resume.pt'), cursor=cursor)); return
        validate(epoch, 1., total); cursor.pop('pending_epoch_validation', None);retain_budget_checkpoint(epoch+1);continuation()
    if ema_state is not None:
        with torch.no_grad():
            for name, parameter in policy.named_parameters(): parameter.copy_(ema_state[name])
    if epochs not in config.get('fixed_epoch_checkpoints',[]) or not (out/'checkpoints/final.pt').is_file():
        weights_checkpoint(out/'checkpoints/final.pt', policy, config, cursor)
    continuation()
    readouts = {}; reloaded = {}
    for label, path in dict(initial=str(out/'checkpoints/initial.pt'), **best_paths,
            final=str(out/'checkpoints/final.pt')).items():
        if not checkpoint_exists(path):
            readouts[label] = dict(status='MISSING_PRIOR_INITIAL_READOUT', checkpoint=path); continue
        if path not in reloaded:
            saved = load_checkpoint(path, map_location=device, weights_only=False)
            policy.load_state_dict(saved['policy_state'], strict=True)
            reloaded[path] = evaluation(policy, config['protocol'], 'val', out/'readouts'/label,
                seed, config['artifact'], config.get('diagnostic_validation_limit'))
        readouts[label] = dict(checkpoint=path, validation=reloaded[path], reloaded=True,
            selection='Independent P1/P2/P3 MSE/MAE; full validation; no test or diagnostic gate')
    selected = selection_records(point_values, best_paths)
    if config.get('dgf_selection_only'):
        selected = {k:v for k,v in selected.items() if k=='P1_val_MSE_best'}
    for label, record in selected.items():
        if record['status'] != 'VALID': continue
        path = selected_checkpoint_path(record, readouts[label])
        value = readouts[label]['validation'].get(record['validation_metric'])
        delta = None if not finite_point(value) else value-record['validation_value']
        comparison = dict(selection_value=record['validation_value'], reloaded_value=value, delta=delta,
            within_original_tolerance=delta is not None and abs(delta) <= max(1e-7, abs(record['validation_value'])*1e-6))
        readouts[label]['selection_comparison'] = comparison
        if not comparison['within_original_tolerance']:
            if resume:
                readouts[label]['cross_device_reload_discrepancy'] = True
                # Keep the old choice; do not substitute a new winner after resumption.
                saved = load_checkpoint(path, map_location=device, weights_only=False)
                policy.load_state_dict(saved['policy_state'], strict=True)
                again = evaluation(policy, config['protocol'], 'val', None, seed,
                    config['artifact'], config.get('diagnostic_validation_limit'))
                reload_check = same_device_readout_check(record, reloaded[path], again)
                readouts[label]['same_device_reload_verified'] = True
                readouts[label]['same_device_reload_checkpoint'] = path
                readouts[label]['same_device_reload_scope'] = reload_check
            else: raise RuntimeError('Point checkpoint reload differs from its own selection: '+label)
    if completion_only: assert cursor['global_step'] == starting_global_step
    best = selected['P1_val_MSE_best']
    result = dict(status='DIAGNOSTIC_ONLY' if config.get('qualification_only') else 'PASS',
        completion_only=completion_only, optimizer_steps_this_attempt=cursor['global_step']-starting_global_step,
        independent_input_exposures_this_attempt=cursor['independent_input_exposures']-starting_exposures,
        first_step_this_attempt=attempt_first,
        meaning='Fixed-budget training and independent point validation complete; formal test remains gated',
        config=config, cursor=cursor, optimizer_trajectories=0 if completion_only else 1,
        first_step=first, readouts=readouts, point_selection=selected,
        selective_treatment=getattr(plugin,'reliability_evidence',None) or (first or {}).get('selective_treatment'),
        actual_gpu=torch.cuda.get_device_name(), seconds=time.time()-began,
        allocated_peak_bytes=torch.cuda.max_memory_allocated(), reserved_peak_bytes=torch.cuda.max_memory_reserved(),
        independent_inputs_per_epoch=total, epochs=epochs, updates_per_input_batch=updates, updates_by_epoch=updates_schedule,
        frozen_execution_contract=execution_contract,
        global_effective_input_batch=global_batch, microbatch=micro, world_size=1, test_used=False,
        checkpoint_interval_batches=checkpoint_interval, continuation_checkpoint=str(out/'resume.pt'),
        best_checkpoint=best.get('checkpoint'), diagnostic_metrics_affect_selection=False,
        validation_cadence='first epoch 1/16, following epochs 1/4',
        preparation_alignment=None if plugin is None else dict(windows=plugin.checked_windows,
            maximum_target_difference=plugin.maximum_target_difference))
    result['probability_selection'] = probability_records(point_values, best_paths)
    atomic_json(out/'result.json', result); return result
