"""Fixed-budget public-weight SFT with six independent point-only readouts."""
from .io_utils import checkpoint_exists, load_checkpoint
import json, random, time
from pathlib import Path
import numpy as np
import torch
from uni2ts.loss.packed import PackedNLLLoss
from .policy import Policy, joints
from .probability import nll
from .data import datasets, batch
from .runtime import seed_all, optimizer_for, checked_step, save_state, restore_state, atomic_json, StopRequested
from .train import weights_checkpoint
from .probe import relative_update
from .evaluation import evaluation
from .point_selection import point_improvements, selection_records, finite_point
from .probability_selection import probability_improvements, probability_records


def supervised_loss(policy, packed, objective):
    result = policy(packed)
    if objective == 'official_PackedNLLLoss':
        if policy.route != 'A0':
            raise ValueError('Native packed NLL applies only to the single-head SFT')
        return PackedNLLLoss()(pred=result[0][0], **{k: packed[k] for k in
            ['target', 'prediction_mask', 'observed_mask', 'sample_id', 'variate_id']})
    if objective == 'legal_trajectory_mixture_NLL':
        experts, target, w, _, _ = joints(result, packed)
        return nll(experts, w, target)
    raise ValueError('Unknown supervised training objective')


def fit(config_path, output, device='cuda', seed=1):
    full = json.loads(Path(config_path).read_text())
    resume = full.get('resume')
    config = {k: v for k, v in full.items() if k != 'resume'}
    if config['route'] not in ['A0', 'D-pre'] or config.get('checkpoint'):
        raise ValueError('New-seed SFT must start from fixed public weights')
    epochs = config.get('fixed_epochs')
    if not isinstance(epochs, int) or not 1 <= epochs <= 200:
        raise ValueError('An explicit fixed_epochs budget is required; no NLL early stopping')
    if config.get('diagnostic_input_limit') is not None and not config.get('qualification_only'):
        raise ValueError('Formal SFT must expose complete training epochs')
    if config.get('seed', seed) != seed:
        raise ValueError('CLI seed differs from frozen configuration')
    seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    stop = StopRequested(); began = time.time()
    policy = Policy(config['weights'], config['route'], revision=config.get('revision'),
        local_files_only=config.get('local_files_only', True)).to(device)
    policy.configure_update(fixed_kappa=True); policy.eval()
    data = datasets(policy, config['dataset'], config['protocol'])
    optimizer = optimizer_for(policy, config['lr'])
    initial = {k: p.detach().cpu().clone() for k, p in policy.named_parameters()}
    global_batch = config['global_input_batch']; micro = config['microbatch']
    total = len(data['train'])
    if config.get('diagnostic_input_limit') is not None:
        total = min(total, config['diagnostic_input_limit'])
    cursor = dict(epoch=0, next_input=0, global_step=0, independent_input_exposures=0)
    best_values = {}; best_paths = {}; curve = []; train_curve = []; first = None
    metric_cache = {}
    if resume:
        cursor, extra = restore_state(resume, policy, optimizer, config)
        best_values = extra['point_values']; best_paths = extra['best_paths']
        curve = extra['curve']; train_curve = extra['train_curve']; first = extra['first']
        if any(not checkpoint_exists(path) for path in best_paths.values()):
            raise RuntimeError('A previously selected point checkpoint is unavailable')
        atomic_json(out/'resumption.json', dict(source=resume, cursor=cursor,
            optimizer_rng_cursor_restored=True, selected_point_states_preserved=True))
    else:
        weights_checkpoint(out/'checkpoints/initial.pt', policy, config, cursor)

    def continuation():
        save_state(out/'resume.pt', policy, optimizer, config, cursor,
            extra=dict(point_values=best_values, best_paths=best_paths, curve=curve,
                train_curve=train_curve, first=first))

    def validate(epoch, fraction, actual_inputs):
        step = cursor['global_step']; cached = step in metric_cache
        if not cached:
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
            try:
                metric_cache[step] = evaluation(policy, config['protocol'], 'val', None,
                    seed, config['artifact'], config.get('diagnostic_validation_limit'))
            finally:
                random.setstate(rng[0]); np.random.set_state(rng[1])
                torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
        metrics = metric_cache[step]
        improved = point_improvements(metrics, best_values)
        improved += probability_improvements(metrics, best_values, config.get('record_probability_checkpoints', False))
        progress = dict(epoch=epoch, scheduled_fraction=fraction, actual_inputs=actual_inputs,
            global_step=step, independent_input_exposures=cursor['independent_input_exposures'])
        if improved:
            path = out/'checkpoints/initial.pt' if step == 0 else out/f'checkpoints/point-best-step{step}.pt'
            if not path.exists():
                weights_checkpoint(path, policy, config, progress)
            for record in improved:
                best_paths[record['label']] = str(path)
        row = dict(**progress, full_validation=metrics['complete'],
            reused_identical_parameter_state=cached, metrics=metrics, selected=improved,
            selection_uses_only='P1/P2/P3 MSE and MAE; no diagnostic gate')
        curve.append(row); atomic_json(out/'validation_curve.json', curve)
        print(json.dumps(dict(event='supervised_point_validation', **progress,
            selected=[x['label'] for x in improved], cached=cached)), flush=True)

    if not resume:
        validate(-1, 0., 0); continuation()
    if cursor.get('pending_epoch_validation') is not None:
        validate(cursor.pop('pending_epoch_validation'), 1., total); continuation()
    for epoch in range(cursor['epoch'], epochs):
        order = torch.randperm(total, generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        start = cursor['next_input'] if epoch == cursor['epoch'] else 0
        denominator = 16 if epoch == 0 else 4
        thresholds = [k/denominator for k in range(1, denominator)
            if int(np.ceil(k/denominator*total)) > start]
        while start < total:
            ids = order[start:start+global_batch]; optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.; batch_began = time.time()
            for offset in range(0, len(ids), micro):
                ix = ids[offset:offset+micro]; packed = batch(data['train'], ix, device)
                loss = supervised_loss(policy, packed, config['supervised_objective'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite supervised training loss, not a diagnostic failure')
                (loss*len(ix)/len(ids)).backward(); loss_sum += float(loss.detach())*len(ix)/len(ids)
            gradient = checked_step(policy, optimizer); cursor['global_step'] += 1
            if first is None:
                change, frozen = relative_update(policy, initial)
                assert change > 0 and frozen == 0
                first = dict(loss=loss_sum, **gradient, relative_parameter_update=change,
                    frozen_max_change=frozen, parameters=sum(p.numel() for p in policy.parameters()),
                    trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad))
                atomic_json(out/'first_step.json', first)
            start += len(ids)
            cursor.update(epoch=epoch, next_input=start, independent_input_exposures=epoch*total+start)
            if start == total:
                cursor.update(epoch=epoch+1, next_input=0, pending_epoch_validation=epoch)
            train_curve.append(dict(epoch=epoch, global_step=cursor['global_step'], inputs=len(ids),
                loss=loss_sum, lr=config['lr'], **gradient, seconds=time.time()-batch_began))
            while thresholds and start >= int(np.ceil(thresholds[0]*total)):
                validate(epoch, thresholds.pop(0), start)
            if cursor['global_step'] % 64 == 0 or cursor['global_step'] == 1 or start == total or stop.requested:
                continuation(); atomic_json(out/'training_curve.json', train_curve)
            if stop.requested:
                atomic_json(out/'result.json', dict(status='INTERRUPTED_WITH_STATE',
                    checkpoint=str(out/'resume.pt'), cursor=cursor)); return
        validate(epoch, 1., total); cursor.pop('pending_epoch_validation', None); continuation()
    weights_checkpoint(out/'checkpoints/final.pt', policy, config, cursor); continuation()
    state_cache = {}; readouts = {}
    for name, path in dict(initial=str(out/'checkpoints/initial.pt'), **best_paths,
            final=str(out/'checkpoints/final.pt')).items():
        if not checkpoint_exists(path):
            readouts[name] = dict(status='MISSING_PRIOR_READOUT', checkpoint=path); continue
        if path not in state_cache:
            state = load_checkpoint(path, map_location=device, weights_only=False)
            policy.load_state_dict(state['policy_state'], strict=True)
            state_cache[path] = evaluation(policy, config['protocol'], 'val', out/'readouts'/name,
                seed, config['artifact'], config.get('diagnostic_validation_limit'))
        readouts[name] = dict(checkpoint=path, validation=state_cache[path], reloaded=True,
            selection='Full validation point metrics only; independent objectives')
    selected = selection_records(best_values, best_paths)
    common_start = selected['P1_val_MSE_best']
    if common_start['status'] != 'VALID':
        raise RuntimeError('Shared public-weight start has no legal P1 validation MSE readout')
    for label, record in selected.items():
        if record['status'] != 'VALID': continue
        repeated = readouts[label]['validation'].get(record['validation_metric'])
        if not finite_point(repeated) or abs(repeated-record['validation_value']) > max(1e-7, abs(record['validation_value'])*1e-6):
            raise RuntimeError('Reloaded point checkpoint differs from its own selection record: '+label)
    result = dict(status='DIAGNOSTIC_ONLY' if config.get('qualification_only') else 'PASS',
        config=config, seed=seed, public_weight_start=True, full_process=True,
        best_checkpoint=common_start['checkpoint'], best_epoch=None, epochs_completed=cursor['epoch'],
        selection_objective='P1 validation MSE; SFT loss is not a selection criterion',
        selection_score=common_start['validation_value'], point_selection=selected,
        cursor=cursor, readouts=readouts, first_step=first, optimizer_trajectories=1,
        actual_gpu=torch.cuda.get_device_name(), seconds=time.time()-began,
        optimizer_steps=cursor['global_step'], test_used=False, early_stop_patience=None,
        fixed_epoch_budget=epochs, global_input_batch=global_batch, microbatch=micro,
        world_size=1, continuation_checkpoint=str(out/'resume.pt'),
        validation_cadence='first epoch 1/16, following epochs 1/4; repeated parameter states cached',
        interpretation='Fixed-budget public SFT; probability/dynamics scores are diagnostics only')
    result['probability_selection'] = probability_records(best_values, best_paths)
    atomic_json(out/'result.json', result); return result
