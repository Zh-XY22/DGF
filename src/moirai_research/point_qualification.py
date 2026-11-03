"""Small worker-only qualification of fixed-budget SFT and point readouts."""
import json
import time
from pathlib import Path
import torch
from .runtime import atomic_json, load_policy
from .point_selection import validate_qualification, finite_point


def audit(config_path, output, device='cuda', seed=1):
    from .supervised import fit
    from . import evaluation as evaluation_module
    config = json.loads(Path(config_path).read_text())
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    proofs = validate_qualification()
    from .selection import validate_qualification as qualify_conditional_selection
    proofs['conditional_selection'] = qualify_conditional_selection()
    from .zero_anchor_check import run_checks as qualify_zero_anchor
    proofs['zero_anchor'] = qualify_zero_anchor(device=device)
    from .point_analysis import validate_qualification as qualify_analysis_records
    proofs['analysis_scalar_record_checks'] = qualify_analysis_records()
    results = []
    supplied = config.get('supervised_configs', [config.get('supervised_config')])
    if not supplied or any(not isinstance(value, dict) for value in supplied):
        raise ValueError('Explicit supervised_configs are required for worker qualification')
    for index, source in enumerate(supplied):
        local = dict(source, fixed_epochs=1, qualification_only=True,
            diagnostic_input_limit=2, diagnostic_validation_limit=2,
            global_input_batch=2, microbatch=1, seed=seed)
        local.pop('resume', None); local.pop('checkpoint', None)
        path = out/f'case-{index}/config.json'; atomic_json(path, local)
        receipt = fit(path, path.parent/'sft', device, seed)
        assert receipt['status'] == 'DIAGNOSTIC_ONLY'
        assert receipt['optimizer_steps'] == 1
        assert receipt['first_step']['relative_parameter_update'] > 0
        assert receipt['first_step']['frozen_max_change'] == 0
        assert all(value['status'] == 'VALID' for value in receipt['point_selection'].values())
        probe = dict(local, checkpoint=receipt['best_checkpoint'], checkpoint_kind='policy_state')
        policy = load_policy(probe, device).eval()
        original = evaluation_module.observation_logp
        def unavailable_probability(*args, **kwargs):
            raise FloatingPointError('Intentionally unavailable diagnostic for qualification')
        try:
            evaluation_module.observation_logp = unavailable_probability
            metrics = evaluation_module.evaluation(policy, local['protocol'], 'val',
                out/f'case-{index}/diagnostic-isolation', seed, local['artifact'], 2)
        finally:
            evaluation_module.observation_logp = original
        assert metrics['joint_NLL_per_position'] is None
        assert metrics['marginal_NLL'] is None
        assert finite_point(metrics['MSE']) and finite_point(metrics['MAP_MSE'])
        assert all(value['status'] == 'VALID' for value in metrics['point_protocol_status'].values())
        profile = None
        if config.get('profile_full_validation'):
            began = time.time()
            complete = evaluation_module.evaluation(policy, local['protocol'], 'val', None,
                seed, local['artifact'])
            profile = dict(wall_seconds=time.time()-began, metric_seconds=complete['seconds'],
                independent_inputs=complete['windows'], complete=complete['complete'],
                protocol=local['protocol'], eval_samples=complete['num_samples'],
                actual_gpu=torch.cuda.get_device_name(), output_arrays_saved=False,
                optimizer_steps=0, optimizer_trajectories=0)
        results.append(dict(route=local['route'], receipt=str(path.parent/'sft/result.json'),
            optimizer_steps=receipt['optimizer_steps'], six_point_readouts=True,
            checkpoint_reloaded=True, unavailable_probability_does_not_gate=True,
            full_validation_profile=profile,
            actual_gpu=torch.cuda.get_device_name()))
        del policy; torch.cuda.empty_cache()
    result = dict(status='PASS', config=config, checks=proofs, cases=results,
        optimizer_trajectories=len(results), optimizer_steps=sum(x['optimizer_steps'] for x in results),
        test_used=False, qualification_only=True,
        scope='Real two-input CUDA updates and reloads plus point/diagnostic isolation; not a formal experiment')
    atomic_json(out/'result.json', result)
    return result
