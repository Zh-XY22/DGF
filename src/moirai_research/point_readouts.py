"""Native-H96 formal readouts behind the new global validation firewall."""
import json, time
from pathlib import Path
import torch
from .io_utils import load_checkpoint
from .evaluation import evaluate_config
from .runtime import atomic_json
from .point_selection import POINT_PROTOCOLS, finite_point
from .point_test_gate import test_release
from .point_scope import native_inference_case, completed_evaluation_scope


def state_identity(case, state):
    # RNG/precision and history protocol are part of the alias, not only weights.
    return json.dumps(dict(dataset=case['dataset'], model=case.get('model'),
        model_id=case.get('model_id'), revision=case.get('revision'), route=case['route'],
        protocol=case['protocol'], artifact=case['artifact'], seed=case['seed'],
        force_fixed_kappa=case.get('force_fixed_kappa',True),
        routing=state.get('routing'), fixed_kappa=state.get('fixed_kappa'),
        precision='float32 model, float64 metrics, TF32 enabled',
        native_horizon=96, eval_samples=100),sort_keys=True)


def same_parameters(left, right):
    return left.keys() == right.keys() and all(torch.equal(left[key],right[key]) for key in left)


def read_metrics(receipt, split):
    record = json.loads(Path(receipt).read_text())
    metrics = record['results'][split]
    if not metrics.get('complete') or metrics.get('split') != split:
        raise ValueError('A formal readout is not complete')
    # Each projected protocol retains its own validity; P2 cannot veto P1/P3.
    # Complete execution and explained point failures are checked independently.
    return metrics


def projected_readout(label, stage, protocol, objective, checkpoint, receipt, split,
        canonical_case, selection=None):
    row = dict(readout=label, stage=stage, protocol=protocol, selection_objective=objective,
        checkpoint=checkpoint, receipt=receipt, split=split, canonical_case=canonical_case,
        selection_status='NOT_APPLICABLE' if objective == 'N/A' else 'VALID' if receipt else 'INVALID')
    if selection is not None:
        row.update(validation_metric=selection.get('validation_metric'),
            validation_value=selection.get('validation_value'))
    if receipt:
        metrics = read_metrics(receipt,split)
        fields = POINT_PROTOCOLS[protocol]
        row.update(point_status='VALID' if all(finite_point(metrics.get(fields[key])) for key in ['MSE','MAE']) else 'INVALID',
            MSE=metrics.get(fields['MSE']),MAE=metrics.get(fields['MAE']),
            CRPS=metrics.get('CRPS'),marginal_NLL=metrics.get('marginal_NLL'),
            joint_NLL_per_position=metrics.get('joint_NLL_per_position'),
            metric_locator='results.'+split, probability_score_scope='Complete checkpoint distribution; not point-output likelihood')
    else:
        row.update(point_status='INVALID',MSE=None,MAE=None,CRPS=None,
            marginal_NLL=None,joint_NLL_per_position=None)
    return row


def evaluate_readouts(config_path, output, device='cuda', seed=1):
    config = json.loads(Path(config_path).read_text())
    out = Path(output); out.mkdir(parents=True,exist_ok=True)
    scientific_path = config['scientific_path']
    release = test_release(config['test_release_receipt'],scientific_path)
    binding = json.loads(Path(config['readout_selection_receipt']).read_text())
    if binding.get('status') != 'PASS' or binding.get('test_used') is not False or binding.get('scientific_path') != scientific_path:
        raise ValueError('The scientific readout binding is not validation-only and frozen')
    if binding['seed'] != seed or config['seed'] != seed:
        raise ValueError('A formal readout seed differs from the frozen scientific path')
    if Path(binding['recipe_selection_receipt']).resolve() != Path(config['test_release_receipt']).resolve():
        raise ValueError('Readout binding points at a different recipe release')
    if set(binding['all_readouts']) != {case['case_id'] for case in config['cases']}:
        raise ValueError('Frozen readout labels differ from supplied cases')
    training = json.loads(Path(binding['training_receipt']).read_text())
    contract = training['config']
    if contract.get('seed') != seed or contract.get('phase') != 'scientific':
        raise ValueError('The bound training receipt is not this scientific seed/phase')
    began = time.time(); cases = []; canonical = []; states = []; numeric_receipts = {}
    prior_root = Path(config['resume_readouts']) if config.get('resume_readouts') else None
    for case in config['cases']:
        native_inference_case(case)
        key = case['case_id']
        if case['checkpoint'] != binding['all_readouts'][key]['checkpoint'] or case['seed'] != seed:
            raise ValueError('A formal case differs from the validation checkpoint/seed binding')
        if case.get('splits') != ['val','test'] or case['protocol']['horizon'] != 96:
            raise ValueError('This entry executes native-H96 complete validation and test only')
        for field in ['weights','revision','route','protocol','artifact','dataset']:
            if case.get(field) != contract.get(field):
                raise ValueError('Formal inference differs from the frozen training data/model contract: '+field)
        if case.get('historical_protocol_replay'):
            raise ValueError('New formal tests cannot bypass release through a historical flag')
        if Path(case['test_release_receipt']).resolve() != Path(config['test_release_receipt']).resolve():
            raise ValueError('A case is not protected by the same final release')
        state = load_checkpoint(case['checkpoint'],map_location='cpu',weights_only=False)
        params = state['policy_state']; identity = state_identity(case,state)
        same = next((old_key for old_key,old_identity,old in states
            if identity == old_identity and same_parameters(params,old)),None)
        if same is not None:
            receipt = numeric_receipts[same]
            record = dict(case_id=key,canonical_case=same,checkpoint=case['checkpoint'],receipt=receipt,
                deduplication='Exact complete policy_state and data/RNG/inference contract equality checked in HOPE',
                exact_parameter_equality=True)
        else:
            states.append((key,identity,params)); directory = out/key; directory.mkdir(exist_ok=True)
            local = directory/'config.json'; atomic_json(local,case)
            previous = prior_root/key if prior_root is not None and (prior_root/key/'config.json').is_file() else None
            result = evaluate_config(local,directory,device,seed,previous)
            if result['status'] != 'PASS':
                raise ValueError('A formal readout did not execute complete windows')
            for split in ['val','test']: read_metrics(directory/'result.json',split)
            receipt = str(directory/'result.json'); numeric_receipts[key] = receipt
            canonical.append(key)
            record = dict(case_id=key,canonical_case=key,checkpoint=case['checkpoint'],receipt=receipt,
                exact_parameter_equality='canonical original')
            torch.cuda.empty_cache()
        record.update(checkpoint_progress=state.get('progress'),
            checkpoint_progress_scope='Read from this complete checkpoint inside HOPE',
            canonical_independent_state=same is None)
        provenance = json.loads(Path(receipt).read_text()).get('readout_provenance',{})
        executed_splits = [split for split,item in provenance.items()
            if item.get('action') == 'EXECUTED_THIS_ATTEMPT'] if same is None else []
        record.update(independent_numerical_execution=bool(executed_splits),
            executed_splits_this_attempt=executed_splits,
            numerical_provenance_scope='Alias and prior completed split reuse are separate from new model execution')
        cases.append(record); atomic_json(out/'progress.json',cases)
    by_label = {case['case_id']:case for case in cases}; readouts = []
    for key,choice in binding['selected'].items():
        protocol = choice['protocol']; objective = choice['selection_objective']
        label = choice.get('readout') or f'{protocol}_val_{objective}_best'
        case = by_label.get(label) if choice.get('status') == 'VALID' else None
        for split in ['val','test']:
            readouts.append(projected_readout(label,'selected',protocol,objective,
                choice.get('checkpoint'),None if case is None else case['receipt'],split,
                None if case is None else case['canonical_case'],choice))
    for label in ['initial','preparation_before','preparation_after','final']:
        case = by_label[label]
        for protocol in POINT_PROTOCOLS:
            for split in ['val','test']:
                readouts.append(projected_readout(label,label,protocol,'N/A',case['checkpoint'],
                    case['receipt'],split,case['canonical_case']))
    atomic_json(out/'checkpoint_readouts.json',readouts)
    result = dict(status='PASS',scientific_path=scientific_path,config=config,seed=seed,
        cases=cases,canonical_cases=canonical,readouts=readouts,selection=binding['selected'],
        evaluation_scope=completed_evaluation_scope(config['cases'],cases),
        selection_frozen_before_test=True,final_freeze_receipt=config['test_release_receipt'],
        test_used_for_selection=False,
        final_freeze_epoch=release['frozen_epoch'],optimizer_trajectories=0,optimizer_steps=0,
        actual_gpu=torch.cuda.get_device_name(),
        world_size=torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1,
        world_size_source='Actual torch.distributed process group, or the uninitialized single-process entry',seconds=time.time()-began,
        model_execution='Native direct H96; no rolling-prefix inference at validation/test',
        diagnostic_metrics_affect_acceptance=False,
        scope='Complete predeclared formal readouts with exact-state aliases; no test selection')
    atomic_json(out/'result.json',result)
    return result
