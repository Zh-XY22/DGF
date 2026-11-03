import json,copy,datetime
from pathlib import Path

def materialize(training_result, data_dir, model_dir, work_dir, label, checkpoint=None, evaluation_seed=1107001, qualification=False):
    result = json.loads(Path(training_result).read_text())
    if (result.get('status') != 'PASS' and not (qualification and result.get('status') == 'DIAGNOSTIC_ONLY' and result.get('config', {}).get('qualification_only'))) or result.get('test_used') is not False:
        raise ValueError('A completed validation-only training result is required')
    c = copy.deepcopy(result['config'])
    data = Path(data_dir).expanduser().resolve()
    work = Path(work_dir).expanduser().resolve()
    protocol = json.loads((data / 'data_protocol.json').read_text())
    for field in ['dataset', 'mode', 'context', 'patch', 'horizon', 'variables', 'train_end', 'val_end', 'test_end', 'ddof', 'windows', 'source_revision', 'scaler_mean', 'scaler_std_ddof1', 'source_first_timestamp', 'source_rows']:
        if protocol[field] != c['protocol'][field]:
            raise ValueError('Data protocol differs from training: ' + field)
    if c['dataset'] != 'ETTh1' or c['protocol']['horizon'] not in [192, 336]:
        raise ValueError('This example covers only the registered ETTh1 H192/336 scope')
    selected = result['readouts'][label]
    state = Path(checkpoint or selected['checkpoint']).expanduser().resolve()
    if not state.is_file():
        raise FileNotFoundError('Supply the selected checkpoint or its explicit transported local copy')
    if work.exists() and any(work.iterdir()):
        raise FileExistsError('Use a fresh configuration output directory')
    work.mkdir(parents=True, exist_ok=True)
    source = 'dgf-qualified-public-entry-0.4-formal-v1'
    c.update(weights=str(Path(model_dir).expanduser().resolve()), artifact=str(data), checkpoint=str(state), checkpoint_source=str(state), checkpoint_kind='policy_state', checkpoint_label=label, splits=['test'], four_horizon_campaign=True, evaluation_source_reference=source, evaluation_artifact_mode='full', common_origin_diagnostics=False)
    for field in ['checkpoint_from_receipt', 'resume', 'diagnostic_limit']:
        c.pop(field, None)
    c['protocol'].update(builder_root=str(data / 'processed'), artifact=str(data))
    contract = {k: v for k, v in c['protocol'].items() if k not in ['builder_root', 'artifact']}
    model = {k: c[k] for k in ['seed', 'revision', 'route', 'K', 'J', 'G', 'force_fixed_kappa', 'method_kind', 'dgf_complete'] if k in c}
    release = dict(version=2, status='FROZEN_BEFORE_FORMAL_TEST', frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), all_validation_decisions_complete=True, source_code_reference=source, scope='Explicit short qualification subset only' if qualification else 'Completed validation MAP selection for this local checkpoint', allowed_checkpoints=[dict(path_id=c['scientific_path'], label=label, checkpoint=str(state), horizon=c['protocol']['horizon'], seed=c['seed'], data_protocol_contract=contract, model_contract=model)], point_evaluation=dict(samples=100, input_block=64, common_origin_diagnostics=False), probability_evaluation=dict(samples=1000, evaluation_seed=evaluation_seed, input_block=4))
    c.update(four_horizon_test_release=str(work / 'release.json'), test_release_receipt=str(work / 'release.json'))
    point = dict(c, evaluation_channel='point100')
    prob = dict(c, evaluation_channel='distribution1000', distribution_samples=1000, distribution_eval_seed=evaluation_seed, distribution_input_block=4)
    for name, value in [('release.json', release), ('point.json', point), ('probability.json', prob)]:
        temporary = work / (name + '.partial')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
        temporary.replace(work / name)
    return dict(status='EXPLICIT_EVALUATION_CONFIGURATIONS_WRITTEN_NOT_EXECUTED', checkpoint_label=label, files=[str(work / n) for n in ['release.json', 'point.json', 'probability.json']], scope='No claim that a transported checkpoint was independently verified; evaluator metadata and original source binding remain required')
