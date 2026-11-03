"""Portable metadata-only configuration assembly for the nine finite point/OPD arms.

This creates configurations and existing CLI commands. It neither loads weights
nor starts a job, and it cannot authorize formal tests before global validation
selection. All IO paths come from caller-supplied public data and receipts.
"""
from copy import deepcopy
import json
from pathlib import Path

from .configuration import write_config_set


ARMS = ('E01', 'E03', 'E16', 'E17', 'E15', 'R15', 'O01', 'O02', 'O03')
OPD_ARMS = ('O01', 'O02', 'O03')
CONDITIONAL_IDS = ('C15', 'CO-N-O01', 'CO-N-O02', 'CO-S-O01', 'CO-S-O02')
MODEL_ID = 'Salesforce/moirai-1.0-R-small'
FIXED_EPOCHS = {'ETTh1': {'A0': 10, 'D-pre': 10}, 'ETTm1': {'A0': 24, 'D-pre': 23}}
PROTOCOL_FIELDS = ('dataset', 'mode', 'context', 'patch', 'horizon', 'variables',
    'train_end', 'val_end', 'test_end', 'ddof', 'windows', 'scaler_mean', 'scaler_std_ddof1')
POINT_CHOICES = tuple(f'{protocol}_val_{objective}_best'
    for protocol in ('P1', 'P2', 'P3') for objective in ('MSE', 'MAE'))


def read(path):
    return json.loads(Path(path).expanduser().read_text())


def path_text(path):
    return str(Path(path).expanduser().resolve())


def check_protocol(protocol):
    dataset = protocol.get('dataset')
    expected = {'ETTh1': ('M', 5000, 64, 8640, 11520, 14400),
                'ETTm1': ('S', 4000, 128, 34560, 46080, 57600)}
    if dataset not in expected or protocol.get('horizon') != 96 or protocol.get('ddof') != 1:
        raise ValueError('The public point campaign fixes ETTh1/ETTm1, H96 and train-only ddof1')
    if tuple(protocol.get(key) for key in ('mode', 'context', 'patch', 'train_end', 'val_end', 'test_end')) != expected[dataset]:
        raise ValueError('Data organization or split differs from the frozen point protocol')
    variables = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    if protocol.get('variables') != variables:
        raise ValueError('All seven official ETT variables, in their native order, are required')
    if protocol.get('observed_zero_values_retained') is not True:
        raise ValueError('Public data metadata must preserve legitimate observed zeros')
    if any(protocol.get(key) != 1 for key in ('train_distance', 'rl_val_distance', 'test_distance')):
        raise ValueError('All input-origin strides are frozen to one')
    groups = 1 if dataset == 'ETTh1' else 7
    windows = dict(train=(protocol['train_end']-protocol['context']-96+1)*groups,
        val=(protocol['val_end']-protocol['train_end']-96+1)*groups,
        test=(protocol['test_end']-protocol['val_end']-96+1)*groups)
    if protocol.get('windows') != windows or protocol.get('eval_samples') != 100:
        raise ValueError('Declared full-window counts or evaluation samples differ')
    if not protocol.get('builder_root') or not protocol.get('scaler_mean') or not protocol.get('scaler_std_ddof1'):
        raise ValueError('Completed public data metadata with builder and training scaler is required')


def matching_protocol(config, protocol):
    if any(config['protocol'].get(key) != protocol.get(key) for key in PROTOCOL_FIELDS):
        raise ValueError('Upstream receipt uses a different model data protocol')


def completed_receipt(path, protocol, seed=None):
    record = read(path)
    if record.get('status') != 'PASS' or record.get('test_used') is not False:
        raise ValueError('A completed validation-only non-diagnostic receipt is required')
    config = record['config']
    matching_protocol(config, protocol)
    if seed is not None and config.get('seed', record.get('seed')) != seed:
        raise ValueError('Upstream seed differs from the requested scientific path')
    return record, config


def conditional_binding(arm, dataset, seed, auxiliary):
    """Bind an existing selected validation recipe; never infer a new trigger."""
    supplied = auxiliary.get('conditional_recipe')
    if arm in ARMS:
        if supplied is not None:
            raise ValueError('A mandatory method cannot be silently replaced by a conditional recipe')
        return None, None
    if arm not in CONDITIONAL_IDS or not isinstance(supplied, dict):
        raise ValueError('Only the finite conditional IDs with a validation-frozen recipe are supported')
    if supplied.get('validation_only') is not True or supplied.get('test_used') is not False:
        raise ValueError('Conditional recipe binding must explicitly be validation-only and test_used=False')
    source = supplied.get('selection_receipt')
    if not source:
        raise ValueError('A real existing conditional validation selection_receipt is required')
    record = read(source)
    if record.get('status') != 'PASS' or record.get('test_used') is not False or record.get('mandatory_paths') != 90:
        raise ValueError('Conditional selection must be a successful validation-only decision after the90 mandatory paths')
    if record.get('validation_only') is not True and record.get('kind') != 'VALIDATION_METADATA_ONLY':
        raise ValueError('Conditional selection receipt does not declare validation-only provenance')
    seeds = record.get('seed_order')
    if not isinstance(seeds, list) or len(seeds) != 5 or len(set(seeds)) != 5 or seed not in seeds:
        raise ValueError('Conditional recipe seed must belong to the five frozen validation seeds')
    selected = record.get('selected')
    if not isinstance(selected, list) or len(selected) > 4:
        raise ValueError('A conditional decision must preserve its finite selected recipe list')
    keys = [(row.get('dataset'), row.get('configuration_id')) for row in selected]
    if len(keys) != len(set(keys)) or any(sum(key[0] == ds for key in keys) > 2 for ds in ('ETTh1', 'ETTm1')):
        raise ValueError('Conditional selected list has duplicate or excessive recipes')
    chosen = [row for row in selected if row.get('dataset') == dataset and row.get('configuration_id') == arm]
    if len(chosen) != 1:
        raise ValueError('The requested conditional recipe was not selected for this dataset; no configuration is instantiated')
    chosen = deepcopy(chosen[0])
    template = 'C15' if arm == 'C15' else 'CO-N' if arm.startswith('CO-N-') else 'CO-S'
    opd = None if template == 'C15' else arm.rsplit('-', 1)[-1]
    components = ['E15', 'E17'] if template == 'C15' else [opd, 'E17' if template == 'CO-N' else 'E15']
    expected = dict(dataset=dataset, configuration_id=arm, template=template, opd_arm=opd, components=components)
    if any(chosen.get(key) != value or supplied.get(key) != value for key, value in expected.items()):
        raise ValueError('Conditional template/opd_arm/components/configuration_id differs from the actual selected validation recipe')
    objectives = chosen.get('triggering_objectives')
    if not isinstance(objectives, list) or not objectives or len(objectives) != len(set(objectives)) or not set(objectives) <= {'MSE', 'MAE'}:
        raise ValueError('Conditional trigger must retain its actual point-only validation objective')
    if chosen.get('conditional_research') is not True:
        raise ValueError('A selected conditional recipe must retain its conditional-research scope')
    return chosen, path_text(source)


def supervised_start(path, protocol, route, seed, revision):
    record, config = completed_receipt(path, protocol, seed)
    if config.get('route') != route or config.get('revision') != revision:
        raise ValueError('Shared supervised model revision or architecture differs')
    identity = (config.get('model_id') or config.get('weights') or '').rstrip('/').split('/')[-1]
    if identity != MODEL_ID.rsplit('/', 1)[-1]:
        raise ValueError('Shared supervised start is not the fixed Moirai-1.0-R-small')
    if not record.get('full_process') or not record.get('public_weight_start') or config.get('checkpoint'):
        raise ValueError('Each new-seed start must be a complete public-weight SFT/warmup')
    epochs = FIXED_EPOCHS[protocol['dataset']][route]
    if config.get('fixed_epochs') != epochs or record.get('epochs_completed') != epochs:
        raise ValueError('Shared SFT/warmup does not have the frozen complete epoch budget')
    selected = record.get('point_selection', {}).get('P1_val_MSE_best', {})
    readout = record.get('readouts', {}).get('P1_val_MSE_best', {})
    checkpoint = selected.get('checkpoint')
    if selected.get('status') != 'VALID' or checkpoint != record.get('best_checkpoint'):
        raise ValueError('Shared start must use the new validation P1-MSE choice, never historical NLL selection')
    if not checkpoint or readout.get('checkpoint') != checkpoint or not readout.get('reloaded') or not readout.get('validation', {}).get('complete'):
        raise ValueError('Shared point-selected start lacks full reloaded validation evidence')
    return record, config, checkpoint


def teacher_from_inputs(teacher_receipt, auxiliary, protocol):
    if teacher_receipt:
        record, source = completed_receipt(teacher_receipt, protocol)
        selected = record.get('point_selection', {}).get('P1_val_MSE_best', {})
        if selected.get('status') != 'VALID' or selected.get('checkpoint') != record.get('best_checkpoint'):
            raise ValueError('A teacher receipt must expose its validation-only P1-MSE checkpoint')
        config = {key: deepcopy(source[key]) for key in
            ('weights', 'revision', 'route', 'local_files_only', 'microbatch', 'protocol', 'artifact', 'dataset', 'model', 'model_id')
            if key in source}
        config.update(checkpoint=selected['checkpoint'], checkpoint_kind='policy_state', force_fixed_kappa=True)
        return config, dict(receipt=path_text(teacher_receipt), selection='Existing full validation P1-MSE; no test ranking')
    config = deepcopy(auxiliary.get('teacher_config'))
    if not isinstance(config, dict) or not auxiliary.get('teacher_validation_only') or not auxiliary.get('teacher_provenance'):
        raise ValueError('OPD requires teacher_receipt or an explicit frozen teacher_config with validation-only provenance')
    matching_protocol(config, protocol)
    if not all(config.get(key) for key in ('weights', 'checkpoint', 'route', 'revision')):
        raise ValueError('Frozen teacher weights, checkpoint, revision and architecture must be explicit')
    kind=config.get('checkpoint_kind','policy_state')
    if kind not in ('legacy','policy_state'):
        raise ValueError('Frozen teacher checkpoint format must be legacy or policy_state')
    config.update(checkpoint_kind=kind, force_fixed_kappa=True)
    return config, dict(selection='Caller supplied prior validation-only frozen choice', provenance=auxiliary['teacher_provenance'])


def make_configs(data_dir, work_dir, arm, model_id, revision, seed, auxiliary_metadata=None,
                 supervised_receipt=None, preparation_receipt=None, teacher_receipt=None, microbatch=None):
    """Generate existing executable stages; no model/data numerical execution.

    ``auxiliary_metadata`` is a JSON file. OPD needs ``opd_coefficients`` with FKL
    and RKL values from {.001,.01,.1}, ``opd_coefficient_probe`` provenance, and a
    teacher source. E15/R15 need a ``selective_artifact`` or ``selective_receipt``
    constructed from the supplied same-seed supervised start. Conditional IDs
    additionally require ``conditional_recipe`` with validation_only=True,
    test_used=False, selection_receipt, dataset, template, opd_arm, components,
    and configuration_id. Their existing shared supervised/preparation receipts
    are mandatory; only their RL is generated for execution.
    """
    if arm not in ARMS + CONDITIONAL_IDS:
        raise ValueError('Only the nine mandatory arms and the five finite selected conditional IDs are supported')
    if not revision or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError('An explicit fixed public revision and nonnegative integer seed are required')
    if str(model_id).rstrip('/').split('/')[-1] != MODEL_ID.rsplit('/', 1)[-1]:
        raise ValueError('This finite configuration generator fixes Moirai-1.0-R-small')
    data = Path(data_dir).expanduser().resolve()
    work = Path(work_dir).expanduser().resolve()
    protocol = read(data / 'data_protocol.json')
    check_protocol(protocol)
    dataset = protocol['dataset']
    route = 'A0' if arm in ('E01', 'E16') else 'D-pre'
    micro = microbatch if microbatch is not None else (16 if dataset == 'ETTh1' else 64)
    if not isinstance(micro, int) or isinstance(micro, bool) or not 1 <= micro <= 512:
        raise ValueError('Microbatch must be an integer in [1,512]; accumulation preserves the global input batch')
    auxiliary = read(auxiliary_metadata) if auxiliary_metadata else {}
    conditional, conditional_source = conditional_binding(arm, dataset, seed, auxiliary)
    if conditional is not None and (not supervised_receipt or not preparation_receipt):
        raise ValueError('Conditional RL must fork completed same-seed supervised and RL-before preparation receipts')
    preparation_opd = arm if arm in OPD_ARMS else None if conditional is None else conditional['opd_arm']
    selective_arm = 'R15' if arm == 'R15' else 'E15' if arm == 'E15' or conditional is not None and conditional['template'] in ('C15', 'CO-S') else None
    algorithm_arm = 'E15' if selective_arm else 'E03' if preparation_opd else arm
    zero_anchor = arm in ('E16', 'E17') or conditional is not None and conditional['template'] in ('C15', 'CO-N')
    scientific_path = f'point-moirai-1.0-R-small-{dataset}-seed{seed}-{arm}'
    common = dict(dataset=dataset, model='moirai-1.0-R-small', model_id=str(model_id), weights=str(model_id),
        revision=revision, local_files_only=False, protocol=deepcopy(protocol), artifact=str(data), route=route, seed=seed,
        scientific_path=scientific_path, force_fixed_kappa=True, microbatch=micro, checkpoint_interval_batches=64,
        objective_version='point-opd-v1-measure-v1', optimizer_version='historical-AdamW-full-v1',
        point_selection_only=True, test_release=False, eval_samples=100,
        point_protocols=['P1', 'P2', 'P3'], point_selection_objectives=['MSE', 'MAE'],
        precision='float32 with TF32; float64 metric accumulation', world_size=1,
        training_order='torch.randperm with seed+complete_epoch',
        validation_cadence='step0; first epoch each1/16; later epochs each1/4; repeated states cached')
    epochs = FIXED_EPOCHS[dataset][route]
    supervised = dict(common, phase='supervised', fixed_epochs=epochs, max_epochs=epochs,
        global_input_batch=64 if dataset == 'ETTh1' else 512,
        lr=1e-7 if route == 'A0' or dataset == 'ETTh1' else 5e-7,
        supervised_objective='official_PackedNLLLoss' if route == 'A0' else 'legal_trajectory_mixture_NLL',
        selection_objective='Full validation P1-MSE, fixed budget; no NLL early stopping')
    supervised_source = path_text(supervised_receipt) if supervised_receipt else str(work / 'supervised/result.json')
    supervised_checkpoint = None
    shared = {}
    if supervised_receipt:
        _, _, supervised_checkpoint = supervised_start(supervised_receipt, protocol, route, seed, revision)
        shared['supervised'] = supervised_source
    preparation = dict(common, phase='preparation', arm='E01' if route == 'A0' else 'E03', scientific_arm=arm,
        preparation='ordinary', lr=5e-7, nll_anchor_coefficient=1., global_input_batch=512,
        epochs=1, updates_per_input_batch=1,
        checkpoint_from_receipt=dict(path=supervised_source, key=['best_checkpoint']))
    teacher = None
    if preparation_opd:
        coefficients = auxiliary.get('opd_coefficients', {})
        direction = 'RKL' if preparation_opd == 'O02' else 'FKL'
        coefficient = coefficients.get(direction)
        if coefficient not in (.001, .01, .1) or not auxiliary.get('opd_coefficient_probe'):
            raise ValueError('OPD direction coefficient must be frozen from the preregistered training-only probe')
        teacher, provenance = teacher_from_inputs(teacher_receipt, auxiliary, protocol)
        preparation.update(arm=preparation_opd, preparation=preparation_opd, preparation_arm=preparation_opd, teacher_config=teacher,
            teacher_provenance=provenance, auxiliary_coefficient=coefficient,
            distillation_samples=8, opd_block_length=32, opd_blocks=3, prefix_state_samples=1,
            prefix_source='frozen_teacher' if preparation_opd == 'O03' else 'current_student',
            prefix_refresh='each optimizer update and input batch; detached local state',
            opd_coefficient_probe=auxiliary['opd_coefficient_probe'],
            preparation_replaces_ordinary=True)
    prep_source = path_text(preparation_receipt) if preparation_receipt else str(work / 'preparation/result.json')
    if preparation_receipt:
        if not supervised_receipt:
            raise ValueError('A shared preparation also requires its exact shared supervised receipt')
        record, source = completed_receipt(preparation_receipt, protocol, seed)
        if source.get('route') != route or source.get('revision') != revision or source.get('checkpoint') != supervised_checkpoint:
            raise ValueError('Prepared state does not branch from the supplied same-model supervised checkpoint')
        if record.get('epochs') != 1 or record.get('updates_per_input_batch') != 1 or record.get('global_effective_input_batch') != 512:
            raise ValueError('Prepared state must complete exactly one epoch, one update per input batch, global512')
        if source.get('phase') != 'preparation' or source.get('lr') != 5e-7 or source.get('nll_anchor_coefficient') != 1.:
            raise ValueError('Prepared state differs from the frozen preparation optimizer/anchor contract')
        if preparation_opd:
            if source.get('preparation_arm') != preparation_opd or source.get('auxiliary_coefficient') != preparation['auxiliary_coefficient']:
                raise ValueError('Ordinary or differently directed distillation cannot substitute for this OPD arm')
            if source.get('teacher_config') != teacher:
                raise ValueError('The shared OPD preparation uses a different frozen teacher')
        elif source.get('preparation') != 'ordinary':
            raise ValueError('Non-OPD controls require the shared ordinary preparation')
        final = record.get('readouts', {}).get('final', {})
        if not final.get('checkpoint') or not final.get('reloaded') or not final.get('validation', {}).get('complete'):
            raise ValueError('Shared preparation final state lacks full reloaded validation')
        shared['preparation'] = prep_source
    training = dict(common, phase='scientific', arm=arm, algorithm_arm=algorithm_arm,
        lr=5e-7, nll_anchor_coefficient=0. if zero_anchor else 1.,
        global_input_batch=512, epochs=4, updates_per_input_batch=2, J=8, G=4, K=1 if route == 'A0' else 8,
        kappa=None if route == 'A0' else 8., credit='position', PPO_clip=.2, reference_beta=.01,
        gradient_norm_clip=1., schedule='constant',
        checkpoint_from_receipt=dict(path=prep_source, key=['readouts', 'final', 'checkpoint']),
        preparation_receipt=prep_source)
    if conditional is not None:
        training.update(conditional_recipe=conditional, conditional_selection_receipt=conditional_source,
            conditional=True, conditional_validation_only=True, component_baselines=conditional['components'])
    if selective_arm:
        if not supervised_receipt:
            raise ValueError('Selective arms require the exact completed same-seed supervised start used to freeze weights')
        selective_source = auxiliary.get('selective_receipt')
        if selective_source:
            record, source = completed_receipt(selective_source, protocol, seed)
            if source.get('checkpoint') != supervised_checkpoint or not record.get('R15_has_nonuniform_treatment') in (True, False):
                raise ValueError('Selective statistics do not bind the supplied supervised start or omit R15 proof')
            artifact = record.get('artifact')
        else:
            artifact = auxiliary.get('selective_artifact')
            if auxiliary.get('selective_reference_checkpoint') != supervised_checkpoint:
                raise ValueError('Explicit selective statistics must name the exact supervised reference checkpoint')
        if not artifact:
            raise ValueError('Frozen training-only selective_artifact/receipt is required')
        training.update(selective_artifact=path_text(artifact), preparation_arm=selective_arm,
            selective_weight='random_weight.npy' if selective_arm == 'R15' else 'soft_weight.npy',
            selective_statistics_receipt=selective_source,
            reliability_used_only_during_RL=True, validation_and_test_use_all_observed=True)
    validation = dict(common, splits=['val'], checkpoint_from_receipt=dict(
        path=str(work / 'training/result.json'), key=['point_selection', 'P1_val_MSE_best', 'checkpoint']))
    configurations = dict(supervised=supervised, preparation=preparation, training=training, validation=validation)
    commands = []
    for stage, command in [('supervised', 'fit-supervised'), ('preparation', 'train'), ('training', 'train'), ('validation', 'evaluate')]:
        if stage not in shared:
            commands.append(['moirai-research', command, '--config', str(work / (stage + '.json')),
                '--output', str(work / stage), '--device', 'cuda', '--seed', str(seed)])
    values = {stage + '.json': config for stage, config in configurations.items()}
    result = dict(status='GENERATED_NOT_EXECUTED', scientific_path=scientific_path,
        commands=commands, shared_receipts=shared, formal_test_included=False,
        conditional_research=conditional is not None, conditional_recipe=conditional,
        conditional_selection_receipt=conditional_source,
        checkpoint_content_verification='Not performed by metadata assembly; executed worker verifies loaded state',
        fixed_supervised_epochs=epochs, point_readout_labels=list(POINT_CHOICES),
        preparation_budget='One complete epoch; OPD replaces ordinary preparation, never follows it',
        RL_budget='Four complete epochs, global independent input batch512, two updates per rollout',
        statistics_scope='All P1/P2/P3 point choices retained by training; this validation command reloads P1-MSE only',
        test_firewall='No generated test command or synthetic release; full registered validation choices must be frozen first')
    values['commands.json'] = result
    work.mkdir(parents=True, exist_ok=True)
    write_config_set(work, values)
    return result
