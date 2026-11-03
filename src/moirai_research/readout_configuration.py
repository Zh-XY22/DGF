"""Validation-frozen six-objective readout assembly; no model or test execution."""
from .hdfs_checkpoints import checkpoint_exists
import copy, json
from pathlib import Path
from .configuration import write_config_set
from .point_selection import POINT_PROTOCOLS, OBJECTIVES, finite_point, point_unavailability_evidence
from .point_test_gate import test_release

READOUTS = tuple(f'{p}_val_{o}_best' for p in POINT_PROTOCOLS for o in OBJECTIVES)
METRICS = ('MSE','MAE','CRPS','MAP_MSE','MAP_MAE','mixture_analytic_MSE',
    'mixture_analytic_MAE','joint_NLL_per_position','marginal_NLL')


def make_readout_configs(training_receipt, preparation_receipt, selection_receipt, work_dir, scientific_path):
    paths = [Path(path).expanduser().resolve() for path in
        (training_receipt, preparation_receipt, selection_receipt)]
    training, preparation, frozen = [json.loads(path.read_text()) for path in paths]
    test_release(paths[2], scientific_path)
    if frozen.get('status') != 'PASS' or frozen.get('test_release') is not True or frozen.get('test_used') is not False:
        raise ValueError('A global validation-frozen method/condition release is required before new tests')
    if not scientific_path: raise ValueError('An explicit scientific path identifier is required')
    for receipt, phase, epochs in [(training,'scientific',4),(preparation,'preparation',1)]:
        if receipt.get('status') != 'PASS' or receipt.get('test_used') is not False or receipt.get('epochs') != epochs:
            raise ValueError('Complete fixed-budget non-test training/preparation receipts are required')
        if receipt['config'].get('phase') != phase or receipt['config'].get('qualification_only'):
            raise ValueError('A diagnostic receipt cannot release formal tests')
    source = training['config']; before = preparation['config']; seed = source['seed']
    for key in ['seed','weights','revision','route','protocol','artifact']:
        if source.get(key) != before.get(key): raise ValueError('Preparation/training contracts differ: '+key)
    if source.get('checkpoint') != preparation['readouts']['final']['checkpoint']:
        raise ValueError('RL does not start at the supplied preparation final state')
    all_readouts = {}; case_sources = {}
    labels = ['initial', *[label for label in READOUTS if label in training['readouts']], 'final']
    for label, receipt, original in [(label,training,label) for label in labels]+[
            ('preparation_before',preparation,'initial'),('preparation_after',preparation,'final')]:
        value = receipt['readouts'][original]; validation = value['validation']
        if not value.get('reloaded') or not validation.get('complete') or validation.get('split') != 'val':
            raise ValueError('Retained readouts require complete independent reload validation')
        if validation.get('seed') != seed or validation.get('windows') != source['protocol']['windows']['val'] or validation.get('num_samples') != 100:
            raise ValueError('A validation seed/window/sample contract differs')
        if not validation.get('independent_marginal_samples'):
            raise ValueError('Engineering evaluation requires independent marginal samples')
        if not finite_point(validation.get('MSE')) or not finite_point(validation.get('MAE')):
            if point_unavailability_evidence(validation,'P2') is None:
                raise ValueError('Engineering point readout is unavailable without independent scalar evidence')
        if not checkpoint_exists(value['checkpoint']): raise ValueError('A retained checkpoint is unavailable')
        origin = paths[0] if receipt is training else paths[1]
        all_readouts[label] = dict(checkpoint=value['checkpoint'], source=str(origin),
            validation={key:validation.get(key) for key in METRICS},
            point_protocol_status=validation.get('point_protocol_status'),
            diagnostic_issues=validation.get('diagnostic_issues',{}))
        case_sources[label] = receipt['config']
    selected = {}
    for protocol, fields in POINT_PROTOCOLS.items():
        for objective in OBJECTIVES:
            label = f'{protocol}_val_{objective}_best'
            chosen = training['point_selection'][label]
            if chosen['status'] != 'VALID':
                selected[f'{protocol}_{objective}'] = dict(chosen, readout=None)
                continue
            if label not in all_readouts or chosen['checkpoint'] != all_readouts[label]['checkpoint']:
                raise ValueError('Saved global validation point choice differs from retained checkpoint')
            selected[f'{protocol}_{objective}'] = dict(chosen, readout=label,
                search_scope='Every scheduled complete validation on this fixed training trajectory')
    work = Path(work_dir).expanduser().resolve(); work.mkdir(parents=True, exist_ok=True)
    binding_path = work/'readout_selection.json'
    binding = dict(status='PASS', scientific_path=scientific_path, seed=seed, selected=selected,
        all_readouts=all_readouts, test_used=False, recipe_selection_receipt=str(paths[2]),
        training_receipt=str(paths[0]), selected_by='Each protocol MSE and MAE independently; diagnostics never gate',
        interpretation='All labels belong to the same trajectory; exact-state aliases are checked in the worker')
    cases = []
    for label, value in all_readouts.items():
        original = case_sources[label]
        case = {key:copy.deepcopy(original[key]) for key in ['weights','revision','route','protocol',
            'artifact','model','model_id','local_files_only','dataset'] if key in original}
        case.update(case_id=label, seed=seed, checkpoint=value['checkpoint'], checkpoint_kind='policy_state',
            force_fixed_kappa=True, splits=['val','test'], test_release_receipt=str(paths[2]),
            readout_selection_receipt=str(binding_path), scientific_path=scientific_path)
        cases.append(case)
    config = dict(seed=seed, scientific_path=scientific_path, cases=cases,
        test_release_receipt=str(paths[2]), readout_selection_receipt=str(binding_path))
    command = ['moirai-research','evaluate-point-readouts','--config',str(work/'readouts.json'),
        '--output',str(work/'evaluation'),'--device','cuda','--seed',str(seed)]
    result = dict(status='GENERATED_NOT_EXECUTED', commands=[command], selected=selected,
        scope='Validation metadata only; no formal numerical testing was executed', optimizer_trajectories=0)
    write_config_set(work, {'readout_selection.json':binding,'readouts.json':config,'commands.json':result})
    return result
