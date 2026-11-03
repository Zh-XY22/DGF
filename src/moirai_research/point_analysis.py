"""Scalar-only HOPE analysis for the finite point/OPD confirmation campaign.

Input JSON contains ``scientific_paths`` (all 90 mandatory paths, including
failed/missing paths), ``selection_freeze_receipt`` and optionally an
``attempt_budget_ledger`` CSV. Each path supplies experiment_id, dataset, arm,
seed, configuration and formal_test_receipt. The latter contains ``readouts``
with readout, protocol, selection_objective, checkpoint, receipt and split.
Each receipt is an evaluate_config result with results[split]. No predictions
or model weights are loaded here. This is record/statistical analysis, never a
substitute for the independent checkpoint and prediction revalidation gate.
"""
import csv
from decimal import Decimal, ROUND_HALF_UP
import json
import math
import os
from pathlib import Path
import statistics
import time

from .point_selection import POINT_PROTOCOLS, point_unavailability_evidence


CORE_METHODS = ('E01', 'E03', 'E16', 'E17', 'E15', 'R15', 'O01', 'O02', 'O03')
DATASETS = ('ETTh1', 'ETTm1')
PAPER_TARGETS = {'ETTh1': {'MSE': .364, 'MAE': .399},
                 'ETTm1': {'MSE': .280, 'MAE': .330}}
CONTRASTS = (('E16', 'E01'), ('E17', 'E03'), ('E15', 'E03'),
             ('E15', 'R15'), ('O01', 'E03'), ('O02', 'E03'),
             ('O03', 'E03'), ('O01', 'O03'), ('O01', 'O02'))
METRICS = ('MSE', 'MAE', 'CRPS', 'marginal_NLL', 'joint_NLL_per_position',
           'coverage_80', 'width_80', 'diff1_L1', 'diff2_L1', 'fft_L1',
           'boundary_change_L1', 'peak_MSE', 'turn_MSE')
T95_DF4 = 2.7764451051977987
PROTOCOL_FIELDS = ('dataset', 'mode', 'context', 'patch', 'horizon', 'variables',
                   'train_end', 'val_end', 'test_end', 'ddof', 'train_distance',
                   'rl_val_distance', 'test_distance', 'eval_samples',
                   'scaler_mean', 'scaler_std_ddof1', 'observed_zero_values_retained',
                   'missing_values', 'builder_normalization', 'model_internal_scaling')
HARDWARE_STAGES = ('training','preparation','formal_evaluation')
STAGE_HARDWARE_COLUMNS = tuple(stage+suffix for stage in HARDWARE_STAGES
    for suffix in ('_actual_gpu','_world_size','_hardware_source','_hardware_status')) + ('hardware_stage_scope',)
CHECKPOINT_COLUMNS = ('experiment_id', 'dataset', 'method', 'seed', 'phase',
    'stage', 'readout', 'split', 'protocol', 'protocol_name', 'selection_objective',
    'checkpoint', 'checkpoint_identity', 'checkpoint_identity_status',
    'checkpoint_identity_source', 'checkpoint_identity_lines', 'canonical_case', 'epoch', 'global_step',
    'independent_input_exposures', 'selection_status',
    'validation_metric', 'validation_value', 'source_receipt', 'metric_source',
    'point_result_status', 'diagnostic_status', 'metric_units', 'num_samples',
    'point_prediction_samples', 'diagnostic_samples', 'probability_score_scope',
    'eval_seed', 'MSE_statistic', 'MAE_statistic', 'actual_gpu', 'actual_gpu_scope') + STAGE_HARDWARE_COLUMNS + METRICS + (
    'raw_metrics_json', 'diagnostic_issues_json')
NONBLOCKING_GAPS = frozenset({
    'MISSING_DIAGNOSTIC_METRICS', 'DIAGNOSTIC_METRIC_ISSUE',
    'SELECTIVE_NONUNIFORM_UNKNOWN', 'SELECTIVE_TREATMENT_DEGENERATE',
    'SELECTIVE_PERMUTATION_UNCHANGED',
})


def _classify_gaps(gaps):
    """Unknown evidence gaps block acceptance; declared diagnostics do not.

    Do not infer a diagnostic exception from words such as NLL in an error:
    a nonfinite training NLL/gradient remains a correctness failure, whereas
    a missing evaluation diagnostic is explicitly identified at its source.
    Numeric paper-target decisions are computed independently of this table.
    """
    for gap in gaps:
        diagnostic = gap.get('kind') in NONBLOCKING_GAPS
        proof = gap.get('point_unavailability_evidence', {})
        independent = (gap.get('kind') == 'PROVEN_UNAVAILABLE_POINT_PROTOCOL' and isinstance(proof, dict)
            and proof.get('status') == 'PROVEN_POINT_PROTOCOL_UNAVAILABLE'
            and proof.get('protocol') in ('P2', 'P3') and bool(proof.get('issues')))
        gap['blocking'] = not (diagnostic or independent)
        gap['classification'] = ('PROVEN_INDEPENDENT_POINT_PROTOCOL_UNAVAILABLE' if independent else
            'DIAGNOSTIC_OR_IDENTIFIABILITY_ONLY' if diagnostic else 'REQUIRED_CORRECTNESS_OR_EXECUTION_EVIDENCE')
        gap['impact'] = ('Only this protocol is unavailable; valid independent point results remain usable' if independent else
                         'Retain the gap; do not veto valid point metrics or strict OR status'
                         if diagnostic else 'Blocks record acceptance until resolved; does not rewrite raw point values')
    return gaps


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _json(value):
    return json.dumps(_clean(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(_json(value) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def _table(path, rows, columns=None):
    columns = list(columns or dict.fromkeys(k for row in rows for k in row))
    temporary = Path(path).with_name(Path(path).name + f'.tmp-{os.getpid()}')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='raise')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _json(v) if isinstance(v, (dict, list, tuple)) else _clean(v)
                             for k, v in row.items()})
    os.replace(temporary, path)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _point(value):
    return _number(value) and value >= 0


def _read(path, base, gaps, owner, required=True):
    if isinstance(path, dict):
        return path
    if not path:
        if required:
            gaps.append(dict(experiment_id=owner, kind='MISSING_REFERENCE', source='', detail='Required receipt reference absent'))
        return {}
    source = Path(path)
    if not source.is_absolute():
        source = base / source
    try:
        value = json.loads(source.read_text())
        if not isinstance(value, dict):
            raise ValueError('Receipt must be a JSON object')
        return value
    except (OSError, ValueError) as error:
        gaps.append(dict(experiment_id=owner, kind='UNREADABLE_RECEIPT', source=str(source), detail=str(error)))
        return {}


def path_hardware_fields(path):
    result={}
    for stage in HARDWARE_STAGES:
        for suffix in ('_actual_gpu','_world_size','_hardware_source','_hardware_status'):
            value=path.get(stage+suffix)
            result[stage+suffix]=value if value is not None else None if suffix=='_hardware_source' else 'UNKNOWN'
    result['hardware_stage_scope']=path.get('hardware_stage_scope',
        'Accepted result-producing stage attempt only; resumed trajectories may span accelerators. Exact attempt evidence remains separate.')
    return result


def _hardware_key(gpu,world_size):
    unknown=(None,'','UNKNOWN','N/A','INVALID')
    if isinstance(gpu,str):
        if gpu.strip().upper() in unknown:return None
        gpu_key=(gpu,)
    elif isinstance(gpu,(list,tuple)) and gpu and all(isinstance(value,str) and value.strip().upper() not in unknown for value in gpu):
        gpu_key=tuple(sorted(gpu))
    else:return None
    if not isinstance(world_size,int) or isinstance(world_size,bool) or world_size<=0:return None
    return gpu_key,world_size


def paired_hardware_fields(method_path,baseline_path,method_readout=None,baseline_readout=None):
    """Descriptive stage matching, never a scientific-pair eligibility gate."""
    method_path=method_path or {};baseline_path=baseline_path or {};method_readout=method_readout or {};baseline_readout=baseline_readout or {}
    result=dict(method_actual_gpu=method_readout.get('actual_gpu'),baseline_actual_gpu=baseline_readout.get('actual_gpu'),
        actual_gpu_scope='Compatibility fields method_actual_gpu/baseline_actual_gpu describe the selected numerical readout evaluation receipt, never training hardware',
        hardware_match_scope='Compatibility hardware_match aliases accepted formal-evaluation stage GPU-model plus world-size matching; not complete resumed-trajectory or precision equivalence')
    for stage in HARDWARE_STAGES:
        for name,path in [('method',method_path),('baseline',baseline_path)]:
            for suffix in ('_actual_gpu','_world_size','_hardware_source','_hardware_status'):
                value=path.get(stage+suffix)
                result[name+'_'+stage+suffix]=value if value is not None else None if suffix=='_hardware_source' else 'UNKNOWN'
        left=_hardware_key(method_path.get(stage+'_actual_gpu'),method_path.get(stage+'_world_size'))
        right=_hardware_key(baseline_path.get(stage+'_actual_gpu'),baseline_path.get(stage+'_world_size'))
        name='evaluation' if stage=='formal_evaluation' else stage
        result[name+'_hardware_match']=left==right if left is not None and right is not None else None
        result[name+'_hardware_match_status']=('UNKNOWN_INCOMPLETE_STAGE_RECEIPT' if left is None or right is None else
            'MATCH_RECORDED_GPU_MODEL_AND_WORLD_SIZE' if left==right else 'DIFFERENT_RECORDED_GPU_MODEL_OR_WORLD_SIZE')
    result['hardware_match']=result['evaluation_hardware_match']
    result['resumption_hardware_scope']='Each stage describes its accepted result-producing attempt; prior failures/resumptions may use other GPUs. Inspect per-attempt worker receipts before claiming an entire trajectory hardware match.'
    return result


def worker_reference_summary(rows):
    keys=[];unbound=0
    for row in rows:
        node=row.get('node_id');attempt=row.get('attempt');run=row.get('run_id')
        if not node or node=='UNKNOWN' or attempt is None:
            unbound+=1;continue
        keys.append((node,str(attempt),run if run not in ('','UNKNOWN') else None))
    unique=set(keys)
    return dict(path_stage_attempt_reference_rows=len(rows),registered_attempt_reference_rows=len(keys),
        unique_node_attempt_run_keys=len(unique),unique_attempts_with_run_id=sum(key[2] is not None for key in unique),
        repeated_shared_attempt_reference_rows=len(keys)-len(unique),unbound_placeholder_reference_rows=unbound,
        scope='Reference rows are keyed by scientific path and stage; the unique (node,attempt,run) count deduplicates shared references. Neither count is independent scientific paths, seeds, or proof of optimizer execution.')


def conditional_budget_audit(config,paths,seeds,freeze,ledger,base,out,gaps):
    """Optional budget omissions are explicit planned slots, never failed science."""
    source=config.get('conditional_budget_admission_receipt')
    admission=_read(source,base,gaps,'OPTIONAL_BUDGET')
    decision=_read(admission.get('validation_decision_receipt'),base,gaps,'OPTIONAL_BUDGET') if admission else {}
    groups=admission.get('groups',[]);by={path['experiment_id']:path for path in paths}
    wanted={(row['dataset'],row['configuration_id']) for row in decision.get('selected',[])}
    recorded={(row.get('dataset'),row.get('configuration_id')) for row in groups}
    admitted=[row for row in groups if row.get('status')=='ADMITTED']
    canceled=[row for row in groups if row.get('status')=='TRIGGERED_NOT_EXECUTED_BUDGET_LIMIT']
    admitted_ids={key for row in admitted for key in row.get('science_ids',[])}
    canceled_ids={key for row in canceled for key in row.get('science_ids',[])+row.get('test_ids',[])}
    ledger_ids={row.get('node_id') for row in ledger}
    checks=dict(raw_validation_decision_is_validation_only=decision.get('status')=='PASS' and decision.get('test_used') is False,
        pretest_frozen=admission.get('status')=='PASS' and admission.get('pretest_frozen') is True and admission.get('test_used') is False,
        every_validation_trigger_resolved=wanted==recorded and len(recorded)==len(groups),
        outcomes_are_admitted_or_budget_canceled=len(admitted)+len(canceled)==len(groups),
        all_groups_have_complete_frozen_seed_slots=all(row.get('seeds')==list(seeds) and len(row.get('science_ids',[]))==5
            and len(set(row['science_ids']))==5 and row.get('test_ids')==['test-'+key for key in row['science_ids']] for row in groups),
        admitted_slots_are_registered_science=all(key in by for key in admitted_ids),
        admitted_path_bindings_match_group=all(identifier in by and by[identifier]['dataset']==row['dataset']
            and by[identifier]['arm']==row['configuration_id'] and by[identifier]['seed']==seed
            for row in admitted for seed,identifier in zip(row.get('seeds',[]),row.get('science_ids',[]))),
        actual_optional_science_equals_admission={path['experiment_id'] for path in paths if path['arm'] not in CORE_METHODS}==admitted_ids,
        canceled_slots_not_registered_or_attempted=not(canceled_ids & (set(by)|ledger_ids)),
        canceled_slots_have_no_optimizer_promise=all(row.get('optimizer_commitment')==0 and row.get('scientific_nodes_permitted') is False for row in canceled),
        frozen_release_resolves_optional_budget=freeze.get('conditional_budget_resolution_complete') is True
            and freeze.get('conditional_budget_admission_receipt')==source)
    slot_rows=[]
    for row in groups:
        for seed,identifier in zip(row.get('seeds',[]),row.get('science_ids',[])):
            actual=by.get(identifier);registered=actual is not None
            slot_rows.append(dict(planned_experiment_id=identifier,dataset=row.get('dataset'),method=row.get('configuration_id'),seed=seed,
                group_id=row.get('group_id'),budget_status=row.get('status'),validation_rule_triggered=True,
                registered_scientific_path=registered,expected_only=not registered,
                actual_status=actual.get('status') if actual else 'NOT_INSTANTIATED_BUDGET_LIMIT',
                formal_test_receipt=actual.get('formal_test_receipt') if actual else None,
                result_status='See actual configuration inventory and checkpoint results' if actual else 'N/A_NOT_EXECUTED',
                checkpoint=None,MSE=None,MAE=None,source_receipt=source,
                reason=row.get('reason'),scope='Planned optional slot, not an extra scientific result or an unsubmitted mandatory path'))
    _table(out/'conditional_budget_decisions.csv',groups,
        list(dict.fromkeys(key for row in groups for key in row)) if groups else ['group_id','dataset','configuration_id','status','validation_rule_triggered','reason'])
    _table(out/'conditional_budget_slot_inventory.csv',slot_rows,
        list(slot_rows[0]) if slot_rows else ['planned_experiment_id','dataset','method','seed','budget_status','registered_scientific_path','expected_only','result_status'])
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',source_receipt=source,checks=checks,
        triggered_groups=len(groups),admitted_groups=len(admitted),budget_canceled_groups=len(canceled),
        admitted_scientific_slots=len(admitted_ids),budget_canceled_scientific_slots=sum(len(row.get('science_ids',[])) for row in canceled),
        scope='Only actual admitted configurations enter the scientific denominator; budget-canceled optional slots are neither untriggered, scientific failures nor fabricated five-seed results')
    if result['status']!='PASS':
        gaps.append(dict(experiment_id='OPTIONAL_BUDGET',kind='CONDITIONAL_BUDGET_RECORD_CONFLICT',source=source or '',
            detail='Optional validation/admission/registration mismatch: '+', '.join(key for key,value in checks.items() if not value)))
    _atomic_json(out/'conditional_budget_audit.json',result)
    return result


def historical_replay_audit(config, base, out, gaps):
    """Keep retrospective comparisons separate from new validation selection."""
    comparisons, sources = [], []
    for reference in config.get('historical_replay_receipts', []):
        receipt = _read(reference.get('acceptance'), base, gaps, 'HISTORICAL_'+reference['dataset'])
        cases=receipt.get('cases', [])
        unchanged=(bool(cases) and receipt.get('selection_recomputed') is False
            and all(case.get('old_selection_preserved') is True and case.get('selection_recomputed') is False for case in cases))
        no_test_selection=receipt.get('test_used_for_selection') is False
        sources.append(dict(dataset=reference['dataset'], status=receipt.get('status', 'UNKNOWN'),
            registered_status=reference.get('status', 'UNKNOWN'),
            acceptance=reference.get('acceptance'), run_id=reference.get('run_id'),
            requested_cases=receipt.get('requested_cases'), canonical_unique_cases=receipt.get('canonical_unique_cases'),
            independent_numerical_execution=receipt.get('executed_unique_cases'),
            old_selection_preserved=unchanged, new_selection_used=receipt.get('test_used_for_selection'),
            selection_provenance_status='RECORDED_UNCHANGED' if unchanged and no_test_selection else 'UNKNOWN_OR_CONFLICT'))
        if receipt and not (unchanged and no_test_selection):
            gaps.append(dict(experiment_id='HISTORICAL_'+reference['dataset'],
                kind='HISTORICAL_SELECTION_PROVENANCE_UNKNOWN_OR_CONFLICT',source=reference.get('acceptance') or '',
                detail='Actual replay receipt does not explicitly preserve every old choice and exclude test-based/new selection; report does not invent those flags'))
        for case in receipt.get('cases', []):
            for value in case.get('comparison', []):
                metric = value.get('metric', '')
                protocol = 'P1' if metric.startswith('MAP_') else 'P3' if metric.startswith('mixture_analytic_') else 'P2'
                comparisons.append(dict(dataset=reference['dataset'], case_id=case['case_id'], seed=case['seed'],
                    protocol=protocol, checkpoint=case.get('checkpoint'), roles=case.get('roles'),
                    numerical_receipt=case.get('receipt'), prior_receipt=case.get('prior_numerical_receipt'), **value))
    groups = []
    for dataset in DATASETS:
        for protocol in ('P1', 'P2', 'P3'):
            for split in ('val', 'test'):
                subset = [r for r in comparisons if (r['dataset'], r['protocol'], r['split']) == (dataset, protocol, split)]
                groups.append(dict(dataset=dataset, protocol=protocol, split=split, comparisons=len(subset),
                    matches=sum(r['status']=='MATCH' for r in subset), conflicts=sum(r['status']=='CONFLICT' for r in subset),
                    unavailable=sum(r['status']=='UNAVAILABLE' for r in subset),
                    maximum_absolute_delta=max((abs(r['delta']) for r in subset if _number(r.get('delta'))), default=None)))
    _table(out/'historical_replay_comparisons.csv', comparisons,
        list(dict.fromkeys(key for row in comparisons for key in row)) if comparisons else ['dataset', 'case_id', 'seed', 'protocol', 'split', 'metric', 'previous', 'reloaded', 'delta', 'status'])
    _table(out/'historical_replay_summary.csv', groups)
    _atomic_json(out/'historical_replay_sources.json', sources)
    return groups


def worker_record_audit(paths, base, out, gaps):
    """Inspect small actual-worker receipts; failed startup is not successful execution."""
    rows = []
    for path in paths:
        for reference in path.get('worker_execution_evidence', []):
            row = dict(experiment_id=path['experiment_id'], **reference)
            loaded = reference.get('loaded_source_receipt')
            records = None
            if loaded:
                source = Path(loaded); source = source if source.is_absolute() else base/source
                try:
                    records = json.loads(source.read_text())
                except (OSError, ValueError) as error:
                    row['loaded_receipt_error'] = str(error)
            verified = (isinstance(records, list) and bool(records)
                and all(isinstance(r, dict) and r.get('content_compared') is True
                        and r.get('verification') == 'frozen_archive_byte_comparison' for r in records))
            # A selected successful receipt can identify its attempt even if a Native
            # status was not preserved. Earlier failed attempts are retained as gaps
            # in their own rows but do not veto a later evidenced successful retry.
            stage = reference.get('stage')
            selected_receipt = path.get({'training':'training_receipt','preparation':'preparation_receipt',
                'formal_evaluation':'formal_test_receipt','shared_SFT':'shared_start_receipt'}.get(stage, ''))
            receipt_parts = Path(selected_receipt).parts if isinstance(selected_receipt,str) else ()
            selected_attempt = bool(reference.get('node_id') in receipt_parts and reference.get('attempt') is not None
                and 'attempt-'+str(reference['attempt']) in receipt_parts)
            successful_attempt = reference.get('native_state') == 'SUCCEEDED' or selected_attempt
            missing = []
            for field in ('loaded_source_receipt','runtime_receipt','frozen_source_receipt'):
                value=reference.get(field)
                file=Path(value) if isinstance(value,str) and value else None
                if file is not None and not file.is_absolute():file=base/file
                if file is None or not file.is_file():missing.append(field)
            row.update(actual_worker_source_members=len(records) if isinstance(records, list) else None,
                worker_byte_comparison_record=verified, audit_kind='EXISTING_SMALL_WORKER_RECORD_INSPECTION',
                successful_attempt_evidence_required=successful_attempt,selected_success_receipt_matches_attempt=selected_attempt,
                missing_actual_worker_references=missing,model_numerical_revalidation=False)
            if successful_attempt and not verified:
                gaps.append(dict(experiment_id=path['experiment_id'], kind='MISSING_ACTUAL_WORKER_SOURCE_PROOF',
                    source=loaded or '', detail='Successful attempt lacks its actual pre-entry frozen-source comparison receipt; current files are not a substitute'))
            missing_runtime = [name for name in missing if name!='loaded_source_receipt']
            if successful_attempt and missing_runtime:
                gaps.append(dict(experiment_id=path['experiment_id'],kind='MISSING_SUCCESSFUL_WORKER_EXECUTION_REFERENCE',
                    source=selected_receipt or loaded or '',
                    detail='Successful attempt is missing actual small receipt reference(s): '+', '.join(missing_runtime)+'; earlier failed startups are not used as a substitute'))
            rows.append(row)
    _table(out/'worker_execution_evidence.csv', rows,
        list(dict.fromkeys(k for row in rows for k in row)) if rows else ['experiment_id', 'stage', 'node_id', 'attempt', 'run_id', 'source_status'])
    _atomic_json(out/'worker_execution_reference_summary.json',worker_reference_summary(rows))
    return rows


def summary(values, expected_seeds):
    """Use seeds as replicates; never treat windows or readouts as seeds."""
    finite = [(s, v) for s, v in values if _number(v)]
    seeds = [s for s, _ in finite]
    if len(seeds) != len(set(seeds)):
        raise ValueError('Duplicate seed in summary')
    numbers = [v for _, v in finite]
    return dict(valid_seeds=len(numbers), seed_order=seeds, per_seed_values=numbers,
                mean=statistics.fmean(numbers) if numbers else None,
                sample_std=statistics.stdev(numbers) if len(numbers) > 1 else None,
                minimum=min(numbers) if numbers else None, maximum=max(numbers) if numbers else None,
                complete_five_seeds=len(expected_seeds) == 5 and set(seeds) == set(expected_seeds))


def paper_decision(dataset, mse_values, mae_values, expected_seeds):
    """Strict unrounded OR and decimal display agreement are distinct decisions."""
    target = PAPER_TARGETS[dataset]
    mse = summary([(s, v) for s, v in mse_values if _point(v)], expected_seeds)
    mae = summary([(s, v) for s, v in mae_values if _point(v)], expected_seeds)
    strict_mse = mse['complete_five_seeds'] and mse['mean'] <= target['MSE']
    strict_mae = mae['complete_five_seeds'] and mae['mean'] <= target['MAE']
    def shown(stats, name):
        return bool(stats['complete_five_seeds'] and
            Decimal(str(stats['mean'])).quantize(Decimal('.001'), rounding=ROUND_HALF_UP)
            == Decimal(str(target[name])).quantize(Decimal('.001'), rounding=ROUND_HALF_UP))
    strict = strict_mse or strict_mae
    display = not strict and (shown(mse, 'MSE') or shown(mae, 'MAE'))
    enough = mse['complete_five_seeds'] or mae['complete_five_seeds']
    return dict(MSE_mean=mse['mean'], MAE_mean=mae['mean'],
        MSE_sample_std=mse['sample_std'], MAE_sample_std=mae['sample_std'],
        MSE_seed_order=mse['seed_order'], MAE_seed_order=mae['seed_order'],
        MSE_per_seed_values=mse['per_seed_values'], MAE_per_seed_values=mae['per_seed_values'],
        MSE_valid_seeds=mse['valid_seeds'], MAE_valid_seeds=mae['valid_seeds'],
        target_MSE=target['MSE'], target_MAE=target['MAE'],
        MSE_gap=mse['mean']-target['MSE'] if mse['mean'] is not None else None,
        MAE_gap=mae['mean']-target['MAE'] if mae['mean'] is not None else None,
        MSE_relative_gap=mse['mean']/target['MSE']-1 if mse['mean'] is not None else None,
        MAE_relative_gap=mae['mean']/target['MAE']-1 if mae['mean'] is not None else None,
        strict_MSE=bool(strict_mse), strict_MAE=bool(strict_mae), strict_OR=bool(strict),
        display_precision_tie_only=bool(display), rounding='ROUND_HALF_UP to three decimals',
        numeric_status='STRICT_OR_REACHED' if strict else 'DISPLAY_PRECISION_TIE_ONLY' if display
            else 'NOT_REACHED' if enough else 'INSUFFICIENT_FIVE_SEED_EVIDENCE',
        diagnostics_used_for_decision=False)


def paired_summary(rows, expected_seeds):
    finite = [row for row in rows if _number(row.get('delta'))]
    stats = summary([(row['seed'], row['delta']) for row in finite], expected_seeds)
    complete = stats['complete_five_seeds']
    radius = T95_DF4*stats['sample_std']/math.sqrt(5) if complete else None
    improvement_flags = [row.get('improved', row['delta'] < 0) for row in finite]
    return dict(**stats, improvement_seed_count=sum(flag is True for flag in improvement_flags)
                    if any(flag is not None for flag in improvement_flags) else None,
                tied_seed_count=sum(row['delta'] == 0 for row in finite),
                mean_difference_t95_lower=stats['mean']-radius if complete else None,
                mean_difference_t95_upper=stats['mean']+radius if complete else None,
                interval_method='Paired-seed mean difference, Student t df=4, descriptive'
                    if complete else 'NOT_COMPUTED: requires all five frozen seeds',
                significance_gate=False)


def _metrics(raw, protocol):
    fields = POINT_PROTOCOLS[protocol]
    prefix = {'P1': 'MAP', 'P2': 'mean', 'P3': 'mixture_analytic'}[protocol]
    answer = {name: raw.get(name) for name in METRICS}
    answer['MSE'] = raw.get(fields['MSE'])
    answer['MAE'] = raw.get(fields['MAE'])
    for name in ('diff1_L1', 'diff2_L1', 'fft_L1', 'boundary_change_L1', 'peak_MSE', 'turn_MSE'):
        answer[name] = raw.get(prefix+'_'+name)
    return {key: value if _number(value) else None for key, value in answer.items()}


def _configuration(path, base, gaps):
    value = path.get('configuration', path.get('config', {}))
    return _read(value, base, gaps, path['experiment_id']) if value else {}


def _stage(item):
    stage = item.get('stage')
    if stage:
        return stage
    name = item.get('readout', '')
    if 'val_' in name or name in {f'{p}_{o}' for p in POINT_PROTOCOLS for o in ('MSE', 'MAE')}:
        return 'selected'
    return name if name in ('initial', 'preparation_before', 'preparation_after', 'final') else 'UNKNOWN'


def _checkpoint_load_journal(reference, base):
    """Index a formal worker's existing load log; never open a checkpoint.

    Each original path and remote URI is retained exactly as written. A
    canonical parameter-state alias is deliberately not a file-byte alias.
    """
    if isinstance(reference, dict):
        source = reference.get('checkpoint_identity_log')
        source = Path(source) if source else None
        if source is not None and not source.is_absolute():
            source = base/source
    elif reference:
        receipt = Path(reference)
        if not receipt.is_absolute():
            receipt = base/receipt
        source = receipt.parents[1]/'checkpoint_identities.jsonl'
    else:
        source = None
    journal = dict(status='UNKNOWN', source=str(source) if source is not None else None,
                   by_reference={}, errors=[])
    if source is None:
        journal['errors'].append('Formal receipt has no worker load-log reference')
        return journal
    try:
        with source.open(encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('Load-log row is not a JSON object')
                    if row.get('operation') != 'actual_load_bytes':
                        continue
                    entry = dict(record=row, source_line=line_number)
                    for key in ('path', 'remote_uri'):
                        if isinstance(row.get(key), str) and row[key]:
                            journal['by_reference'].setdefault(row[key], []).append(dict(entry, matched_field=key))
                except ValueError as error:
                    journal['errors'].append(dict(line=line_number, detail=str(error)))
        journal['status'] = 'PARSED' if not journal['errors'] else 'MALFORMED_LOG'
    except OSError as error:
        journal['errors'].append(str(error))
    return journal


def _checkpoint_load_identity(checkpoint, journal, owner, gaps, required=True):
    """Join only recorded actual-load bytes using the exact checkpoint path."""
    answer = dict(status='UNKNOWN', checkpoint=checkpoint, source=journal.get('source'), source_lines=[],
                  operation='actual_load_bytes', content_read_by_analysis=False,
                  identity_scope='Serialized file bytes; policy-parameter equality is a separate relation')
    records = journal.get('by_reference', {}).get(str(checkpoint), []) if checkpoint else []
    signatures = set()
    valid = []
    for entry in records:
        row = entry['record']
        value = row.get('sha256')
        size = row.get('bytes')
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in '0123456789abcdefABCDEF' for character in value)
                or not isinstance(size, int) or isinstance(size, bool) or size <= 0):
            continue
        signatures.add((value.lower(), size))
        valid.append(entry)
    if journal.get('status') == 'PARSED' and records and len(valid) == len(records) and len(signatures) == 1:
        value, size = next(iter(signatures))
        answer.update(status='ACTUAL_LOAD_RECORDED', sha256=value, bytes=size,
            source_lines=sorted({row['source_line'] for row in valid}),
            observations=[dict(path=row['record'].get('path'), remote_uri=row['record'].get('remote_uri'),
                epoch=row['record'].get('epoch'), source_line=row['source_line'], matched_field=row['matched_field'])
                for row in valid])
    elif len(signatures) > 1:
        answer.update(status='CONFLICTING_LOAD_RECORDS', source_lines=sorted({row['source_line'] for row in records}),
                      detail='The exact checkpoint reference has multiple recorded serialized identities; no last-record substitution')
    else:
        answer['detail'] = ('No complete, unambiguous actual-load record for this exact checkpoint reference; '
                            'save records, file size and canonical parameter aliases were not substituted')
    if required and answer['status'] != 'ACTUAL_LOAD_RECORDED':
        gaps.append(dict(experiment_id=owner, kind='CHECKPOINT_ACTUAL_LOAD_IDENTITY_NOT_PROVEN',
            source=journal.get('source') or 'UNKNOWN', detail=f'{checkpoint}: '+answer.get('detail', 'Required byte provenance absent')))
    return answer


def _extract(path, base, gaps, configuration):
    identifier = path['experiment_id']
    if path.get('scientific_failure_terminal') is True and not path.get('formal_test_receipt'):
        _read(path.get('failure_receipt'), base, gaps, identifier)
        return []
    receipt = _read(path.get('formal_test_receipt'), base, gaps, identifier)
    binding = {}
    if receipt:
        if receipt.get('selection_frozen_before_test') is not True:
            gaps.append(dict(experiment_id=identifier, kind='TEST_SELECTION_FIREWALL_NOT_PROVEN',
                             source=path.get('formal_test_receipt', ''), detail='Formal receipt does not prove validation-frozen selection'))
        binding = _read(receipt.get('config', {}).get('readout_selection_receipt'), base, gaps, identifier)
    items = receipt.get('readouts', path.get('readouts', []))
    if isinstance(items, dict):
        items = [dict(value, readout=name) for name, value in items.items()]
    rows = []
    load_journal = _checkpoint_load_journal(path.get('formal_test_receipt'), base)
    identity_cache = {}
    for item in items:
        protocol_values = [item['protocol']] if item.get('protocol') in POINT_PROTOCOLS else list(POINT_PROTOCOLS)
        split = item.get('split', 'test')
        detail = _read(item.get('receipt'), base, gaps, identifier, required=item.get('selection_status') != 'INVALID')
        raw = item.get('metrics') or detail.get('results', {}).get(split)
        if raw is None:
            source_readout = item.get('source_readout', item.get('readout'))
            raw = detail.get('readouts', {}).get(source_readout, {}).get('validation', {}) if split == 'val' else {}
        if not raw and item.get('selection_status') != 'INVALID':
            gaps.append(dict(experiment_id=identifier, kind='MISSING_METRICS', source=item.get('receipt', ''),
                             detail=f"No exact metrics for {item.get('readout')}/{split}; another checkpoint was not substituted"))
        if raw:
            expected_windows = configuration.get('protocol', {}).get('windows', {}).get(split)
            if raw.get('complete') is not True or raw.get('split') != split or (
                    expected_windows is not None and raw.get('windows') != expected_windows):
                gaps.append(dict(experiment_id=identifier, kind='INCOMPLETE_OR_MISMATCHED_SPLIT',
                    source=item.get('receipt', ''), detail=f'Expected complete {split} with {expected_windows} windows'))
        bound = binding.get('all_readouts', {}).get(item.get('readout'), {})
        if item.get('selection_status') != 'INVALID' and bound.get('checkpoint') != item.get('checkpoint'):
            gaps.append(dict(experiment_id=identifier, kind='CHECKPOINT_BINDING_MISMATCH',
                source=receipt.get('config', {}).get('readout_selection_receipt', ''),
                detail=f"Readout {item.get('readout')} does not match its exact frozen checkpoint reference"))
        case = next((case for case in receipt.get('cases', []) if case.get('case_id') == item.get('readout')), {})
        progress = case.get('checkpoint_progress') or {}
        identity_key = (item.get('checkpoint'), item.get('selection_status') != 'INVALID')
        if identity_key not in identity_cache:
            identity_cache[identity_key] = _checkpoint_load_identity(item.get('checkpoint'), load_journal, identifier, gaps,
                required=bool(item.get('checkpoint')) and item.get('selection_status') != 'INVALID')
        checkpoint_identity = identity_cache[identity_key]
        for protocol in protocol_values:
            values = _metrics(raw, protocol)
            stats = dict(MSE_statistic='independent 100-sample mean', MAE_statistic='independent 100-sample median')
            if protocol != 'P2':
                value = 'selected-mode legal analytic mean' if protocol == 'P1' else 'full-mixture legal analytic mean'
                stats = dict(MSE_statistic=value, MAE_statistic=value)
            row = dict(experiment_id=identifier, dataset=path['dataset'], method=path['arm'], seed=path['seed'],
                phase=path.get('phase', 'core'), stage=_stage(item), readout=item.get('readout', 'UNKNOWN'),
                split=split, protocol=protocol, protocol_name=POINT_PROTOCOLS[protocol]['name'],
                selection_objective=item.get('selection_objective', item.get('objective', 'N/A')),
                checkpoint=item.get('checkpoint'), checkpoint_identity=checkpoint_identity,
                checkpoint_identity_status=checkpoint_identity['status'], checkpoint_identity_source=checkpoint_identity['source'],
                checkpoint_identity_lines=checkpoint_identity['source_lines'],
                canonical_case=item.get('canonical_case', 'UNKNOWN'),
                epoch=progress.get('epoch', progress.get('raw_epoch')), global_step=progress.get('global_step'),
                independent_input_exposures=progress.get('independent_input_exposures'),
                selection_status=item.get('selection_status', 'UNKNOWN'),
                validation_metric=item.get('validation_metric'), validation_value=item.get('validation_value'),
                source_receipt=item.get('receipt', ''), metric_source=item.get('metric_source', f'results.{split}'),
                point_result_status='VALID' if _point(values['MSE']) and _point(values['MAE']) else 'INVALID_OR_MISSING',
                diagnostic_status='COMPLETE' if all(_number(values[k]) for k in METRICS if k not in ('MSE', 'MAE')) else 'DECLARED_GAPS',
                metric_units=raw.get('units', 'UNKNOWN'), num_samples=raw.get('num_samples'), eval_seed=raw.get('seed'),
                point_prediction_samples=raw.get('num_samples') if protocol == 'P2' else 0,
                diagnostic_samples=raw.get('num_samples'),
                probability_score_scope='CRPS, NLL and interval calibration score the complete checkpoint mixture distribution; not the deterministic P1/P3 output or a selected-mode conditional distribution',
                actual_gpu=detail.get('actual_gpu', 'UNKNOWN'),
                actual_gpu_scope='Exact numerical readout receipt evaluation GPU; not the training device',
                **path_hardware_fields(path),raw_metrics_json=raw,
                diagnostic_issues_json=raw.get('diagnostic_issues', []), **stats, **values)
            if (protocol in ('P1', 'P2') and item.get('selection_status') != 'INVALID'
                    and row['point_result_status'] != 'VALID'):
                proof = point_unavailability_evidence(raw, protocol)
                gaps.append(dict(experiment_id=identifier,
                    kind='PROVEN_UNAVAILABLE_POINT_PROTOCOL' if proof else 'INVALID_REQUIRED_POINT_METRICS',
                    point_unavailability_evidence=proof,
                    source=item.get('receipt', ''), detail=f'{item.get("readout")}/{split}/{protocol}: expected valid MSE/MAE, '
                    'but a point value is absent, negative or nonfinite; independent evidence='+_json(proof)))
            missing_diagnostics = [key for key in METRICS if key not in ('MSE', 'MAE') and values[key] is None]
            if missing_diagnostics:
                gaps.append(dict(experiment_id=identifier, kind='MISSING_DIAGNOSTIC_METRICS',
                    source=item.get('receipt', ''), detail=f'{item.get("readout")}/{split}/{protocol}: absent or nonfinite diagnostic values: '
                    +', '.join(missing_diagnostics)))
            if raw.get('diagnostic_issues'):
                gaps.append(dict(experiment_id=identifier, kind='DIAGNOSTIC_METRIC_ISSUE',
                    source=item.get('receipt', ''), detail=f'{item.get("readout")}/{split}/{protocol}: '+_json(raw['diagnostic_issues'])))
            rows.append(row)
    return rows


def _recipes(freeze):
    recipes = freeze.get('deliveries', freeze.get('recipes', []))
    if not isinstance(recipes, list):
        return [], 'UNKNOWN_RECIPE_SCHEMA'
    answer = []
    for item in recipes:
        row = dict(item)
        row['selection_objective'] = row.get('selection_objective', row.get('objective'))
        row['method'] = row.get('method', row.get('arm', row.get('recipe_id')))
        answer.append(row)
    return answer, 'READ_FROZEN_RECIPES_NO_TEST_SELECTION'


def _frozen(recipe, dataset, method, protocol, objective):
    return any(row.get('dataset') == dataset and row.get('method') == method
               and row.get('protocol') == protocol and row.get('selection_objective') == objective for row in recipe)


def _format(value):
    return f'{value:.9f}' if _number(value) else 'N/A'


def _markdown(headers, rows):
    def cell(value):
        return str(value if value is not None else 'N/A').replace('|', '\\|').replace('\n', ' ')
    return ['| '+' | '.join(map(cell, headers))+' |', '| '+' | '.join('---' for _ in headers)+' |'] + [
        '| '+' | '.join(cell(x) for x in row)+' |' for row in rows]


def _protocol_key(configuration):
    protocol = configuration.get('protocol', {})
    return _json({key: protocol.get(key) for key in PROTOCOL_FIELDS})


def _comparison_compatible(method, baseline, configurations):
    left, right = configurations[method['experiment_id']], configurations[baseline['experiment_id']]
    if not left.get('protocol') or not right.get('protocol'):
        return False, 'Missing data protocol; equality cannot be inferred from dataset name'
    for key in ('model', 'objective_version', 'optimizer_version'):
        if left.get(key) and right.get(key) and left[key] != right[key]:
            return False, f'{key} differs'
    if _protocol_key(left) != _protocol_key(right):
        return False, 'Data protocol differs or is missing asymmetrically'
    return True, 'Same dataset/seed/protocol/selection objective; declared treatment differences retained'


def _load_ledger(reference, base, gaps):
    if not reference:
        return []
    source = Path(reference)
    if not source.is_absolute():
        source = base/source
    try:
        csv.field_size_limit(20_000_000)
        with source.open(encoding='utf-8', newline='') as handle:
            return list(csv.DictReader(handle))
    except OSError as error:
        gaps.append(dict(experiment_id='CAMPAIGN', kind='MISSING_ATTEMPT_LEDGER', source=str(source), detail=str(error)))
        return []


def _reference_text(reference, base):
    """Normalize an existing metadata reference without opening its arrays."""
    if not isinstance(reference, (str, os.PathLike)) or not reference:
        return None
    value = str(reference)
    if '://' in value:
        return value.rstrip('/')
    path = Path(value)
    return os.path.normpath(str(path if path.is_absolute() else base/path))


def _selective_treatment_audit(paths, configurations, base, gaps):
    """Review frozen statistics/proofs only; never load weights or predictions.

    ``preparation_arm`` overrides ``algorithm_arm`` inside Preparation. R15
    legitimately shares the E15 RL objective while reading random weights.
    An executed configuration is distinct from a numerical load receipt.
    """
    audit = []
    from .point_scope import rl_only_reliability_scope, completed_evaluation_scope
    for path in paths:
        identifier = path['experiment_id']
        planned = configurations[identifier]
        if not (path['arm'] in ('E15', 'R15') or planned.get('selective_artifact')):
            continue
        actual = _read(path.get('training_receipt'), base, [], identifier, required=False)
        loaded = actual.get('config') or planned
        artifact = loaded.get('selective_artifact')
        statistic_source = loaded.get('selective_statistics_receipt')
        if not statistic_source and artifact:
            statistic_source = str(Path(artifact)/'result.json')
        statistics_receipt = _read(statistic_source, base, [], identifier, required=False)
        statistic_config = statistics_receipt.get('config', {})
        proof_source = statistics_receipt.get('random_weight_proof')
        if not proof_source and artifact:
            proof_source = str(Path(artifact)/'random_weight_proof.json')
        proof = _read(proof_source, base, [], identifier, required=False)
        protocol = loaded.get('protocol', {})
        horizon = protocol.get('horizon')
        variables = protocol.get('variables')
        expected_strata = {(variable, position) for variable in range(len(variables))
                           for position in range(horizon)} if isinstance(variables, list) and isinstance(horizon, int) else None
        strata = proof.get('strata')
        strata = strata if isinstance(strata, list) else []
        strata_keys = [(row.get('variable'), row.get('horizon')) for row in strata if isinstance(row, dict)]
        strata_complete = (len(strata_keys) == len(set(strata_keys)) and set(strata_keys) == expected_strata
                           if expected_strata is not None and strata else None)
        exact_distribution = (proof.get('status') == 'PASS' and proof.get('exact_sorted_values_equal') is True
                              and strata_complete is True
                              and all(row.get('exact_weight_distribution') is True for row in strata)) if proof else None
        nonuniform = proof.get('nonuniform', statistics_receipt.get('R15_has_nonuniform_treatment'))
        nonuniform = nonuniform if isinstance(nonuniform, bool) else None
        same_fraction = proof.get('same_value_fraction')
        permutation_changes_weights = same_fraction < 1 if _number(same_fraction) and 0 <= same_fraction <= 1 else None
        weight_arm = loaded.get('preparation_arm', loaded.get('algorithm_arm', loaded.get('arm')))
        expected_filename = 'random_weight.npy' if weight_arm == 'R15' else 'soft_weight.npy' if weight_arm == 'E15' else None
        direct = actual.get('selective_treatment', {})
        preparation_source=path.get('preparation_receipt') or loaded.get('preparation_receipt')
        preparation=_read(preparation_source,base,[],identifier,required=False)
        formal_source=path.get('formal_test_receipt')
        formal=_read(formal_source,base,[],identifier,required=False)
        runtime_RL_scope=rl_only_reliability_scope(actual,preparation) if actual and preparation else None
        runtime_evaluation_scope=None;scope_detail='Formal execution scope is not recorded yet'
        if formal.get('status')=='PASS' and formal.get('evaluation_scope',{}).get('status')=='PASS':
            try:
                verified=completed_evaluation_scope(formal.get('config',{}).get('cases',[]),formal.get('cases',[]))
                runtime_evaluation_scope=verified['all_observed_native_evaluator_contract'] and verified['training_reliability_inputs_absent']
                scope_detail=verified['evidence_level']
            except (ValueError,KeyError,OSError) as error:
                runtime_evaluation_scope=False;scope_detail=type(error).__name__+': '+str(error)
        # Missing declarations cannot prove execution; explicit contradictions remain errors.
        if loaded.get('reliability_used_only_during_RL') is False:runtime_RL_scope=False
        if loaded.get('validation_and_test_use_all_observed') is False:runtime_evaluation_scope=False
        actual_filename = direct.get('weight_filename') if isinstance(direct, dict) else None
        requested_filename = loaded.get('selective_weight')
        config_file_selection_consistent = (requested_filename == expected_filename
            if requested_filename and expected_filename else None)
        numerical_file_selection_consistent = (actual_filename == expected_filename
            if actual_filename and expected_filename else None)
        statistics_artifact_matches = (_reference_text(artifact, base) == _reference_text(statistics_receipt.get('artifact'), base)
            if artifact and statistics_receipt.get('artifact') else None)
        statistics_binding = (statistic_config.get('dataset') == path['dataset']
            and statistic_config.get('seed') == path['seed']
            and _protocol_key(statistic_config) == _protocol_key(loaded)) if statistic_config else None
        train_only = statistics_receipt.get('train_only_statistics')
        reference_frozen = statistics_receipt.get('reference_parameters_unchanged')
        no_candidate_selection = proof.get('independent_of_candidate_forecast')
        no_test_use = proof.get('test_used') is False if proof else None
        if nonuniform is False:
            treatment = 'NO_EFFECTIVE_SELECTIVE_TREATMENT'
            equivalence = ('E03 with the same remaining configuration' if path['arm'] in ('E15', 'R15')
                           else 'the same conditional method with selective weighting disabled')
            interpretation = ('Frozen weights are globally constant. Normalized weighted reward/loss reduce to the '
                              'unweighted objective; this is not evidence that selective learning is ineffective, '
                              'and score differences cannot establish a new selective benefit.')
        elif nonuniform is True and path['arm'] == 'R15' and permutation_changes_weights is False:
            treatment = 'RANDOMIZATION_DID_NOT_CHANGE_WEIGHT_VALUES'
            equivalence = 'E15 with the same remaining configuration'
            interpretation = 'The recorded permutation leaves all weight values unchanged; E15 versus R15 cannot identify difficulty targeting.'
        elif nonuniform is True:
            treatment, equivalence = 'NONUNIFORM_WEIGHT_VALUES_RECORDED', None
            interpretation = 'Nonuniform values are recorded; this alone does not prove useful selective learning or actual runtime loading.'
        else:
            treatment, equivalence = 'UNKNOWN', None
            interpretation = 'No recorded nonuniformity proof; do not infer treatment strength from the method name.'
        audit.append(dict(experiment_id=identifier, dataset=path['dataset'], method=path['arm'], seed=path['seed'],
            status=path.get('status', 'UNKNOWN'), artifact=artifact,
            statistics_receipt=statistic_source, random_weight_proof=proof_source,
            statistics_status=statistics_receipt.get('status', 'UNKNOWN'), proof_status=proof.get('status', 'UNKNOWN'),
            statistics_reference_checkpoint=statistic_config.get('checkpoint', 'UNKNOWN'),
            algorithm_arm=loaded.get('algorithm_arm', loaded.get('arm')), preparation_arm=weight_arm,
            declared_weight_filename=requested_filename, config_selected_weight_filename=expected_filename,
            recorded_loaded_weight_filename=actual_filename or 'UNKNOWN',
            configuration_bound_to_completed_training=actual.get('status') == 'PASS',
            config_file_selection_consistent=config_file_selection_consistent,
            numerical_file_selection_consistent=numerical_file_selection_consistent,
            file_selection_evidence='EXPLICIT_TRAINING_RECEIPT' if actual_filename else 'CONFIGURATION_ONLY; direct file-load evidence UNKNOWN',
            declared_reliability_used_only_during_RL=loaded.get('reliability_used_only_during_RL'),
            declared_validation_and_test_use_all_observed=loaded.get('validation_and_test_use_all_observed'),
            reliability_used_only_during_RL=runtime_RL_scope,
            validation_and_test_use_all_observed=runtime_evaluation_scope,
            scope_evidence_source='ACTUAL_PREPARATION_TRAINING_AND_FORMAL_RECEIPTS',
            scope_preparation_receipt=preparation_source,scope_formal_receipt=formal_source,scope_evidence_detail=scope_detail,
            statistics_artifact_matches=statistics_artifact_matches, statistics_seed_dataset_protocol_match=statistics_binding,
            train_only_statistics=train_only, frozen_reference_unchanged=reference_frozen,
            same_weights_for_all_candidates=no_candidate_selection, randomization_test_unused=no_test_use,
            observed_constraint=proof.get('observed_constraint', 'UNKNOWN'),
            permutation_scope=proof.get('permutation_scope', 'UNKNOWN'), permutation_seed=proof.get('permutation_seed'),
            strata_recorded=len(strata), strata_expected=len(expected_strata) if expected_strata is not None else None,
            unique_complete_variable_horizon_strata=strata_complete,
            exact_matched_within_stratum_weight_distribution=exact_distribution,
            global_weights_nonuniform=nonuniform, same_value_fraction=same_fraction,
            permutation_changes_weights=permutation_changes_weights,
            fallback_window_variables=statistics_receipt.get('fallback_window_variables'),
            retention_by_horizon_variable=statistics_receipt.get('retention_by_horizon_variable', 'UNKNOWN'),
            peak_retention=statistics_receipt.get('peak_retention', 'UNKNOWN'),
            jump_pair_retention=statistics_receipt.get('jump_pair_retention', 'UNKNOWN'),
            treatment_status=treatment, semantic_equivalence_if_other_settings_match=equivalence,
            interpretation=interpretation, source_match_to_E15=None, source_match_to_R15=None,
            cross_arm_source_status='UNKNOWN', array_loading_performed=False, numeric_revalidation=False))
    by_key = {(row['dataset'], row['seed'], row['method']): row for row in audit}
    for row in audit:
        for peer in ('E15', 'R15'):
            other = by_key.get((row['dataset'], row['seed'], peer))
            a, b = _reference_text(row['artifact'], base), _reference_text(other['artifact'], base) if other else None
            row['source_match_to_'+peer] = a == b if a and b else None
        peer_results = [row['source_match_to_E15'], row['source_match_to_R15']]
        row['cross_arm_source_status'] = ('MISMATCH' if False in peer_results else 'SAME_FROZEN_ARTIFACT'
                                          if all(x is True for x in peer_results) else 'UNKNOWN')
        if row['global_weights_nonuniform'] is False:
            gaps.append(dict(experiment_id=row['experiment_id'], kind='SELECTIVE_TREATMENT_DEGENERATE',
                source=row['random_weight_proof'], detail=row['interpretation']))
        elif row['global_weights_nonuniform'] is None:
            gaps.append(dict(experiment_id=row['experiment_id'], kind='SELECTIVE_NONUNIFORM_UNKNOWN',
                source=row['random_weight_proof'], detail=row['interpretation']))
        elif row['treatment_status'] == 'RANDOMIZATION_DID_NOT_CHANGE_WEIGHT_VALUES':
            gaps.append(dict(experiment_id=row['experiment_id'], kind='SELECTIVE_PERMUTATION_UNCHANGED',
                source=row['random_weight_proof'], detail=row['interpretation']))
        if row['status'] in ('ACCEPTED', 'COMPLETED', 'SUCCEEDED_ACCEPTED'):
            required = {'statistics_status': row['statistics_status'] == 'PASS',
                'completed_training_configuration_bound': row['configuration_bound_to_completed_training'],
                'statistics_artifact_matches': row['statistics_artifact_matches'],
                'statistics_seed_dataset_protocol_match': row['statistics_seed_dataset_protocol_match'],
                'train_only_statistics': row['train_only_statistics'],
                'frozen_reference_unchanged': row['frozen_reference_unchanged'],
                'same_weights_for_all_candidates': row['same_weights_for_all_candidates'],
                'randomization_test_unused': row['randomization_test_unused'],
                'RL_only_reliability': row['reliability_used_only_during_RL'],
                'evaluation_all_observed': row['validation_and_test_use_all_observed'],
                'same_E15_R15_source': row['cross_arm_source_status'] == 'SAME_FROZEN_ARTIFACT',
                'matched_variable_horizon_distributions': row['exact_matched_within_stratum_weight_distribution']}
            if row['config_file_selection_consistent'] is False or row['numerical_file_selection_consistent'] is False:
                required['weight_file_selection'] = False
            missing = [key for key, value in required.items() if value is not True]
            if missing:
                gaps.append(dict(experiment_id=row['experiment_id'], kind='SELECTIVE_REQUIRED_EVIDENCE_NOT_PROVEN',
                    source=row['statistics_receipt'], detail='Required frozen/matched treatment evidence is missing or inconsistent: '+', '.join(missing)))
    return audit


def _opd_logs(reference, base):
    """Read existing scalar logs from this preparation's retained attempts only."""
    if not reference or isinstance(reference, dict):
        return dict(status='UNKNOWN', source_logs=[])
    receipt = Path(reference)
    if not receipt.is_absolute():
        receipt = base/receipt
    candidates = [receipt.parent/'micro_diagnostics.jsonl']
    if receipt.parent.name == 'result' and receipt.parent.parent.name.startswith('attempt-'):
        candidates += sorted(receipt.parent.parent.parent.glob('attempt-*/result/micro_diagnostics.jsonl'))
    sources, errors, versions, refresh_calls = [], [], set(), set()
    records = valid_triplets = 0
    for source in dict.fromkeys(candidates):
        if not source.is_file():
            continue
        sources.append(str(source))
        try:
            with source.open() as handle:
                for line in handle:
                    record = json.loads(line)
                    for diagnostic in record.get('diagnostics', []):
                        if 'native_calls' not in diagnostic:
                            continue
                        records += 1
                        events = diagnostic.get('events', [])
                        valid = diagnostic.get('native_calls') == 3 and len(events) == 3
                        valid &= all(event.get('block') == index and event.get('teacher_student_same_state') is True
                                     and event.get('real_future_input_nonzero') == 0
                                     and event.get('generated_prefix_visible') is True for index, event in enumerate(events))
                        valid_triplets += bool(valid)
                        if diagnostic.get('policy_version') is not None:
                            versions.add(diagnostic['policy_version'])
                        if diagnostic.get('prefix_refresh_call') is not None:
                            refresh_calls.add(diagnostic['prefix_refresh_call'])
        except (OSError, ValueError) as error:
            errors.append(dict(source=str(source), detail=str(error)))
    return dict(status='THREE_ROLLING_CALLS_RECORDED' if records and valid_triplets == records and not errors else 'MISSING_OR_INCOMPLETE',
                source_logs=sources, recorded_microbatches=records, valid_three_state_records=valid_triplets,
                policy_versions=sorted(versions), distinct_policy_versions=len(versions),
                refresh_call_count=len(refresh_calls), errors=errors,
                scope='Retained preparation-attempt scalar logs; no model execution or future-label reconstruction')


def record_consistency(paths, rows, seeds, freeze):
    """Record checks only; execution belongs in a HOPE qualification worker."""
    identifiers = [path['experiment_id'] for path in paths]
    expected = {(dataset, method, seed) for dataset in DATASETS for method in CORE_METHODS for seed in seeds}
    actual = {(path['dataset'], path['arm'], path['seed']) for path in paths if path['arm'] in CORE_METHODS}
    keys = [(row['experiment_id'], row['readout'], row['split'], row['protocol'], row['selection_objective']) for row in rows]
    checks = dict(unique_scientific_ids=len(identifiers) == len(set(identifiers)),
        unique_scientific_configuration_seeds=len(paths) == len({(p['dataset'], p['arm'], p['seed']) for p in paths}),
        five_distinct_frozen_seeds=len(seeds) == 5 and len(set(seeds)) == 5,
        all_90_mandatory_paths_registered=actual == expected and len(expected) == 90,
        unique_readout_keys=len(keys) == len(set(keys)),
        row_foreign_keys=all(row['experiment_id'] in identifiers for row in rows),
        conditional_path_limit=sum(path['arm'] not in CORE_METHODS for path in paths) <= 20,
        test_release_was_validation_frozen=freeze.get('status') == 'PASS' and freeze.get('test_release') is True
            and freeze.get('final_selection_frozen') is True and freeze.get('test_used') is False)
    return dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
        missing_mandatory_keys=sorted(expected-actual), unexpected_core_keys=sorted(actual-expected),
        kind='RECORD_CONSISTENCY_ONLY', model_numerical_revalidation=False)


def recipe_gap_evidence(paths, freeze, seeds, base):
    """Prove each missing recipe using existing JSON only; never infer from INVALID.

    A P2/P3 exception needs the complete fixed validation schedule and all retained
    reloads. A finite value at any such state disproves an unavailable choice.
    Scientific terminals retain the coordinator's narrow executed-failure policy,
    but must explain every otherwise incomplete method cohort in this exact key.
    P1 fallback is deliberately outside this exception.
    """
    seeds, base = tuple(seeds), Path(base)
    cache, reviews, errors, accepted = {}, [], [], []
    expected_schedule = {(-1, 0.)} | {(epoch, k/(16 if epoch == 0 else 4))
        for epoch in range(4) for k in range(1, (16 if epoch == 0 else 4)+1)}
    def source(reference):
        if not isinstance(reference, (str, Path)) or not str(reference):
            return None, None, 'SOURCE_REFERENCE_MISSING'
        file = Path(reference); file = file if file.is_absolute() else base/file
        name = str(file)
        if name not in cache:
            try:
                if file.suffix != '.json' or file.stat().st_size > 16*1024*1024:
                    raise ValueError('Expected a bounded existing JSON receipt')
                cache[name] = (json.loads(file.read_text()), name, None)
            except (OSError, ValueError) as error:
                cache[name] = (None, name, str(error))
        return cache[name]
    def scientific(path):
        record, ref, error = source(path.get('failure_receipt'))
        record = record if isinstance(record, dict) else {}
        cfg = record.get('config', record.get('configuration', {}))
        cfg = cfg if isinstance(cfg, dict) else {}
        classification = str(record.get('failure_class', '')).lower()
        excluded = ('platform', 'permission', 'network', 'budget', 'unsubmitted', 'queue', 'storage',
            'mount', 'preempt', 'checkpoint_missing', 'teacher_missing', 'code_bug', 'implementation_bug',
            'oom', 'machine_failure', 'node_failure', 'timeout', 'evict', 'auth_failure', 'cuda_error',
            'diagnostic', 'crps', 'calibration', 'metric_nonfinite')
        repair = record.get('finite_repair_exhausted') is True or (
            record.get('repair_policy') == 'not_applicable_inherent_numerical_failure'
            and bool(record.get('root_diagnosis_reason')) and bool(record.get('repair_attempt_evidence')))
        execution = any(record.get(key) for key in ('model_execution_evidence', 'first_step_evidence',
            'training_curve_evidence', 'runtime_evidence'))
        submitted = record.get('run_id') or any(item.get('run_id') for item in path.get('worker_execution_evidence', []))
        valid = (not error and path.get('status') in ('FAILED_SCIENTIFIC_TERMINAL', 'DIVERGED')
            and path.get('scientific_failure_terminal') is True and record.get('scientific_failure_terminal') is True
            and record.get('model_execution_started') is True and execution and submitted and repair
            and record.get('failure_domain') in ('training_loss', 'training_gradient', 'invalid_policy_distribution', 'nonfinite_prediction')
            and classification and not any(token in classification for token in excluded)
            and any(token in classification for token in ('numerical', 'nonfinite', 'non_finite', 'nan', 'diverg', 'overflow', 'underflow'))
            and cfg.get('dataset') == path['dataset'] and cfg.get('seed') == path['seed']
            and cfg.get('phase') == 'scientific' and cfg.get('scientific_path') == path['experiment_id'])
        return dict(status='PROVEN_SCIENTIFIC_FAILURE' if valid else 'UNPROVEN_SCIENTIFIC_FAILURE',
                    source=ref, problem=error if error else None)
    def training(path, protocol, objective):
        if path.get('scientific_failure_terminal') is True:
            return scientific(path)
        record, ref, error = source(path.get('training_receipt'))
        result = dict(status='UNPROVEN', source=ref, problems=[])
        if error or not isinstance(record, dict):
            result['problems'].append(error or 'TRAINING_RECEIPT_SCHEMA'); return result
        cfg = record.get('config', {}); cursor = record.get('cursor', {})
        if not isinstance(cfg, dict) or not isinstance(cursor, dict):
            result['problems'].append('TRAINING_CONFIGURATION_OR_CURSOR_SCHEMA'); return result
        windows = cfg.get('protocol', {}).get('windows', {})
        if not isinstance(windows, dict):
            result['problems'].append('TRAINING_WINDOWS_SCHEMA'); return result
        total, val_count = windows.get('train'), windows.get('val')
        if not (path.get('status') in ('ACCEPTED', 'COMPLETED', 'SUCCEEDED_ACCEPTED')
                and record.get('status') == 'PASS' and record.get('test_used') is False
                and record.get('epochs') == 4 and cfg.get('phase') == 'scientific'
                and cfg.get('dataset') == path['dataset'] and cfg.get('seed') == path['seed']
                and cfg.get('scientific_path') == path['experiment_id']
                and isinstance(total, int) and not isinstance(total, bool) and total > 0
                and isinstance(val_count, int) and not isinstance(val_count, bool) and val_count > 0
                and cursor.get('epoch') == 4 and cursor.get('next_input') == 0
                and _number(cursor.get('global_step')) and cursor['global_step'] > 0
                and record.get('independent_inputs_per_epoch') == total
                and cursor.get('independent_input_exposures') == 4*total):
            result['problems'].append('COMPLETE_SAME_PATH_SCIENTIFIC_TRAINING_NOT_PROVEN'); return result
        label = f'{protocol}_val_{objective}_best'
        metric = POINT_PROTOCOLS[protocol][objective]
        selections = record.get('point_selection', {})
        choice = selections.get(label, {}) if isinstance(selections, dict) else {}
        choice = choice if isinstance(choice, dict) else {}
        if not (choice.get('protocol') == protocol and choice.get('selection_objective') == objective
                and choice.get('validation_metric') == metric and choice.get('status') in ('VALID', 'INVALID')):
            result['problems'].append('SELECTION_RECORD_MISSING_OR_MISMATCHED'); return result
        curve, curve_ref, error = source(Path(ref).parent/'validation_curve.json')
        result['validation_curve'] = curve_ref
        if error or not isinstance(curve, list):
            result['problems'].append('COMPLETE_VALIDATION_SCHEDULE_MISSING'); return result
        schedule = [(row.get('epoch'), row.get('scheduled_fraction')) for row in curve if isinstance(row, dict)]
        schedule_scalars = all(isinstance(epoch, int) and not isinstance(epoch, bool) and _number(fraction)
                               for epoch, fraction in schedule)
        if not schedule_scalars or len(schedule) != len(curve) or len(schedule) != len(expected_schedule) or set(schedule) != expected_schedule:
            result['problems'].append('COMPLETE_VALIDATION_SCHEDULE_NOT_PROVEN'); return result
        readouts = record.get('readouts', {})
        if not isinstance(readouts, dict) or not all(key in readouts for key in ('initial', 'final')):
            result['problems'].append('INITIAL_OR_FINAL_RELOAD_MISSING'); return result
        observations = [(curve_ref+':'+str(index), row.get('metrics'), row.get('full_validation') is True)
                        for index, row in enumerate(curve)]
        observations += [(ref+':readouts.'+key+'.validation', row.get('validation') if isinstance(row, dict) else None,
                          isinstance(row, dict) and row.get('reloaded') is True) for key, row in readouts.items()]
        result['validation_records'] = len(observations)
        for locator, metrics, full in observations:
            if not (isinstance(metrics, dict) and full and metrics.get('complete') is True
                    and metrics.get('split') == 'val' and metrics.get('seed') == path['seed']
                    and metrics.get('windows') == val_count and metrics.get('num_samples') == 100
                    and metrics.get('independent_marginal_samples') is True):
                result['problems'].append('INCOMPLETE_OR_MISMATCHED_VALIDATION: '+locator)
        if result['problems']:return result
        if choice['status'] == 'VALID':
            selected = readouts.get(label, {})
            if not (_point(choice.get('validation_value')) and choice.get('checkpoint')
                    and selected.get('checkpoint') == choice['checkpoint']
                    and _point(selected.get('validation', {}).get(metric))):
                result['problems'].append('VALID_CHOICE_HAS_NO_MATCHING_COMPLETE_RELOAD'); return result
            result['status'] = 'VALID_POINT_SEED'; return result
        if protocol not in ('P2', 'P3'):
            result['problems'].append('P1_FALLBACK_IS_NOT_INFERRED'); return result
        if choice.get('checkpoint') is not None or choice.get('validation_value') is not None:
            result['problems'].append('INVALID_SELECTION_HAS_A_CHECKPOINT_OR_VALUE'); return result
        reasons = []
        for locator, metrics, _ in observations:
            if _point(metrics.get(metric)):
                result['problems'].append('FINITE_POINT_VALUE_OMITTED_FROM_SELECTION: '+locator); continue
            proof = point_unavailability_evidence(metrics, protocol, objective)
            if proof is None:
                result['problems'].append('POINT_PROTOCOL_UNAVAILABILITY_REASON_NOT_PROVEN: '+locator)
            else:
                reasons.extend(dict(source=locator, issue=reason['metric'], first_origin=reason['first_origin'],
                    occurrences=reason['occurrences'], first_message=reason['first_message']) for reason in proof['issues'])
        if not result['problems']:
            result.update(status='PROVEN_P2_SAMPLED_UNAVAILABLE' if protocol == 'P2' else 'PROVEN_P3_ANALYTIC_UNAVAILABLE',
                          unavailable_reasons=reasons)
        return result
    recipes = freeze.get('deliveries', freeze.get('recipes', []))
    gaps = freeze.get('recipe_gaps', [])
    if not isinstance(recipes, list) or not isinstance(gaps, list):
        return dict(status='FAIL', accepted_gap_keys=[], gap_reviews=[], errors=['RECIPE_GAP_SCHEMA'])
    frozen_keys = {(row.get('dataset'), row.get('protocol'), row.get('selection_objective', row.get('objective')))
                   for row in recipes if isinstance(row, dict)}
    seen = set()
    for gap in gaps:
        key = (gap.get('dataset'), gap.get('protocol'), gap.get('selection_objective', gap.get('objective'))) if isinstance(gap, dict) else (None, None, None)
        review = dict(dataset=key[0], protocol=key[1], objective=key[2], status='BLOCKED', problems=[], seed_evidence=[])
        reviews.append(review)
        if key in seen:review['problems'].append('DUPLICATE_DECLARED_GAP')
        seen.add(key)
        if key in frozen_keys:review['problems'].append('GAP_OVERLAPS_FROZEN_RECIPE')
        if key[0] not in DATASETS or key[1] not in POINT_PROTOCOLS or key[2] not in ('MSE', 'MAE'):
            review['problems'].append('UNKNOWN_GAP_KEY')
        if len(seeds) != 5 or len(set(seeds)) != 5:review['problems'].append('FIVE_FROZEN_SEEDS_NOT_PROVEN')
        if review['problems']:continue
        methods = set(CORE_METHODS) | {path['arm'] for path in paths if path.get('dataset') == key[0]}
        for method in sorted(methods):
            cohort = [path for path in paths if path.get('dataset') == key[0] and path.get('arm') == method]
            if len(cohort) != 5 or {path.get('seed') for path in cohort} != set(seeds):
                review['problems'].append('FIVE_REGISTERED_METHOD_SEEDS_MISSING_OR_DUPLICATE: '+method); continue
            statuses = []
            for seed in seeds:
                path = next(path for path in cohort if path['seed'] == seed)
                evidence = training(path, key[1], key[2])
                review['seed_evidence'].append(dict(experiment_id=path['experiment_id'], method=method, seed=seed, **evidence))
                statuses.append(evidence['status'])
                if evidence['status'] not in ('VALID_POINT_SEED', 'PROVEN_SCIENTIFIC_FAILURE',
                        'PROVEN_P2_SAMPLED_UNAVAILABLE', 'PROVEN_P3_ANALYTIC_UNAVAILABLE'):
                    review['problems'].append('METHOD_SEED_CAUSE_NOT_PROVEN: '+method+'/'+str(seed))
            if all(value == 'VALID_POINT_SEED' for value in statuses):
                review['problems'].append('VALID_FIVE_SEED_CANDIDATE_OMITTED')
        if not review['problems']:
            review['status'] = 'PROVEN_DECLARED_GAP'
            review['explanation'] = (key[1]+' independently proven readout unavailability and/or executed scientific terminal, checked per method/seed'
                if key[1] in ('P2','P3') else 'Executed scientific terminals checked per method/seed; no protocol fallback inferred')
            accepted.append(list(key))
    return dict(status='PASS' if all(row['status'] == 'PROVEN_DECLARED_GAP' for row in reviews) else 'FAIL',
                accepted_gap_keys=accepted, gap_reviews=reviews, errors=errors,
                scope='Existing bounded JSON receipts and scalar records only; no new numerical execution; P1 fallback unchanged')


def validate_qualification():
    """Run this targeted scalar decision check in HOPE, never on coordinator."""
    seeds = (101, 211, 307, 401, 503)
    mse = [(seed, .365) for seed in seeds]
    mae = [(seed, .399282) for seed in seeds]
    tied = paper_decision('ETTh1', mse, mae, seeds)
    assert not tied['strict_OR'] and tied['display_precision_tie_only']
    reached = paper_decision('ETTh1', [(seed, .364) for seed in seeds], [(seed, .5) for seed in seeds], seeds)
    assert reached['strict_OR'] and reached['strict_MSE'] and not reached['strict_MAE']
    assert paper_decision('ETTh1', mse[:4], mae[:4], seeds)['numeric_status'] == 'INSUFFICIENT_FIVE_SEED_EVIDENCE'
    assert Decimal('.3995').quantize(Decimal('.001'), rounding=ROUND_HALF_UP) == Decimal('.400')
    stats = paired_summary([dict(seed=seed, delta=value) for seed, value in zip(seeds, (-2., -1., 0., 1., 2.))], seeds)
    assert stats['mean'] == 0 and stats['improvement_seed_count'] == 2
    assert math.isclose(stats['sample_std'], math.sqrt(2.5))
    assert math.isclose(stats['mean_difference_t95_upper'], T95_DF4/math.sqrt(2))
    diagnostic_gap = _metrics(dict(MAP_MSE=.364, MAP_MAE=.5, CRPS=float('nan')), 'P1')
    assert diagnostic_gap['MSE'] == .364 and diagnostic_gap['CRPS'] is None
    classified = _classify_gaps([
        dict(kind='MISSING_DIAGNOSTIC_METRICS'), dict(kind='DIAGNOSTIC_METRIC_ISSUE'),
        dict(kind='SELECTIVE_TREATMENT_DEGENERATE'), dict(kind='SELECTIVE_NONUNIFORM_UNKNOWN'),
        dict(kind='INVALID_REQUIRED_POINT_METRICS'), dict(kind='CHECKPOINT_BINDING_MISMATCH'),
        dict(kind='MISSING_OPD_ROLLING_EXECUTION_EVIDENCE'), dict(kind='UNKNOWN_UNCLASSIFIED_GAP')])
    assert all(not row['blocking'] for row in classified[:4])
    assert all(row['blocking'] for row in classified[4:])
    selective_check = _qualify_selective_record_schema()
    return dict(status='PASS', kind='SCALAR_DECISION_AND_RECORD_CHECK', model_numerical_revalidation=False,
        strict_unrounded_OR=True, ROUND_HALF_UP_separate=True, missing_seed_not_five=True,
        diagnostic_gaps_do_not_veto_points=True, unknown_necessary_gaps_block=True,
        selective_record_schema=selective_check, paired_t_df4_descriptive=True)


def _qualify_selective_record_schema():
    """Synthetic metadata fixtures only, executed with the HOPE qualification.

    These are deliberately not experimental results and contain no weight
    arrays. The R15 fixture tests the real preparation_arm override contract.
    """
    import tempfile
    from .point_scope import completed_evaluation_scope
    with tempfile.TemporaryDirectory(prefix='point-selective-record-check-') as temporary:
        root = Path(temporary)
        protocol = dict(dataset='ETTh1', mode='M', variables=['v0', 'v1'], horizon=96,windows=dict(val=2,test=2))
        common = dict(dataset='ETTh1', seed=101, protocol=protocol,phase='scientific',
                      checkpoint='fixture-preparation-final.pt',selective_artifact=str(root),
                      selective_statistics_receipt=str(root/'result.json'))
        preparation=root/'preparation.json'
        _atomic_json(preparation,dict(status='PASS',config=dict(dataset='ETTh1',seed=101,phase='preparation'),
            readouts=dict(final=dict(checkpoint='fixture-preparation-final.pt')),test_used=False))
        case=dict(case_id='final',dataset='ETTh1',route='D-pre',seed=101,protocol=protocol,artifact=str(root),splits=['val','test'])
        numeric=root/'native-readout.json'
        _atomic_json(numeric,dict(status='PASS',config=case,
            results={split:dict(complete=True,split=split,seed=101,windows=2) for split in ('val','test')}))
        records=[dict(case_id='final',canonical_case='final',receipt=str(numeric))]
        formal=root/'formal.json'
        _atomic_json(formal,dict(status='PASS',config=dict(cases=[case]),cases=records,
            evaluation_scope=completed_evaluation_scope([case],records)))
        configs = {arm: dict(common, arm=arm, algorithm_arm='E15', preparation_arm=arm,
                            selective_weight='random_weight.npy' if arm == 'R15' else 'soft_weight.npy')
                   for arm in ('E15', 'R15')}
        paths = []
        for arm, config in configs.items():
            receipt = root/(arm+'-training.json')
            _atomic_json(receipt, dict(status='PASS', config=config,test_used=False,
                selective_treatment=dict(status='ACTUALLY_LOADED_IN_RL',frozen=True,
                    same_weights_for_all_J_G_candidates=True,
                    weight_filename='random_weight.npy' if arm=='R15' else 'soft_weight.npy')))
            paths.append(dict(experiment_id=arm, arm=arm, dataset='ETTh1', seed=101,
                              status='ACCEPTED', training_receipt=str(receipt),
                              preparation_receipt=str(preparation),formal_test_receipt=str(formal)))
        statistics_receipt = dict(status='PASS', config=dict(common, checkpoint='fixture-reference'), artifact=str(root),
            train_only_statistics=True, reference_parameters_unchanged=True,
            random_weight_proof=str(root/'random_weight_proof.json'), R15_has_nonuniform_treatment=False)
        proof = dict(status='PASS', nonuniform=False, same_value_fraction=1., exact_sorted_values_equal=True,
            independent_of_candidate_forecast=True, test_used=False,
            strata=[dict(variable=v, horizon=h, exact_weight_distribution=True, minimum=.2, maximum=.2)
                    for v in range(2) for h in range(96)])
        _atomic_json(root/'result.json', statistics_receipt)
        _atomic_json(root/'random_weight_proof.json', proof)
        gaps = []
        rows = _selective_treatment_audit(paths, configs, root, gaps)
        assert all(row['treatment_status'] == 'NO_EFFECTIVE_SELECTIVE_TREATMENT' for row in rows)
        assert all(row['cross_arm_source_status'] == 'SAME_FROZEN_ARTIFACT' for row in rows)
        assert all(row['exact_matched_within_stratum_weight_distribution'] is True for row in rows)
        assert next(row for row in rows if row['method'] == 'R15')['config_selected_weight_filename'] == 'random_weight.npy'
        assert not any(row['blocking'] for row in _classify_gaps(gaps))
        proof.update(nonuniform=True, same_value_fraction=.5)
        for row in proof['strata']:
            row['maximum'] = 1.
        _atomic_json(root/'random_weight_proof.json', proof)
        gaps = []
        rows = _selective_treatment_audit(paths, configs, root, gaps)
        assert all(row['global_weights_nonuniform'] is True and row['permutation_changes_weights'] is True for row in rows)
        assert not gaps
        proof['strata'].pop()
        _atomic_json(root/'random_weight_proof.json', proof)
        gaps = []
        rows = _selective_treatment_audit(paths, configs, root, gaps)
        assert all(row['unique_complete_variable_horizon_strata'] is False for row in rows)
        assert any(row['blocking'] for row in _classify_gaps(gaps))
    return dict(status='PASS', fixture_kind='JSON_METADATA_SCHEMA_ONLY', arrays_or_models_loaded=False,
        preparation_arm_override_preserved=True, constant_weights_not_a_new_selective_effect=True,
        exact_stratum_coverage_required=True, diagnostic_gap_not_acceptance_veto=True)


def shared_start_outputs(shared_starts,scientific_paths,seeds,out,gaps):
    """Auxiliary validation evidence; never enter scientific-path test pairing."""
    from .point_shared_results import collect_shared_starts
    result=collect_shared_starts(shared_starts,scientific_paths,seeds=seeds)
    for name,key in [('shared_sft_inventory.csv','inventory'),
                     ('shared_sft_checkpoint_results.csv','checkpoint_results'),
                     ('shared_sft_curve_index.csv','curve_index'),
                     ('shared_start_lineage.csv','lineage')]:
        values=result[key]
        _table(out/name,values,list(dict.fromkeys(field for row in values for field in row)) if values else ['shared_start_node_id','status','source_receipt'])
    with (out/'shared_sft_configurations_full.jsonl').open('w',encoding='utf-8') as handle:
        for row in result['configurations']:handle.write(_json(row)+'\n')
    _atomic_json(out/'shared_sft_record_consistency.json',result['summary'])
    gaps.extend(result['gaps'])
    return result


def analyze(config_path, output, device='cuda', seed=1):
    """Generate all registered results; never choose recipes using test values."""
    began = time.time()
    source = Path(config_path)
    config = json.loads(source.read_text())
    base = source.parent
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    gaps = []
    paths = config['scientific_paths']
    seeds = tuple(config.get('seeds', (101, 211, 307, 401, 503)))
    shared_starts=config.get('shared_starts',[])
    shared=shared_start_outputs(shared_starts,paths,seeds,out,gaps)
    historical_groups = historical_replay_audit(config, base, out, gaps)
    shared_worker_paths=[dict(experiment_id=entry['node_id'],shared_start_receipt=entry.get('result_receipt'),
        worker_execution_evidence=entry.get('worker_execution_evidence',[])) for entry in shared_starts]
    worker_records = worker_record_audit(paths+shared_worker_paths, base, out, gaps)
    worker_reference_counts = worker_reference_summary(worker_records)
    freeze = _read(config.get('selection_freeze_receipt'), base, gaps, 'CAMPAIGN')
    frozen, frozen_status = _recipes(freeze)
    ledger = _load_ledger(config.get('attempt_budget_ledger'), base, gaps)
    optional_budget = conditional_budget_audit(config,paths,seeds,freeze,ledger,base,out,gaps)
    configurations = {path['experiment_id']: _configuration(path, base, gaps) for path in paths}
    rows = [row for path in paths for row in _extract(path, base, gaps, configurations[path['experiment_id']])]
    selective = _selective_treatment_audit(paths, configurations, base, gaps)
    selective_by_id = {row['experiment_id']: row for row in selective}
    _table(out/'selective_treatment_audit.csv', selective,
           list(selective[0]) if selective else ['experiment_id', 'dataset', 'method', 'seed', 'treatment_status', 'interpretation'])
    validation = record_consistency(paths, rows, seeds, freeze)
    validation['checks'].update({'shared_SFT_'+key:value for key,value in shared['summary']['checks'].items()})
    validation['checks']['optional_budget_admission_consistent']=optional_budget['status']=='PASS'
    recipe_keys = [(r.get('dataset'), r.get('protocol'), r.get('selection_objective')) for r in frozen]
    recipe_gaps = freeze.get('recipe_gaps', [])
    gap_proof = recipe_gap_evidence(paths, freeze, seeds, base)
    _atomic_json(out/'recipe_gap_evidence.json', gap_proof)
    _table(out/'recipe_gap_audit.csv', gap_proof['gap_reviews'],
        ['dataset', 'protocol', 'objective', 'status', 'problems', 'seed_evidence', 'explanation'])
    for review in gap_proof['gap_reviews']:
        if review['status'] != 'PROVEN_DECLARED_GAP':
            gaps.append(dict(experiment_id='CAMPAIGN', kind='RECIPE_GAP_NOT_PROVEN',
                source=config.get('selection_freeze_receipt', ''),
                detail=_json({key:review[key] for key in ('dataset', 'protocol', 'objective', 'problems')})))
    validation['checks']['unique_frozen_delivery_keys'] = len(recipe_keys) == len(set(recipe_keys))
    validation['checks']['declared_recipe_gaps_have_original_execution_evidence'] = gap_proof['status'] == 'PASS'
    validation['checks']['all_delivery_objectives_frozen_or_scientific_gap_declared'] = (
        set(recipe_keys) | {tuple(key) for key in gap_proof['accepted_gap_keys']}) == {
        (dataset, protocol, objective) for dataset in DATASETS for protocol in POINT_PROTOCOLS for objective in ('MSE', 'MAE')}
    validation['checks']['configuration_seed_dataset_bindings'] = all(
        configurations[p['experiment_id']].get('seed') == p['seed']
        and configurations[p['experiment_id']].get('dataset') == p['dataset'] for p in paths)
    qualification = validate_qualification()
    _table(out/'checkpoint_results.csv', rows, CHECKPOINT_COLUMNS)
    aliases = []
    for path in paths:
        formal = _read(path.get('formal_test_receipt'), base, [], path['experiment_id'], required=False)
        for case in formal.get('cases', []):
            aliases.append(dict(experiment_id=path['experiment_id'], dataset=path['dataset'], method=path['arm'],
                seed=path['seed'], readout=case.get('case_id'), checkpoint=case.get('checkpoint'),
                canonical_case=case.get('canonical_case'), receipt=case.get('receipt'),
                exact_parameter_equality=case.get('exact_parameter_equality', 'UNKNOWN'),
                independent_numerical_execution=case.get('independent_numerical_execution', 'UNKNOWN'),
                checkpoint_progress=case.get('checkpoint_progress'), scope='Within this formal worker; not an extra independent training path'))
    _table(out/'checkpoint_aliases.csv', aliases,
        ['experiment_id', 'dataset', 'method', 'seed', 'readout', 'checkpoint', 'canonical_case',
         'receipt', 'exact_parameter_equality', 'independent_numerical_execution', 'checkpoint_progress', 'scope'])
    for protocol in POINT_PROTOCOLS:
        _table(out/f'checkpoint_results_{POINT_PROTOCOLS[protocol]["name"]}.csv',
               [row for row in rows if row['protocol'] == protocol], CHECKPOINT_COLUMNS)
    selected = [row for row in rows if row['split'] == 'test' and row['selection_objective'] in ('MSE', 'MAE')
                and row['stage'] not in ('initial', 'preparation_before', 'preparation_after', 'final')]
    selected_index = {}
    for row in selected:
        key = (row['dataset'], row['method'], row['seed'], row['protocol'], row['selection_objective'])
        if key in selected_index:
            validation['checks']['unique_selected_test_keys'] = False
            continue
        selected_index[key] = row
    validation['checks'].setdefault('unique_selected_test_keys', True)
    successful_paths = [p for p in paths if p.get('status') in ('ACCEPTED', 'COMPLETED', 'SUCCEEDED_ACCEPTED')]
    validation['checks']['successful_paths_have_all_six_test_readouts'] = all(
        all((p['dataset'], p['arm'], p['seed'], protocol, objective) in selected_index
            for protocol in POINT_PROTOCOLS for objective in ('MSE', 'MAE')) for p in successful_paths)
    validation['status'] = 'PASS' if all(validation['checks'].values()) else 'FAIL'
    methods = list(dict.fromkeys(CORE_METHODS + tuple(path['arm'] for path in paths)))
    groups, target_rows = [], []
    for dataset in DATASETS:
        for method in methods:
            if not any(path['dataset'] == dataset and path['arm'] == method for path in paths):
                continue
            for protocol in POINT_PROTOCOLS:
                for objective in ('MSE', 'MAE'):
                    samples = [selected_index.get((dataset, method, s, protocol, objective), {}) for s in seeds]
                    for metric in METRICS:
                        values = [(s, row.get(metric)) for s, row in zip(seeds, samples)]
                        groups.append(dict(dataset=dataset, method=method, protocol=protocol,
                            selection_objective=objective, metric=metric, **summary(values, seeds)))
                    decision = paper_decision(dataset, [(s, r.get('MSE')) for s, r in zip(seeds, samples)],
                        [(s, r.get('MAE')) for s, r in zip(seeds, samples)], seeds)
                    alignment = config.get('paper_protocol_status', 'INFORMATION_INSUFFICIENT') if protocol == 'P1' else 'EXPLICIT_DECISION_PROTOCOL_DIFFERENCE'
                    target_rows.append(dict(dataset=dataset, method=method, protocol=protocol,
                        protocol_name=POINT_PROTOCOLS[protocol]['name'], selection_objective=objective,
                        validation_frozen_delivery_recipe=_frozen(frozen, dataset, method, protocol, objective),
                        protocol_status=alignment, paper_version='2606.10592v1', paper_MSE_table=9,
                        paper_MAE_table=10, full_paper_reproduction_claim=False, **decision))
    _table(out/'confirmation_summary.csv', groups)
    _table(out/'confirmation_by_seed.csv', selected, CHECKPOINT_COLUMNS)
    _table(out/'paper_target_comparison.csv', target_rows)
    performance = []
    for protocol in POINT_PROTOCOLS:
        dataset_status = {}
        for dataset in DATASETS:
            deliverable = [r for r in target_rows if r['dataset'] == dataset and r['protocol'] == protocol
                           and r['validation_frozen_delivery_recipe']]
            reached = any(r['strict_OR'] for r in deliverable)
            sufficient = bool(deliverable) and all(r['numeric_status'] != 'INSUFFICIENT_FIVE_SEED_EVIDENCE' for r in deliverable)
            dataset_status[dataset] = 'STRICT_OR_REACHED' if reached else 'NOT_REACHED' if sufficient else 'INSUFFICIENT_EVIDENCE'
        count = sum(value == 'STRICT_OR_REACHED' for value in dataset_status.values())
        performance.append(dict(protocol=protocol, dataset_status=dataset_status,
            status='BOTH_DATASETS_NUMERIC_TARGET_REACHED' if count == 2 else 'PARTIAL_ONE_DATASET_REACHED' if count == 1
            else 'INSUFFICIENT_EVIDENCE' if 'INSUFFICIENT_EVIDENCE' in dataset_status.values() else 'NUMERIC_TARGET_NOT_REACHED',
            full_paper_reproduction_claim=False, selection_source='Frozen validation recipes only; no test reselection'))
    _atomic_json(out/'performance_status.json', performance)

    comparisons = list(CONTRASTS)
    for path in paths:
        if path['arm'] not in CORE_METHODS:
            components = path.get('component_baselines', [])
            if len(components) != 2:
                gaps.append(dict(experiment_id=path['experiment_id'], kind='MISSING_COMBINATION_COMPONENTS',
                    source='', detail='Conditional configuration must name its two validation-frozen component baselines'))
            for component in components:
                pair = (path['arm'], component)
                if pair not in comparisons:
                    comparisons.append(pair)
    paths_by_id = {path['experiment_id']: path for path in paths}
    paired = []
    paired_groups = []
    for dataset in DATASETS:
        for method, baseline in comparisons:
            if not any(path['dataset'] == dataset and path['arm'] == method for path in paths):
                continue
            for protocol in POINT_PROTOCOLS:
                for objective in ('MSE', 'MAE'):
                    seed_pairs = []
                    for s in seeds:
                        left = selected_index.get((dataset, method, s, protocol, objective))
                        right = selected_index.get((dataset, baseline, s, protocol, objective))
                        compatible, reason = _comparison_compatible(left, right, configurations) if left and right else (False, 'Missing selected method or baseline result')
                        left_treatment = selective_by_id.get(left['experiment_id'], {}) if left else {}
                        right_treatment = selective_by_id.get(right['experiment_id'], {}) if right else {}
                        for metric in METRICS:
                            a = left.get(metric) if left else None
                            b = right.get(metric) if right else None
                            valid = compatible and _number(a) and _number(b)
                            direction = 'toward_0.8' if metric == 'coverage_80' else 'interpret_with_coverage' if metric == 'width_80' else 'lower'
                            improved = (abs(a-.8) < abs(b-.8) if metric == 'coverage_80' else None
                                if metric == 'width_80' else a < b) if valid else None
                            row = dict(dataset=dataset, method=method, baseline=baseline, protocol=protocol,
                                selection_objective=objective, seed=s, metric=metric,
                                method_experiment_id=left['experiment_id'] if left else None,
                                baseline_experiment_id=right['experiment_id'] if right else None,
                                method_value=a, baseline_value=b, delta=a-b if valid else None,
                                relative_improvement=(b-a)/b if valid and b != 0 else None,
                                valid_pair=valid, reason=reason,
                                metric_direction=direction, improved=improved,
                                **paired_hardware_fields(paths_by_id.get(left['experiment_id']) if left else None,
                                    paths_by_id.get(right['experiment_id']) if right else None,left,right),
                                method_allocated_gpu_hours=paths_by_id[left['experiment_id']].get('allocated_gpu_hours') if left else None,
                                baseline_allocated_gpu_hours=paths_by_id[right['experiment_id']].get('allocated_gpu_hours') if right else None,
                                method_allocated_gpu_hours_lower_bound=paths_by_id[left['experiment_id']].get('allocated_gpu_hours_lower_bound') if left else None,
                                baseline_allocated_gpu_hours_lower_bound=paths_by_id[right['experiment_id']].get('allocated_gpu_hours_lower_bound') if right else None,
                                method_cost_measurement_status=paths_by_id[left['experiment_id']].get('allocated_gpu_hours_measurement_status', 'UNKNOWN') if left else 'UNKNOWN',
                                baseline_cost_measurement_status=paths_by_id[right['experiment_id']].get('allocated_gpu_hours_measurement_status', 'UNKNOWN') if right else 'UNKNOWN',
                                cost_scope='Marginal RL node including its attempts; settled Native allocation bounds, not optimizer-only time; cumulative/shared costs remain separate',
                                method_source=left['source_receipt'] if left else None,
                                baseline_source=right['source_receipt'] if right else None,
                                method_selective_treatment_status=left_treatment.get('treatment_status', 'NOT_APPLICABLE'),
                                baseline_selective_treatment_status=right_treatment.get('treatment_status', 'NOT_APPLICABLE'),
                                selective_effect_interpretation=('No effective weighting treatment: descriptive score differences are not a selective benefit or a selective-learning negative result'
                                    if left_treatment.get('global_weights_nonuniform') is False
                                    or right_treatment.get('global_weights_nonuniform') is False else
                                    'Randomization left weights unchanged; this pair cannot identify window-difficulty targeting'
                                    if any(t.get('treatment_status') == 'RANDOMIZATION_DID_NOT_CHANGE_WEIGHT_VALUES' for t in (left_treatment, right_treatment)) else
                                    'Treatment strength UNKNOWN; do not infer selective causality from point scores'
                                    if any(t.get('treatment_status') == 'UNKNOWN' for t in (left_treatment, right_treatment)) else
                                    'See selective_treatment_audit.csv for matched treatment and execution evidence'
                                    if left_treatment or right_treatment else 'NOT_APPLICABLE'),
                                delta_direction='method minus baseline; lower point/error scores favor method',
                                relative_formula='(baseline-method)/baseline; diagnostic directions require interpretation')
                            paired.append(row)
                            seed_pairs.append(row)
                    for metric in METRICS:
                        metric_rows = [row for row in seed_pairs if row['metric'] == metric]
                        paired_groups.append(dict(dataset=dataset, method=method, baseline=baseline,
                            protocol=protocol, selection_objective=objective, metric=metric,
                            no_effective_selective_treatment_seeds=sum('NO_EFFECTIVE_SELECTIVE_TREATMENT' in
                                (r['method_selective_treatment_status'], r['baseline_selective_treatment_status']) for r in metric_rows),
                            unchanged_permutation_seeds=sum('RANDOMIZATION_DID_NOT_CHANGE_WEIGHT_VALUES' in
                                (r['method_selective_treatment_status'], r['baseline_selective_treatment_status']) for r in metric_rows),
                            selective_treatment_unknown_seeds=sum('UNKNOWN' in
                                (r['method_selective_treatment_status'], r['baseline_selective_treatment_status']) for r in metric_rows),
                            **paired_summary(metric_rows, seeds)))
    _table(out/'paired_effects.csv', paired)
    _table(out/'paired_effects_summary.csv', paired_groups)

    stage_effects = []
    for path in paths:
        for protocol in POINT_PROTOCOLS:
            phase_rows = [row for row in rows if row['experiment_id'] == path['experiment_id']
                          and row['protocol'] == protocol and row['split'] == 'test'
                          and row['selection_objective'] == 'N/A']
            for newer, older in [('preparation_after', 'preparation_before'), ('final', 'preparation_after')]:
                a = next((row for row in phase_rows if row['stage'] == newer), None)
                b = next((row for row in phase_rows if row['stage'] == older), None)
                for metric in METRICS:
                    va = a.get(metric) if a else None
                    vb = b.get(metric) if b else None
                    stage_effects.append(dict(experiment_id=path['experiment_id'], dataset=path['dataset'],
                        method=path['arm'], seed=path['seed'], protocol=protocol,
                        contrast=newer+' minus '+older, metric=metric, method_value=va, baseline_value=vb,
                        delta=va-vb if _number(va) and _number(vb) else None,
                        method_source=a['source_receipt'] if a else None,
                        baseline_source=b['source_receipt'] if b else None,
                        status='AVAILABLE' if _number(va) and _number(vb) else 'MISSING_STAGE_READOUT'))
    _table(out/'checkpoint_stage_effects.csv', stage_effects)

    inventory, opd = [], []
    for path in paths:
        identifier = path['experiment_id']
        c = configurations[identifier]
        present = [r for r in selected if r['experiment_id'] == identifier]
        inventory.append(dict(experiment_id=identifier, dataset=path['dataset'], method=path['arm'], seed=path['seed'],
            shared_start_node_id=path.get('shared_start_node_id'),
            phase=path.get('phase', 'core'), status=path.get('status', 'UNKNOWN'),
            scientific_failure_terminal=path.get('scientific_failure_terminal', False),
            formal_test_receipt=path.get('formal_test_receipt'), selected_test_rows=len(present),
            registered_readout_combinations=6, configuration=c,
            training_receipt=path.get('training_receipt'), preparation_receipt=path.get('preparation_receipt'),
            selective_treatment_status=selective_by_id.get(identifier, {}).get('treatment_status', 'NOT_APPLICABLE'),
            selective_weights_nonuniform=selective_by_id.get(identifier, {}).get('global_weights_nonuniform'),
            allocated_gpu_hours=path.get('allocated_gpu_hours'), marginal_RL_gpu_hours=path.get('marginal_RL_gpu_hours'),
            cumulative_preparation_and_RL_gpu_hours=path.get('cumulative_preparation_and_RL_gpu_hours')))
        inventory[-1].update(path_hardware_fields(path))
        for cost_name in ('allocated_gpu_hours', 'marginal_RL_gpu_hours',
                          'cumulative_preparation_and_RL_gpu_hours', 'shared_SFT_gpu_hours'):
            for suffix in ('_lower_bound', '_measurement_status', '_scope', '_source'):
                inventory[-1][cost_name+suffix] = path.get(cost_name+suffix, 'UNKNOWN' if suffix == '_measurement_status' else None)
        inventory[-1]['cost_interpretation'] = 'Upper/lower allocation observations, not exact optimizer time; do not sum repeated shared-ancestor costs across scientific paths'
        if path['arm'].startswith('O0'):
            preparation = _read(path.get('preparation_receipt'), base, gaps, identifier,
                                required=path.get('scientific_failure_terminal') is not True)
            prep_config = preparation.get('config', {})
            rolling = _opd_logs(path.get('preparation_receipt'), base)
            if path.get('status') in ('ACCEPTED', 'COMPLETED', 'SUCCEEDED_ACCEPTED') and rolling.get('status') != 'THREE_ROLLING_CALLS_RECORDED':
                gaps.append(dict(experiment_id=identifier, kind='MISSING_OPD_ROLLING_EXECUTION_EVIDENCE',
                    source=path.get('preparation_receipt', ''), detail='Accepted OPD path lacks recorded three-state same-prefix/no-future execution evidence'))
            opd.append(dict(experiment_id=identifier, dataset=path['dataset'], method=path['arm'], seed=path['seed'],
                status=path.get('status', 'UNKNOWN'), preparation_receipt=path.get('preparation_receipt'),
                training_receipt=path.get('training_receipt'), formal_test_receipt=path.get('formal_test_receipt'),
                teacher=prep_config.get('teacher_config', prep_config.get('teacher', prep_config.get('teacher_checkpoint', 'UNKNOWN'))),
                prefix_source='frozen teacher' if path['arm'] == 'O03' else 'current student',
                actual_rollout_evidence=rolling,
                actual_optimizer_steps=preparation.get('cursor', {}).get('global_step', preparation.get('optimizer_steps')),
                optimizer_steps_this_attempt=preparation.get('optimizer_steps_this_attempt'),
                first_step=preparation.get('first_step', 'UNKNOWN'),
                readouts=preparation.get('readouts', {}), interpretation='Rolling-prefix adaptation during preparation only; native H96 downstream RL/inference'))
    _table(out/'configuration_inventory.csv', inventory)
    _table(out/'cost_results.csv', [{key:value for key,value in item.items() if key in ('experiment_id','dataset','method','seed','phase','status','cost_interpretation') or 'gpu_hours' in key} for item in inventory])
    with (out/'configurations_full.jsonl').open('w', encoding='utf-8') as handle:
        for path in paths:
            handle.write(_json(dict(experiment_id=path['experiment_id'], planned=path.get('planned_configuration'),
                                    effective=configurations[path['experiment_id']], source=path))+'\n')
    _table(out/'opd_execution_evidence.csv', opd)
    _table(out/'attempt_and_budget_ledger.csv', ledger, list(ledger[0]) if ledger else ['node_id', 'attempt', 'run_id', 'status', 'allocated_gpu_hours', 'optimizer_commitments'])
    _classify_gaps(gaps)
    blocking_gaps = [gap for gap in gaps if gap['blocking']]
    diagnostic_gaps = [gap for gap in gaps if not gap['blocking']]
    _table(out/'evidence_gaps.csv', gaps, ['experiment_id', 'kind', 'source', 'detail', 'blocking',
        'classification', 'impact', 'point_unavailability_evidence'])
    _atomic_json(out/'record_consistency.json', validation)
    _atomic_json(out/'scalar_decision_qualification.json', qualification)

    map_lines = ['# MAP-mode及论文数值目标核对', '',
        'P1使用期望路由权重的最大模式，输出该模式合法解析均值；同一预测计算MSE与MAE。并列固定取最小索引。',
        '本文件读取已冻结checkpoint的正式收据；没有在测试上选择方法、epoch或推理通道。',
        '论文目标为small/H96：ETTh1 0.364/0.399；ETTm1 0.280/0.330，分别来自v1附录Table 9/10。',
        '严格OR采用未舍入五种子均值。显示精度持平使用十进制ROUND_HALF_UP三位，单列且不等于严格达标。',
        'CRPS、NLL及其他诊断不参与数值达标、选点或晋级。缺种子不补造五种子结论。', '']
    map_lines += _markdown(['数据集', '方法', '验证选点', '五seed MSE均值', '五seed MAE均值', 'MSE有效seed', 'MAE有效seed', '数值状态', '冻结交付', '协议状态'],
        [[r['dataset'], r['method'], r['selection_objective'], _format(r['MSE_mean']), _format(r['MAE_mean']),
          r['MSE_valid_seeds'], r['MAE_valid_seeds'], r['numeric_status'], r['validation_frozen_delivery_recipe'], r['protocol_status']]
         for r in target_rows if r['protocol'] == 'P1'])
    map_lines += ['', '数值达标与协议一致性分开。工程通道达标不能称原论文复现；即使MAP数字达到，也不能外推到其他模型、数据集和horizon。',
                  '旧联合MSE/MAE/CRPS门槛结论不被回写。当前新目标明确为点指标OR。', '']
    (out/'MAP_MODE_AUDIT.md').write_text('\n'.join(map_lines), encoding='utf-8')

    effects = ['# 逐设置配对效果', '',
        '差值为方法减对照；相对改善为(对照−方法)/对照。所有配对固定数据集、种子、推理协议及验证选点目标。',
        '以下区间是五个训练种子配对均值差的Student t区间，df=4，仅描述不确定性；不作为达标或晋级门槛。',
        '共享起点、准备、原生RL和恢复成本分别保留，不能把等RL输入预算称为等总FLOPs或等墙钟。', '']
    effects += _markdown(['数据集', '配对', '通道', '验证选点', '指标', '有效seed', '均值差', '样本标准差', '均值差95%下界', '上界', '改善seed'],
        [[r['dataset'], r['method']+'−'+r['baseline'], r['protocol'], r['selection_objective'], r['metric'],
          r['valid_seeds'], _format(r['mean']), _format(r['sample_std']), _format(r['mean_difference_t95_lower']),
          _format(r['mean_difference_t95_upper']), r['improvement_seed_count']]
         for r in paired_groups if r['metric'] in ('MSE', 'MAE', 'CRPS')])
    effects += ['', '准备前后和固定预算final的差异另见checkpoint_stage_effects.csv；缺失读出保持缺失，不以其他checkpoint替代。', '']
    effects += ['## 选择性权重与匹配随机对照', '',
        '统计及随机置换证明来自既有JSON收据；本分析不读取权重数组，也不将配置声明包装为数值执行证明。',
        '全局权重恒定时，归一化加权目标退化为关闭选择权重的相应方法；E15/R15在其他设置匹配时退化为E03。',
        '此时可以保留点指标，但不能据差值声称选择性学习有效、无效或产生独立新收益。仅置换未改变权重值时，E15−R15也不可辨识窗口难度匹配。', '']
    effects += _markdown(['数据集', '方法', 'seed', '非均匀', '随机化改变数值', '同源状态', '分层分布精确匹配', '处理状态', '说明'],
        [[r['dataset'], r['method'], r['seed'], r['global_weights_nonuniform'], r['permutation_changes_weights'],
          r['cross_arm_source_status'], r['exact_matched_within_stratum_weight_distribution'],
          r['treatment_status'], r['interpretation']] for r in selective])
    effects += ['', '逐配置统计来源、训练配置覆盖优先级、每变量/预测位置覆盖和真实文件加载证据强弱见selective_treatment_audit.csv。', '']
    (out/'setting_effects.md').write_text('\n'.join(effects), encoding='utf-8')

    failed = [p for p in paths if p.get('scientific_failure_terminal') is True]
    successful = [p for p in paths if p.get('status') in ('ACCEPTED', 'COMPLETED', 'SUCCEEDED_ACCEPTED')]
    unfinished = [p for p in paths if p not in failed and p not in successful]
    complete = validation['status'] == 'PASS' and not unfinished and not blocking_gaps
    failure_lines = ['# 失败、未知与缺口', '',
        '科学失败终态只按显式标记和失败收据列示；平台失败、未提交和预算不足不自动转为科学负结果。', '']
    failure_lines += _markdown(['配置', '数据集', '方法', 'seed', '状态', '失败证据'],
        [[p['experiment_id'], p['dataset'], p['arm'], p['seed'], p.get('status', 'UNKNOWN'), p.get('failure_receipt', 'UNKNOWN')]
         for p in failed+unfinished])
    failure_lines += ['', '## 尚缺证据', '',
        '必要点指标、身份/起点绑定、无泄漏、真实OPD和缺失路径等缺口阻塞记录验收。CRPS、NLL、校准、尾部等评测诊断缺口仅保留报告，不否决有效点指标或OR达标。',
        '训练loss/梯度非有限不属于诊断例外；此处的非否决分类只适用于明确标识的评测诊断及处理量可辨识性限制。', '']
    failure_lines += _markdown(['配置', '类型', '阻塞', '来源', '缺口'],
        [[g['experiment_id'], g['kind'], g['blocking'], g['source'], g['detail']] for g in gaps])
    (out/'failures_and_gaps.md').write_text('\n'.join(failure_lines)+'\n', encoding='utf-8')
    report = ['# 本轮点预测确认与滚动前缀OPD最终汇总', '',
        f'报告基准UTC秒：{time.time():.3f}。独立科学路径{len(paths)}条；强制应有90条，条件最多20条。',
        f'当前成功标记{len(successful)}条，有证据的科学失败终态{len(failed)}条，其他未完成/未知{len(unfinished)}条。',
        f'登记/记录检查：{validation["status"]}；冻结选择读取状态：{frozen_status}；必要证据缺口{len(blocking_gaps)}条、非否决的诊断/可辨识性缺口{len(diagnostic_gaps)}条。',
        '本分析只读取既有标量，不构成模型数值重载验证。独立重载及指标复算必须引用其独立HOPE收据。',
        '历史测试集已暴露；五个新种子检验训练随机性，不创造新的未见测试集。',
        'P1/P3表中的CRPS、NLL及区间校准仍评估同一checkpoint的完整预测混合分布；不是确定性点输出的概率，也不是仅选定模式的概率评分。各协议的点指标按其单独预测定义计算。',
        '共享SFT起点单独展开，不计入科学路径分母：shared_sft_inventory.csv、shared_sft_checkpoint_results.csv、shared_sft_configurations_full.jsonl、shared_sft_curve_index.csv及shared_start_lineage.csv。主configuration_inventory.csv通过shared_start_node_id关联；主checkpoint_results.csv仍只列科学路径的准备/RL读出。',
        '共享起点表只整理已有验证结果，没有补造其测试结果。epoch保留原始来源语义，final的cursor表示完成后的游标；总optimizer_steps不能当作恢复后单个attempt的新增更新数。共享SFT的ETTh1 batch为64、ETTm1 batch为512；ETTm1/D-pre冻结LR为5e-7，其余共享组1e-7，原预注册概括文字勘误独立保留，准备/RL均为batch512及LR5e-7。',
        '共享节点摘要：'+_json(shared['summary']), '', '## 所有方法逐seed点指标及CRPS', '']
    report += _markdown(['数据集', '方法', 'seed', '通道', '验证选点', 'MSE', 'MAE', 'CRPS', '状态'],
        [[r['dataset'], r['method'], r['seed'], r['protocol'], r['selection_objective'],
          _format(r['MSE']), _format(r['MAE']), _format(r['CRPS']), r['point_result_status']] for r in selected])
    report += ['', '## 全部方法五种子均值', '']
    group_lookup = {(r['dataset'], r['method'], r['protocol'], r['selection_objective'], r['metric']): r for r in groups}
    report += _markdown(['数据集', '方法', '通道', '验证选点', 'MSE均值', 'MAE均值', 'CRPS均值', '点指标有效seed'],
        [[r['dataset'], r['method'], r['protocol'], r['selection_objective'], _format(r['MSE_mean']), _format(r['MAE_mean']),
          _format(group_lookup[(r['dataset'], r['method'], r['protocol'], r['selection_objective'], 'CRPS')]['mean']),
          f"{r['MSE_valid_seeds']}/{r['MAE_valid_seeds']}"] for r in target_rows])
    report += ['', '## 论文OR目标与协议限制', '', *map_lines[2:], '',
        '固定教师在五种子之间共享时，种子区间条件于这份教师，不包含教师训练或教师选择的不确定性。',
        '选择性处理审计见selective_treatment_audit.csv。权重恒定或随机置换未改变数值时，保留点指标但不得解释为有效新处理、独立收益或选择性学习的否定证据。',
        f'选择性配置共{len(selective)}条，其中全局恒定权重{sum(r["global_weights_nonuniform"] is False for r in selective)}条，处理量未知{sum(r["global_weights_nonuniform"] is None for r in selective)}条。',
        '## 历史checkpoint独立重载', '',
        '本节只比较原checkpoint的旧值与本轮独立重载值，原选点和历史验收不回写，也不进入新方法选择器。超容差不能直接解释为代码错误、硬件原因或新训练收益。工程采样通道与MAP/解析通道分别呈现；完整逐点旧值、新值、差值及原始收据见historical_replay_comparisons.csv。',
        *_markdown(['数据集','协议','split','比较数','容差内','超容差','不可用','最大绝对差'],
            [[r['dataset'],r['protocol'],r['split'],r['comparisons'],r['matches'],r['conflicts'],r['unavailable'],_format(r['maximum_absolute_delta'])] for r in historical_groups]),
        '## 实际执行源码证据', '',
        f'科学路径或共享节点×stage×attempt引用共{worker_reference_counts["path_stage_attempt_reference_rows"]}行；按(node,attempt,run)去重后为{worker_reference_counts["unique_node_attempt_run_keys"]}个登记attempt键，其中{worker_reference_counts["unique_attempts_with_run_id"]}个有run ID。共享准备被多路径引用，引用行数不等于实际attempt数或独立训练数；未绑定占位{worker_reference_counts["unbound_placeholder_reference_rows"]}行。见worker_execution_evidence.csv与worker_execution_reference_summary.json。小收据检查不构成模型数值复验，未用当前磁盘源码替代历史worker。',
        '硬件分训练、准备、正式评测三阶段分别记录，GPU型号和world_size任一缺失则匹配状态为UNKNOWN，不把两个UNKNOWN判成相同。兼容actual_gpu字段只表示数值读出评测设备；配对表另列training/preparation/evaluation_hardware_match。恢复轨迹可能跨卡，最终成功进程的硬件不能证明整条轨迹始终使用同类设备，须查逐attempt原始收据；硬件匹配仅供描述，不作为点指标或配对资格门槛。',
        '## 成本与完成状态', '', '本轮硬上限：1500实际分配GPU小时、320条更新轨迹提交承诺、同时最多24张GPU。',
        '成本状态来自协调器的逐attempt账本及冻结快照；未知项不能填0，不能重复累加共享祖先成本。闭合作业仍报告Native分配时间的观测上下界，不冒充精确优化器时间。逐路径成本及计量状态见cost_results.csv。',
        '唯一节点/尝试成本快照：'+_json(config.get('unique_attempt_cost_summary', config.get('budget_status', {}).get('allocation_summary', {'status':'UNKNOWN'}))),
        '成本收据：'+str(config.get('cost_snapshot_receipt','UNKNOWN')),
        '```json', _json(config.get('budget_status', {'status': 'UNKNOWN', 'source': config.get('attempt_budget_ledger')})), '```',
        f'可选预算裁定：触发{optional_budget["triggered_groups"]}组，完整准入{optional_budget["admitted_groups"]}组，预算未执行{optional_budget["budget_canceled_groups"]}组（{optional_budget["budget_canceled_scientific_slots"]}个计划种子槽位）。预算未执行项不进入实际科学路径分母，不阻塞核心90正式测试，不等于未触发或科学失败；完整依据见conditional_budget_decisions.csv和conditional_budget_slot_inventory.csv。',
        f'强制/条件状态快照：{_json(config.get("core_conditional_status", {}))}',
        '各预登记协议的性能状态：'+_json(performance),
        '配方缺口逐项核验（科学失败或P3解析均值不可用）：'+_json([
            {key:row[key] for key in ('dataset', 'protocol', 'objective', 'status', 'problems')}
            for row in gap_proof['gap_reviews']])+'。原始证据定位见recipe_gap_evidence.json及recipe_gap_audit.csv。',
        f'报告交付判定：{"记录齐备；仍需结合独立模型重载和代码验收收据" if complete else "未完成；保留未完成节点与证据缺口，不以负结果掩盖缺跑"}。',
        '纯评测诊断缺口不阻塞以上记录交付判定；必要正确性或执行证据缺口仍阻塞。论文点值OR状态单独从未舍入五种子结果计算，不由缺口总数决定。',
        '性能成功状态独立见paper_target_comparison.csv，不以完成数量、诊断改善或开源可运行性替代。',
        '逐配对结论见setting_effects.md，全部逐seed差值及原始来源见paired_effects.csv；不以测试最优格替换验证冻结交付配方。', '']
    (out/'FINAL_REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    result = dict(status='PASS' if complete else 'INCOMPLETE', optimizer_trajectories=0,
        analysis_kind='SCALAR_RECORD_AND_SEED_EFFECT_ANALYSIS', model_numerical_revalidation=False,
        scientific_paths=len(paths), successful_paths=len(successful), scientific_failed_terminal=len(failed),
        unfinished_paths=len(unfinished), checkpoint_rows=len(rows), selected_rows=len(selected),
        gap_rows=len(gaps), blocking_gap_rows=len(blocking_gaps), diagnostic_gap_rows=len(diagnostic_gaps),
        selective_treatment_rows=len(selective),worker_reference_counts=worker_reference_counts,optional_budget_admission=optional_budget,
        shared_SFT_nodes=shared['summary']['provided_shared_nodes'],shared_SFT_validation_roles=shared['summary']['checkpoint_validation_roles'],
        record_consistency=validation['status'], selection_freeze_receipt=config.get('selection_freeze_receipt'),
        recipe_gap_evidence_status=gap_proof['status'], proven_recipe_gap_keys=gap_proof['accepted_gap_keys'],
        performance_status=performance,
        device_argument=device, seed_argument=seed, seconds=time.time()-began,
        point_only_OR=True, diagnostic_scores_never_gate=True, test_values_used_for_selection=False)
    _atomic_json(out/'result.json', result)
    return result
