"""Finite validation-only conditional combinations and final recipe freezing."""
from decimal import Decimal, ROUND_HALF_UP
from statistics import mean
from .point_selection import POINT_PROTOCOLS, OBJECTIVES, finite_point

CORE_METHODS = ('E01', 'E03', 'E16', 'E17', 'E15', 'R15', 'O01', 'O02', 'O03')
DATASETS = ('ETTh1', 'ETTm1')
CANDIDATES = ('E15', 'E16', 'E17', 'R15', 'O01', 'O02', 'O03')
PREPARATION = {'O01', 'O02', 'O03'}
REWARD = {'E15', 'E16', 'E17', 'R15'}


def compatible(pair):
    return set(pair) in [{'E15', 'E17'}, {'O01', 'E17'}, {'O02', 'E17'}, {'O01', 'E15'}, {'O02', 'E15'}]


def validation_index(rows):
    indexed = {}
    for row in rows:
        if row.get('split', 'val') != 'val' or row.get('test_used', False):
            raise ValueError('Selectors may consume validation metadata only')
        key = (row['dataset'], row['method'], int(row['seed']), row['protocol'], row['selection_objective'])
        if key in indexed:
            raise ValueError('Duplicate validation selection record: '+str(key))
        indexed[key] = row
    return indexed


def cohort(indexed, dataset, method, seeds, protocol, objective):
    records = [indexed.get((dataset, method, seed, protocol, objective)) for seed in seeds]
    if any(record is None or not finite_point(record.get('validation_value')) for record in records):
        return dict(valid=False, count=sum(record is not None and finite_point(record.get('validation_value')) for record in records), records=records,
            scientific_terminal_seeds=[seed for seed,record in zip(seeds,records)
                if record is not None and record.get('scientific_failure_terminal') is True])
    return dict(valid=True, count=len(seeds), records=records,
        values=[record['validation_value'] for record in records],
        mean=mean(record['validation_value'] for record in records))


def component_signal(indexed, dataset, method, seeds, protocol, objective):
    treatment = cohort(indexed, dataset, method, seeds, protocol, objective)
    baseline = cohort(indexed, dataset, 'E03', seeds, protocol, objective)
    if not treatment['valid'] or not baseline['valid']:
        return dict(method=method, baseline='E03', eligible=False,
            reason='Five valid paired validation seeds are unavailable',
            reason_code='NOT_TRIGGERED_INSUFFICIENT_VALID_COHORT'
                if treatment.get('scientific_terminal_seeds') or baseline.get('scientific_terminal_seeds')
                else 'WAITING_VALIDATION_EVIDENCE',
            scientific_terminal_seeds=sorted(set(treatment.get('scientific_terminal_seeds',[])+baseline.get('scientific_terminal_seeds',[]))),
            treatment_count=treatment['count'], baseline_count=baseline['count'])
    improved = sum(a < b for a, b in zip(treatment['values'], baseline['values']))
    relative = (baseline['mean']-treatment['mean'])/baseline['mean'] if baseline['mean'] > 0 else None
    eligible = treatment['mean'] < baseline['mean'] and improved >= 3
    return dict(method=method, baseline='E03', eligible=eligible,
        mean=treatment['mean'], baseline_mean=baseline['mean'], improved_seeds=improved,
        relative_improvement=relative, per_seed_deltas=[a-b for a,b in zip(treatment['values'],baseline['values'])],
        reason='Mean improves and at least 3/5 paired seeds improve' if eligible else 'Mean improvement or 3/5 sign rule failed')


def conditional_combinations(rows, seeds, p1_unavailable=None):
    """At most one MSE and one MAE recipe per dataset, then semantic dedup.

    P2 fallback needs explicit implementation-invalid P1 evidence supplied before
    test release. Missing seeds or poor P1 values are never fallback reasons.
    """
    seeds = tuple(seeds)
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError('The five new seeds must already be frozen')
    indexed = validation_index(rows); p1_unavailable = p1_unavailable or {}
    missing = [(ds, arm, seed) for ds in DATASETS for arm in CORE_METHODS for seed in seeds
        if (ds, arm, seed, 'P1', 'MSE') not in indexed]
    if missing:
        raise ValueError('All 90 mandatory validation/terminal records must precede conditional selection: '+str(missing))
    selected = {}; decisions = []; student_choices = []
    for ds in DATASETS:
        protocol = 'P2' if ds in p1_unavailable else 'P1'
        if protocol == 'P2' and not p1_unavailable[ds]:
            raise ValueError('P1 fallback requires explicit unavailable-implementation evidence')
        for objective in OBJECTIVES:
            students = [(arm, cohort(indexed, ds, arm, seeds, protocol, objective)) for arm in ['O01','O02']]
            student_terminal = any(value.get('scientific_terminal_seeds') for _,value in students)
            students = [(arm, value) for arm, value in students if value['valid']]
            chosen = min(students, key=lambda pair:(pair[1]['mean'],pair[0]))[0] if students else None
            student_choices.append(dict(dataset=ds, protocol=protocol, objective=objective, selected_student_opd=chosen))
            specs = [('C15', ['E15','E17'], None), ('CO-N', [chosen,'E17'], chosen), ('CO-S', [chosen,'E15'], chosen)]
            eligible = []
            for template, components, opd in specs:
                config_id = template if opd is None else template+'-'+opd
                signals = [component_signal(indexed, ds, arm, seeds, protocol, objective) for arm in components if arm is not None]
                passes = len(signals) == 2 and all(signal['eligible'] for signal in signals)
                score = min(signal['relative_improvement'] for signal in signals) if passes else None
                record = dict(dataset=ds, objective=objective, protocol=protocol,
                    configuration_id=config_id, template=template, opd_arm=opd,
                    components=components, component_signals=signals, eligible=passes,
                    ranking_weaker_relative_improvement=score,
                    reason='Eligible under same-target paired rules' if passes else 'Rule not triggered; no cross-metric substitution',
                    status='ELIGIBLE' if passes else 'NOT_TRIGGERED_INSUFFICIENT_VALID_COHORT'
                    if any(signal.get('reason_code')=='NOT_TRIGGERED_INSUFFICIENT_VALID_COHORT' for signal in signals)
                        or (template!='C15' and chosen is None and student_terminal)
                    else 'NOT_TRIGGERED')
                decisions.append(record)
                if passes: eligible.append(record)
            if eligible:
                winner = min(eligible, key=lambda row:(-row['ranking_weaker_relative_improvement'],row['configuration_id']))
                key = (ds, winner['configuration_id'])
                if key not in selected:
                    selected[key] = dict(dataset=ds, configuration_id=winner['configuration_id'],
                        template=winner['template'], opd_arm=winner['opd_arm'],
                        components=winner['components'], triggering_objectives=[],
                        parent_stage='RL-before preparation checkpoint; full four-epoch RL restarts',
                        conditional_research=True)
                selected[key]['triggering_objectives'].append(objective)
    assert all(sum(row['dataset'] == ds for row in selected.values()) <= 2 for ds in DATASETS)
    return dict(status='PASS', test_used=False, mandatory_paths=90, seed_order=list(seeds),
        selected=list(selected.values()), candidate_decisions=decisions, student_opd_choices=student_choices,
        fallback_evidence=p1_unavailable, diagnostic_metrics_used=False,
        ranking_rule='Per objective: descending weaker component relative mean improvement, configuration ID tie break')


def freeze_final_recipes(rows, seeds, required_methods_by_dataset):
    indexed = validation_index(rows); frozen = []; exclusions = []
    for ds in DATASETS:
        methods = sorted(required_methods_by_dataset[ds])
        for protocol in POINT_PROTOCOLS:
            for objective in OBJECTIVES:
                candidates = []
                for method in methods:
                    group = cohort(indexed, ds, method, seeds, protocol, objective)
                    if group['valid']:
                        candidates.append((method, group))
                    else:
                        exclusions.append(dict(dataset=ds, method=method, protocol=protocol,
                            objective=objective, valid_seeds=group['count'], reason='Incomplete finite point cohort'))
                if not candidates:
                    frozen.append(dict(dataset=ds, protocol=protocol, objective=objective,
                        status='INSUFFICIENT_EVIDENCE', method=None, valid_seeds=0))
                    continue
                method, group = min(candidates, key=lambda pair:(pair[1]['mean'],pair[0]))
                if any(not row.get('checkpoint') for row in group['records']):
                    raise ValueError('A frozen selected validation record has no checkpoint')
                frozen.append(dict(dataset=ds, protocol=protocol, objective=objective, status='FROZEN',
                    method=method, validation_mean=group['mean'], valid_seeds=len(seeds),
                    seed_order=list(seeds), checkpoints=[dict(seed=seed, experiment_id=row.get('experiment_id'),
                        checkpoint=row['checkpoint'], validation_value=row['validation_value'],
                        source=row.get('source')) for seed,row in zip(seeds,group['records'])]))
    eligible = [recipe for recipe in frozen if recipe['status']=='FROZEN']
    gaps = [recipe for recipe in frozen if recipe['status']!='FROZEN']
    return dict(status='PASS', test_used=False, recipes=eligible, recipe_gaps=gaps,
        recipe_state='COMPLETE_POINT_RECIPES' if len(eligible)==12 else
            'PARTIAL_COMPLETE_FIVE_SEED_RECIPES' if eligible else 'NO_COMPLETE_FIVE_SEED_RECIPE', exclusions=exclusions,
        criterion='Each channel and MSE/MAE independently; no probability-score gate')


def paper_target_status(mse_values, mae_values, target_mse, target_mae, expected_seeds=5):
    if len(mse_values) != expected_seeds or len(mae_values) != expected_seeds or not all(finite_point(x) for x in list(mse_values)+list(mae_values)):
        return dict(status='INSUFFICIENT_EVIDENCE', strict_OR=None, display_tie=None,
            effective_mse_seeds=len(mse_values), effective_mae_seeds=len(mae_values))
    mse = mean(mse_values); mae = mean(mae_values)
    strict = mse <= target_mse or mae <= target_mae
    rounded = lambda value: Decimal(str(value)).quantize(Decimal('.001'), rounding=ROUND_HALF_UP)
    display_tie = not strict and (rounded(mse) == rounded(target_mse) or rounded(mae) == rounded(target_mae))
    return dict(status='STRICT_NUMERIC_OR' if strict else 'DISPLAY_PRECISION_TIE' if display_tie else 'NOT_REACHED',
        strict_OR=strict, display_tie=display_tie, mean_MSE=mse, mean_MAE=mae,
        target_MSE=target_mse, target_MAE=target_mae,
        delta_MSE=mse-target_mse, delta_MAE=mae-target_mae,
        protocol_equivalence='Separate audit required; this function establishes numeric status only')


def validate_qualification():
    """Worker-only regression proof for conditional rules and strict/display OR."""
    seeds = (101,211,307,401,503)
    rows = []
    values = {
        'MSE': {'E15':.98,'E17':.97,'O01':.96,'O02':.95},
        'MAE': {'E15':.94,'E17':.97,'O01':.95,'O02':.96},
    }
    for ds in DATASETS:
        for arm in CORE_METHODS:
            for seed in seeds:
                for protocol in POINT_PROTOCOLS:
                    for objective in OBJECTIVES:
                        rows.append(dict(dataset=ds, method=arm, seed=seed, protocol=protocol,
                            selection_objective=objective, validation_value=values[objective].get(arm,1.),
                            checkpoint=f'qualification-only/{ds}/{arm}/{seed}/{objective}',
                            split='val', CRPS=float('nan')))
    selected = conditional_combinations(rows,seeds)
    for ds in DATASETS:
        assert {r['configuration_id'] for r in selected['selected'] if r['dataset']==ds} == {'CO-N-O02','CO-S-O01'}
    frozen = freeze_final_recipes(rows,seeds,{ds:CORE_METHODS for ds in DATASETS})
    assert len(frozen['recipes']) == 12
    assert paper_target_status([.5]*5,[.399282]*5,.364,.399)['status'] == 'DISPLAY_PRECISION_TIE'
    assert paper_target_status([.364]*5,[.9]*5,.364,.399)['status'] == 'STRICT_NUMERIC_OR'
    assert paper_target_status([.5]*4,[.39]*4,.364,.399)['status'] == 'INSUFFICIENT_EVIDENCE'
    try:
        conditional_combinations([dict(rows[0],split='test')]+rows[1:],seeds)
    except ValueError:
        pass
    else:
        raise AssertionError('Test metadata was not rejected')
    return dict(status='PASS', ninety_path_gate=True, paired_same_objective_sign_rule=True,
        at_most_two_conditional_recipes_per_dataset=True, diagnostic_values_ignored=True,
        strict_or_separate_from_display_rounding=True, test_inputs_rejected=True)
