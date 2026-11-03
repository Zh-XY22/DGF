"""Point-only validation choices for the finite confirmation study.

This module reads validation scalars only. Probability and dynamics diagnostics
never affect promotion, checkpoint selection, or conditional combinations.
"""
import math

POINT_PROTOCOLS = {
    'P1': dict(name='paper_MAP', MSE='MAP_MSE', MAE='MAP_MAE'),
    'P2': dict(name='engineering_100', MSE='MSE', MAE='MAE'),
    'P3': dict(name='mixture_mean', MSE='mixture_analytic_MSE', MAE='mixture_analytic_MAE'),
}
OBJECTIVES = ('MSE', 'MAE')


def finite_point(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def point_unavailability_evidence(metrics, protocol, objective=None):
    """Existing scalar proof for an unavailable secondary readout, never a fallback.

    P1 must remain valid. Missing execution, a negative finite score, a contradictory
    VALID flag, or an unrelated probability diagnostic cannot explain missing points.
    """
    if protocol not in ('P2', 'P3') or not isinstance(metrics, dict):
        return None
    if objective is not None and objective not in OBJECTIVES:
        raise ValueError('Unknown point selection objective')
    states = metrics.get('point_protocol_status', {})
    issues = metrics.get('diagnostic_issues', {})
    if not isinstance(states, dict) or not isinstance(issues, dict):
        return None
    if any(not isinstance(value, dict) for key, value in states.items() if key in POINT_PROTOCOLS):
        return None
    if states.get(protocol, {}).get('status') != 'INVALID' or states.get('P1', {}).get('status') != 'VALID':
        return None
    for key, fields in POINT_PROTOCOLS.items():
        valid = all(finite_point(metrics.get(fields[name])) for name in OBJECTIVES)
        if states.get(key, {}).get('status') == 'VALID' and not valid:
            return None
    if not all(finite_point(metrics.get(POINT_PROTOCOLS['P1'][name])) for name in OBJECTIVES):
        return None
    fields = [POINT_PROTOCOLS[protocol][name] for name in ((objective,) if objective else OBJECTIVES)]
    unavailable = [key for key in fields if not finite_point(metrics.get(key))]
    if not unavailable:
        return None
    reasons = []
    for key in unavailable:
        value = metrics.get(key)
        if key not in metrics or value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool) or math.isfinite(value)):
            return None
        issue = issues.get(key, {})
        if not (isinstance(issue, dict) and finite_point(issue.get('occurrences')) and issue['occurrences'] > 0
                and finite_point(issue.get('first_origin')) and isinstance(issue.get('first_message'), str)
                and issue['first_message'].strip()):
            return None
        reasons.append(dict(metric=key, **{name:issue[name] for name in ('occurrences','first_origin','first_message')}))
    return dict(status='PROVEN_POINT_PROTOCOL_UNAVAILABLE', protocol=protocol,
                unavailable_metrics=unavailable, supporting_protocols=['P1'], issues=reasons)


def point_improvements(metrics, best_values):
    """Update six independent minima; a missing protocol stays explicitly invalid."""
    improved = []
    for protocol, fields in POINT_PROTOCOLS.items():
        for objective in OBJECTIVES:
            field = fields[objective]
            value = metrics.get(field)
            label = f'{protocol}_val_{objective}_best'
            if finite_point(value) and (label not in best_values or value < best_values[label]):
                best_values[label] = value
                improved.append(dict(label=label, protocol=protocol, protocol_name=fields['name'],
                    selection_objective=objective, validation_metric=field, validation_value=value))
    return improved


def selection_records(best_values, best_paths):
    records = {}
    for protocol, fields in POINT_PROTOCOLS.items():
        for objective in OBJECTIVES:
            label = f'{protocol}_val_{objective}_best'
            if label in best_paths and finite_point(best_values.get(label)):
                records[label] = dict(status='VALID', checkpoint=best_paths[label], protocol=protocol,
                    protocol_name=fields['name'], selection_objective=objective,
                    validation_metric=fields[objective], validation_value=best_values[label])
            else:
                records[label] = dict(status='INVALID', checkpoint=None, protocol=protocol,
                    protocol_name=fields['name'], selection_objective=objective,
                    validation_metric=fields[objective], validation_value=None,
                    reason='No finite valid point readout was available; no diagnostic substitution')
    return records


def selected_checkpoint_path(record, readout):
    path = record.get('checkpoint')
    if not path or path != readout.get('checkpoint'):
        raise RuntimeError('Selected checkpoint and its reload readout must identify the same path')
    return path


def same_device_readout_check(record, first, repeated):
    """Verify the selected metric itself, preserving the existing reload tolerance."""
    metric = record.get('validation_metric')
    point_fields = {fields[objective] for fields in POINT_PROTOCOLS.values() for objective in OBJECTIVES}
    if metric not in point_fields:
        raise ValueError('Reload verification requires the selected point metric')
    if not finite_point(first.get(metric)) or not finite_point(repeated.get(metric)):
        raise RuntimeError('Selected point metric unavailable during same-device reload: ' + metric)
    checked = []
    for key in dict.fromkeys(['MSE', 'MAE', 'MAP_MSE', 'MAP_MAE', metric]):
        a, b = repeated.get(key), first.get(key)
        if finite_point(a) and finite_point(b):
            if abs(a-b) > max(1e-9, abs(a)*1e-7):
                raise RuntimeError('Same-device point checkpoint reload differs: ' + key)
            checked.append(key)
    return dict(selected_metric=metric, checked_metrics=checked,
                tolerance='max(1e-9, abs(repeated_value)*1e-7)')


def validate_qualification():
    """Run only in a HOPE numerical qualification worker, never on coordinator."""
    best = {}
    six = dict(MSE=2., MAE=3., MAP_MSE=4., MAP_MAE=5.,
        mixture_analytic_MSE=6., mixture_analytic_MAE=7., CRPS=float('nan'))
    assert len(point_improvements(six, best)) == 6
    assert len(point_improvements(dict(six, CRPS=-1.e99), best)) == 0
    assert [r['label'] for r in point_improvements(dict(six, MAP_MAE=4.), best)] == ['P1_val_MAE_best']
    invalid = point_improvements(dict(six, MAP_MSE=float('nan'), MAP_MAE=None), {})
    assert len(invalid) == 4
    return dict(status='PASS', six_independent_point_choices=True,
        diagnostic_values_do_not_gate=True, invalid_protocol_does_not_reject_other_protocols=True)
