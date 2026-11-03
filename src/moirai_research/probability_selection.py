"""Separate distribution-score readouts, never a gate on point-only choices."""
import math

LABEL='distribution_val_CRPS_best'

def probability_improvements(metrics,values,enabled):
    value=metrics.get('CRPS')
    valid=isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0
    if enabled and valid and (LABEL not in values or value<values[LABEL]):
        values[LABEL]=value
        return [dict(label=LABEL,protocol='distribution',selection_objective='CRPS',validation_metric='CRPS',validation_value=value,
            scope='Separate probability checkpoint only; point minima and point promotion are unchanged')]
    return []

def probability_records(values,paths):
    if LABEL not in paths:return dict(status='UNAVAILABLE',checkpoint=None,reason='No valid distribution score readout; does not invalidate point results')
    return dict(status='VALID',checkpoint=paths[LABEL],validation_metric='CRPS',validation_value=values[LABEL],
        estimator='Historical100 iid sample fair empirical CRPS; same estimator under explicit fair_CRPS alias; high-budget confirmation separate')
