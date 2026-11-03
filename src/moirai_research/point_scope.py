"""Evidence for native evaluation and RL-only reliability; metadata, not masks."""
import json
from pathlib import Path

FILTER_INPUTS=('selective_artifact','selective_weight','reliability_weights','training_mask','label_filter')


def native_inference_case(case):
    if any(case.get(key) is not None for key in FILTER_INPUTS):
        raise ValueError('Formal inference cannot receive training reliability or label-filter inputs')
    if case.get('splits')!=['val','test'] or case.get('protocol',{}).get('horizon')!=96:
        raise ValueError('Native inference scope requires complete H96 validation and test')


def completed_evaluation_scope(cases,records):
    """The native evaluator asserts complete futures; verify its executed scope.

    This is executed-entry/config/complete-window evidence. It does not claim a
    separate recomputation of prediction arrays or per-element masks.
    """
    by={row['case_id']:row for row in records}
    if not cases or len(by)!=len(records) or {c['case_id'] for c in cases}!=set(by):
        raise ValueError('Incomplete or duplicate native evaluation scope')
    evidence=[]
    for case in cases:
        native_inference_case(case)
        source=by[case['case_id']]['receipt'];receipt=json.loads(Path(source).read_text())
        actual=receipt.get('config',{})
        native_inference_case(actual)
        for key in ('dataset','seed','route','protocol','artifact'):
            if actual.get(key)!=case.get(key):raise ValueError('Native readout receipt contract differs: '+key)
        if receipt.get('status')!='PASS':raise ValueError('Native readout receipt is not complete')
        for split in ('val','test'):
            value=receipt.get('results',{}).get(split,{})
            if value.get('complete') is not True or value.get('split')!=split or value.get('seed')!=case['seed'] \
                    or value.get('windows')!=case['protocol']['windows'][split]:
                raise ValueError('Native readout did not complete its bound windows/seed')
        evidence.append(dict(case_id=case['case_id'],canonical_case=by[case['case_id']].get('canonical_case'),
            receipt=source,complete_splits=['val','test']))
    return dict(status='PASS',all_observed_native_evaluator_contract=True,
        training_reliability_inputs_absent=True,cases=evidence,
        evidence_level='Actual complete native evaluator receipts with its all-observed future assertion; not independent mask-array recomputation')


def rl_only_reliability_scope(training,preparation):
    config=training.get('config',{});before=preparation.get('config',{});direct=training.get('selective_treatment') or {}
    required=(training.get('status')=='PASS',preparation.get('status')=='PASS',
        config.get('phase')=='scientific',before.get('phase')=='preparation',
        config.get('seed')==before.get('seed'),config.get('dataset')==before.get('dataset'),
        config.get('checkpoint')==preparation.get('readouts',{}).get('final',{}).get('checkpoint'),
        before.get('selective_artifact') is None,not preparation.get('selective_treatment'),
        direct.get('status')=='ACTUALLY_LOADED_IN_RL',direct.get('frozen') is True,
        direct.get('same_weights_for_all_J_G_candidates') is True,
        training.get('test_used') is False,preparation.get('test_used') is False)
    return all(required)
