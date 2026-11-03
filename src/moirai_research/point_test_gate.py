"""Metadata-only new-campaign formal-test authorization checks."""
import json
from pathlib import Path

CAMPAIGN = 'point_opd_confirmation_20261024_130931'
WARMUP_CAMPAIGN = 'sft_warmup_etth1_20261028_000001'


def test_release(path, scientific_path):
    record = json.loads(Path(path).read_text())
    if record.get('status') != 'PASS' or record.get('test_release') is not True or record.get('test_used') is not False:
        raise ValueError('Formal tests require a successful validation-only release')
    if record.get('campaign') == WARMUP_CAMPAIGN:
        if (record.get('dataset') != 'ETTh1' or record.get('horizon') != 96
                or record.get('model_id') != 'Salesforce/moirai-1.0-R-small'):
            raise ValueError('The warmup release differs from its finite data/model scope')
        for field in ['core_validation_complete','conditional_validation_complete','final_selection_frozen']:
            if record.get(field) is not True:
                raise ValueError('Warmup validation/test firewall is not released: '+field)
        paths=record.get('authorized_scientific_paths')
        if (not isinstance(paths,list) or not paths or len(paths)>40
                or len(paths)!=len(set(paths)) or scientific_path not in paths):
            raise ValueError('Warmup path is absent, duplicated, or outside the finite release')
        if not isinstance(record.get('frozen_epoch'),(int,float)) or not isinstance(record.get('recipes'),list):
            raise ValueError('Warmup frozen time and recipes are required')
        return record
    if record.get('campaign') != CAMPAIGN or record.get('mandatory_paths') != 90:
        raise ValueError('A historical or different-campaign release cannot authorize new tests')
    for field in ['core_validation_complete','conditional_validation_complete','final_selection_frozen']:
        if record.get(field) is not True:
            raise ValueError('The global validation/test firewall is not released: '+field)
    paths = record.get('authorized_scientific_paths')
    if not isinstance(paths,list) or len(paths) != len(set(paths)) or len(paths) > 110 or scientific_path not in paths:
        raise ValueError('This scientific path is not uniquely authorized in the frozen finite matrix')
    if not isinstance(record.get('frozen_epoch'),(int,float)) or not isinstance(record.get('recipes'),list):
        raise ValueError('Frozen time and recipe records are required')
    return record
