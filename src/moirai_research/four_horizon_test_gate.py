"""Fail closed until validation-only research and per-state choices are frozen."""
from datetime import datetime,timezone
import json
from pathlib import Path


def check_release(config):
    file=config.get('four_horizon_test_release')
    if not file:raise ValueError('Four-horizon formal test remains sealed')
    release=json.loads(Path(file).read_text())
    if release.get('status')!='FROZEN_BEFORE_FORMAL_TEST' or release.get('version') not in [1,2]:
        raise ValueError('Invalid four-horizon test release')
    if not release.get('all_validation_decisions_complete'):
        raise ValueError('Exploration/conditional/confirmation choices are not frozen')
    frozen=datetime.fromisoformat(release['frozen_at'])
    if frozen.tzinfo is None or frozen>datetime.now(timezone.utc):raise ValueError('Invalid freeze time')
    path=config.get('scientific_path');label=config.get('checkpoint_label')
    source=config.get('checkpoint_source',config.get('checkpoint'))
    found=[r for r in release['allowed_checkpoints'] if r['path_id']==path and r['label']==label and r['checkpoint']==source
        and r['horizon']==config['protocol']['horizon'] and r['seed']==config['seed']]
    if len(found)!=1:raise ValueError('Checkpoint/path/selection label is not in the frozen release')
    if found[0]['horizon']!=config['protocol']['horizon']:raise ValueError('Frozen horizon differs')
    if found[0]['seed']!=config['seed']:raise ValueError('Frozen training seed differs')
    if release['version']==2:
        if config.get('evaluation_source_reference')!=release['source_code_reference']:raise ValueError('Frozen public evaluator source identity differs')
        frozen_protocol=found[0]['data_protocol_contract']
        for key,value in frozen_protocol.items():
            if config['protocol'].get(key)!=value:raise ValueError('Frozen data/scaler/variable protocol differs: '+key)
        model=found[0]['model_contract']
        for key,value in model.items():
            if config.get(key)!=value:raise ValueError('Frozen model inference identity differs: '+key)
        channel=config.get('evaluation_channel')
        if channel=='point100':
            if config['protocol'].get('eval_input_block',64)!=release['point_evaluation']['input_block']:raise ValueError('Frozen point evaluation block/RNG rule changed')
            if 'common_origin_diagnostics' in release['point_evaluation'] and bool(config.get('common_origin_diagnostics',False))!=release['point_evaluation']['common_origin_diagnostics']:
                raise ValueError('Frozen common-origin diagnostic contract changed')
        elif channel=='distribution1000':
            rule=release['probability_evaluation']
            if config.get('distribution_samples')!=rule['samples'] or config.get('distribution_eval_seed')!=rule['evaluation_seed'] or config.get('distribution_input_block')!=rule['input_block']:
                raise ValueError('Frozen distribution sampling budget or RNG changed')
        else:raise ValueError('Unregistered frozen evaluation channel')
    return found[0]
