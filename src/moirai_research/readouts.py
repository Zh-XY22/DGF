"""Frozen, independently chosen inference readouts; no test-time selection."""
from .io_utils import load_checkpoint
import json
import time
from pathlib import Path
import torch
from .evaluation import evaluate_config
from .runtime import atomic_json


def evaluate_readouts(config_path, output, device='cuda', seed=1):
    config=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    frozen=json.loads(Path(config['test_release_receipt']).read_text())
    binding=json.loads(Path(config['readout_selection_receipt']).read_text())
    assert frozen['status']=='PASS' and frozen['test_release'] and not frozen['test_used']
    assert binding['status']=='PASS' and not binding['test_used'] and binding['scientific_path']==config['scientific_path']
    assert binding['seed']==seed and config['seed']==seed
    assert set(binding['all_readouts'])=={c['case_id'] for c in config['cases']}
    began=time.time();canonical=[];cases=[]
    prior_root=Path(config['resume_readouts']) if config.get('resume_readouts') else None
    # Equality of complete stored parameters is checked inside this HOPE worker.
    # Aliases preserve every requested readout without repeating predictions.
    states=[]
    for case in config['cases']:
        key=case['case_id'];assert binding['all_readouts'][key]['checkpoint']==case['checkpoint']
        assert case['seed']==seed and case['splits']==['test']
        state=load_checkpoint(case['checkpoint'],map_location='cpu',weights_only=False)
        params=state['policy_state'];identity=(case['route'],case['protocol'],case.get('force_fixed_kappa'))
        same=None
        for old_key,old_identity,old in states:
            if identity==old_identity and params.keys()==old.keys() and all(torch.equal(params[k],old[k]) for k in params):
                same=old_key;break
        if same is not None:
            record=dict(case_id=key,canonical_case=same,checkpoint=case['checkpoint'],
                deduplication='Exact complete stored policy parameters and data/inference protocol equality checked in worker')
        else:
            states.append((key,identity,params));directory=out/key;directory.mkdir(exist_ok=True)
            local=directory/'config.json';atomic_json(local,case)
            prior=prior_root/key if prior_root is not None and (prior_root/key/'config.json').is_file() else None
            result=evaluate_config(str(local),directory,device,seed,prior)
            assert result['status']=='PASS' and result['results']['test']['complete']
            record=dict(case_id=key,canonical_case=key,checkpoint=case['checkpoint'],receipt=str(directory/'result.json'))
            canonical.append(key);torch.cuda.empty_cache()
        record.update(checkpoint_progress=state.get('progress'),
            checkpoint_progress_status='STORED_METADATA_RECORDED' if isinstance(state.get('progress'),dict) else 'UNAVAILABLE',
            checkpoint_progress_scope='Read inside the numerical worker; not inferred from the checkpoint filename')
        cases.append(record);atomic_json(out/'progress.json',cases)
    diagnostic_receipt=None
    if config.get('mode_diagnostic'):
        from .mode_diagnostics import diagnose
        path=out/'mode_diagnostic_config.json';atomic_json(path,config['mode_diagnostic'])
        diagnostic=diagnose(path,out/'mode_diagnostics',device,seed)
        assert diagnostic['status']=='PASS' and not diagnostic['test_used']
        diagnostic_receipt=str(out/'mode_diagnostics/result.json')
    result=dict(status='PASS',scientific_path=config['scientific_path'],config=config,cases=cases,
        canonical_cases=canonical,selection=binding['selected'],selection_frozen_before_test=True,
        mode_diagnostics_receipt=diagnostic_receipt,
        optimizer_trajectories=0,optimizer_steps=0,seed=seed,seconds=time.time()-began,
        actual_gpu=torch.cuda.get_device_name(),scope='Complete formal test for predeclared readouts; test does not choose a checkpoint or method')
    atomic_json(out/'result.json',result);return result
