"""Finite historical checkpoint replay with per-case RNG and exact-state aliases."""
import json, time
from pathlib import Path
import torch
from .io_utils import load_checkpoint
from .runtime import atomic_json
from .evaluation import evaluate_config
from .point_selection import POINT_PROTOCOLS, finite_point
from .point_readouts import state_identity, same_parameters, read_metrics, projected_readout

POINT_FIELDS = ('MSE','MAE','MAP_MSE','MAP_MAE','mixture_analytic_MSE','mixture_analytic_MAE')
OLD_PROTOCOL = {'MAP':'P1','engineering':'P2','mixture_analytic':'P3'}


def inputs(config):
    if config.get('index_file'):
        index = json.loads(Path(config['index_file']).read_text())
        cases = index['cases']
        source = config['index_file']
    else:
        cases = config.get('cases',[]); source = 'explicit frozen cases'
    if config.get('dataset'):
        cases = [case for case in cases if case.get('dataset',case.get('config',{}).get('dataset')) == config['dataset']]
    if not cases or len({case['case_id'] for case in cases}) != len(cases):
        raise ValueError('A nonempty, unique finite historical-case list is required')
    for case in cases:
        c = case['config']
        if c.get('dataset') not in ['ETTh1','ETTm1'] or c['protocol']['horizon'] != 96:
            raise ValueError('Historical replay is restricted to the declared H96 ETT scope')
        if not c.get('historical_protocol_replay') or c.get('splits') != ['val','test']:
            raise ValueError('Historical cases must explicitly request complete retrospective val/test')
        if not case.get('roles'):
            raise ValueError('A historical numerical replay requires its old scientific-role lineage')
        for role in case['roles']:
            allowed = role.get('stage') == 'confirm' and role.get('method') in ['E00','E01','E02','E03','NEW1'] and c['seed'] in [17,29,43,59,71]
            allowed |= role.get('stage') == 'explore' and role.get('method') in ['E15','E16','E17'] and c['seed'] == 1
            if not allowed:
                raise ValueError('A case is outside the finite historical replay authorization')
    return cases, source


def point_comparisons(old, current, split):
    result = []
    for key in POINT_FIELDS:
        expected = old.get(key); actual = current.get(key)
        tolerance = None if not finite_point(expected) else max(1e-5,abs(expected)*1e-4)
        valid = finite_point(expected) and finite_point(actual)
        delta = actual-expected if valid else None
        result.append(dict(split=split,metric=key,previous=expected,reloaded=actual,delta=delta,
            tolerance=tolerance,within_tolerance=valid and abs(delta) <= tolerance,
            status='MATCH' if valid and abs(delta) <= tolerance else 'CONFLICT' if valid else 'UNAVAILABLE',
            scope='Historical full-checkpoint replay; cross-device discrepancies retained without reselection'))
    return result


def reload(config_path, output, device='cuda', seed=1):
    config = json.loads(Path(config_path).read_text())
    requested, index_source = inputs(config)
    out = Path(output); out.mkdir(parents=True,exist_ok=True); began = time.time()
    prior_root = Path(config['resume_readouts']) if config.get('resume_readouts') else None
    groups = {}
    for item in requested:
        c = item['config']; key = (c['dataset'],c['seed'],c['route'])
        groups.setdefault(key,[]).append(item)
    cases = []; readouts = []; canonical = []; all_conflicts = []
    for group, items in sorted(groups.items()):
        states = []; numerical_receipts = {}
        for item in items:
            c = item['config']; key = item['case_id']; case_seed = c['seed']
            state = load_checkpoint(c['checkpoint'],map_location='cpu',weights_only=False)
            identity = state_identity(c,state); params = state['policy_state']
            same = next((old_key for old_key,old_identity,old in states
                if identity == old_identity and same_parameters(params,old)),None)
            if same is not None:
                receipt = numerical_receipts[same]
            else:
                states.append((key,identity,params)); directory = out/key; directory.mkdir(exist_ok=True)
                local = directory/'config.json'; atomic_json(local,c)
                prior = prior_root/key if prior_root is not None and (prior_root/key/'config.json').is_file() else None
                numerical = evaluate_config(local,directory,device,case_seed,prior)
                if numerical['status'] != 'PASS':
                    raise ValueError('A retrospective readout did not complete all windows')
                receipt = str(directory/'result.json'); numerical_receipts[key] = receipt; canonical.append(key)
                torch.cuda.empty_cache()
            comparison = []
            for split,reference_key in [('val','historical_validation_metrics'),('test','historical_test_metrics')]:
                actual = read_metrics(receipt,split)
                comparison.extend(point_comparisons(c.get(reference_key,{}),actual,split))
            conflicts = [dict(case_id=key,**value) for value in comparison if value['status']=='CONFLICT']
            all_conflicts.extend(conflicts)
            record = dict(case_id=key,canonical_case=same or key,checkpoint=c['checkpoint'],receipt=receipt,
                seed=case_seed,dataset=c['dataset'],roles=item['roles'],
                prior_numerical_receipt=item.get('prior_numerical_receipt'),comparison=comparison,
                checkpoint_progress=state.get('progress'),
                checkpoint_progress_scope='Read from original checkpoint inside new HOPE worker',
                canonical_independent_state=same is None,
                content_equality='Canonical state loaded' if same is None else 'Exact full policy_state plus history/RNG/inference contract equality in worker',
                requires_investigation=bool(conflicts),
                old_selection_preserved=True,selection_recomputed=False)
            provenance = json.loads(Path(receipt).read_text()).get('readout_provenance',{})
            executed_splits = [split for split,value in provenance.items()
                if value.get('action') == 'EXECUTED_THIS_ATTEMPT'] if same is None else []
            record.update(independent_numerical_execution=bool(executed_splits),
                executed_splits_this_attempt=executed_splits)
            cases.append(record)
            for role in item['roles']:
                protocol = OLD_PROTOCOL[role['protocol']]
                for split in ['val','test']:
                    row = projected_readout(role['readout'],'historical_selected',protocol,'MSE',
                        role['checkpoint'],receipt,split,same or key)
                    row.update(experiment_id=role['experiment_id'],method=role['method'],
                        dataset=c['dataset'],seed=case_seed,
                        historical_selection_scope='Old validation MSE among four retained states; no new checkpoint selection',
                        old_metric_receipt=item.get('prior_numerical_receipt'))
                    readouts.append(row)
            atomic_json(out/'progress.json',cases)
        del states; torch.cuda.empty_cache()
    atomic_json(out/'checkpoint_readouts.json',readouts)
    result = dict(status='PASS',config=config,index_source=index_source,cases=cases,
        canonical_cases=canonical,readouts=readouts,requested_cases=len(requested),
        canonical_unique_cases=len(canonical),
        executed_unique_cases=sum(case['independent_numerical_execution'] for case in cases),
        scientific_role_count=sum(len(item['roles']) for item in requested),
        coordinator_seed=seed,actual_case_seeds=sorted({item['config']['seed'] for item in requested}),
        per_case_seed_preserved=True,optimizer_trajectories=0,optimizer_steps=0,
        actual_gpu=torch.cuda.get_device_name(),seconds=time.time()-began,
        point_parity_conflicts=all_conflicts,requires_investigation=bool(all_conflicts),
        execution_acceptance_scope='Complete independent numerical replay and point diagnostics, not a claim that every old value matched',
        diagnostic_metrics_affect_point_acceptance=False,selection_recomputed=False,
        test_used_for_selection=False,
        limitation='Historical MAP/analytic choices searched four retained states only; benchmark test exposure predates this campaign')
    atomic_json(out/'result.json',result)
    return result
