"""HOPE/standalone worker analysis of validation-frozen, complete test receipts.

No model selection occurs here. Seed and time-block uncertainty are reported
separately; overlapping forecast windows are never treated as independent seeds.
"""
import csv
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .runtime import atomic_json
from .analysis_contract import EXPLORATION_CONTRASTS, decision_metrics, PAPER_TARGETS, PAPER_REFERENCE,generation_evidence


SEEDS = (17, 29, 43, 59, 71)
DECISIONS = {
    'engineering': ('MSE', 'MAE', 'sample_mean', 'sample_median'),
    'MAP': ('MAP_MSE', 'MAP_MAE', 'MAP', 'MAP'),
    'mixture_analytic': ('mixture_analytic_MSE', 'mixture_analytic_MAE', 'mixture_analytic', 'mixture_analytic'),
}
METRICS = ('MSE', 'MAE', 'CRPS', 'marginal_NLL', 'joint_NLL_per_position')


def read(path):
    return json.loads(Path(path).read_text())


def table(path, rows, columns):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='raise')
        writer.writeheader(); writer.writerows(rows)


def group_forecast_origins(origin,variable,value,variables):
    order=np.lexsort((variable,origin));origin=origin[order];variable=variable[order];value=value[order]
    assert len(origin)%variables==0
    assert np.array_equal(variable.reshape(-1,variables),np.tile(np.arange(variables),(len(origin)//variables,1)))
    times=origin.reshape(-1,variables)
    assert (times==times[:,:1]).all() and (np.diff(times[:,0])==1).all()
    return times[:,0],value.reshape(-1,variables,value.shape[-1]).mean(1)


def time_errors(directory, protocol_name, expected, device):
    """Recompute primary point errors; aggregate all variables at each origin.

    S mode's seven rows at a time belong to one time block, not seven independent
    resampling units. M mode's seven simultaneous columns obey the same rule.
    """
    mse_key, mae_key, mse_pred, mae_pred = DECISIONS[protocol_name]
    origins = []; columns = []; values = [];variables=len(expected['per_variable']);assert variables>0
    for path in sorted(Path(directory).glob('block_*.npz')):
        with np.load(path, allow_pickle=False) as block:
            idx = block['indices']; y = torch.as_tensor(block['target'], device=device)
            observed = block['observed']
            if not observed.all():
                raise ValueError('Current ETT analysis requires complete masks; missing-label extension is not implicit')
            mse = (torch.as_tensor(block[mse_pred], device=device) - y).square().mean(1)
            mae = (torch.as_tensor(block[mae_pred], device=device) - y).abs().mean(1)
            for key, recomputed in [(mse_key, mse), (mae_key, mae)]:
                recorded = torch.as_tensor(block['metric_' + key], device=device)
                torch.testing.assert_close(recomputed, recorded, rtol=1e-8, atol=1e-10)
            n, d = mse.shape
            assert d in (1, variables)
            joint = torch.as_tensor(block['joint_NLL_per_position'], device=device)[:, None].expand(n, d)
            metrics = torch.stack([mse, mae,
                torch.as_tensor(block['metric_CRPS'], device=device),
                torch.as_tensor(block['metric_marginal_NLL'], device=device), joint], -1)
            assert torch.isfinite(metrics).all()
            origins.append(np.repeat(idx[:, 0], d))
            columns.append(np.tile(np.arange(variables), n) if d == variables else idx[:, 1])
            values.append(metrics.reshape(-1, len(METRICS)).cpu().numpy())
    if not values:
        raise ValueError('No independently recomputable prediction blocks')
    origin = np.concatenate(origins); variable = np.concatenate(columns); value = np.concatenate(values)
    times,result=group_forecast_origins(origin,variable,value,variables)
    for i, name in enumerate(METRICS):
        source = mse_key if name == 'MSE' else mae_key if name == 'MAE' else name
        assert math.isclose(float(result[:, i].mean()), expected[source], rel_tol=1e-8, abs_tol=1e-9), source
    return times, result


def block_interval(delta, length, device, rng_seed, replicates=999):
    """Paired circular moving blocks, on forecast-origin time, all variables together."""
    x = torch.as_tensor(delta, device=device, dtype=torch.float64)
    n = len(x); length = min(length, n); blocks = math.ceil(n / length)
    generator = torch.Generator(device=device).manual_seed(rng_seed)
    samples = []
    for first in range(0, replicates, 32):
        batch = min(32, replicates - first)
        starts = torch.randint(n, (batch, blocks), device=device, generator=generator)
        indices = (starts[..., None] + torch.arange(length, device=device)) % n
        indices = indices.reshape(batch, -1)[:, :n]
        samples.append(x[indices].mean(1))
    values = torch.cat(samples)
    return torch.quantile(values, torch.tensor([.025, .975], device=device, dtype=torch.float64), dim=0).cpu().numpy()


def exact_sign_flip(values):
    x = np.asarray(values, dtype=float)
    observed = abs(float(x.mean()))
    null = [abs(float((x * signs).mean())) for signs in itertools.product((-1., 1.), repeat=len(x))]
    return sum(v >= observed - 1e-15 for v in null) / len(null)


def numerical_checks(device):
    constant = np.ones((96, 2)) * np.asarray([.25, -.5])
    interval = block_interval(constant, 24, device, 1, replicates=31)
    np.testing.assert_allclose(interval, np.tile([.25, -.5], (2, 1)), atol=1e-12)
    assert exact_sign_flip([1, 1, 1, 1, 1]) == .0625
    assert exact_sign_flip([0, 0, 0, 0, 0]) == 1.
    paired=paired_scalar_summary([0.,1.,2.,3.,4.],[0.,0.,0.,0.,0.])
    assert paired['mean_delta']==2. and math.isclose(paired['sample_std'],math.sqrt(2.5),rel_tol=1e-12)
    missing=paired_scalar_summary([None,1.],[0.,0.]);assert missing['mean_delta'] is None and missing['deltas'][0] is None
    order=np.asarray([4,0,5,1,3,2]);origin=np.repeat([101,102],3);variable=np.tile(np.arange(3),2);value=np.arange(30).reshape(6,5)
    times,grouped=group_forecast_origins(origin[order],variable[order],value[order],3)
    np.testing.assert_array_equal(times,[101,102]);np.testing.assert_array_equal(grouped,[[5,6,7,8,9],[20,21,22,23,24]])
    return dict(constant_paired_blocks='PASS',exact_five_seed_sign_flip='PASS',secondary_pair_mean_std_and_missing='PASS',
        complete_variable_origin_grouping='PASS',device=str(device))


def mode_tables(items, output):
    """Copy qualified diagnostic scalars and complete cluster rows; do not refit."""
    modes=[]; clusters=[]; fixed=[]
    for item in items:
        receipt=read(item['receipt']);source=receipt.get('mode_diagnostics_receipt')
        if not source:raise ValueError('Complete test receipt lacks its qualified validation mode diagnostics')
        result=read(source)
        assert result['status']=='PASS' and result['parameters_unchanged'] and not result['test_used']
        identity={k:item[k] for k in ['scientific_path','phase','model','dataset','route','arm','seed']}
        for group in ['router_calibration','training_prior_reference','uniform_reference']:
            data=result[group]
            if data is None:
                modes.append(dict(**identity,group=group,metric='applicability',value=None,detail=result['calibration_nonapplicability'],source=source))
                continue
            for name,value in data.items():
                modes.append(dict(**identity,group=group,metric=name,value=value if isinstance(value,(float,int)) else None,
                    detail='' if isinstance(value,(float,int)) else json.dumps(value,ensure_ascii=False),source=source))
        for name in ['soft_mode_cluster_mutual_information','permutation_invariant_normalized_MI',
                     'mode_occupancy','mean_router_weights','mean_mode_pair_squared_distance',
                     'kappa_quantiles','maximum_absolute_prediction','mode_to_cluster','training_cluster_prior']:
            value=result[name]
            modes.append(dict(**identity,group='mode_structure',metric=name,value=value if isinstance(value,(float,int)) else None,
                detail='' if isinstance(value,(float,int)) else json.dumps(value,ensure_ascii=False),source=source))
        for name in ['qualification_version','historical_reproduction_status','historical_output_parity','same_device_entry_parity']:
            modes.append(dict(**identity,group='qualification_boundary',metric=name,value=None,
                detail=json.dumps(result[name],ensure_ascii=False),source=source))
        cluster_file=Path(source).parent/'validation_cluster_errors.csv'
        with cluster_file.open() as handle:
            for row in csv.DictReader(handle):
                clusters.append(dict(**identity,cluster_record=json.dumps(row,ensure_ascii=False),source=str(cluster_file)))
        fixed.append(dict(**identity,diagnostics=json.dumps(result['fixed_validation_diagnostics'],ensure_ascii=False),
            worst_windows=json.dumps(result['worst_windows'],ensure_ascii=False),source=source))
    common=['scientific_path','phase','model','dataset','route','arm','seed']
    table(output/'mode_diagnostics.csv',modes,common+['group','metric','value','detail','source'])
    table(output/'validation_cluster_errors.csv',clusters,common+['cluster_record','source'])
    table(output/'fixed_window_diagnostics.csv',fixed,common+['diagnostics','worst_windows','source'])
    return dict(mode_rows=len(modes),cluster_rows=len(clusters),fixed_diagnostic_paths=len(fixed))


def metric_deltas(method, baseline, protocol):
    """Existing qualified scalar differences, executed only in the analysis worker."""
    for metric, field, direction in decision_metrics(protocol):
        a, b = method.get(field), baseline.get(field)
        finite = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in [a, b])
        yield dict(metric=metric, stored_field=field, method_value=a, baseline_value=b,
            delta=a-b if finite else None,
            relative_improvement=(b-a)/b if finite and b>0 and direction=='lower' and 'NLL' not in metric else None,
            direction=direction, validity='FINITE_EXISTING_SCALARS' if finite else 'MISSING_OR_NONFINITE_RETAINED')


def exploration_tables(selected, per_case, output):
    """All fixed mechanisms and stage changes; seed1 never gains seed uncertainty."""
    contrasts=[]; stages=[]
    for ds in ('ETTh1', 'ETTm1'):
        for method, baseline, label, caveat in EXPLORATION_CONTRASTS:
            for protocol in DECISIONS:
                a=selected[('explore',ds,1,method,protocol)];b=selected[('explore',ds,1,baseline,protocol)]
                assert a['model']==b['model'] and a['route']==b['route']
                assert a['data_contract']==b['data_contract']
                _,av,_=per_case[(a['scientific_path'],a['readout'])]
                _,bv,_=per_case[(b['scientific_path'],b['readout'])]
                identical=a['canonical_scientific_path']==b['canonical_scientific_path']
                identity=dict(dataset=ds,seed=1,model=a['model'],route=a['route'],protocol=protocol,
                    method=method,baseline=baseline,contrast=label,method_readout=a['readout'],baseline_readout=b['readout'],
                    method_source=a['source'],baseline_source=b['source'],same_trajectory_reused=identical,
                    method_gpu=a['prediction_gpu'],baseline_gpu=b['prediction_gpu'],
                    hardware_match='UNKNOWN' if 'UNKNOWN' in [a['prediction_gpu'],b['prediction_gpu']] else a['prediction_gpu']==b['prediction_gpu'],
                    scope='DESCRIPTIVE_EXPLORATION_SINGLE_SEED; not independent confirmation',
                    limitation=('Identical trajectory reused; no identifiable treatment effect. ' if identical else '')+caveat)
                contrasts.extend(dict(**identity,**v) for v in metric_deltas(av,bv,protocol))
    for key,choice in selected.items():
        phase,ds,seed,arm,protocol=key
        comparisons=[('preparation_after','preparation_before','preparation_change'),
            (choice['readout'],'preparation_after','validation_selected_additional_training_change'),
            ('final','initial','final_additional_training_change'),
            (choice['readout'],'final','validation_selected_vs_final')]
        for method,baseline,label in comparisons:
            detail_a,av,source_a=per_case[(choice['scientific_path'],method)]
            detail_b,bv,source_b=per_case[(choice['scientific_path'],baseline)]
            hardware_a=generation_evidence(detail_a)['prediction_gpu'];hardware_b=generation_evidence(detail_b)['prediction_gpu']
            identity=dict(scientific_path=choice['scientific_path'],phase=phase,dataset=ds,seed=seed,
                model=choice['model'],route=choice['route'],arm=arm,protocol=protocol,
                method_readout=method,baseline_readout=baseline,contrast=label,
                method_source=source_a,baseline_source=source_b,
                method_gpu=hardware_a,baseline_gpu=hardware_b,
                hardware_match='UNKNOWN' if 'UNKNOWN' in [hardware_a,hardware_b] else hardware_a==hardware_b,
                scope='WITHIN_TRAJECTORY_READOUT_CHANGE; no new independent training or test selection')
            stages.extend(dict(**identity,**v) for v in metric_deltas(av,bv,protocol))
    delta_columns=['metric','stored_field','method_value','baseline_value','delta','relative_improvement','direction','validity']
    table(output/'exploration_paired_effects.csv',contrasts,
        ['dataset','seed','model','route','protocol','method','baseline','contrast','method_readout','baseline_readout',
         'method_source','baseline_source','same_trajectory_reused','method_gpu','baseline_gpu','hardware_match','scope','limitation']+delta_columns)
    table(output/'checkpoint_stage_effects.csv',stages,
        ['scientific_path','phase','dataset','seed','model','route','arm','protocol','method_readout','baseline_readout',
         'contrast','method_source','baseline_source','method_gpu','baseline_gpu','hardware_match','scope']+delta_columns)
    return dict(exploration_contrasts=len(EXPLORATION_CONTRASTS),exploration_metric_rows=len(contrasts),
        stage_metric_rows=len(stages),no_seed1_significance_claim=True)


def confirmation_absolute_tables(selected, per_case, candidates, output):
    """All five fresh seeds, absolute scores and cautious paper-number reference."""
    rows=[];paper=[]
    for ds in ('ETTh1','ETTm1'):
        for arm in ['E00','E01','E02','E03']+candidates:
            for protocol in DECISIONS:
                choices=[selected[('confirm',ds,s,arm,protocol)] for s in SEEDS]
                assert len({x['model'] for x in choices})==1
                data=[per_case[(x['scientific_path'],x['readout'])][1] for x in choices]
                summaries={}
                for metric,field,direction in decision_metrics(protocol):
                    values=[r.get(field) for r in data]
                    finite=all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in values)
                    mean=float(np.mean(values)) if finite else None
                    sd=float(np.std(values,ddof=1)) if finite else None
                    summaries[metric]=mean
                    rows.append(dict(model=choices[0]['model'],dataset=ds,arm=arm,protocol=protocol,metric=metric,
                        stored_field=field,direction=direction,seeds=5,seed_order=json.dumps(SEEDS),
                        per_seed_values=json.dumps(values,allow_nan=True),mean=mean,sample_std=sd,
                        validity='ALL_FIVE_FINITE' if finite else 'MISSING_OR_NONFINITE_RETAINED',
                        sources=json.dumps([x['source'] for x in choices]),
                        scope='Fresh full-process seeds only; seed1 excluded; no across-window standard deviation'))
                if protocol=='MAP':
                    target=PAPER_TARGETS[ds];mse=summaries['MSE'];mae=summaries['MAE']
                    paper.append(dict(model=choices[0]['model'],dataset=ds,arm=arm,seeds=5,horizon=96,
                        inference=protocol,paper_MSE=target['MSE'],paper_MAE=target['MAE'],mean_MSE=mse,mean_MAE=mae,
                        relative_MSE_gap=None if mse is None else mse/target['MSE']-1,
                        relative_MAE_gap=None if mae is None else mae/target['MAE']-1,
                        within_one_percent_rounded_numbers=None if mse is None or mae is None else
                            abs(mse/target['MSE']-1)<=.01 and abs(mae/target['MAE']-1)<=.01,
                        full_protocol_equivalence_established=False,reproduction_or_superiority_claim=False,
                        reference=json.dumps(PAPER_REFERENCE),
                        scope='Numeric reference only; matching a rounded number does not establish paper reproduction'))
    table(output/'confirmation_absolute_summary.csv',rows,
        ['model','dataset','arm','protocol','metric','stored_field','direction','seeds','seed_order',
         'per_seed_values','mean','sample_std','validity','sources','scope'])
    table(output/'paper_numeric_reference.csv',paper,
        ['model','dataset','arm','seeds','horizon','inference','paper_MSE','paper_MAE','mean_MSE','mean_MAE',
         'relative_MSE_gap','relative_MAE_gap','within_one_percent_rounded_numbers',
         'full_protocol_equivalence_established','reproduction_or_superiority_claim','reference','scope'])
    return dict(absolute_metric_rows=len(rows),paper_reference_rows=len(paper),paper_reference_only=True)


def paired_scalar_summary(method_values,baseline_values):
    """Preserve missing values; summarize only a complete paired seed vector."""
    assert len(method_values)==len(baseline_values)
    valid=[isinstance(a,(int,float)) and not isinstance(a,bool) and math.isfinite(a) and
           isinstance(b,(int,float)) and not isinstance(b,bool) and math.isfinite(b)
           for a,b in zip(method_values,baseline_values)]
    deltas=[float(a-b) if keep else None for a,b,keep in zip(method_values,baseline_values,valid)]
    complete=all(valid) and len(deltas)>1
    return dict(deltas=deltas,mean_delta=float(np.mean(deltas)) if complete else None,
        sample_std=float(np.std(deltas,ddof=1)) if complete else None,
        validity='COMPLETE_PAIRED_VECTOR' if complete else 'MISSING_OR_NONFINITE_RETAINED')


def confirmation_secondary_pairs(selected,per_case,candidates,frozen,output):
    """Predeclared secondary metrics: paired descriptions, no new significance tests."""
    rows=[];summaries=[]
    for ds in ('ETTh1','ETTm1'):
        strong=frozen['controls'][ds]['arm']
        contrasts=[('E01','E00','RL_vs_continue_NLL_A0'),('E03','E02','RL_vs_continue_NLL_D_pre')]
        contrasts += [(new,base,kind) for new in candidates for base,kind in [(strong,'primary_vs_frozen_strong'),('E02','mechanism_vs_D_pre_NLL'),('E03','mechanism_vs_D_pre_R')]]
        for method,baseline,contrast in contrasts:
            for protocol in DECISIONS:
                pairs=[]
                for seed in SEEDS:
                    a=selected[('confirm',ds,seed,method,protocol)];b=selected[('confirm',ds,seed,baseline,protocol)]
                    assert a['model']==b['model'] and a['dataset']==b['dataset'] and a['seed']==b['seed']
                    assert a['data_contract']==b['data_contract']
                    pairs.append((a,b,per_case[(a['scientific_path'],a['readout'])][1],per_case[(b['scientific_path'],b['readout'])][1]))
                for metric,field,direction in decision_metrics(protocol):
                    if metric in METRICS:continue
                    av=[r[2].get(field) for r in pairs];bv=[r[3].get(field) for r in pairs];summary=paired_scalar_summary(av,bv)
                    identity=dict(model=pairs[0][0]['model'],dataset=ds,method=method,baseline=baseline,contrast=contrast,
                        protocol=protocol,metric=metric,stored_field=field,direction=direction)
                    for index,(a,b,_,_) in enumerate(pairs):
                        rows.append(dict(**identity,seed=SEEDS[index],method_value=av[index],baseline_value=bv[index],delta=summary['deltas'][index],
                            method_readout=a['readout'],baseline_readout=b['readout'],method_source=a['source'],baseline_source=b['source'],
                            method_gpu=a['prediction_gpu'],baseline_gpu=b['prediction_gpu'],
                            hardware_match='UNKNOWN' if 'UNKNOWN' in [a['prediction_gpu'],b['prediction_gpu']] else a['prediction_gpu']==b['prediction_gpu'],
                            scope='Predeclared secondary descriptive pairing; no per-metric significance claim'))
                    summaries.append(dict(**identity,seeds=5,seed_order=json.dumps(SEEDS),per_seed_values=json.dumps(summary['deltas'],allow_nan=True),
                        mean_delta=summary['mean_delta'],sample_std=summary['sample_std'],validity=summary['validity'],
                        scope='Fresh-seed paired differences only. Repeated contrast labels are contexts, not independent replications; no secondary hypothesis test.'))
    common=['model','dataset','method','baseline','contrast','protocol','metric','stored_field','direction']
    table(output/'secondary_paired_effects.csv',rows,common+['seed','method_value','baseline_value','delta','method_readout','baseline_readout',
        'method_source','baseline_source','method_gpu','baseline_gpu','hardware_match','scope'])
    table(output/'secondary_confirmation_summary.csv',summaries,common+['seeds','seed_order','per_seed_values','mean_delta','sample_std','validity','scope'])
    return dict(paired_rows=len(rows),summary_rows=len(summaries),scope='Predeclared secondary metrics; descriptive paired seed summaries, not additional confirmatory tests')


def analyze(config_path, output, device='cuda', seed=1):
    config = read(config_path); out = Path(output); out.mkdir(parents=True, exist_ok=True)
    assert config['analysis_version']=='finite-effects-v3-secondary-pairs', 'Analysis configuration version is not current'
    frozen = read(config['selection_receipt'])
    assert frozen['status'] == 'PASS' and frozen['test_release'] and not frozen['test_used']
    assert tuple(config['confirmation_seeds']) == SEEDS
    assert config['block_lengths'] == [192, 384, 768] and config['bootstrap_replicates'] == 999
    began = time.time(); checks = numerical_checks(device)
    diagnostic_tables=mode_tables(config['test_nodes'],out)
    long_rows = []; selected_rows = []; lineage_rows=[]; reward_rows=[]; per_case = {}; selected = {}; arrays = {}
    for item in config['test_nodes']:
        receipt = read(item['receipt']); assert receipt['status'] == 'PASS'
        assert receipt['selection_frozen_before_test'] and receipt['scientific_path'] == item['scientific_path']
        cases = {c['case_id']: c for c in receipt['cases']}
        binding=read(receipt['config']['readout_selection_receipt'])
        for label, case in cases.items():
            canonical = cases[case['canonical_case']]
            detail = read(canonical['receipt']); metric = detail['results']['test']
            assert detail['status'] == 'PASS' and metric['complete'] and metric['split'] == 'test'
            assert metric['seed'] == item['seed'] and metric['num_samples'] == 100
            assert metric['independent_marginal_samples']
            identity = dict(scientific_path=item['scientific_path'], phase=item['phase'], model=item['model'],
                dataset=item['dataset'], route=item['route'], arm=item['arm'], seed=item['seed'], readout=label)
            training_source=binding['all_readouts'][label]['source'];training=read(training_source)
            source_config=training['config'];progress=case.get('checkpoint_progress') or {}
            provenance=training.get('semantic_alias',{}).get('canonical_receipt',training_source)
            worker_source=Path(provenance).parent.parent/'loaded_source_identity.json'
            lineage_rows.append(dict(**identity,checkpoint=case['checkpoint'],canonical_case=case['canonical_case'],
                parent_checkpoint=source_config.get('checkpoint'),base_weights=source_config['weights'],
                public_revision=source_config.get('revision'),training_phase=source_config.get('phase'),
                objective_version=source_config.get('objective_version'),optimizer_version=source_config.get('optimizer_version'),
                raw_epoch=progress.get('epoch'),global_step=progress.get('global_step'),
                independent_input_exposures=progress.get('independent_input_exposures'),
                stored_progress=json.dumps(progress),progress_status=case.get('checkpoint_progress_status','UNAVAILABLE'),
                training_receipt=training_source,worker_source_record=str(worker_source) if worker_source.is_file() else None,
                readout_receipt=item['receipt'],scope='Stored lineage and checkpoint progress; missing fields remain unknown'))
            per_case[(item['scientific_path'], label)] = (detail, metric, canonical['receipt'])
            diagnostic_source=detail.get('group_reward_diagnostic_receipt')
            if diagnostic_source:
                diagnostic=read(diagnostic_source)
                assert diagnostic['status']=='PASS' and diagnostic['test_used'] is False and diagnostic['optimizer_steps']==0
                for channel in ['accuracy_reward','full_router_reward']:
                    for name,value in diagnostic[channel].items():
                        if isinstance(value,(float,int)) and not isinstance(value,bool):
                            reward_rows.append(dict(**identity,channel=channel,metric=name,value=value,source=diagnostic_source,
                                scope=diagnostic['scope'],cohort_inputs=diagnostic['independent_inputs'],J=diagnostic['J'],G=diagnostic['G']))
            for name, value in metric.items():
                if isinstance(value, (float, int)) and not isinstance(value, bool):
                    long_rows.append(dict(**identity, metric=name, variable='ALL', value=value, source=canonical['receipt']))
            for variable, data in metric['per_variable'].items():
                for name, value in data.items():
                    long_rows.append(dict(**identity, metric=name, variable=variable, value=value, source=canonical['receipt']))
        for protocol, choice in receipt['selection'].items():
            assert protocol in DECISIONS
            label = choice['readout']; detail, metric, source = per_case[(item['scientific_path'], label)]
            mse_key, mae_key, _, _ = DECISIONS[protocol]
            record = dict(**item, protocol=protocol, readout=label, checkpoint=choice['checkpoint'],
                MSE=metric[mse_key], MAE=metric[mae_key], CRPS=metric['CRPS'],
                marginal_NLL=metric['marginal_NLL'], joint_NLL_per_position=metric['joint_NLL_per_position'],
                source=source, validation_metric=choice['validation_metric'], validation_value=choice['validation_value'],
                **generation_evidence(detail))
            selected_rows.append({k:v for k,v in record.items() if k != 'receipt'})
            record['data_contract']=(detail['config']['protocol'],detail['config']['artifact'],metric['units'])
            canonical=receipt.get('semantic_alias',{}).get('canonical_node')
            record['canonical_scientific_path']=canonical.removeprefix('test-') if canonical else item['scientific_path']
            key = (item['phase'], item['dataset'], item['seed'], item['arm'], protocol)
            assert key not in selected
            selected[key] = record
            if item['phase'] == 'confirm':
                directory = detail['readout_provenance']['test']['outputs']
                cache_key = (directory, protocol)
                if cache_key not in arrays:
                    arrays[cache_key] = time_errors(directory, protocol, metric, device)
                record['time_values'] = arrays[cache_key]
        atomic_json(out/'progress.json', dict(completed_paths=len(selected_rows)//3, expected_paths=len(config['test_nodes'])))
    common = ['scientific_path','phase','model','dataset','route','arm','seed']
    table(out/'results_long.csv',long_rows,common+['readout','metric','variable','value','source'])
    table(out/'checkpoint_readouts.csv',selected_rows,common+['protocol','readout','checkpoint',*METRICS,'source','validation_metric','validation_value',
        'prediction_gpu','receipt_gpu','model_precision','metric_precision','tf32_matmul','tf32_cudnn'])
    table(out/'checkpoint_lineage.csv',lineage_rows,common+['readout','checkpoint','canonical_case','parent_checkpoint','base_weights',
        'public_revision','training_phase','objective_version','optimizer_version','raw_epoch','global_step',
        'independent_input_exposures','stored_progress','progress_status','training_receipt','worker_source_record','readout_receipt','scope'])
    table(out/'group_reward_diagnostics.csv',reward_rows,common+['readout','channel','metric','value','source','scope','cohort_inputs','J','G'])
    exploratory_tables=exploration_tables(selected,per_case,out)
    pairs=[]; intervals=[]; summaries=[]; seed_deltas={}; primary=[]
    candidates=[x['slot'] for x in frozen['selected']]
    absolute_tables=confirmation_absolute_tables(selected,per_case,candidates,out)
    secondary_tables=confirmation_secondary_pairs(selected,per_case,candidates,frozen,out)
    for ds in ('ETTh1','ETTm1'):
        strong=frozen['controls'][ds]['arm']
        contrasts=[('E01','E00','RL_vs_continue_NLL_A0'),('E03','E02','RL_vs_continue_NLL_D_pre')]
        contrasts += [(new,base,kind) for new in candidates for base,kind in [(strong,'primary_vs_frozen_strong'),('E02','mechanism_vs_D_pre_NLL'),('E03','mechanism_vs_D_pre_R')]]
        for method,baseline,contrast in contrasts:
            for protocol in DECISIONS:
                deltas=[]
                for s in SEEDS:
                    a=selected[('confirm',ds,s,method,protocol)];b=selected[('confirm',ds,s,baseline,protocol)]
                    assert a['model']==b['model'] and a['dataset']==b['dataset'] and a['seed']==b['seed']
                    assert a['data_contract']==b['data_contract'],'Cannot pair different data/scaling/probability-measure contracts'
                    # Configuration contracts are checked in the metadata manifest
                    # and actual prediction times/labels are checked below.
                    at,av=a['time_values'];bt,bv=b['time_values'];assert np.array_equal(at,bt)
                    delta=av-bv;point=delta.mean(0);deltas.append(point)
                    identity=dict(dataset=ds,method=method,baseline=baseline,contrast=contrast,protocol=protocol,seed=s)
                    for i,name in enumerate(METRICS):
                        pairs.append(dict(**identity,metric=name,method_value=a[name],baseline_value=b[name],delta=float(point[i]),
                            relative_improvement=(b[name]-a[name])/b[name] if name not in ['marginal_NLL','joint_NLL_per_position'] and b[name]>0 else None,
                            method_gpu=a['prediction_gpu'],baseline_gpu=b['prediction_gpu'],
                            hardware_match='UNKNOWN' if 'UNKNOWN' in [a['prediction_gpu'],b['prediction_gpu']] else a['prediction_gpu']==b['prediction_gpu'],
                            method_source=a['source'],baseline_source=b['source']))
                    for length in config['block_lengths']:
                        ci=block_interval(delta,length,device,seed+1009*s+length,config['bootstrap_replicates'])
                        for i,name in enumerate(METRICS):
                            intervals.append(dict(**identity,metric=name,block_length=length,replicates=config['bootstrap_replicates'],
                                origins=len(at),delta=float(point[i]),lower=float(ci[0,i]),upper=float(ci[1,i]),
                                scope='time uncertainty conditional on this seed, circular paired forecast-origin blocks'))
                data=np.asarray(deltas)
                for i,name in enumerate(METRICS):
                    vector=data[:,i];mean=float(vector.mean());sd=float(vector.std(ddof=1));half=2.7764451051977987*sd/math.sqrt(5)
                    summaries.append(dict(dataset=ds,method=method,baseline=baseline,contrast=contrast,protocol=protocol,metric=name,
                        seeds=5,mean_delta=mean,sample_std=sd,t95_lower=mean-half,t95_upper=mean+half,
                        exact_sign_flip_p=exact_sign_flip(vector),assumption='t interval assumes independent approximately normal seed-level paired differences; sign flip assumes symmetry'))
                seed_deltas[(ds,method,baseline,protocol,contrast)]=data
    for method in candidates:
        logs=[]
        for s in SEEDS:
            ratios=[]
            for ds in ('ETTh1','ETTm1'):
                baseline=frozen['controls'][ds]['arm']
                a=selected[('confirm',ds,s,method,'engineering')];b=selected[('confirm',ds,s,baseline,'engineering')]
                ratios.append(math.log(a['MSE']/b['MSE']))
            logs.append(sum(ratios)/2)
        mean=float(np.mean(logs))
        primary.append(dict(method=method,seeds=5,mean_log_MSE_ratio=mean,geometric_MSE_ratio=math.exp(mean),
            relative_improvement=1-math.exp(mean),exact_sign_flip_p=exact_sign_flip(logs),
            diagnostic_only=frozen['diagnostic_only']))
    ordered=sorted(range(len(primary)),key=lambda i:primary[i]['exact_sign_flip_p']);last=0.
    for rank,i in enumerate(ordered):
        last=max(last,min(1.,(len(primary)-rank)*primary[i]['exact_sign_flip_p']));primary[i]['holm_adjusted_p']=last
    table(out/'paired_effects.csv',pairs,['dataset','method','baseline','contrast','protocol','seed','metric','method_value','baseline_value','delta','relative_improvement',
        'method_gpu','baseline_gpu','hardware_match','method_source','baseline_source'])
    table(out/'time_block_intervals.csv',intervals,['dataset','method','baseline','contrast','protocol','seed','metric','block_length','replicates','origins','delta','lower','upper','scope'])
    table(out/'confirmation_seed_summary.csv',summaries,['dataset','method','baseline','contrast','protocol','metric','seeds','mean_delta','sample_std','t95_lower','t95_upper','exact_sign_flip_p','assumption'])
    table(out/'primary_confirmation.csv',primary,['method','seeds','mean_log_MSE_ratio','geometric_MSE_ratio','relative_improvement','exact_sign_flip_p','diagnostic_only','holm_adjusted_p'])
    result=dict(status='PASS',scope='Complete retained-test tables and paired confirmation analysis; not an efficacy or paper-reproduction verdict',
        numerical_checks=checks,mode_diagnostic_tables=diagnostic_tables,exploration_tables=exploratory_tables,
        analysis_version=config['analysis_version'],secondary_confirmation_tables=secondary_tables,
        confirmation_absolute_tables=absolute_tables,checkpoint_lineage_rows=len(lineage_rows),group_reward_diagnostic_rows=len(reward_rows),scientific_paths=len(config['test_nodes']),selected_readouts=len(selected_rows),result_rows=len(long_rows),
        paired_rows=len(pairs),primary=primary,seconds=time.time()-began,optimizer_trajectories=0,
        actual_gpu=torch.cuda.get_device_name() if str(device).startswith('cuda') else None,
        limitations=['ETT tests are historically exposed benchmarks','No seed1 in confirmation uncertainty',
            'MAP and analytic validation choices cover only four retained states',
            'Five-seed two-sided exact sign-flip minimum p is 0.0625; no invented significance',
            'Block intervals condition on each training seed and do not represent across-seed uncertainty',
            'Per-pair GPU matches are reported; a different or undocumented GPU is a numerical-comparability limitation, not another training seed',
            'Secondary dynamic/interval/raw-scale paired summaries are descriptive; they are not new confirmatory significance tests',
            'CRPS/NLL describe the predictive distribution of each independently selected checkpoint, not a point-decision score'])
    atomic_json(out/'result.json',result);return result
