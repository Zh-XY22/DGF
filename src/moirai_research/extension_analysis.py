"""HOPE-only, fixed-method five-seed Solar transfer analysis."""
import json,math,time
from pathlib import Path
import numpy as np
import torch
from .runtime import atomic_json
from .analysis import (read,table,time_errors,block_interval,exact_sign_flip,numerical_checks,
    mode_tables,paired_scalar_summary,DECISIONS,METRICS,SEEDS)
from .analysis_contract import decision_metrics,generation_evidence


def analyze(config_path,output,device='cuda',seed=1):
    config=read(config_path);out=Path(output);out.mkdir(parents=True,exist_ok=True);began=time.time()
    assert config['analysis_version']=='solar-transfer-v1' and tuple(config['confirmation_seeds'])==SEEDS
    assert config['block_lengths']==[192,384,768] and config['bootstrap_replicates']==999
    frozen=read(config['selection_receipt']);assert frozen['status']=='PASS' and not frozen['test_used'] and frozen['test_release']
    assert frozen['methods']==['E02','E03','NEW1'] and frozen['recipe']['components']==['E14']
    assert len(config['test_nodes'])==15
    checks=numerical_checks(device);diagnostics=mode_tables(config['test_nodes'],out)
    selected={};contracts={};all_metrics={};long_rows=[];readouts=[];lineage=[]
    for item in config['test_nodes']:
        receipt=read(item['receipt']);assert receipt['status']=='PASS' and receipt['selection_frozen_before_test']
        assert receipt['scientific_path']==item['scientific_path'] and item['dataset']=='SolarAL_hourly_mean'
        cases={c['case_id']:c for c in receipt['cases']}
        for label,case in cases.items():
            canonical=cases[case['canonical_case']];detail=read(canonical['receipt']);metric=detail['results']['test']
            assert detail['status']=='PASS' and metric['complete'] and metric['windows']==1657
            assert metric['seed']==item['seed'] and metric['num_samples']==100 and metric['independent_marginal_samples']
            assert len(metric['per_variable'])==137
            assert detail['config']['model']==item['model']=='moirai-1.0-R-small'
            assert detail['config']['dataset']==item['dataset'] and detail['config']['route']==item['route']
            identity={k:item[k] for k in ['scientific_path','phase','model','dataset','route','arm','seed']}
            all_metrics[(item['scientific_path'],label)]=(metric,detail,canonical['receipt'])
            for name,value in metric.items():
                if isinstance(value,(int,float)) and not isinstance(value,bool):long_rows.append(dict(**identity,readout=label,metric=name,variable='ALL',value=value,source=canonical['receipt']))
            for variable,values in metric['per_variable'].items():
                for name,value in values.items():long_rows.append(dict(**identity,readout=label,metric=name,variable=variable,value=value,source=canonical['receipt']))
            lineage.append(dict(**identity,readout=label,checkpoint=case['checkpoint'],canonical_case=case['canonical_case'],
                checkpoint_progress=json.dumps(case.get('checkpoint_progress')),source=canonical['receipt']))
        for protocol,choice in receipt['selection'].items():
            metric,detail,source=all_metrics[(item['scientific_path'],choice['readout'])];mse,mae,_,_=DECISIONS[protocol]
            record=dict(**identity,protocol=protocol,readout=choice['readout'],checkpoint=choice['checkpoint'],
                MSE=metric[mse],MAE=metric[mae],CRPS=metric['CRPS'],marginal_NLL=metric['marginal_NLL'],
                joint_NLL_per_position=metric['joint_NLL_per_position'],source=source,
                validation_metric=choice['validation_metric'],validation_value=choice['validation_value'],**generation_evidence(detail))
            readouts.append(record)
            time_values=time_errors(detail['readout_provenance']['test']['outputs'],protocol,metric,device)
            selected[(item['seed'],item['arm'],protocol)]=(record,metric,time_values)
            cfg=detail['config']
            contracts[(item['seed'],item['arm'],protocol)]=(cfg['weights'],cfg['revision'],cfg['protocol'],cfg['artifact'],metric['units'],metric['seed'])
        atomic_json(out/'progress.json',dict(completed_paths=len(readouts)//3,expected_paths=15))
    common=['scientific_path','phase','model','dataset','route','arm','seed']
    table(out/'results_long.csv',long_rows,common+['readout','metric','variable','value','source'])
    table(out/'checkpoint_readouts.csv',readouts,common+['protocol','readout','checkpoint',*METRICS,'source','validation_metric','validation_value',
        'prediction_gpu','receipt_gpu','model_precision','metric_precision','tf32_matmul','tf32_cudnn'])
    table(out/'checkpoint_lineage.csv',lineage,common+['readout','checkpoint','canonical_case','checkpoint_progress','source'])
    absolute=[];pairs=[];intervals=[];summaries=[];secondary=[];secondary_summary=[];primary=[]
    ds='SolarAL_hourly_mean';contrasts=[('E03','E02','RL_vs_continue_NLL_D_pre'),('NEW1','E03','frozen_diagnostic_vs_R'),('NEW1','E02','frozen_diagnostic_vs_NLL')]
    for protocol in DECISIONS:
        for arm in ['E02','E03','NEW1']:
            records=[selected[(s,arm,protocol)] for s in SEEDS]
            for metric,field,direction in decision_metrics(protocol):
                values=[r[1].get(field) for r in records]
                valid=all(isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) for v in values)
                absolute.append(dict(model='moirai-1.0-R-small',dataset=ds,arm=arm,protocol=protocol,metric=metric,stored_field=field,
                    direction=direction,seeds=5,seed_order=json.dumps(SEEDS),per_seed_values=json.dumps(values,allow_nan=True),
                    mean=float(np.mean(values)) if valid else None,sample_std=float(np.std(values,ddof=1)) if valid else None,
                    validity='ALL_FIVE_FINITE' if valid else 'MISSING_OR_NONFINITE_RETAINED'))
        for method,baseline,contrast in contrasts:
            vectors=[];paired=[]
            for s in SEEDS:
                a,am,(at,av)=selected[(s,method,protocol)];b,bm,(bt,bv)=selected[(s,baseline,protocol)]
                assert contracts[(s,method,protocol)]==contracts[(s,baseline,protocol)],'Unmatched model/data/seed/probability contract'
                np.testing.assert_array_equal(at,bt);delta=av-bv;point=delta.mean(0);vectors.append(point);paired.append((a,b,am,bm))
                for i,metric in enumerate(METRICS):
                    pairs.append(dict(dataset=ds,method=method,baseline=baseline,contrast=contrast,protocol=protocol,seed=s,metric=metric,
                        method_value=a[metric],baseline_value=b[metric],delta=float(point[i]),
                        relative_improvement=(b[metric]-a[metric])/b[metric] if 'NLL' not in metric and b[metric]>0 else None,
                        method_gpu=a['prediction_gpu'],baseline_gpu=b['prediction_gpu'],hardware_match=a['prediction_gpu']==b['prediction_gpu'],
                        method_source=a['source'],baseline_source=b['source']))
                for length in config['block_lengths']:
                    ci=block_interval(delta,length,device,seed+1009*s+length,999)
                    for i,metric in enumerate(METRICS):intervals.append(dict(dataset=ds,method=method,baseline=baseline,contrast=contrast,protocol=protocol,
                        seed=s,metric=metric,block_length=length,origins=len(at),delta=float(point[i]),lower=float(ci[0,i]),upper=float(ci[1,i]),replicates=999))
            data=np.asarray(vectors)
            for i,metric in enumerate(METRICS):
                vector=data[:,i];mean=float(vector.mean());sd=float(vector.std(ddof=1));half=2.7764451051977987*sd/math.sqrt(5)
                summaries.append(dict(dataset=ds,method=method,baseline=baseline,contrast=contrast,protocol=protocol,metric=metric,
                    seeds=5,mean_delta=mean,sample_std=sd,t95_lower=mean-half,t95_upper=mean+half,exact_sign_flip_p=exact_sign_flip(vector)))
            for metric,field,direction in decision_metrics(protocol):
                if metric in METRICS:continue
                av=[r[2].get(field) for r in paired];bv=[r[3].get(field) for r in paired];summary=paired_scalar_summary(av,bv)
                identity=dict(dataset=ds,method=method,baseline=baseline,contrast=contrast,protocol=protocol,metric=metric,stored_field=field,direction=direction)
                for i,s in enumerate(SEEDS):secondary.append(dict(**identity,seed=s,method_value=av[i],baseline_value=bv[i],delta=summary['deltas'][i]))
                secondary_summary.append(dict(**identity,seeds=5,seed_order=json.dumps(SEEDS),per_seed_values=json.dumps(summary['deltas'],allow_nan=True),
                    mean_delta=summary['mean_delta'],sample_std=summary['sample_std'],validity=summary['validity']))
    for method,baseline in [('NEW1','E03'),('E03','E02')]:
        values=[math.log(selected[(s,method,'engineering')][0]['MSE']/selected[(s,baseline,'engineering')][0]['MSE']) for s in SEEDS]
        mean=float(np.mean(values));primary.append(dict(method=method,baseline=baseline,seeds=5,mean_log_MSE_ratio=mean,
            geometric_MSE_ratio=math.exp(mean),relative_improvement=1-math.exp(mean),exact_sign_flip_p=exact_sign_flip(values),diagnostic_only=True))
    last=0.
    for rank,index in enumerate(sorted(range(len(primary)),key=lambda i:primary[i]['exact_sign_flip_p'])):
        last=max(last,min(1.,(len(primary)-rank)*primary[index]['exact_sign_flip_p']));primary[index]['holm_adjusted_p']=last
    table(out/'confirmation_absolute_summary.csv',absolute,['model','dataset','arm','protocol','metric','stored_field','direction','seeds','seed_order','per_seed_values','mean','sample_std','validity'])
    table(out/'paired_effects.csv',pairs,['dataset','method','baseline','contrast','protocol','seed','metric','method_value','baseline_value','delta','relative_improvement','method_gpu','baseline_gpu','hardware_match','method_source','baseline_source'])
    table(out/'time_block_intervals.csv',intervals,['dataset','method','baseline','contrast','protocol','seed','metric','block_length','origins','delta','lower','upper','replicates'])
    table(out/'confirmation_seed_summary.csv',summaries,['dataset','method','baseline','contrast','protocol','metric','seeds','mean_delta','sample_std','t95_lower','t95_upper','exact_sign_flip_p'])
    table(out/'secondary_paired_effects.csv',secondary,['dataset','method','baseline','contrast','protocol','metric','stored_field','direction','seed','method_value','baseline_value','delta'])
    table(out/'secondary_confirmation_summary.csv',secondary_summary,['dataset','method','baseline','contrast','protocol','metric','stored_field','direction','seeds','seed_order','per_seed_values','mean_delta','sample_std','validity'])
    table(out/'primary_confirmation.csv',primary,['method','baseline','seeds','mean_log_MSE_ratio','geometric_MSE_ratio','relative_improvement','exact_sign_flip_p','holm_adjusted_p','diagnostic_only'])
    # Explicit empty tables prevent fabricated comparisons with paper H96 values
    # or a new seed1 exploration that never happened on this held-out dataset.
    table(out/'exploration_paired_effects.csv',[],['dataset','method','baseline','metric','value','scope'])
    table(out/'paper_numeric_reference.csv',[],['dataset','paper_MSE','paper_MAE','scope'])
    result=dict(status='PASS',analysis_version='solar-transfer-v1',optimizer_trajectories=0,scientific_paths=15,selected_readouts=45,
        result_rows=len(long_rows),numerical_checks=checks,mode_diagnostic_tables=diagnostics,actual_gpu=torch.cuda.get_device_name(),seconds=time.time()-began,
        limitations=['Engineering hourly Solar aggregation is not established identical to paper Solar',
            'No Solar H96 paper result is supplied or fabricated','Foundation-model training exposure is UNKNOWN',
            'Five full-process seeds; seed1 is not confirmation','Time blocks are forecast origins and condition on each seed',
            'Fixed diagnostic recipe was selected on ETT validation before Solar; no test-based replacement',
            'Five-seed exact two-sided sign-flip minimum p0.0625; paired secondary results are descriptive',
            'MAP/analytic checkpoint choice covers four retained states only'],
        empty_table_reasons={'exploration_paired_effects.csv':'No seed1 method search on this transfer dataset',
            'paper_numeric_reference.csv':'No equivalent published Solar H96 target has been established'},
        scope='Fixed-method transfer evidence, not an automatic superiority or paper-equivalence verdict')
    atomic_json(out/'result.json',result);return result
