"""Actual installed frozen-case CLI on validation only; zero optimizer work."""
from pathlib import Path
import json,subprocess,sys
from .runtime import atomic_json,seed_all
from .policy import Policy
from .public_data import prepare_csv
from .train import weights_checkpoint

def qualify(model_root,csv_file,output,revision,common_origin_diagnostics=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);scratch=out.parent.parent/'manifest_entry_work';scratch.mkdir(exist_ok=True)
    cases=[]
    for h in [96,192,336,720]:
        data=scratch/f'data_H{h}';protocol=prepare_csv(str(csv_file),'ETTh1',data,horizon=h,source_revision=revision);protocol=dict(protocol,eval_input_block=4)
        for route in ['A0','D-pre']:
            seed_all(198719);policy=Policy(str(model_root),route,local_files_only=True)
            checkpoint=scratch/f'{route}_H{h}.pt';weights_checkpoint(checkpoint,policy,dict(route=route),dict(global_step=0));del policy
            config=dict(dataset='ETTh1',weights=str(model_root),route=route,checkpoint=str(checkpoint),checkpoint_kind='policy_state',
                local_files_only=True,force_fixed_kappa=True,protocol=protocol,artifact=str(data),seed=198719,diagnostic_limit=6,four_horizon_campaign=True)
            base=dict(checkpoint_source='fixture-'+route+'-H'+str(h),data_key='public-data-H'+str(h),path_id='interface-'+route+'-H'+str(h),
                stage='ENTRY_FIXTURE_NOT_SCIENTIFIC_TRAINING',horizon=h,train_seed=198719,source_training_receipt='OFFICIAL_PRETRAINED_INTERFACE_FIXTURE',evaluation_configuration=config)
            aliases=[dict(base,case_id='MSE',label='P1_val_MSE_best'),dict(base,case_id='MAE',label='P1_val_MAE_best')]
            settings=dict(execution_group_id='entry-'+route+'-H'+str(h),source_code_reference='installed-frozen-manifest-entry-v1',
                cases=aliases,validation_only=True,distribution_samples=1000,distribution_evaluation_seed=198367,common_origin_diagnostics=common_origin_diagnostics)
            file=out/f'{route}_H{h}_manifest.json';directory=scratch/f'{route}_H{h}';atomic_json(file,settings)
            command=[sys.executable,'-m','moirai_research.cli','evaluate-frozen-manifest','--config',str(file),'--output',str(directory),'--device','cuda']
            run=subprocess.run(command,capture_output=True,text=True,timeout=2400)
            if run.returncode:
                (out/'manifest-entry-error.txt').write_text((run.stdout+'\n'+run.stderr)[-16000:]);raise RuntimeError('Manifest validation CLI failed')
            value=json.loads((directory/'result.json').read_text())
            assert value['status']=='DIAGNOSTIC_ONLY' and not value['test_used'] and value['unique_numerical_readouts']==1
            assert value['completed_alias_count']==2 and value['point_CLI_executions_this_attempt']==1 and value['distribution_CLI_executions_this_attempt']==1
            actual=value['canonical_results']['readout_0'];point=actual['point_result']['results']['val'];probability=actual['distribution_result']['results']['val']
            assert point['windows']==probability['windows']==6 and probability['samples']==1000
            assert point['artifact_mode']=='compact_statistics' and actual['distribution_status']=='COMPLETE'
            if common_origin_diagnostics:
                common=actual['point_result']['common_origin_diagnostics']['val']
                assert common['common_windows']==6 and abs(common['metrics']['MAP_MSE']-point['MAP_MSE'])<1e-12
            # The exact completed readout is reused without a model CLI call.
            resume_config=dict(settings,previous_completed_cases=str(directory/'completed_cases.json'))
            resumed=scratch/f'{route}_H{h}_resumed';resume_file=out/f'{route}_H{h}_resume_manifest.json';atomic_json(resume_file,resume_config)
            resume_command=[sys.executable,'-m','moirai_research.cli','evaluate-frozen-manifest','--config',str(resume_file),'--output',str(resumed),'--device','cuda']
            run=subprocess.run(resume_command,capture_output=True,text=True,timeout=120)
            if run.returncode:raise RuntimeError('Completed readout reuse CLI failed')
            reused=json.loads((resumed/'result.json').read_text())
            assert reused['point_CLI_executions_this_attempt']==0 and reused['distribution_CLI_executions_this_attempt']==0
            assert reused['canonical_results']['readout_0']['point_result']==actual['point_result']
            cases.append(dict(horizon=h,route=route,public_command=command,resume_command=resume_command,windows=6,
                unique_point_readouts=1,aliases=2,point_metrics=point,distribution_metrics=probability,
                resumed_point_model_executions=0,resumed_probability_model_executions=0,test_used=False))
            atomic_json(out/'progress.json',dict(cases=cases))
    result=dict(status='PASS',cases=cases,common_origin_diagnostics=common_origin_diagnostics,optimizer_updates=0,optimizer_trajectories=0,test_used=False,
        scope='Installed public manifest CLI, A0/D-pre and four H, two validation batches, point100/distribution1000, exact alias reuse and zero-model completed-case resume. No formal test or full-budget scientific replication.')
    atomic_json(out/'result.json',result);return result
