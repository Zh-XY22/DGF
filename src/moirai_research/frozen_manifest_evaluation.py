"""Portable frozen-case orchestration over the already qualified public CLIs.

No model or metric implementation lives here. Aliases refer to one canonical
readout; probability scores remain one distribution record per checkpoint.
"""
from pathlib import Path
import copy,json,subprocess,sys


def write(file,value):
    file=Path(file);file.parent.mkdir(parents=True,exist_ok=True)
    temporary=file.with_suffix(file.suffix+'.new');temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(file)

def identity(case,validation_only,distribution_samples,evaluation_seed):
    config=case['evaluation_configuration'];protocol=config['protocol']
    return dict(checkpoint_source=case['checkpoint_source'],horizon=protocol['horizon'],seed=config['seed'],
        data_key=case['data_key'],data_protocol_contract={k:v for k,v in protocol.items() if k not in ['builder_root','artifact']},route=config['route'],revision=config.get('revision'),transport_mode=config.get('transport_mode'),
        force_fixed_kappa=config.get('force_fixed_kappa',True),point_samples=100,
        point_input_block=protocol.get('eval_input_block',64),
        validation_only=validation_only,diagnostic_limit=config.get('diagnostic_limit'),
        common_origin_diagnostics=bool(config.get('common_origin_diagnostics',False)),
        distribution_samples=distribution_samples,distribution_evaluation_seed=evaluation_seed,
        distribution_input_block=4)

def check_checkpoint_header(header,config):
    if header is None:return 'UNKNOWN_LEGACY_HEADER'
    for saved,expected in [('saved_seed','seed'),('saved_route','route'),('saved_revision','revision')]:
        value=header.get(saved)
        if value is not None and value!=config.get(expected):raise ValueError('Loaded checkpoint header differs from the frozen case: '+saved)
    return 'AVAILABLE_HEADER_FIELDS_MATCH'

def execute(config_path,output,device='cuda'):
    settings=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    validation_only=settings.get('validation_only',False);large_samples=settings.get('distribution_samples',0)
    if large_samples not in [0,1000]:raise ValueError('Manifest large-sample budget must be0 or1000')
    evaluation_seed=settings.get('distribution_evaluation_seed',198367)
    if evaluation_seed!=198367:raise ValueError('Frozen distribution evaluation RNG changed')
    if len({v['case_id'] for v in settings['cases']})!=len(settings['cases']):raise ValueError('Duplicate frozen case identity')
    previous={}
    if settings.get('previous_completed_cases'):
        prior=json.loads(Path(settings['previous_completed_cases']).read_text())
        if prior['execution_group_id']!=settings['execution_group_id']:raise ValueError('Resume belongs to a different frozen task')
        if prior.get('source_code_reference')!=settings.get('source_code_reference'):raise ValueError('Completed readout source version differs; no automatic numerical reuse')
        previous={json.dumps(v['identity'],sort_keys=True):v for v in prior['canonical_results'].values() if v.get('point_status')=='COMPLETE'}
    canonical={};by_identity={};aliases=[];failures=[]
    for case in settings['cases']:
        active=copy.deepcopy(case['evaluation_configuration'])
        if active['seed']!=case['train_seed'] or active['protocol']['horizon']!=case['horizon']:raise ValueError('Case and actual model seed/H disagree')
        active.update(checkpoint_source=case['checkpoint_source'],scientific_path=case['path_id'],checkpoint_label=case['label'],
            splits=['val'] if validation_only else ['val','test'],four_horizon_campaign=True,evaluation_artifact_mode='compact_statistics')
        active['common_origin_diagnostics']=settings.get('common_origin_diagnostics',False)
        if not validation_only:
            if active.get('diagnostic_limit') is not None:raise ValueError('A limited diagnostic cannot become a formal test')
            active.update(test_release_receipt=settings['test_release_file'],four_horizon_test_release=settings['test_release_file'],
                evaluation_channel='point100',evaluation_source_reference=settings['source_code_reference'],test_release=True)
            from .four_horizon_test_gate import check_release
            check_release(active)  # Check every alias before numerical reuse.
        binding=identity(dict(case,evaluation_configuration=active),validation_only,large_samples,evaluation_seed)
        key=json.dumps(binding,sort_keys=True)
        if key in by_identity:
            name=by_identity[key]
        else:
            name='readout_'+str(len(canonical));directory=out/name;directory.mkdir(exist_ok=True)
            file=directory/'point_configuration.json';write(file,active)
            expected='DIAGNOSTIC_ONLY' if active.get('diagnostic_limit') is not None else 'PASS'
            if key in previous:
                value=copy.deepcopy(previous[key]);value.update(resume_source='PREVIOUS_COMPLETE_FROZEN_CASE',numerical_execution='REUSED_PREVIOUS_COMPLETE_CASE_NO_MODEL_EXECUTION',distribution_execution='REUSED_PREVIOUS_COMPLETE_CASE' if value['distribution_status']=='COMPLETE' else 'NOT_EXECUTED_THIS_ATTEMPT')
                canonical[name]=value
            else:
                command=[sys.executable,'-m','moirai_research.cli','evaluate','--config',str(file),'--output',str(directory/'point'),'--device',device,'--seed',str(active['seed'])]
                run=subprocess.run(command,capture_output=True,text=True,timeout=12*3600)
                if run.returncode:
                    write(directory/'failure.json',dict(status='POINT_EVALUATION_FAILED',return_code=run.returncode,command=command))
                    (directory/'point-error.txt').write_text((run.stdout+'\n'+run.stderr)[-18000:])
                    failures.append(dict(case_id=case['case_id'],stage='point100',return_code=run.returncode))
                    break
                result=json.loads((directory/'point/result.json').read_text())
                header_status=check_checkpoint_header(result.get('checkpoint_metadata'),active)
                if active['common_origin_diagnostics']:
                    from .common_origin_statistics import reduce_blocks
                    result['common_origin_diagnostics']={split:reduce_blocks(directory/'point'/split,active['protocol'],split,require_complete=active.get('diagnostic_limit') is None)
                        for split in result['results']}
                expected='DIAGNOSTIC_ONLY' if active.get('diagnostic_limit') is not None else 'PASS'
                if result['status']!=expected:raise ValueError('Point evaluator completeness differs from frozen manifest')
                value=dict(identity=binding,point_status='COMPLETE',point_result=result,point_command=command,distribution_status='NOT_REQUESTED',
                    checkpoint_header_status=header_status,numerical_execution='EXECUTED_THIS_ATTEMPT',test_used=not validation_only)
                canonical[name]=value
            if large_samples and value['distribution_status']!='COMPLETE' and (key not in previous or settings.get('retry_distribution_diagnostics')):
                probability=dict(active,distribution_samples=large_samples,distribution_eval_seed=evaluation_seed,distribution_input_block=4,evaluation_channel='distribution1000')
                probability_file=directory/'distribution_configuration.json';write(probability_file,probability)
                command=[sys.executable,'-m','moirai_research.cli','evaluate-distribution','--config',str(probability_file),'--output',str(directory/'distribution'),'--device',device,'--seed',str(active['seed'])]
                run=subprocess.run(command,capture_output=True,text=True,timeout=12*3600)
                value['distribution_command']=command
                value['distribution_execution']='EXECUTED_THIS_ATTEMPT'
                if run.returncode:
                    value.update(distribution_status='DIAGNOSTIC_UNAVAILABLE_REQUIRES_CLASSIFICATION',distribution_return_code=run.returncode)
                    (directory/'distribution-error.txt').write_text((run.stdout+'\n'+run.stderr)[-18000:])
                else:
                    distribution=json.loads((directory/'distribution/result.json').read_text())
                    if distribution['status']!=expected:raise ValueError('Distribution completeness differs from frozen manifest')
                    value.update(distribution_status='COMPLETE',distribution_result=distribution)
            canonical[name]=value
            by_identity[key]=name
        aliases.append(dict(case_id=case['case_id'],path_id=case['path_id'],label=case['label'],stage=case['stage'],
            horizon=case['horizon'],train_seed=case['train_seed'],checkpoint_source=case['checkpoint_source'],
            source_training_receipt=case['source_training_receipt'],canonical_readout=name,
            scientific_training_replicates_added=0))
        write(out/'completed_cases.json',dict(execution_group_id=settings['execution_group_id'],source_code_reference=settings.get('source_code_reference'),canonical_results=canonical,aliases=aliases))
    status='FAILED' if failures else 'DIAGNOSTIC_ONLY' if any(v['identity']['diagnostic_limit'] is not None for v in canonical.values()) else 'PASS'
    result=dict(status=status,execution_group_id=settings['execution_group_id'],source_code_reference=settings.get('source_code_reference'),canonical_results=canonical,aliases=aliases,
        expected_case_count=len(settings['cases']),completed_alias_count=len(aliases),unique_numerical_readouts=len(canonical),
        point_CLI_executions_this_attempt=sum(v['numerical_execution']=='EXECUTED_THIS_ATTEMPT' for v in canonical.values()),
        distribution_CLI_executions_this_attempt=sum(v.get('distribution_execution')=='EXECUTED_THIS_ATTEMPT' for v in canonical.values()),
        distribution_unavailable_cases=[k for k,v in canonical.items() if v['distribution_status'].startswith('DIAGNOSTIC_UNAVAILABLE')],
        failures=failures,optimizer_updates=0,optimizer_trajectories=0,test_used=not validation_only,
        interpretation='One canonical numerical record may serve multiple separately frozen labels. No alias, protocol or evaluation draw is a new training seed. Missing probability diagnostics never revoke valid point scores.')
    write(out/'result.json',result);return result
