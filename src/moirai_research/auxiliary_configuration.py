"""Portable auxiliary-stage assembly from completed metadata, without model IO."""
from .hdfs_checkpoints import checkpoint_exists
import copy
import json
import math
from pathlib import Path


PROTOCOL_FIELDS = ('mode','context','patch','horizon','variables','train_end','val_end','test_end','ddof','windows',
                   'scaler_mean','scaler_std_ddof1')


def read(path):
    return json.loads(Path(path).expanduser().read_text())


def checked_receipt(path, protocol, kind, seed=None):
    result=read(path)
    if result.get('status')!='PASS' or result.get('test_used') is not False:
        raise ValueError('A completed, non-test scientific receipt is required')
    config=result['config']
    dataset=config.get('dataset',config['protocol'].get('dataset'))
    if dataset!=protocol['dataset'] or any(config['protocol'].get(k)!=protocol.get(k) for k in PROTOCOL_FIELDS):
        raise ValueError('Upstream data protocol differs')
    if seed is not None and result.get('seed',config.get('seed'))!=seed:
        raise ValueError('Student seed differs from the requested seed')
    if kind=='supervised':
        selected=result.get('readouts',{}).get('val_NLL_best',{})
        if not result.get('full_process') or not result.get('public_weight_start') or not selected.get('reloaded'):
            raise ValueError('A complete public-weight supervised start is required')
        validation=selected.get('validation',{})
        if not validation.get('complete') or validation.get('split')!='val':
            raise ValueError('The supervised checkpoint must have a complete validation readout')
        checkpoint=result['best_checkpoint']
        if selected.get('checkpoint')!=checkpoint:
            raise ValueError('The selected supervised checkpoint differs from its validation readout')
    elif kind=='estimator':
        if not result.get('training_only_fit') or not result.get('frozen'):
            raise ValueError('A frozen training-only estimator is required')
        checkpoint=result['checkpoint']
    else:
        raise ValueError('Unknown upstream receipt kind')
    if not checkpoint_exists(checkpoint):raise ValueError('The selected upstream checkpoint is unavailable')
    return result,checkpoint


def make_auxiliary_configs(data_dir,work_dir,stage,seed=1,coefficient=None,
                           supervised_receipt=None,estimator_receipt=None,teacher_receipt=None):
    """Produce commands for existing numerical entries; no fitting or selection."""
    data=Path(data_dir).expanduser().resolve();work=Path(work_dir).expanduser().resolve()
    protocol=read(data/'data_protocol.json')
    if protocol.get('dataset') not in ['ETTh1','ETTm1'] or protocol.get('horizon')!=96:
        raise ValueError('The registered auxiliary workflow is ETT horizon96')
    if stage not in ['neighbors','estimator','selective','distillation']:
        raise ValueError('Unknown auxiliary stage')
    if protocol['dataset']=='SolarAL_hourly_mean' and stage!='neighbors':
        raise ValueError('Conditional Solar scope only uses the training-frozen dynamic taxonomy; new selective/teacher axes are not registered')
    if stage!='estimator' and (not isinstance(coefficient,(int,float)) or not math.isfinite(coefficient) or coefficient<0):
        raise ValueError('An explicit frozen nonnegative auxiliary coefficient is required')
    work.mkdir(parents=True,exist_ok=True);common=dict(dataset=protocol['dataset'],protocol=protocol,artifact=str(data),seed=seed)
    commands=[];auxiliary=None;provenance={};config=None
    values={}
    if stage=='neighbors':
        config=dict(common,fold_width=512)
        destination=work/'neighbors';command='build-neighbors'
        auxiliary=dict(auxiliary_coefficient=coefficient,neighbor_artifact=str(destination/'training'))
    elif stage=='estimator':
        config=dict(common,lr=1e-4,max_epochs=20,patience=3,batch_size=128,kernel=25,
            shared_across_variables=True,selection='full validation MSE, no test',
            training_choices='Fixed study estimator settings; not a coefficient search')
        destination=work/'estimator';command='train-estimator'
    else:
        if not supervised_receipt:raise ValueError('A completed student supervised receipt is required')
        student,checkpoint=checked_receipt(supervised_receipt,protocol,'supervised',seed)
        if student['config']['route']!='D-pre':raise ValueError('These auxiliary treatments require the D-pre carrier')
        source=student['config']
        carrier={k:copy.deepcopy(source[k]) for k in ['weights','revision','route','local_files_only','microbatch'] if k in source}
        carrier.update(common,checkpoint=checkpoint,checkpoint_kind='policy_state',force_fixed_kappa=True)
        provenance['student_supervised_receipt']=str(Path(supervised_receipt).resolve())
        if stage=='selective':
            if not estimator_receipt:raise ValueError('A completed frozen estimator receipt is required')
            _,estimator_checkpoint=checked_receipt(estimator_receipt,protocol,'estimator')
            config=dict(carrier,estimator_checkpoint=estimator_checkpoint,
                estimator_receipt=str(Path(estimator_receipt).resolve()))
            destination=work/'selective';command='build-selective-statistics'
            auxiliary=dict(auxiliary_coefficient=coefficient,selective_artifact=str(destination))
            provenance['estimator_receipt']=str(Path(estimator_receipt).resolve())
        else:
            if not teacher_receipt:raise ValueError('A completed validation-selected teacher receipt is required')
            teacher,teacher_checkpoint=checked_receipt(teacher_receipt,protocol,'supervised')
            teacher_config={k:copy.deepcopy(teacher['config'][k]) for k in
                ['weights','revision','route','local_files_only','microbatch','protocol','artifact','dataset'] if k in teacher['config']}
            teacher_config.update(checkpoint=teacher_checkpoint,checkpoint_kind='policy_state',force_fixed_kappa=True)
            auxiliary=dict(auxiliary_coefficient=coefficient,teacher_config=teacher_config)
            provenance.update(teacher_receipt=str(Path(teacher_receipt).resolve()),
                teacher_selection='User-supplied frozen validation choice; generator does not rank teachers',
                student_validation=student['readouts']['val_NLL_best']['validation'],
                teacher_validation=teacher['readouts']['val_NLL_best']['validation'])
    if config is not None:
        path=work/(stage+'.json');values[stage+'.json']=config
        commands.append(['moirai-research',command,'--config',str(path),'--output',str(destination),
            '--device','cuda','--seed',str(seed)])
    if auxiliary is not None:
        values['auxiliary_metadata.json']=auxiliary
    result=dict(status='GENERATED_NOT_EXECUTED',stage=stage,seed=seed,commands=commands,
        auxiliary_metadata=str(work/'auxiliary_metadata.json') if auxiliary is not None else None,
        upstream=provenance,
        limitations=['Configuration assembly is not numerical acceptance',
            'Run each numerical command on an authorized GPU worker',
            'Coefficients and teachers must be frozen using training probes and validation before formal tests',
            'Neighbor futures are weak labels; strict-coverage fallback and all negative results remain required'])
    values['commands.json']=result
    from .configuration import write_config_set
    write_config_set(work,values)
    return result
