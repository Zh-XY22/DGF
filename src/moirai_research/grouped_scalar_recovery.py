"""Recover original scalar curves from full continuation states; zero inference.

This repairs lost publication receipts. It does not invent original reload logs
or relabel a CPU state read as model prediction or independent metric recompute.
"""
from pathlib import Path
import json,gc
import torch
from .point_selection import selection_records
from .probability_selection import probability_records
from .runtime import atomic_json

def recover(cases,output,exists=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);exists=exists or (lambda p:Path(p).is_file());recovered=[]
    for case in cases:
        models={};parameter_checks={}
        for method,local in case['models'].items():
            state=torch.load(local['resume_local'],map_location='cpu',weights_only=False)
            final=torch.load(local['final_local'],map_location='cpu',weights_only=False)
            config=state['config'];cursor=state['cursor'];extra=state['extra'];joint=case['kind']=='train-joint'
            n=config['common_train_windows'] if joint else config['protocol']['windows']['train'];batch=config['global_input_batch'];epochs=config['epochs']
            updates=config['updates_per_input_batch'];expected=((n+batch-1)//batch)*epochs*updates
            if cursor['epoch']!=epochs or cursor['next_input']!=0 or cursor['global_step']!=expected or cursor['independent_input_exposures']!=n*epochs:raise ValueError('Continuation state is not a completed budget')
            if set(state['policy_state'])!=set(final['policy_state']):raise ValueError('Final weights/state parameter names differ')
            if not all(torch.equal(value,final['policy_state'][name]) for name,value in state['policy_state'].items()):raise ValueError('Final weights differ from full continuation state')
            if any(not torch.isfinite(value).all() for value in state['policy_state'].values() if torch.is_floating_point(value)):raise FloatingPointError('Nonfinite saved model parameters')
            first=extra['first'] if joint else extra['first_step']
            if first['relative_parameter_change']<=0:raise ValueError('No effective original update')
            curve=extra['curve'];metrics={}
            for row in curve:metrics[row['global_step']]=row['metrics_by_horizon'] if joint else row['metrics']
            selected={'joint_selected':extra['best_path']} if joint else extra['best_paths']
            original_training_root=Path(next(iter(selected.values()))).parent.parent
            local['original_training_root']=str(original_training_root)
            readouts={}
            for label,old in dict(initial=str(Path(local['original_training_root'])/'checkpoints/initial.pt'),**selected,
                final=str(Path(local['original_training_root'])/'checkpoints/final.pt')).items():
                name=Path(old).name
                step=0 if name=='initial.pt' else cursor['global_step'] if name=='final.pt' else int(name.removesuffix('.pt').rsplit('step',1)[1])
                uri=local['state_uri']+'/'+name
                if not exists(uri):raise FileNotFoundError('Selected original checkpoint is missing: '+uri)
                if step not in metrics:raise ValueError('No original validation scalar record for selected checkpoint step')
                readouts[label]=dict(checkpoint=uri,**{'validation_by_horizon' if joint else 'validation':metrics[step]},
                    reloaded=False,reload_evidence='ORIGINAL_RELOAD_RECEIPT_UNPUBLISHED; source control flow and final-state postcondition retained, formal independent reload still required',
                    metric_origin='Original complete validation curve saved inside optimizer continuation state; no new prediction',global_step=step)
            result=dict(status='PASS',config=config,cursor=cursor,first_step=first,readouts=readouts,optimizer_trajectories=1,
                optimizer_steps=expected,optimizer_updates=expected,continuation_checkpoint=local['state_uri']+'/resume.pt',test_used=False,
                recovered_from_complete_state=True,independent_reload_receipt_status='MISSING_ORIGINAL_RECEIPT_PENDING_FORMAL_RELOAD',
                actual_gpu='UNKNOWN_ORIGINAL_ENVIRONMENT_RECEIPT_NOT_PUBLISHED',
                raw_training_curve_source=case['original_attempt']+'/states/'+method+'/resume.pt#extra',
                scalar_recovery_scope='Existing validation scalars and exact final-weight/full-state identity; zero new optimizer updates and zero model inference')
            if joint:
                result.update(selection_objective=config['joint_selection_objective'],selection_score=extra['best'],best_checkpoint=readouts['joint_selected']['checkpoint'])
            else:
                convert=lambda values:{k:dict(v,checkpoint=local['state_uri']+'/'+Path(v['checkpoint']).name) if v.get('checkpoint') else v for k,v in values.items()}
                result.update(point_selection=convert(selection_records(extra['point_values'],extra['best_paths'])),
                    probability_selection=probability_records(extra['point_values'],extra['best_paths']),best_checkpoint=readouts['P1_val_MSE_best']['checkpoint'])
                if result['probability_selection'].get('checkpoint'):
                    result['probability_selection']['checkpoint']=local['state_uri']+'/'+Path(result['probability_selection']['checkpoint']).name
            models[method]=result;parameter_checks[method]=dict(status='EXACT_SAVED_PARAMETER_MATCH',optimizer_state_present=bool(state['optimizer_state']['state']),
                RNG_state_present=all(k in state['rng'] for k in ['python','numpy','torch','cuda']),cursor=cursor)
            atomic_json(out/case['node_id']/method/'original_validation_curve.json',curve)
            atomic_json(out/case['node_id']/method/'original_training_curve.json',extra.get('training',extra.get('training_curve',[])))
            del state,final;gc.collect()
        result=models['joint'] if case['kind']=='train-joint' else dict(status='PASS',results=models,optimizer_trajectories=2,optimizer_process_starts=8,
            optimizer_updates=sum(v['optimizer_steps'] for v in models.values()),shared_parent=case['parent_checkpoint'],
            paired_action_evidence='Original per-batch JSON proof was not published. Same cache roles/path and completed postcondition support the frozen code execution; no invented raw events.',test_used=False)
        recovered.append(dict(node_id=case['node_id'],original_attempt=case['original_attempt'],original_attempt_id=case['original_attempt_id'],original_app_ids=case['original_app_ids'],original_run_ids=case['original_run_ids'],
            original_native_status=case['original_native_status'],original_source_directory=case['original_source_directory'],kind=case['kind'],
            parent_checkpoint=case.get('parent_checkpoint'),result=result,parameter_state_checks=parameter_checks,
            evidence_level='SAVED_COMPLETE_STATE_AND_ORIGINAL_VALIDATION_SCALARS; missing original final scalar publication is declared',
            raw_runtime_matching_log='MISSING_ORIGINAL_PUBLICATION' if case['kind']=='train-credit-pair' else 'NA'))
    result=dict(status='PASS',recovered_groups=recovered,optimizer_updates=0,optimizer_trajectories=0,model_inference_calls=0,
        scope='Recover existing complete-state scalar evidence; do not overwrite old attempts or claim new prediction/reload metrics',test_used=False)
    atomic_json(out/'result.json',result);return result
