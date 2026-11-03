"""Read-only shared-SFT scalar export, separate from scientific-path results.

Only caller-provided JSON receipts/curves are read. No model, tensor, array,
checkpoint, filesystem discovery, network or job interfaces are used.
"""
import copy
import json
import math
from pathlib import Path

PROTOCOLS={'P1':('MAP_MSE','MAP_MAE'),'P2':('MSE','MAE'),
           'P3':('mixture_analytic_MSE','mixture_analytic_MAE')}
DATASETS=('ETTh1','ETTm1')
ROUTES=('A0','D-pre')
PROGRESS_FIELDS=('epoch','global_step','independent_input_exposures','scheduled_fraction','actual_inputs')


def _clean(value):
    if isinstance(value,float) and not math.isfinite(value):return 'NaN' if math.isnan(value) else 'Infinity' if value>0 else '-Infinity'
    if isinstance(value,dict):return {str(key):_clean(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [_clean(item) for item in value]
    return value


def _finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def _gap(gaps,owner,kind,source,detail):
    gaps.append(dict(experiment_id=owner,kind=kind,source=str(source or ''),detail=detail))


def _read(path,owner,gaps,kind,expected,required=False):
    if not path:
        if required:_gap(gaps,owner,kind,path,'Required existing JSON reference is absent')
        return None
    try:
        file=Path(path)
        if file.suffix not in ('.json','.jsonl') or file.stat().st_size>64*1024*1024:
            raise ValueError('Only bounded scalar JSON metadata is permitted')
        text=file.read_text(encoding='utf-8')
        value=[json.loads(line) for line in text.splitlines() if line.strip()] if file.suffix=='.jsonl' else json.loads(text)
        if not isinstance(value,expected):raise ValueError('Unexpected JSON metadata shape')
        return value
    except (OSError,ValueError,TypeError) as error:
        _gap(gaps,owner,kind,path,str(error));return None


def _difference(planned,effective,prefix=''):
    result=[]
    for key in sorted(set(planned)|set(effective)):
        name=prefix+'.'+key if prefix else key;a=planned.get(key);b=effective.get(key)
        if isinstance(a,dict) and isinstance(b,dict):result.extend(_difference(a,b,name))
        elif _clean(a)!=_clean(b):result.append(dict(field=name,planned=a,effective=b,
            planned_present=key in planned,effective_present=key in effective))
    return result


def _project(progress):
    return {key:progress.get(key) for key in PROGRESS_FIELDS}


def _progress(label,readout,receipt,curve,choice,owner,source,gaps):
    for field in ('progress','checkpoint_progress'):
        direct=readout.get(field)
        if isinstance(direct,dict) and any(key in direct for key in PROGRESS_FIELDS):
            return _project(direct),'readouts.'+label+'.'+field,None
    if label=='initial':
        matches=[(i,row) for i,row in enumerate(curve) if isinstance(row,dict)
            and row.get('global_step')==0 and row.get('independent_input_exposures')==0]
        if matches:return _project(matches[0][1]),'validation_curve',matches[0][0]
    elif label=='final':
        cursor=receipt.get('cursor')
        if isinstance(cursor,dict) and cursor.get('global_step') is not None:return _project(cursor),'receipt.cursor',None
    else:
        events=[(i,row,event) for i,row in enumerate(curve) if isinstance(row,dict)
            for event in row.get('selected',[]) if isinstance(event,dict) and event.get('label')==label]
        if events:
            i,row,event=events[-1]
            if (_finite(choice.get('validation_value')) and event.get('validation_value')==choice['validation_value']
                and event.get('validation_metric')==choice.get('validation_metric')):
                return _project(row),'validation_curve.selected',i
            _gap(gaps,owner,'SHARED_PROGRESS_SELECTION_MISMATCH',source,
                'Latest selected event for '+label+' differs from the receipt selection; no filename or earlier event substituted')
            return _project({}),'UNKNOWN_SELECTION_MISMATCH',None
    _gap(gaps,owner,'SHARED_PROGRESS_UNKNOWN',source,'No exact existing progress record for '+label)
    return _project({}),'UNKNOWN',None


def collect_shared_starts(shared_starts,scientific_paths=(),seeds=(101,211,307,401,503)):
    """Return JSON/CSV-ready shared inventory, val readouts and lineage metadata.

    Accepted nodes produce twelve declared validation roles: initial/final each
    in P1/P2/P3, plus six independently selected best roles. Invalid choices stay
    explicit. Unaccepted nodes retain configuration and curve-prefix evidence,
    but never acquire a final checkpoint result merely from planned budgets.
    """
    starts=list(shared_starts);seeds=tuple(seeds);paths=list(scientific_paths)
    inventory=[];results=[];configurations=[];curves=[];lineage=[];gaps=[];by={};receipts={}
    expected={(dataset,route,seed) for dataset in DATASETS for route in ROUTES for seed in seeds}
    actual=[(row.get('dataset'),row.get('route'),row.get('seed')) for row in starts]
    accepted_metadata_complete=True;accepted_progress_complete=True
    for entry in starts:
        owner=entry.get('node_id');status=entry.get('status','UNKNOWN');accepted=status=='ACCEPTED'
        if not owner or owner in by:_gap(gaps,owner or 'SHARED_UNKNOWN','SHARED_DUPLICATE_OR_MISSING_NODE_ID',None,'Node ID is missing or duplicated')
        else:by[owner]=entry
        source=entry.get('result_receipt');receipt=_read(source,owner,gaps,'SHARED_RESULT_UNAVAILABLE',dict,accepted) or {}
        receipts[owner]=receipt
        effective=receipt.get('config') if isinstance(receipt.get('config'),dict) else None
        planned=entry.get('planned_configuration') if isinstance(entry.get('planned_configuration'),dict) else None
        validation=_read(entry.get('validation_curve'),owner,gaps,'SHARED_VALIDATION_CURVE_UNAVAILABLE',list,accepted) or []
        training=_read(entry.get('training_curve'),owner,gaps,'SHARED_TRAINING_CURVE_UNAVAILABLE',list,False) or []
        for kind,rows in [('validation',validation),('training',training)]:
            prefix=next(((i,row) for i,row in reversed(list(enumerate(rows))) if isinstance(row,dict)),None)
            curves.append(dict(node_id=owner,curve_kind=kind,source=entry.get(kind+'_curve'),records=len(rows),
                last_record_index=prefix[0] if prefix else None,last_progress=_project(prefix[1]) if prefix else None,
                status='EXISTING_SCALAR_PREFIX' if rows else 'UNKNOWN_OR_NOT_YET_AVAILABLE',
                scope='Existing curve reference; no interpolation or final-state inference'))
        valid_receipt=accepted and receipt.get('status')=='PASS' and effective is not None
        if accepted and not valid_receipt:
            accepted_metadata_complete=False;_gap(gaps,owner,'SHARED_ACCEPTED_RECEIPT_NOT_PROVEN',source,'Accepted node lacks a PASS receipt and actual configuration')
        if effective is not None and any(effective.get(key)!=entry.get(key) for key in ('dataset','route','seed')):
            accepted_metadata_complete=False;_gap(gaps,owner,'SHARED_CONFIGURATION_BINDING_MISMATCH',source,'Actual dataset/route/seed differs from shared-node registration')
        differences=_difference(planned,effective) if planned is not None and effective is not None else None
        configurations.append(dict(node_id=owner,dataset=entry.get('dataset'),route=entry.get('route'),seed=entry.get('seed'),
            planned=copy.deepcopy(planned),planned_configuration_source=entry.get('planned_configuration_source'),
            effective=copy.deepcopy(effective),effective_source=source,
            effective_locator='config',plan_effective_differences=differences))
        readouts=receipt.get('readouts',{}) if isinstance(receipt.get('readouts'),dict) else {}
        selected=receipt.get('point_selection',{}) if isinstance(receipt.get('point_selection'),dict) else {}
        common=receipt.get('best_checkpoint');common_choice=selected.get('P1_val_MSE_best',{})
        if valid_receipt and (not common or common_choice.get('checkpoint')!=common):
            _gap(gaps,owner,'SHARED_COMMON_START_REFERENCE_CONFLICT',source,'best_checkpoint does not match the declared P1 validation-MSE branch start')
        inventory.append(dict(node_id=owner,node_kind='SHARED_SFT_OR_MULTIMODE_WARMUP',scientific_path=False,
            dataset=entry.get('dataset'),route=entry.get('route'),seed=entry.get('seed'),status=status,
            result_receipt=source,receipt_status=receipt.get('status','UNKNOWN'),accepted_receipt_available=valid_receipt,
            actual_configuration_status='RECORDED' if effective is not None else 'UNKNOWN',
            actual_lr=effective.get('lr') if effective else None,actual_epochs_completed=receipt.get('epochs_completed'),
            actual_optimizer_steps=receipt.get('optimizer_steps'),cursor=receipt.get('cursor'),
            actual_global_input_batch=receipt.get('global_input_batch'),actual_microbatch=receipt.get('microbatch'),
            actual_world_size=receipt.get('world_size'),actual_gpu=receipt.get('actual_gpu','UNKNOWN'),
            first_step=receipt.get('first_step'),common_branch_checkpoint=common,
            validation_curve=entry.get('validation_curve'),training_curve=entry.get('training_curve'),
            attempts=entry.get('attempts',[]),worker_execution_evidence=entry.get('worker_execution_evidence',[]),
            planned_readout_roles=12,exported_readout_roles=0,readout_scope='Validation only; not a formal test result',
            plan_effective_differences=differences))
        lineage.append(dict(edge='PUBLIC_INITIALIZATION_TO_SHARED_NODE',shared_start_node_id=owner,
            dataset=entry.get('dataset'),route=entry.get('route'),seed=entry.get('seed'),
            public_model_id=effective.get('public_model_id',effective.get('model_id')) if effective else None,
            public_revision=effective.get('revision') if effective else None,
            initialization_weights_reference=effective.get('weights') if effective else None,
            initial_checkpoint=readouts.get('initial',{}).get('checkpoint'),source_receipt=source,
            relation_scope='Recorded construction/training lineage, not parameter equality or a new public-weight verification'))
        if not valid_receipt:continue
        labels=[('initial',protocol,'N/A') for protocol in PROTOCOLS]
        labels += [(f'{protocol}_val_{objective}_best',protocol,objective) for protocol in PROTOCOLS for objective in ('MSE','MAE')]
        labels += [('final',protocol,'N/A') for protocol in PROTOCOLS]
        aliases={}
        for label,protocol,objective in labels:
            readout=readouts.get(label,{})
            if not isinstance(readout,dict):readout={}
            choice=selected.get(label,{}) if objective!='N/A' else {}
            invalid=objective!='N/A' and choice.get('status')=='INVALID'
            if objective!='N/A' and not invalid and (choice.get('status')!='VALID'
                or choice.get('checkpoint')!=readout.get('checkpoint')
                or choice.get('protocol')!=protocol or choice.get('selection_objective')!=objective
                or choice.get('validation_metric')!=PROTOCOLS[protocol][0 if objective=='MSE' else 1]):
                accepted_metadata_complete=False
                _gap(gaps,owner,'SHARED_SELECTION_READOUT_BINDING_MISMATCH',source,
                    label+': checkpoint/protocol/objective/metric does not match its actual readout role')
            metrics=readout.get('validation',{}) if isinstance(readout.get('validation'),dict) else {}
            complete=bool(readout.get('reloaded') is True and metrics.get('complete') is True and metrics.get('split')=='val')
            if not complete and not invalid:
                accepted_metadata_complete=False;_gap(gaps,owner,'SHARED_REQUIRED_READOUT_MISSING',source,'No complete reloaded validation for '+label)
            if complete:
                progress,progress_source,index=_progress(label,readout,receipt,validation,choice,owner,entry.get('validation_curve') or source,gaps)
                if progress.get('global_step') is None:accepted_progress_complete=False
            else:progress,progress_source,index=_project({}),'NOT_APPLICABLE_INVALID_CHOICE' if invalid else 'UNKNOWN',None
            checkpoint=readout.get('checkpoint')
            canonical=aliases.setdefault(checkpoint,label) if checkpoint else None
            mse,mae=(metrics.get(key) for key in PROTOCOLS[protocol])
            point_valid=complete and _finite(mse) and mse>=0 and _finite(mae) and mae>=0
            diagnostics={key:value for key,value in metrics.items() if key in ('CRPS','marginal_NLL','joint_NLL_per_position','coverage_80','width_80')}
            missing=[key for key in ('CRPS','marginal_NLL','joint_NLL_per_position') if not _finite(metrics.get(key))]
            if complete and missing:_gap(gaps,owner,'MISSING_DIAGNOSTIC_METRICS',source,label+'/'+protocol+': '+', '.join(missing)+'; valid point values are retained')
            if metrics.get('diagnostic_issues'):_gap(gaps,owner,'DIAGNOSTIC_METRIC_ISSUE',source,label+': '+str(metrics['diagnostic_issues']))
            results.append(dict(shared_start_node_id=owner,shared_readout_id=str(owner)+'::'+label+'::'+protocol,
                dataset=entry.get('dataset'),route=entry.get('route'),seed=entry.get('seed'),
                readout=label,protocol=protocol,selection_objective=objective,split='val',
                checkpoint=checkpoint,canonical_reference_readout=canonical,checkpoint_reference_alias=canonical is not None and canonical!=label,
                alias_scope='Identical checkpoint reference only; no weight-content comparison was performed',
                selection_status=choice.get('status','N/A') if objective!='N/A' else 'N/A',
                validation_metric=choice.get('validation_metric'),validation_value=choice.get('validation_value'),
                result_status='VALID_POINT_METRICS' if point_valid else 'INVALID_SELECTION' if invalid else 'UNKNOWN_OR_INVALID_POINT_METRICS',
                MSE=mse,MAE=mae,metric_units=metrics.get('units','UNKNOWN'),
                probability_score_scope='CRPS/NLL/calibration diagnose the complete checkpoint mixture distribution, not the deterministic P1/P3 point output or selected-mode conditional distribution',
                diagnostics=diagnostics,raw_metrics=metrics,
                progress=progress,progress_source=progress_source,validation_curve_record_index=index,
                epoch=progress.get('epoch'),global_step=progress.get('global_step'),independent_input_exposures=progress.get('independent_input_exposures'),
                source_receipt=source,source_locator='readouts.'+label+'.validation',
                num_samples=metrics.get('num_samples'),eval_seed=metrics.get('seed'),
                MSE_statistic='100 independent marginal-sample mean' if protocol=='P2' else 'legal mode analytic mean' if protocol=='P1' else 'legal mixture analytic mean',
                MAE_statistic='100 independent marginal-sample median' if protocol=='P2' else 'same analytic prediction as MSE',
                actual_gpu=receipt.get('actual_gpu','UNKNOWN'),actual_world_size=receipt.get('world_size'),
                scope='Existing shared-node validation role; not an independent scientific path or new numerical revalidation'))
        inventory[-1]['exported_readout_roles']=len(labels)
    # Retain missing registered-input slots explicitly, without inventing node IDs.
    for dataset,route,seed in sorted(expected-set(actual)):
        inventory.append(dict(node_id=None,dataset=dataset,route=route,seed=seed,status='UNKNOWN_NOT_PROVIDED',
            scientific_path=False,expected_only=True,result_receipt=None,exported_readout_roles=0))
        _gap(gaps,'SHARED_EXPECTED_SLOT','SHARED_PLANNED_NODE_NOT_PROVIDED',None,f'{dataset}/{route}/seed{seed}')
    for path in paths:
        parent=path.get('shared_start_node_id');shared=by.get(parent);receipt=receipts.get(parent,{})
        if shared is None:
            _gap(gaps,path.get('experiment_id','UNKNOWN'),'SHARED_LINEAGE_FOREIGN_KEY_UNKNOWN',None,'No matching shared_start_node_id; no other node was borrowed')
        elif any(path.get(key)!=shared.get(key) for key in ('dataset','seed')) or (
            path.get('route',path.get('configuration',{}).get('route')) is not None
            and path.get('route',path.get('configuration',{}).get('route'))!=shared.get('route')):
            _gap(gaps,path['experiment_id'],'SHARED_LINEAGE_BINDING_MISMATCH',None,'Scientific dataset/route/seed differs from its referenced shared node')
        prep=_read(path.get('preparation_receipt'),path.get('experiment_id'),gaps,'SHARED_PREPARATION_LINEAGE_UNKNOWN',dict,False) or {}
        train=_read(path.get('training_receipt'),path.get('experiment_id'),gaps,'SHARED_TRAINING_LINEAGE_UNKNOWN',dict,False) or {}
        common=receipt.get('best_checkpoint');prep_parent=prep.get('config',{}).get('checkpoint')
        prep_final=prep.get('readouts',{}).get('final',{}).get('checkpoint');train_parent=train.get('config',{}).get('checkpoint')
        if common and prep_parent and common!=prep_parent:_gap(gaps,path['experiment_id'],'SHARED_PREPARATION_PARENT_REFERENCE_MISMATCH',path.get('preparation_receipt'),'Preparation references another checkpoint; no content-equivalence assumption made')
        if prep_final and train_parent and prep_final!=train_parent:_gap(gaps,path['experiment_id'],'SHARED_RL_PARENT_REFERENCE_MISMATCH',path.get('training_receipt'),'RL does not reference this preparation final checkpoint')
        lineage.append(dict(edge='SHARED_TO_PREPARATION_TO_SCIENTIFIC_PATH',shared_start_node_id=parent,
            scientific_path_id=path.get('experiment_id'),dataset=path.get('dataset'),seed=path.get('seed'),
            shared_node_known=shared is not None,shared_common_checkpoint=common,
            preparation_receipt=path.get('preparation_receipt'),preparation_parent_checkpoint=prep_parent,
            preparation_final_checkpoint=prep_final,training_receipt=path.get('training_receipt'),training_parent_checkpoint=train_parent,
            relation_scope='Explicit node foreign key and existing configuration references; missing relations remain unknown'))
    summary=dict(expected_shared_nodes=len(expected),provided_shared_nodes=len(starts),inventory_rows=len(inventory),
        accepted_shared_nodes=sum(row.get('status')=='ACCEPTED' for row in starts),checkpoint_validation_roles=len(results),
        scientific_paths_count=len(paths),shared_nodes_are_scientific_paths=False,
        checks=dict(five_distinct_seeds=len(seeds)==5 and len(set(seeds))==5,
            all_twenty_shared_slots_provided=len(starts)==20 and set(actual)==expected and len(set(actual))==20,
            unique_shared_node_ids=len(by)==len(starts),all_shared_nodes_accepted=all(row.get('status')=='ACCEPTED' for row in starts),
            accepted_required_readout_metadata_complete=accepted_metadata_complete,
            accepted_progress_from_existing_records=accepted_progress_complete,
            scientific_shared_foreign_keys_known=all(path.get('shared_start_node_id') in by for path in paths),
            scientific_shared_bindings_consistent=not any(gap['kind']=='SHARED_LINEAGE_BINDING_MISMATCH' for gap in gaps)),
        scope='Twenty shared stages, not ninety scientific paths. Only existing validation scalars are exported; no formal-test metric or numerical check was generated.')
    return _clean(dict(inventory=inventory,checkpoint_results=results,configurations=configurations,curve_index=curves,lineage=lineage,gaps=gaps,summary=summary))
