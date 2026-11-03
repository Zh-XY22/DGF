"""HOPE-only real-entry tests and explicitly retrospective train-input probes."""
from pathlib import Path
import json,time,statistics
from .runtime import atomic_json,load_policy,seed_all,optimizer_for,restore_state
from .train import train
from .adaptive_dgf import adopt_prefix,qualification_settings

def qualify(config,output,prefix=None,download=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    c,auxiliary=qualification_settings(config)
    file=out/'schedule.json';atomic_json(file,dict(c,stop_after_epoch=2))
    train(file,out/'schedule','cuda',c['seed'])
    interrupted=json.loads((out/'schedule/result.json').read_text());assert interrupted['status']=='INTERRUPTED_WITH_STATE'
    atomic_json(file,dict(c,resume=str(out/'schedule/resume.pt')))
    result=train(file,out/'schedule','cuda',c['seed']);assert result['status']=='DIAGNOSTIC_ONLY'
    rows=json.loads((out/'schedule/training_curve.json').read_text());rates=[r['lr'] for r in rows]
    assert rates==[1.5e-6,1.5e-6,1.5e-7,1.5e-7],rates
    diagnostics=[json.loads(x) for x in (out/'schedule/micro_diagnostics.jsonl').open()]
    assert all(x['diagnostics'][0].get('adaptive_component_gradient_evidence') for x in diagnostics)
    assert all(not d.get('adaptive_component_gradient_evidence') for x in diagnostics for d in x['diagnostics'][1:])
    file2=out/'auxiliary.json';atomic_json(file2,auxiliary)
    result2=train(file2,out/'auxiliary','cuda',c['seed']);assert result2['cursor']['global_step']==1
    evidence=json.loads((out/'auxiliary/micro_diagnostics.jsonl').open().readline())['diagnostics'][0]
    assert evidence['auxiliary_NLL_coefficient']==1. and evidence['auxiliary_NLL'] is not None
    prefix_proof=None
    if prefix:
        new=dict(config,lr=1.5e-6,adaptive_dgf=dict(learning_rate=dict(kind='piecewise',turn_step=prefix['cursor']['global_step'],final_lr=5e-7),component_gradient_steps=[7,14,21,27]),retain_all_validation_checkpoints=True)
        uri=out/'inherited_prefix.pt';download(prefix['resume'],uri)
        path,prefix_proof=adopt_prefix(uri,new,prefix['checkpoint_map'],download,out/'prefix',prefix['cursor']['global_step'])
        policy=load_policy(new,'cuda');optimizer=optimizer_for(policy,new['lr'])
        cursor,_=restore_state(path,policy,optimizer,new)
        assert cursor==prefix['cursor'] and optimizer.state
        prefix_proof['restored_optimizer_RNG_cursor_into_real_policy']=True
    receipt=dict(status='PASS',scope='Four resumed schedule updates plus one NLL-extension update on two real training histories; two-window validation only; full-state historical-prefix adoption and real policy restore; not scientific performance',optimizer_updates=5,optimizer_trajectories=2,learning_rates=rates,prefix=prefix_proof,all_core_features_enabled=True,diagnostic_first_microbatch_scoping=True,formal_test_used=False)
    atomic_json(out/'result.json',receipt);return receipt

def diagnose(cases,output):
    from .data import datasets,batch
    from .dgf_reference_rl import make_rollout,objective
    from .probability_evaluation import evaluate_distribution
    out=Path(output);out.mkdir(parents=True,exist_ok=True);results=[]
    for case in cases:
        config=case['config'];policy=load_policy(config,'cuda').eval();data=datasets(policy,'ETTh1',config['protocol'])['train']
        # Fixed before model execution, exclusively training prediction origins.
        ids=[0,127,1023];b=batch(data,ids,'cuda');step=case['step'];probes=[]
        for group_size in [100,32,16]:
            for repeat in range(3):
                c=dict(config,G=group_size,_microbatch_index=0,_optimizer_step=step,_paired_rollout_key='posthoc/train-fixed/0',adaptive_dgf=dict(learning_rate={'kind':'constant'},component_gradient_steps=[step],calibrate_NLL_gradient=True))
                seed_all(271110+repeat);began=time.time();record=make_rollout(policy,None,b,c);loss,diag=objective(policy,b,record,c)
                assert loss.isfinite() and diag['adaptive_component_gradient_evidence']
                probes.append(dict(G=group_size,repeat=repeat,seconds=time.time()-began,diagnostics=diag))
                del loss,record
        scores=[]
        for rng in [2719001,2719003]:
            metrics=evaluate_distribution(policy,config['protocol'],'val',config['artifact'],samples=1000,evaluation_seed=rng,input_block=4)
            assert metrics['complete'] and metrics['independent_marginal_draws']
            scores.append(metrics)
        row=dict(case_id=case['case_id'],checkpoint=case['source_checkpoint'],horizon=config['protocol']['horizon'],seed=config['seed'],global_step=step,training_input_indices=ids,gradient_scope='POSTHOC_FIXED_TRAIN_INPUTS_NOT_ORIGINAL_FULL_TRAIN_GRADIENT',probes=probes,full_validation_probability=scores,formal_test_used=False)
        atomic_json(out/(case['case_id']+'.json'),row);results.append(row)
        del policy
    result=dict(status='PASS',kind='POSTHOC_TRAIN_DIAGNOSTICS_AND_FULL_VALIDATION_PROBABILITY_NO_TEST',cases=results,optimizer_updates=0,optimizer_trajectories=0)
    atomic_json(out/'result.json',result);return result
