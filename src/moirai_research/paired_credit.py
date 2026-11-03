"""Literal common actions for global/local credit; transient one-epoch cache.

Two policies resume separately at epoch boundaries. The common behavior remains
frozen at the branch parent: this is an explicitly clipped off-policy surrogate.
"""
from pathlib import Path
import json,sys,subprocess,shutil,gc
import torch
from .runtime import atomic_json
from .data import structured,forecast_dimensions
from .rewards import relative
from .continuous_estimators import structured_credits
from .rollout_diagnostics import group_statistics

def common_record(reference,batch,config):
    from .policy import joints,rollout,expert_log_prob
    from .train import to_cpu,to_device
    key=config['_paired_rollout_key'];mode=config['paired_action_role']
    if mode not in ['WRITE','READ','REGENERATE']:raise ValueError('Unknown common-action role')
    file=None if mode=='REGENERATE' else Path(config['paired_action_cache'])/(key+'.pt')
    indices=batch['_index'].detach().cpu()
    if file is not None and file.exists():
        saved=torch.load(file,map_location='cpu',weights_only=False)
        assert torch.equal(saved['indices'],indices)
    else:
        if mode not in ['WRITE','REGENERATE']:raise RuntimeError('Reader is missing a common action record: '+key)
        seed=int(config['seed'])*1000003+int(config['_paired_epoch'])*10007+int(indices[0])*13+len(indices)
        with torch.no_grad(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            experts,_,weights,alpha,_=joints(reference(batch),batch)
            if alpha is not None or len(experts)!=1:raise ValueError('Paired credit currently qualifies conditional independent A0 factors only')
            actions=rollout(experts,weights,None,config.get('J',8),config.get('G',4))
            old=expert_log_prob(experts,actions).detach()
        saved=dict(indices=indices,actions=to_cpu(actions),old=old.cpu(),source='Frozen common behavior',action_seed=seed)
        saved['actions']['component']=saved['actions']['component'].to(torch.int8)
        if file is not None:
            file.parent.mkdir(parents=True,exist_ok=True);tmp=file.with_suffix('.writing');torch.save(saved,tmp);tmp.replace(file)
    actions=to_device(saved['actions'],batch['target'].device);actions['component']=actions['component'].long()
    old=saved['old'].to(batch['target'].device)
    with torch.no_grad():
        experts,target,weights,_,_=joints(reference(batch),batch)
        check=expert_log_prob(experts,actions);torch.testing.assert_close(check,old,rtol=1e-6,atol=1e-6)
        h=config['protocol']['horizon'];d=forecast_dimensions(config['protocol'])
        prediction=structured(actions['forecast'],d,h);truth=structured(target,d,h)
        error=-(prediction.double()-truth[:,None,None].double()).square();reward=error.mean((-2,-1))
        term=structured_credits(prediction,truth,config['difference_reward_first'],config['difference_reward_second'])
        return dict(actions=actions,old=old,reference_lp=old,advantage=relative(error),observed=torch.ones_like(truth,dtype=torch.bool),
            weight=torch.ones_like(truth),mode_advantage=relative(reward),dir_advantage=relative(reward.mean(2),axis=1),
            reference_w=weights,reference_alpha=None,old_router=None,reward_mean=float(reward.mean()),
            reward_zero_std_fraction=float((reward.std(2,correction=0)==0).double().mean()),structured=term,
            behavior_scope='Stateless deterministic reconstruction of the original matched action stream; shared frozen parent' if mode=='REGENERATE' else 'Same transient serialized actions and behavior log probabilities, shared frozen parent',
            paired_action_key=key,paired_action_role=mode,paired_action_seed=saved['action_seed'],
            reward_statistics=dict(version='fixed-actions-v1',accuracy=group_statistics(reward),full_router=group_statistics(reward),
                position_accuracy_variance_G=float(error.var(2,correction=0).mean())))

def fit_pair(configs,outputs,cache_directory,device='cuda'):
    configs={k:dict(v) for k,v in configs.items()};outputs={k:Path(v) for k,v in outputs.items()};cache=Path(cache_directory)
    assert set(configs)=={'R12','R13'}
    comparable=[{k:v for k,v in c.items() if k not in ['recipe_id','continuous_recipe','scientific_path','paired_action_role']} for c in configs.values()]
    if comparable[0]!=comparable[1]:raise ValueError('Paired arms differ beyond the credit assignment and identity')
    epochs=configs['R12']['epochs'];launches=[];proofs=[]
    for epoch in range(epochs):
        for method in ['R12','R13']:
            base=configs[method];directory=outputs[method];config=dict(base,paired_action_cache=str(cache),paired_action_role='WRITE' if method=='R12' else 'READ')
            if epoch:config['resume']=str(directory/'resume.pt')
            if epoch+1<epochs:config['stop_after_epoch']=epoch+1
            file=directory.parent/(method+'_execution.json');atomic_json(file,config)
            command=[sys.executable,'-m','moirai_research.cli','train','--config',str(file),'--output',str(directory),'--device',device,'--seed',str(base['seed'])]
            run=subprocess.run(command,capture_output=True,text=True,timeout=24*3600)
            launches.append(dict(method=method,epoch=epoch,command=command,return_code=run.returncode))
            if run.returncode:
                atomic_json(directory.parent/'pair_launches.json',launches);(directory.parent/'pair_error.txt').write_text((run.stdout+'\n'+run.stderr)[-18000:]);raise RuntimeError('Paired credit stage failed: '+method)
            result=json.loads((directory/'result.json').read_text());assert result['cursor']['epoch']==epoch+1
            if epoch+1<epochs:assert result['status']=='INTERRUPTED_WITH_STATE'
            else:assert result['status'] in ['PASS','DIAGNOSTIC_ONLY']
        # Both optimizers consumed the exact same serialized records before removal.
        read_keys=[];write_keys=[]
        for method,dest in [('R12',write_keys),('R13',read_keys)]:
            for line in (outputs[method]/'micro_diagnostics.jsonl').read_text().splitlines():
                for item in json.loads(line)['diagnostics']:
                    if item['paired_action_key'].startswith(str(epoch)+'/'):dest.append(item['paired_action_key'])
        assert read_keys==write_keys and read_keys
        proofs.append(dict(epoch=epoch,matched_record_keys=len(set(read_keys)),optimizer_consumptions_per_arm=len(read_keys),
            equality='Both arms loaded the same serialized actions, rewards/common normalizer and behavior probability records; credit assignment differs',status='PASS'))
        shutil.rmtree(cache/str(epoch));atomic_json(outputs['R12'].parent/'paired_action_evidence.json',proofs)
        # Immutable common epoch boundary: no ensemble cache needs to survive a
        # platform recovery, and neither policy resumes ahead of its partner.
        boundary=outputs['R12'].parent/'paired_boundaries'/str(epoch+1)
        for method,directory in outputs.items():
            target=boundary/method;target.mkdir(parents=True,exist_ok=False)
            saved=torch.load(directory/'resume.pt',map_location='cpu',weights_only=False)
            selected={Path(p) for p in saved['extra']['best_paths'].values()}
            selected.add(directory/'checkpoints/initial.pt')
            if (directory/'checkpoints/final.pt').exists():selected.add(directory/'checkpoints/final.pt')
            del saved;gc.collect()
            for file in [directory/'resume.pt']+sorted(selected):
                destination=target/file.relative_to(directory);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,destination)
        atomic_json(boundary/'boundary.json',dict(completed_common_epoch=epoch+1,methods=['R12','R13'],status='BOTH_OPTIMIZERS_COMPLETE',
            original_training_roots={k:str(v) for k,v in outputs.items()},actions_consumed_identically=True))

    results={method:json.loads((path/'result.json').read_text()) for method,path in outputs.items()}
    return dict(status='PASS',results=results,paired_action_evidence=proofs,launches=launches,
        optimizer_trajectories=2,optimizer_process_starts=len(launches),
        optimizer_updates=sum(r['cursor']['global_step'] for r in results.values()),
        recovery_policy='Immutable boundaries after both policies finish an epoch; discard only later uncommitted work on platform recovery',
        scope='Matched global/local credit, frozen behavior with clipped IS; transient ensemble cache removed after paired consumption')
