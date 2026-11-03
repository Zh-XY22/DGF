"""One policy and one selected checkpoint, four normalized horizon losses.

The independent unit is a common prediction origin. Each optimizer update
averages four horizon losses; it is not four independent trained models.
"""
from pathlib import Path
import json,math,random,time,gc
import numpy as np
import torch
from .runtime import seed_all,load_policy,optimizer_for,checked_step,atomic_json,save_state,restore_state,StopRequested
from .policy import Policy,joints
from .data import datasets,batch
from .train import weights_checkpoint,to_cpu,to_device
from .supervised import supervised_loss
from .probability import nll
from .rl import make_rollout,objective
from .evaluation import evaluation
from .preparation import Preparation
from .probe import relative_update

TARGET_MSE={96:.364,192:.401,336:.419,720:.413}
TARGET_MAE={96:.399,192:.415,336:.426,720:.440}

def fit_joint(config_path,output,device='cuda'):
    full=json.loads(Path(config_path).read_text());resume=full.pop('resume',None);stop_after_epoch=full.pop('stop_after_epoch',None);config=full
    if set(config['protocols'])!={'96','192','336','720'}:raise ValueError('Joint branch must cover all four H')
    if config.get('test_release'):raise ValueError('Training cannot open the formal test')
    out=Path(output);out.mkdir(parents=True,exist_ok=True);seed_all(config['seed']);stop=StopRequested();began=time.time()
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    phase=config['phase'];initializing=phase=='supervised'
    if config.get('updates_by_epoch') and config['updates_by_epoch']!=[config['updates_per_input_batch']]*config['epochs']:raise ValueError('Varying joint update schedules require their own qualified execution; never ignore this field')
    if phase in ['warmup_DIRECT','scientific_DIRECT'] and not config.get('direct_objective'):raise ValueError('Joint direct phase must load its actual objective')
    if config.get('continuous_recipe')=='R09' and config.get('G')!=8:raise ValueError('Joint R09 must actually use G=8')
    if config.get('route','A0')!='A0':raise ValueError('Joint D-pre requires its own qualified architecture entry')
    rollout_function,objective_function=make_rollout,objective
    if config.get('continuous_recipe'):
        if phase!='scientific_RL':raise ValueError('Continuous-policy operators require joint RL phase')
        from .four_horizon_rl import make_rollout as rollout_function,objective as objective_function
    policy=Policy(config['weights'],'A0',revision=config.get('revision'),local_files_only=True).to(device) if initializing else load_policy(config,device)
    policy.eval();reference=load_policy(config,device).eval().requires_grad_(False) if phase=='scientific_RL' else None
    optimizer=optimizer_for(policy,config['lr']);original={k:p.detach().cpu().clone() for k,p in policy.named_parameters()}
    protocols={int(h):p for h,p in config['protocols'].items()};data={h:datasets(policy,'ETTh1',p)['train'] for h,p in protocols.items()}
    total=min(len(v) for v in data.values())
    assert total==config['common_train_windows']
    if config.get('qualification_only'):total=min(total,config['diagnostic_input_limit'])
    plugins={h:Preparation(dict(config,protocol=p,artifact=p['artifact'],preparation_arm='S06',arm='S06'),device) for h,p in protocols.items()} if phase=='warmup_C02' else {}
    if plugins and config.get('direct_objective'):raise ValueError('C02 auxiliary loss cannot be silently discarded by a direct objective; use separate explicit stages')
    cursor=dict(epoch=0,next_input=0,global_step=0,independent_input_exposures=0,horizon_input_exposures={str(h):0 for h in protocols})
    best=None;best_path=None;curve=[];training=[];first=None
    if resume:
        cursor,extra=restore_state(resume,policy,optimizer,config);best=extra['best'];best_path=extra['best_path'];curve=extra['curve'];training=extra['training'];first=extra['first']
    else:weights_checkpoint(out/'checkpoints/initial.pt',policy,config,cursor)
    def continuation():save_state(out/'resume.pt',policy,optimizer,config,cursor,extra=dict(best=best,best_path=best_path,curve=curve,training=training,first=first))
    def metrics():
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        try:return {str(h):evaluation(policy,p,'val',None,config['seed'],p['artifact'],config.get('diagnostic_validation_limit')) for h,p in protocols.items()}
        finally:
            random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
    cache={}
    def validate(epoch,fraction):
        nonlocal best,best_path
        step=cursor['global_step']
        if step not in cache:cache[step]=metrics()
        values=cache[step];objective_name=config.get('joint_selection_objective','MSE')
        if objective_name in ['MSE','MAE']:
            key='MAP_'+objective_name;den=TARGET_MSE if objective_name=='MSE' else TARGET_MAE
        elif objective_name=='CRPS':key='CRPS';den={int(h):v for h,v in config['joint_CRPS_reference'].items()}
        else:raise ValueError('Unknown frozen joint selection objective')
        ratios=[values[str(h)][key]/den[h] for h in protocols]
        score=(sum(ratios)/4,max(ratios)) if objective_name=='CRPS' else (max(ratios),sum(ratios)/4)
        if not all(math.isfinite(v) for v in score):raise FloatingPointError('Joint primary validation objective is invalid')
        selected=best is None or score<tuple(best)
        if selected:
            best=list(score);path=out/'checkpoints/initial.pt' if step==0 else out/f'checkpoints/joint-best-step{step}.pt'
            if not path.exists():weights_checkpoint(path,policy,config,cursor)
            best_path=str(path)
        curve.append(dict(epoch=epoch,fraction=fraction,global_step=step,score=list(score),selected=selected,metrics_by_horizon=values))
        atomic_json(out/'validation_curve.json',curve)
    if not resume:validate(-1,0.);continuation()
    if cursor.get('pending_epoch_validation') is not None:validate(cursor.pop('pending_epoch_validation'),1.);continuation()
    epochs=config['epochs'];global_batch=config['global_input_batch'];micro=config['microbatch'];updates=config['updates_per_input_batch']
    for epoch in range(cursor['epoch'],epochs):
        order=torch.randperm(total,generator=torch.Generator().manual_seed(config['seed']+epoch)).tolist()
        start=cursor['next_input'] if epoch==cursor['epoch'] else 0;denominator=16 if epoch==0 else 4
        thresholds=[k/denominator for k in range(1,denominator) if math.ceil(k/denominator*total)>start]
        while start<total:
            ids=order[start:start+global_batch];chunks=[ids[k:k+micro] for k in range(0,len(ids),micro)];rollouts={};packed={}
            for h,p in protocols.items():
                packed[h]=[to_cpu(batch(data[h],ix,device)) for ix in chunks]
                if reference is not None:
                    active=dict(config,protocol=p,arm='E16')
                    rollouts[h]=[to_cpu(rollout_function(policy,reference,to_device(b,device),dict(active,
                        _paired_epoch=epoch,_paired_rollout_key=f'joint/{epoch}/{start}/{h}/{chunk}'))) for chunk,b in enumerate(packed[h])]
            if config.get('continuous_recipe'):
                from .four_horizon_rl import prepare_complete_batch
                for h,p in protocols.items():prepare_complete_batch(rollouts[h],dict(config,protocol=p))
            weights=config.get('horizon_weights_by_epoch',[[.25]*4]*epochs)[epoch]
            if len(weights)!=4 or abs(sum(weights)-1.)>1e-9 or min(weights)<=0:raise ValueError('Every H must receive positive normalized exposure')
            for update in range(updates):
                if config.get('continuous_recipe')=='R08' and update>0:
                    for h,p in protocols.items():rollouts[h]=[to_cpu(rollout_function(policy,reference,to_device(b,device),dict(config,protocol=p,arm='E16'))) for b in packed[h]]
                optimizer.zero_grad(set_to_none=True);loss_sum=0.;diagnostics=[]
                for wi,(h,p) in zip(weights,protocols.items()):
                    for i,ix in enumerate(chunks):
                        b=to_device(packed[h][i],device)
                        if reference is not None:
                            active=dict(config,protocol=p,arm='E16',_optimizer_step=cursor['global_step'],_total_optimizer_steps=math.ceil(total/global_batch)*epochs*updates)
                            loss,diag=objective_function(policy,b,to_device(rollouts[h][i],device),active)
                        elif initializing:loss=supervised_loss(policy,b,'official_PackedNLLLoss');diag=dict(objective='native packed NLL')
                        else:
                            experts,target,w,_,_=joints(policy(b),b);loss=nll(experts,w,target);diag=dict(NLL=float(loss.detach()))
                            if h in plugins:
                                auxiliary,extra=plugins[h].auxiliary(b,experts,target,w);loss=loss+.1*auxiliary;diag.update(extra,auxiliary_coefficient=.1)
                            if config.get('direct_objective'):
                                from .four_horizon_losses import direct_objective
                                loss,diag=direct_objective(experts,w,target,dict(config,protocol=p))
                        if config.get('point_probability_mix'):
                            from .point_probability_mix import add_probability_term
                            loss,diag=add_probability_term(policy,b,loss,diag,dict(config,protocol=p))
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite joint training loss')
                        factor=wi*len(ix)/len(ids);(loss*factor).backward();loss_sum+=factor*float(loss.detach());diagnostics.append(dict(horizon=h,weight=wi,**diag))
                gradient=checked_step(policy,optimizer);cursor['global_step']+=1
                if first is None:
                    change,frozen=relative_update(policy,original);assert change>0 and frozen==0
                    first=dict(**gradient,relative_parameter_change=change,parameters=sum(v.numel() for v in policy.parameters()),trainable_parameters=sum(v.numel() for v in policy.parameters() if v.requires_grad))
                    atomic_json(out/'first_step.json',first)
                training.append(dict(epoch=epoch,global_step=cursor['global_step'],inputs=len(ids),loss=loss_sum,gradient=gradient,horizon_weights=weights,diagnostics=diagnostics))
            start+=len(ids);cursor.update(epoch=epoch,next_input=start,independent_input_exposures=epoch*total+start,
                horizon_input_exposures={str(h):epoch*total+start for h in protocols})
            if start==total:cursor.update(epoch=epoch+1,next_input=0,pending_epoch_validation=epoch)
            while thresholds and start>=math.ceil(thresholds[0]*total):validate(epoch,thresholds.pop(0))
            atomic_json(out/'training_curve.json',training);continuation()
            if stop_after_epoch is not None and cursor['epoch']>=int(stop_after_epoch):stop.requested=True
            if stop.requested:atomic_json(out/'result.json',dict(status='INTERRUPTED_WITH_STATE',cursor=cursor,checkpoint=str(out/'resume.pt')));return
        validate(epoch,1.);cursor.pop('pending_epoch_validation',None);continuation()
    weights_checkpoint(out/'checkpoints/final.pt',policy,config,cursor);continuation();readouts={};reloaded={}
    for label,path in dict(initial=str(out/'checkpoints/initial.pt'),joint_selected=best_path,final=str(out/'checkpoints/final.pt')).items():
        if path not in reloaded:
            state=torch.load(path,map_location=device,weights_only=False);policy.load_state_dict(state['policy_state'],strict=True);reloaded[path]=metrics()
        readouts[label]=dict(checkpoint=path,validation_by_horizon=reloaded[path],reloaded=True)
    result=dict(status='DIAGNOSTIC_ONLY' if config.get('qualification_only') else 'PASS',config=config,cursor=cursor,first_step=first,
        readouts=readouts,selection_objective=config['joint_selection_objective'],selection_score=best,best_checkpoint=best_path,
        optimizer_steps=cursor['global_step'],optimizer_updates=cursor['global_step'],optimizer_trajectories=1,continuation_checkpoint=str(out/'resume.pt'),
        actual_gpu=torch.cuda.get_device_name(),seconds=time.time()-began,test_used=False,
        interpretation='One policy, one joint-selected checkpoint across four H; each H position-normalized before macro weighting',
        common_training_origins=[5000,5000+total-1],horizon_input_exposures=cursor['horizon_input_exposures'],
        horizon_loss_weighted_optimizer_updates={str(h):sum(row['horizon_weights'][i] for row in training) for i,h in enumerate(protocols)})
    atomic_json(out/'result.json',result);return result
