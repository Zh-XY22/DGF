"""Single-card, fixed-global-batch measured admission profile; not a method arm."""
import json
from pathlib import Path
import time
import torch
from .runtime import seed_all,load_policy,optimizer_for,checked_step,save_state,atomic_json
from .data import datasets,batch
from .probe import to_cpu,to_device,relative_update
from .rl import make_rollout,objective
from .evaluation import evaluation


def profile(config_path,output,device='cuda',seed=1):
    config=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    seed_all(seed);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    policy=load_policy(config,device).eval();reference=load_policy(config,device).eval().requires_grad_(False)
    data=datasets(policy,config['dataset'],config['protocol'])['train'];optimizer=optimizer_for(policy,config['lr'])
    initial={n:p.detach().cpu().clone() for n,p in policy.named_parameters()}
    schedule=config.get('profile_microbatches',[config['microbatch']]*4)
    if not schedule or any(not isinstance(v,int) or v<1 for v in schedule):raise ValueError('Invalid fixed profile microbatch schedule')
    inputs=512*len(schedule);assert len(data)>=inputs
    seed_all(seed);indices=torch.randperm(len(data),generator=torch.Generator().manual_seed(seed)).tolist()[:inputs]
    records=[];step=0;began=time.time();torch.cuda.reset_peak_memory_stats()
    for block,micro in enumerate(schedule):
        ids=indices[block*512:(block+1)*512];pieces=[];row=dict(block=block,microbatch=micro,inputs=len(ids),fetch_seconds=0.,rollout_seconds=0.,update_seconds=0.,optimizer_seconds=0.)
        block_began=time.time()
        for begin in range(0,len(ids),micro):
            ix=ids[begin:begin+micro];t=time.time();b=batch(data,ix,device);torch.cuda.synchronize();row['fetch_seconds']+=time.time()-t
            t=time.time();record=make_rollout(policy,reference,b,config);torch.cuda.synchronize();row['rollout_seconds']+=time.time()-t
            pieces.append((to_cpu(b),to_cpu(record),len(ix)))
        row['updates']=[]
        for update in range(2):
            optimizer.zero_grad(set_to_none=True);t=time.time();total_loss=0.;max_ratio=float('-inf')
            for packed,saved,count in pieces:
                b=to_device(packed,device);record=to_device(saved,device)
                loss,diag=objective(policy,b,record,config);(loss*count/len(ids)).backward()
                total_loss+=float(loss.detach())*count/len(ids);max_ratio=max(max_ratio,diag['max_position_log_ratio'])
            torch.cuda.synchronize();row['update_seconds']+=time.time()-t;t=time.time()
            gradient=checked_step(policy,optimizer);torch.cuda.synchronize();row['optimizer_seconds']+=time.time()-t;step+=1
            row['updates'].append(dict(global_step=step,loss=total_loss,max_position_log_ratio=max_ratio,**gradient))
        row['compute_wall_seconds']=time.time()-block_began
        row['allocated_peak_bytes']=torch.cuda.max_memory_allocated();row['reserved_peak_bytes']=torch.cuda.max_memory_reserved()
        t=time.time();save_state(out/'resume.pt',policy,optimizer,config,dict(block=block+1,global_step=step,input_count=(block+1)*512))
        row['checkpoint_seconds']=time.time()-t;records.append(row);atomic_json(out/'timing.json',records);print(json.dumps(row),flush=True)
    supervised_timing=None;supervised_timings=[]
    if config.get('include_supervised_timing'):
        # One continuous discarded probe trajectory: no model/optimizer reset,
        # independent fork, or scientific checkpoint reuse after the RL probe.
        from .supervised import supervised_loss
        for group in optimizer.param_groups:group['lr']=config['supervised_lr']
        for micro in config.get('supervised_profile_microbatches',[config['microbatch']]):
            optimizer.zero_grad(set_to_none=True);ids=indices[:512]
            supervised_timing=dict(inputs=len(ids),microbatch=micro,fetch_seconds=0.,forward_backward_seconds=0.,
                objective='legal_trajectory_mixture_NLL',reused_optimizer_state=True,loss=0.)
            for begin in range(0,len(ids),micro):
                ix=ids[begin:begin+micro];t=time.time();b=batch(data,ix,device)
                torch.cuda.synchronize();supervised_timing['fetch_seconds']+=time.time()-t;t=time.time()
                loss=supervised_loss(policy,b,'legal_trajectory_mixture_NLL')
                (loss*len(ix)/len(ids)).backward();torch.cuda.synchronize()
                supervised_timing['forward_backward_seconds']+=time.time()-t
                supervised_timing['loss']+=float(loss.detach())*len(ix)/len(ids)
            t=time.time();gradient=checked_step(policy,optimizer);torch.cuda.synchronize();step+=1
            supervised_timing.update(optimizer_seconds=time.time()-t,gradient=gradient,
                global_step=step,lr=config['supervised_lr'],scientific_start_reused=False)
            supervised_timings.append(supervised_timing)
            atomic_json(out/'supervised_timings.json',supervised_timings)
    t=time.time();validation=evaluation(policy,config['protocol'],'val',None,seed,config['artifact']);torch.cuda.synchronize();validation_seconds=time.time()-t
    change,frozen=relative_update(policy,initial);assert frozen==0 and change>0
    result=dict(status='PASS',scope='Resource admission and finite-update qualification only; checkpoint never used as a scientific starting point',
        config=config,optimizer_trajectories=1,optimizer_steps=step,independent_inputs=inputs,records=records,
        steady_batch_seconds=sum(r['compute_wall_seconds'] for r in records[1:])/max(1,len(records)-1),
        checkpoint_seconds=sum(r['checkpoint_seconds'] for r in records)/len(records),full_validation_seconds=validation_seconds,
        validation_windows=validation['windows'],validation_diagnostic_only=validation,
        relative_parameter_update=change,frozen_max_change=frozen,
        actual_gpu=torch.cuda.get_device_name(),requested_gpus=1,world_size=1,
        allocated_peak_bytes=torch.cuda.max_memory_allocated(),reserved_peak_bytes=torch.cuda.max_memory_reserved(),
        cpu_threads=torch.get_num_threads(),microbatch=config['microbatch'],global_input_batch=512,J=8,G=4,
        seconds=time.time()-began)
    if supervised_timing is not None:
        result['supervised_timing']=supervised_timing;result['supervised_timings']=supervised_timings
    if 'profile_microbatches' in config:
        result['profile_microbatches']=schedule
        result['steady_batch_seconds_interpretation']='Mixed fixed microbatch timings; use per-microbatch records for admission, not this aggregate'
        result['total_input_exposures']=inputs+sum(r['inputs'] for r in supervised_timings)
    atomic_json(out/'result.json',result);return result
