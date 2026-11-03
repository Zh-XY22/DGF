"""Independent DLinear estimator for selective-learning reference errors.

Algorithm source: cure-lab/LTSF-Linear models/DLinear.py (trend/seasonal
linear projections, replicated-edge moving average 25). No upstream source copy.
The estimator is a heuristic reference, not an irreducible-error oracle.
"""
from .io_utils import load_checkpoint
import json
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from torch import nn
from .runtime import seed_all,atomic_json,checked_step,StopRequested

class DLinear(nn.Module):
    def __init__(self,context,horizon):
        super().__init__();self.trend=nn.Linear(context,horizon);self.seasonal=nn.Linear(context,horizon)
    def forward(self,history):
        x=history.transpose(1,2)
        trend=torch.nn.functional.avg_pool1d(torch.nn.functional.pad(x,(12,12),mode='replicate'),25,stride=1)
        return (self.trend(trend)+self.seasonal(x-trend)).transpose(1,2)

def train_estimator(config_path,output,device='cuda',seed=1):
    config=json.loads(Path(config_path).read_text());p=config['protocol'];out=Path(output);out.mkdir(parents=True,exist_ok=True)
    seed_all(seed);stop=StopRequested();started=time.time()
    values=torch.tensor(np.load(Path(config['artifact'])/'normalized.npy',allow_pickle=False),device=device,dtype=torch.float32)
    assert values.shape[1]==7 and torch.isfinite(values).all()
    context=p['context'];h=p['horizon'];size=config['batch_size']
    train_origins=torch.arange(context,p['train_end']-h+1,device=device)
    val_origins=torch.arange(p['train_end'],p['val_end']-h+1,device=device)
    assert len(train_origins)* (7 if p['mode']=='S' else 1)==p['windows']['train']
    assert len(val_origins)* (7 if p['mode']=='S' else 1)==p['windows']['val']
    history_offsets=torch.arange(-context,0,device=device);future_offsets=torch.arange(h,device=device)
    def data(origins):return values[origins[:,None]+history_offsets],values[origins[:,None]+future_offsets]
    model=DLinear(context,h).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    epoch=0;cursor=0;step=0;best=float('inf');bad=0;best_epoch=None;curve=[];train_sum=0.;train_count=0;first=None
    def save(path):
        payload=dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),config=config,
          epoch=epoch,cursor=cursor,global_step=step,best=best,best_epoch=best_epoch,bad_epochs=bad,
          train_sum=train_sum,train_count=train_count,curve=curve,first_step=first,
          rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate()))
        tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)
    if config.get('resume'):
        state=load_checkpoint(config['resume'],map_location='cpu',weights_only=False)
        comparable={k:v for k,v in config.items() if k!='resume'}
        if {k:v for k,v in state['config'].items() if k!='resume'}!=comparable:raise ValueError('Estimator resume config mismatch')
        model.load_state_dict(state['model_state']);optimizer.load_state_dict(state['optimizer_state'])
        prior_best=Path(config['resume']).parent/'best.pt'
        if prior_best.exists() and prior_best.resolve()!=(out/'best.pt').resolve():shutil.copy2(prior_best,out/'best.pt')
        epoch=state['epoch'];cursor=state['cursor'];step=state['global_step'];best=state['best'];best_epoch=state['best_epoch'];bad=state['bad_epochs']
        train_sum=state['train_sum'];train_count=state['train_count'];curve=state['curve'];first=state['first_step']
        torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state_all(state['rng']['cuda']);np.random.set_state(state['rng']['numpy']);random.setstate(state['rng']['python'])
    @torch.no_grad()
    def validate():
        model.eval();total=0.;count=0
        for begin in range(0,len(val_origins),size):
            x,y=data(val_origins[begin:begin+size]);error=(model(x)-y).double().square()
            total+=float(error.sum());count+=error.numel()
        return total/count
    for epoch in range(epoch,config['max_epochs']):
        order=torch.randperm(len(train_origins),generator=torch.Generator().manual_seed(seed+epoch)).to(device)
        model.train()
        while cursor<len(order):
            selected=train_origins[order[cursor:cursor+size]];x,y=data(selected)
            optimizer.zero_grad(set_to_none=True);pred=model(x);loss=(pred-y).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('DLinear nonfinite loss')
            loss.backward();diag=checked_step(model,optimizer)
            if first is None:
                first=dict(loss=float(loss.detach()),**diag,parameter_update=sum(float((v.detach().cpu()-initial[k]).double().square().sum()) for k,v in model.state_dict().items())**.5)
                assert first['parameter_update']>0
            train_sum+=float(loss.detach())*len(selected);train_count+=len(selected);cursor+=len(selected);step+=1
            if step%64==0 or stop.requested:save(out/'resume.pt')
            if stop.requested:
                atomic_json(out/'result.json',dict(status='INTERRUPTED_WITH_STATE',checkpoint=str(out/'resume.pt'),global_step=step))
                return
        val=validate();improved=val<best
        if improved:best=val;best_epoch=epoch;bad=0
        else:bad+=1
        row=dict(epoch=epoch,global_step=step,train_MSE=train_sum/train_count,validation_MSE=val,
                 best=improved,seconds=time.time()-started,independent_variable_windows=train_count*7)
        curve.append(row);print(json.dumps(row),flush=True);atomic_json(out/'curve.json',curve)
        if improved:save(out/'best.pt')
        cursor=0;train_sum=0.;train_count=0
        # Saved next-epoch cursor avoids repeating a completed epoch on restart.
        current_epoch=epoch;epoch+=1;save(out/'resume.pt');epoch=current_epoch
        if bad>=config['patience']:break
    best_state=load_checkpoint(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(best_state['model_state'])
    for parameter in model.parameters():parameter.requires_grad_(False)
    reloaded=validate();assert abs(reloaded-best)<=1e-8
    result=dict(status='PASS',best_validation_MSE=best,reloaded_validation_MSE=reloaded,best_epoch=best_epoch,
        checkpoint=str(out/'best.pt'),config=config,first_step=first,parameters=sum(p.numel() for p in model.parameters()),
        optimizer_trajectories=1,optimizer_steps=step,actual_gpu=torch.cuda.get_device_name(),seconds=time.time()-started,
        frozen=True,test_used=False,training_only_fit=True,
        caveat='DLinear reference error is a heuristic; it does not identify irreducible noise')
    atomic_json(out/'result.json',result);return result
