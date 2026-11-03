"""Portable optimizer and complete continuation state, no cluster dependency."""
from .io_utils import load_checkpoint
import json
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from .policy import Policy
from .io_utils import buffered_torch_save

class StopRequested:
    def __init__(self):
        self.requested=False
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,self.handle)
    def handle(self,*_):self.requested=True

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def optimizer_for(policy,lr):
    from uni2ts.module.ts_embed import MultiInSizeLinear,MultiOutSizeLinear
    from uni2ts.module.position import LearnedProjection
    decay=set()
    for module in policy.modules():
        if isinstance(module,(torch.nn.Linear,MultiInSizeLinear,MultiOutSizeLinear,LearnedProjection)):
            decay.update(id(p) for name,p in module.named_parameters(recurse=False) if name.endswith('weight'))
    return torch.optim.AdamW([
        dict(params=[p for p in policy.parameters() if p.requires_grad and id(p) in decay],weight_decay=.1),
        dict(params=[p for p in policy.parameters() if p.requires_grad and id(p) not in decay],weight_decay=0.)],
        lr=lr,betas=(.9,.98),eps=1e-6)

def load_policy(config,device):
    if config.get('transport_mode'):
        from .distribution_transport import TransportPolicy
        return TransportPolicy(config).to(device)
    p=Policy(config['weights'],config['route'],config['checkpoint'] if config['route']=='A0' and config.get('checkpoint_kind','legacy')=='legacy' else None,
             revision=config.get('revision'),local_files_only=config.get('local_files_only',True))
    if config['route']!='A0' or config.get('checkpoint_kind','legacy')!='legacy':
        state=load_checkpoint(config['checkpoint'],map_location='cpu',weights_only=False)
        p.load_state_dict(state['policy_state'],strict=True)
        p.fixed_kappa=state.get('fixed_kappa',True);p.routing=state.get('routing',p.routing)
    p.configure_update(fixed_kappa=True if config.get('force_fixed_kappa',True) else p.fixed_kappa)
    if config.get('dgf_complete'):
        p.configure_update(fixed_kappa=config['phase']=='preparation' or bool(config['dgf_complete'].get('fixed_total_concentration',False)))
    return p.to(device)

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def save_state(path,policy,optimizer,config,cursor,scheduler=None,extra=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state=dict(format_version=1,policy_state=policy.state_dict(),optimizer_state=optimizer.state_dict(),
        transport_mode=getattr(policy,'transport_mode',None),
        scheduler_state=None if scheduler is None else scheduler.state_dict(),fixed_kappa=policy.fixed_kappa,
        route=policy.route,routing=policy.routing,config=config,cursor=cursor,
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
        saved_epoch=time.time(),extra=extra or {})
    buffered_torch_save(state,path)

def restore_state(path,policy,optimizer,config,scheduler=None):
    state=load_checkpoint(path,map_location='cpu',weights_only=False)
    if state['config']!=config:raise ValueError('Continuation config differs; explicitly start a new scientific attempt')
    policy.load_state_dict(state['policy_state'],strict=True);optimizer.load_state_dict(state['optimizer_state'])
    if scheduler is not None:scheduler.load_state_dict(state['scheduler_state'])
    random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy']);torch.set_rng_state(state['rng']['torch'])
    if torch.cuda.is_available():torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state['cursor'],state['extra']

def checked_step(policy,optimizer,allow_zero_gradient=False):
    trainable=[p for p in policy.parameters() if p.requires_grad]
    named=[(n,p) for n,p in policy.named_parameters() if p.grad is not None]
    if not named:raise RuntimeError('No gradient chain')
    if not all(torch.isfinite(p.grad).all() for _,p in named):raise FloatingPointError('Nonfinite parameter gradient')
    norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
    if float(norm)==0 and not allow_zero_gradient:raise RuntimeError('Zero policy gradient')
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in trainable):raise FloatingPointError('Nonfinite updated parameter')
    return dict(gradient_norm=float(norm),parameters_with_gradient=len(named),zero_total_gradient=float(norm)==0)
