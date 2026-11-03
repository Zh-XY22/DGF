"""Real-policy gradient-only qualification for the formal objective module."""
import json
from pathlib import Path
import time
import torch
from .runtime import seed_all,load_policy,atomic_json
from .data import datasets,batch
from .rl import make_rollout,objective
from .probe import frozen_rollout,position_loss


def audit(config_path,output,device='cuda',seed=1):
    settings=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    reports=[];began=time.time()
    for case in settings['cases']:
        seed_all(seed);policy=load_policy(case,device).eval();reference=load_policy(case,device).eval().requires_grad_(False)
        training=datasets(policy,case['dataset'],case['protocol'])['train'];b=batch(training,case.get('audit_input_indices',[0,1]),device)
        config=dict(case,arm='E01' if case['route']=='A0' else 'E03',nll_anchor_coefficient=1.,difference_coefficients={'1':.01,'2':.01})
        seed_all(seed);old_record=frozen_rollout(policy,b)
        seed_all(seed);new_record=make_rollout(policy,reference,b,config)
        assert all(torch.equal(old_record['action'][k],new_record['actions'][k]) for k in old_record['action'])
        old_loss,_,_,_=position_loss(policy,b,old_record,1.)
        new_loss,diagnostic=objective(policy,b,new_record,config)
        pars=[p for p in policy.parameters() if p.requires_grad]
        old_grad=torch.autograd.grad(old_loss,pars,allow_unused=True)
        new_grad=torch.autograd.grad(new_loss,pars,allow_unused=True)
        assert abs(float(old_loss-new_loss))<1e-6
        differences=[]
        for a,z in zip(old_grad,new_grad):
            assert (a is None)==(z is None)
            if a is not None:differences.append(float((a-z).abs().max()))
        assert max(differences)<1e-5
        reports.append(dict(dataset=case['dataset'],route=case['route'],check='baseline_objective_parity',
            loss_error=abs(float(old_loss-new_loss)),gradient_max_error=max(differences),
            position_log_ratio_max=diagnostic['max_position_log_ratio'],reference_regularizer=diagnostic['reference_regularizer']))
        del old_grad,new_grad,old_loss,new_loss
        for arm in ['E12','E13','E14','E15','E16' if case['route']=='A0' else 'E17']:
            config['arm']=arm;reliability=None
            if arm=='E15':
                dim=len(case['protocol']['variables']) if case['protocol']['mode']=='M' else 1
                reliability=torch.ones(b['target'].shape[0],case['protocol']['horizon'],dim,device=device);reliability[:,32:64]=.2
            seed_all(seed);record=make_rollout(policy,reference,b,config,reliability)
            loss,diagnostic=objective(policy,b,record,config)
            gradients=torch.autograd.grad(loss,pars,allow_unused=True)
            assert all(torch.isfinite(g).all() for g in gradients if g is not None)
            norm=sum(float(g.double().square().sum()) for g in gradients if g is not None)**.5
            assert norm>0
            assert diagnostic['max_position_log_ratio']==0 and abs(diagnostic['reference_regularizer'])<1e-6
            assert all(p.grad is None for p in reference.parameters())
            reports.append(dict(dataset=case['dataset'],route=case['route'],check=arm,
                finite_nonzero_gradient_norm=norm,reference_frozen=True,**diagnostic))
            atomic_json(out/'checks.json',reports)
            del gradients,loss
        del policy,reference,training;torch.cuda.empty_cache()
    result=dict(status='PASS',checks=reports,optimizer_trajectories=0,optimizer_steps=0,seconds=time.time()-began,
        scope='Real fixed training inputs; no optimizer update or validation/test performance selection',actual_gpu=torch.cuda.get_device_name())
    atomic_json(out/'result.json',result);return result
