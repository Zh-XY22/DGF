"""One SAM perturbation per complete accumulated input batch, warmup only."""
import math
import torch
from .policy import joints
from .probability import nll


def second_gradient(policy,optimizer,packed_batches,chunks,total_inputs,plugin,context,config,to_device,device):
    if config['phase']!='preparation' or config.get('preparation_arm')!='S06':
        raise ValueError('SAM qualification is limited to the registered C02 preparation')
    rho=float(config['sam_rho'])
    if rho<=0:raise ValueError('SAM radius must be positive')
    parameters=[p for p in policy.parameters() if p.requires_grad and p.grad is not None]
    norm=math.sqrt(sum(float(p.grad.double().square().sum()) for p in parameters))
    if not math.isfinite(norm) or norm<=0:raise FloatingPointError('Invalid SAM first gradient')
    original=[p.detach().clone() for p in parameters]
    try:
        with torch.no_grad():
            for p in parameters:p.add_(p.grad,alpha=rho/(norm+1e-12))
        optimizer.zero_grad(set_to_none=True);loss_sum=0.
        for packed,ids in zip(packed_batches,chunks):
            b=to_device(packed,device);experts,target,w,_,_=joints(policy(b),b)
            anchor=nll(experts,w,target)
            auxiliary,_=plugin.auxiliary(b,experts,target,w,context,student_policy=policy)
            loss=anchor+config['auxiliary_coefficient']*auxiliary
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite perturbed SAM loss')
            (loss*len(ids)/total_inputs).backward();loss_sum+=float(loss.detach())*len(ids)/total_inputs
    finally:
        with torch.no_grad():
            for p,saved in zip(parameters,original):p.copy_(saved)
    return dict(SAM_radius=rho,SAM_first_gradient_norm=norm,SAM_second_loss=loss_sum,
        SAM_gradient_evaluations_per_optimizer_update=2,SAM_restored_base_parameters_before_optimizer=True)
