"""Known conditional two-mode recovery, including x=0 irreducible ambiguity."""
from .io_utils import load_checkpoint
from pathlib import Path
import json
import time
import torch

def audit_modes(output,device='cuda',seed=1):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);d=torch.device(device);started=time.time()
    h=8;time_axis=torch.linspace(-1,1,h,device=d)
    true_means=torch.stack([-1-.3*time_axis,1+.3*time_axis])
    means=torch.nn.Parameter(torch.stack([-.4-.1*time_axis,.4+.1*time_axis]))
    router=torch.nn.Linear(1,2,device=d)
    torch.nn.init.normal_(router.weight,std=.01);torch.nn.init.zeros_(router.bias)
    parameters=[means]+list(router.parameters());optimizer=torch.optim.Adam(parameters,lr=.025)
    before={k:v.detach().clone() for k,v in [('means',means),('router_weight',router.weight)]}
    curve=[];first_gradient=None
    for step in range(400):
        x=(torch.randint(0,3,(256,1),device=d)-1).float()
        truth_w=torch.stack([torch.sigmoid(2*x[:,0]),1-torch.sigmoid(2*x[:,0])],-1)
        mode=torch.distributions.Categorical(probs=truth_w).sample()
        y=true_means[mode]+torch.randn((256,h),device=d)*.15
        logp=torch.distributions.Normal(means,.15).log_prob(y[:,None]).sum(-1)
        loss=-torch.logsumexp(router(x).log_softmax(-1)+logp,-1).mean()
        optimizer.zero_grad();loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(parameters,10.)
        if step==0:first_gradient=float(norm)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('Synthetic mixture failed')
        optimizer.step()
        if step%20==0:curve.append(dict(step=step,loss=float(loss),gradient_norm=float(norm)))
    x=torch.tensor([[-1.],[0.],[1.]],device=d)
    with torch.no_grad():
        expected=torch.stack([torch.sigmoid(2*x[:,0]),1-torch.sigmoid(2*x[:,0])],-1)
        weight=router(x).softmax(-1)
        identity_error=(means-true_means).square().mean()
        permuted_error=(means.flip(0)-true_means).square().mean()
        permutation=[0,1] if identity_error<=permuted_error else [1,0]
        aligned_mean=means[permutation];aligned_weight=weight[:,permutation]
        mean_mse=float((aligned_mean-true_means).square().mean())
        route_rmse=float((aligned_weight-expected).square().mean().sqrt())
    payload=dict(means=means.detach().cpu(),router=router.state_dict(),optimizer=optimizer.state_dict(),
                 rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if d.type=='cuda' else None,
                 step=400,seed=seed)
    torch.save(payload,out/'full_state.pt')
    reload=load_checkpoint(out/'full_state.pt',map_location=d,weights_only=False)
    assert torch.equal(reload['means'].to(d),means.detach())
    result=dict(status='PASS' if mean_mse<.02 and route_rmse<.08 and first_gradient>0 else 'FAILED_RECOVERY',
        seed=seed,optimizer_steps=400,optimizer_trajectories=1,mean_MSE=mean_mse,
        router_probability_RMSE=route_rmse,expected_weights=expected.cpu().tolist(),learned_weights=aligned_weight.cpu().tolist(),
        permutation=permutation,first_gradient=first_gradient,
        parameter_change=dict(means=float((means-before['means']).norm()),router=float((router.weight-before['router_weight']).norm())),
        ambiguous_x0_expected_weight=.5,reloaded_exact=True,seconds=time.time()-started,
        actual_gpu=torch.cuda.get_device_name(d) if d.type=='cuda' else None,
        limitation='Synthetic Gaussian modes gate only; not proof of useful ETT/Moirai modes')
    (out/'curve.json').write_text(json.dumps(curve,indent=2))
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if result['status']!='PASS':raise AssertionError('Synthetic mode recovery failed; diagnose before full release')
    return result
