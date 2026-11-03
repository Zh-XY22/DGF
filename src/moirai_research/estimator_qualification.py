"""Finite mathematical checks for the new continuous-policy surrogates.

Executed inside HOPE. Tests use fixed actions or known independent Gaussian
factors; they do not assert standardized local credit is unbiased.
"""
from pathlib import Path
import torch
from .runtime import atomic_json, seed_all
from .continuous_estimators import (cispo, ppo_asymmetric, geometric_surrogate,
    global_batch_scale, fixed_scale_advantage, exact_reference_kl_surrogate,
    local_dependency_score, structured_credits, map_baseline_advantage)


def qualify_estimators(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);seed_all(164103)
    device='cuda';dtype=torch.float64;records=[]
    current=torch.tensor([-.1,.4,1.2],device=device,dtype=dtype,requires_grad=True)
    behavior=torch.zeros_like(current);adv=torch.tensor([2.,-3.,4.],device=device,dtype=dtype)
    value=cispo(current,behavior,adv);gradient=torch.autograd.grad(value.sum(),current)[0]
    expected=current.detach().exp().clamp(0,1.2)*adv
    assert torch.allclose(gradient,expected,atol=1e-12,rtol=1e-12)
    assert not torch.allclose(gradient,expected*(1+current.detach()))
    records.append(dict(check='CISPO stopped importance weight',gradient=gradient.tolist(),expected=expected.tolist()))
    current=torch.zeros((2,8,4,11,3),device=device,dtype=dtype,requires_grad=True)
    previous=torch.zeros_like(current);a=torch.randn((2,8,4),device=device,dtype=dtype)
    whole=geometric_surrogate(current,previous,a,.005)
    blocks=geometric_surrogate(current,previous,a,.005,block_length=4)
    g1=torch.autograd.grad(whole,current,retain_graph=True)[0]
    g2=torch.autograd.grad(blocks,current)[0]
    assert torch.allclose(g1,g2,atol=1e-12,rtol=1e-12)
    expected=a[...,None,None].expand_as(current)/current.numel()
    assert torch.allclose(g1,expected,atol=1e-12,rtol=1e-12)
    records.append(dict(check='Geometric ratio old=current and uneven final block',max_gradient_difference=float((g1-g2).abs().max()),exact_joint_IS_claim=False))
    chunks=[torch.tensor([1.,2.],device=device),torch.tensor([10.,15.,21.],device=device)]
    scale=global_batch_scale(chunks);truth=torch.cat(chunks).double().std(correction=0)
    assert torch.allclose(scale,truth,atol=1e-12,rtol=1e-12)
    constant=torch.ones((2,8,4,7,3),device=device,dtype=dtype)
    assert fixed_scale_advantage(constant,2.).abs().max()==0
    rewards=torch.randn_like(constant);baseline=torch.randn((2,7,3),device=device,dtype=dtype,requires_grad=True)
    assert not map_baseline_advantage(rewards,baseline,2.).requires_grad
    records.append(dict(check='Full-input population scale and detached independent baseline',scale=float(scale)))
    # Exact categorical enumeration verifies the explicit reference-KL score
    # gradient; equality only in expectation, not for one Monte Carlo sample.
    logits=torch.tensor([-.3,.2,.8],device=device,dtype=dtype,requires_grad=True)
    p=logits.softmax(-1);q=torch.tensor([.2,.5,.3],device=device,dtype=dtype)
    actual=(p*(p.log()-q.log())).sum();ga=torch.autograd.grad(actual,logits,retain_graph=True)[0]
    lp=logits.log_softmax(-1);old=lp.detach()
    surrogate=(p.detach()*(lp-q.log()+1).detach()*lp).sum()
    gs=torch.autograd.grad(surrogate,logits)[0]
    assert torch.allclose(ga,gs,atol=1e-12,rtol=1e-12)
    zero=exact_reference_kl_surrogate(lp,old,lp.detach())[1]
    assert abs(float(zero))<1e-12
    records.append(dict(check='Reference KL exact finite-support gradient',gradient=ga.tolist(),same_distribution_estimate=float(zero)))
    # Independent Gaussian actions: raw total reward and dependency-local score
    # must have the same expected gradient. Gaussian analytic MSE/difference
    # gradients are known and avoid comparing one implementation to itself.
    n=240000;h=5;seed_all(195183)
    mean=torch.tensor([-.4,.2,.8,-.1,.5],device=device,dtype=dtype)
    target=torch.tensor([.5,-.1,.2,.6,-.2],device=device,dtype=dtype)
    action=mean+torch.randn((n,h),device=device,dtype=dtype)*.7
    score=(action-mean)/(.7**2);err=action-target
    reward=-err.square().mean(-1)
    local=-err.square()*score/h
    expected=-2*(mean-target)/h
    for order,coef in [(1,.1),(2,.1)]:
        residual=err.diff(n=order,dim=-1);part=-residual.square()
        reward=reward+coef*part.mean(-1)
        for position in range(h-order):
            for offset in range(order+1):
                local[:,position+offset]+=coef*part[:,position]*score[:,position+offset]/(h-order)
        # Independent analytic derivative of finite differences.
        residual_mean=(mean-target).diff(n=order)
        coefficients=[-1.,1.] if order==1 else [1.,-2.,1.]
        for position in range(h-order):
            for offset,coefficient in enumerate(coefficients):
                expected[position+offset]+=-2*coef*residual_mean[position]*coefficient/(h-order)
    full=reward[:,None]*score
    for name,estimates in [('global',full),('dependency_local',local)]:
        estimate=estimates.mean(0);se=estimates.std(0,correction=1)/(n**.5)
        assert ((estimate-expected).abs()<=6*se+.002).all()
        records.append(dict(check='Raw independent-factor structured reward '+name,estimate=estimate.tolist(),analytic=expected.tolist(),standard_error=se.tolist(),standardized_unbiased_claim=False))
    prediction=action[:32].reshape(2,2,8,h,1)
    truth=target.expand(2,h)[...,None]
    terms=structured_credits(prediction,truth)
    lp=torch.randn_like(prediction)
    assert terms['total_reward'].shape==(2,2,8)
    assert local_dependency_score(lp,3).shape[-2]==h-2
    constant_terms=structured_credits(torch.zeros_like(prediction),torch.zeros_like(truth))
    assert constant_terms['trajectory_advantage'].abs().max()==0
    assert all(t['advantage'].abs().max()==0 for t in constant_terms['local'])
    result=dict(status='PASS',cases=records,optimizer_updates=0,optimizer_trajectories=0,
        scope='Mathematical surrogates only; no production trainer qualification; standardized local/global objectives remain explicit different surrogates',test_used=False)
    atomic_json(out/'result.json',result);return result
