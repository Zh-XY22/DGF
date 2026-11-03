"""Executable mathematical checks. Run via CLI in a compute job, not at submission."""
from .io_utils import load_checkpoint
from pathlib import Path
import json
import time
import torch
from .rewards import relative,block_advantages,block_policy_loss,dependency_log_prob,difference_terms,ppo_surrogate
from .distillation import forward_kl_surrogate,reverse_kl_surrogate,mixed_measure_log_prob,trajectory_mixture_log_prob
from .selective import absolute_time_statistics,finalize_uncertainty,dual_mask,weighted_auxiliary_loss,retention_matched_random_mask
from .neighbor_contract import admissible,temporal_unique,route_from_logits

def core_audit(output,device='cuda',seed=1):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed)
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    d=torch.device(device);dtype=torch.float64;records=[];started=time.time()
    def t(value,grad=False):return torch.tensor(value,device=d,dtype=dtype,requires_grad=grad)
    def check(name,passed,**evidence):
        record=dict(name=name,passed=bool(passed),**evidence);records.append(record)
        with (out/'checks.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record,allow_nan=False),flush=True)
        if not passed:raise AssertionError(name)
    constant=t([1.,1.,1.,1.]).reshape(1,1,4)
    check('constant_group_zero',torch.equal(relative(constant),torch.zeros_like(constant)))
    reward=t([1.,3.,7.,9.]).reshape(1,1,4)
    p=t(.2,True);loss=-(p*t([.2,.7,-.8,.1]).reshape(1,1,4)*relative(reward)).mean();loss.backward()
    check('detached_actions_nonzero_score_gradient',p.grad.isfinite() and p.grad.abs()>1e-6,gradient=float(p.grad))
    target=torch.zeros((2,96,2),device=d,dtype=dtype);observed=torch.ones_like(target,dtype=torch.bool)
    observed[0,32:64,0]=False;observed[1,64:]=False
    pred=torch.randn((2,2,4,96,2),device=d,dtype=dtype)
    record=block_advantages(pred,target,observed)
    combined=(record['advantage']*record['weights'][:,None,None]).sum(-1)
    check('common_block_normalizer_matches_total_advantage',torch.allclose(combined,relative(record['total_reward']),atol=1e-10),
          max_error=float((combined-relative(record['total_reward'])).abs().max()),valid_zero_targets=int(observed.sum()))
    check('empty_block_weight_zero',float(record['weights'][1,2])==0.)
    current=torch.randn_like(pred,requires_grad=True);old=current.detach().clone()
    block_policy_loss(current,old,observed,record,True,clip=None).backward()
    check('local_credit_mask_and_finite_gradient',torch.isfinite(current.grad).all() and current.grad[~observed[:,None,None].expand_as(pred)].abs().sum()==0,
          norm=float(current.grad.norm()))
    # Exact enumeration: independent Bernoulli actions, two positions. No Monte Carlo.
    states=t([[0,0],[0,1],[1,0],[1,1]])
    theta=t([-.4,.7],True);prob=torch.distributions.Bernoulli(logits=theta)
    lp=prob.log_prob(states);mass=lp.sum(-1).exp().detach();r=-states.square()
    global_obj=(mass*r.sum(-1)*lp.sum(-1)).sum()
    local_obj=(mass*(r*lp).sum(-1)).sum()
    gg=torch.autograd.grad(global_obj,theta,retain_graph=True)[0]
    gl=torch.autograd.grad(local_obj,theta)[0]
    expected=-theta.sigmoid()*(1-theta.sigmoid())
    check('independent_local_score_exact_expectation',torch.allclose(gg,gl,atol=1e-12) and torch.allclose(gl,expected,atol=1e-12),
          global_gradient=gg.tolist(),local_gradient=gl.tolist(),analytic=expected.detach().tolist(),
          limitation='Unstandardized expectation only; group-normalized PPO is a separately weighted objective')
    # A coupled action deliberately requires the global joint score; do not drop cross terms.
    joint_theta=t(.3,True);coupled=torch.distributions.Bernoulli(logits=joint_theta)
    y=t([0.,1.]);lp=coupled.log_prob(y);mass=lp.exp().detach()
    coupled_grad=torch.autograd.grad((mass*(-2*y)*lp).sum(),joint_theta)[0]
    check('coupled_distribution_global_score',torch.allclose(coupled_grad,-2*joint_theta.sigmoid()*(1-joint_theta.sigmoid()),atol=1e-12),gradient=float(coupled_grad))
    hole=torch.ones((1,5,1),device=d,dtype=torch.bool);hole[:,2]=False
    dp=torch.randn((1,1,4,5,1),device=d,dtype=dtype);truth=torch.zeros((1,5,1),device=d,dtype=dtype)
    _,m1=difference_terms(dp,truth,hole,1);_,m2=difference_terms(dp,truth,hole,2)
    check('difference_masks_do_not_bridge_holes',m1.flatten().tolist()==[True,False,False,True] and not m2.any())
    lp=t([1.,2.,3.,4.,5.]).reshape(1,1,1,5,1)
    check('difference_score_dependency_sum',dependency_log_prob(lp,2).flatten().tolist()==[6.,9.,12.])
    # Gaussian KL derivatives are available analytically for the numerical estimator check.
    n=250000;teacher=torch.distributions.Normal(t(-.5),t(.7))
    mu=t(.3,True);log_scale=t(-.2,True);student=torch.distributions.Normal(mu,log_scale.exp())
    actions=teacher.sample((n,)).detach();fkl=forward_kl_surrogate(student.log_prob(actions))
    fg=torch.autograd.grad(fkl,(mu,log_scale))
    fa=((mu-teacher.loc)/log_scale.exp().square(),1-((teacher.loc-mu).square()+teacher.scale.square())/log_scale.exp().square())
    ferrors=[abs(float(a-b)) for a,b in zip(fg,fa)]
    check('FKL_analytic_gradient',max(ferrors)<.025,gradient=[float(x) for x in fg],analytic=[float(x) for x in fa],samples=n,tolerance=.025)
    mu=t(.3,True);log_scale=t(-.2,True);student=torch.distributions.Normal(mu,log_scale.exp())
    actions=student.sample((n,)).detach();slp=student.log_prob(actions);tlp=teacher.log_prob(actions)
    rkl=reverse_kl_surrogate(slp,tlp);rg=torch.autograd.grad(rkl,(mu,log_scale))
    ra=((mu-teacher.loc)/teacher.scale.square(),-1+log_scale.exp().square()/teacher.scale.square())
    check('RKL_score_function_analytic_gradient',max(abs(float(a-b)) for a,b in zip(rg,ra))<.04,
          gradient=[float(x) for x in rg],analytic=[float(x) for x in ra],samples=n,tolerance=.04)
    mu=t(.3,True);student=torch.distributions.Normal(mu,t(.8));actions=student.sample((n,)).detach()
    wrong=torch.autograd.grad((student.log_prob(actions)-teacher.log_prob(actions)).mean(),mu)[0]
    check('incorrect_detached_RKL_regression',abs(float(wrong))<.03 and abs(float(wrong-ra[0]))>.5,
          incorrect_gradient=float(wrong),correct_gradient=float(ra[0]))
    normal=torch.distributions.Normal(t(0.),t(1.));nb=torch.distributions.NegativeBinomial(total_count=t(2.),logits=t(0.))
    raw=t([0.,.25]);atomic=torch.tensor([True,False],device=d)
    measure=mixed_measure_log_prob([(normal,False),(nb,True)],t([.4,.6]).log(),raw,t(1.),atomic)
    check('mixed_measure_keeps_mass_and_density_separate',torch.allclose(measure,t([.6,.4]).log()+torch.stack([nb.log_prob(t(0.)),normal.log_prob(t(.25))])),values=measure.tolist())
    mode_lp=t([[[[[0.],[-20.]],[[-20.],[0.]]]]])
    mix=trajectory_mixture_log_prob(mode_lp,t([[.5,.5]]),torch.ones((1,2,1),device=d,dtype=torch.bool))
    point=torch.logsumexp(mode_lp+t([.5,.5]).log().reshape(1,1,2,1,1),2).sum((-2,-1))
    check('trajectory_mixture_differs_from_point_mixture',float((mix-point).abs())>10,trajectory=float(mix),pointwise=float(point))
    old=t([0.,0.]);cur=t([0.,0.],True);adv=t([1.,-1.])
    check('old_equals_current_ratio_one',torch.equal((cur-old).exp(),torch.ones_like(cur)))
    delta=old-cur.detach();kl=delta.exp()-delta-1
    check('reference_equals_current_KL_zero',float(kl.abs().max())==0)
    clip_param=t([2.],True);clipped=-ppo_surrogate(clip_param,t([0.]),t([1.]),.2).mean()
    cg=torch.autograd.grad(clipped,clip_param)[0]
    check('PPO_clipping_piecewise_gradient',float(cg)==0)
    # One actual optimizer trajectory in this audit; counted in the campaign node budget.
    parameter=torch.nn.Parameter(t([0.]));optimizer=torch.optim.SGD([parameter],lr=.3);fixed_old=parameter.detach().clone()
    ratios=[]
    for i in range(2):
        ratios.append(float((parameter-fixed_old).exp()));optimizer.zero_grad()
        loss=-ppo_surrogate(parameter,fixed_old,t([1.]),.2).mean();loss.backward();optimizer.step()
    check('second_update_reuses_old_policy',ratios[0]==1 and ratios[1]>1.2,ratios=ratios,optimizer_steps=2)
    gradients=[]
    for j,g in [(1,1),(8,4),(8,100)]:
        scalar=t(.2,True);loss=-(scalar.expand(2,j,g)*t(2.)).mean();loss.backward();gradients.append(float(scalar.grad))
    check('mean_reduction_not_scaled_by_J_G',max(gradients)-min(gradients)<1e-12,gradients=gradients)
    # Rank mean with unequal tail counts must be weighted by count*world/global_count.
    x=t([1.,2.,3.,4.,5.]);serial=x.mean();world=2
    ddp=((x[:3].mean()*3*world/5)+(x[3:].mean()*2*world/5))/world
    check('DDP_tail_weight_algebra',torch.allclose(serial,ddp),scope='Numerical reduction identity; not a multi-rank CUDA execution claim')
    alpha=t([[2.,2.]],True);pi=torch.distributions.Dirichlet(alpha).sample((8,)).transpose(0,1).detach()
    group_reward=pi[...,0];a=relative(group_reward,axis=1)
    lp=torch.distributions.Dirichlet(alpha[:,None]).log_prob(pi)
    correct=-(a*lp).mean();ag=torch.autograd.grad(correct,alpha,retain_graph=True)[0]
    within=t([[-1.,1.,-1.,1.]]).expand(8,4)
    canceled=-(lp.reshape(8,1)*within).mean();wrongg=torch.autograd.grad(canceled,alpha)[0]
    check('Dirichlet_across_J_gradient_not_within_G',ag.norm()>1e-6 and wrongg.abs().max()<1e-12,
          correct_gradient=ag.tolist(),incorrect_gradient=wrongg.tolist())
    weights,concentration=route_from_logits(t([[2.,0.,-1.,0.,0.,0.,0.,0.]]))
    check('fixed_kappa_allows_nonuniform_weights',torch.allclose(concentration.sum(-1),t([8.])) and weights.max()-weights.min()>.1,
          sum_alpha=float(concentration.sum()),weights=weights.tolist())
    origins=torch.arange(0,1000,device=d);valid=admissible(700,origins,96,8640)
    check('neighbor_label_end_before_query_with_gap',int(origins[valid].max())+192<=700 and not valid[700])
    chosen=temporal_unique(torch.arange(500,0,-1,device=d),96)
    check('neighbor_temporal_dedup',all(abs(int(a-b))>=192 for i,a in enumerate(chosen) for b in chosen[i+1:]),chosen=chosen.tolist())
    error=t([[[1.],[2.]],[[4.],[8.]]]);times=torch.tensor([[[4],[5]],[[5],[6]]],device=d)
    variable=torch.zeros_like(times);obs=torch.ones_like(times,dtype=torch.bool)
    stats=absolute_time_statistics(error,times,variable,obs,8,1)
    check('selective_absolute_time_aggregation',float(stats['count'][5,0])==2 and float(stats['sum'][5,0])==6)
    keep,info=finalize_uncertainty(stats)
    check('selective_insufficient_statistics_retained',bool(keep[4,0]) and bool(keep[6,0]))
    y=torch.zeros((8,10,2),device=d,dtype=dtype);prediction=torch.randn_like(y);estimate=torch.zeros_like(y)
    obs=torch.ones_like(y,dtype=torch.bool);u=torch.ones_like(obs);u[:,0]=False
    mask,detail=dual_mask(prediction,y,estimate,obs,u)
    check('dual_mask_AND',torch.equal(mask,u&detail['anomaly']),fallback_count=int(detail['fallback'].sum()))
    generator=torch.Generator(device=d);generator.manual_seed(seed)
    random_mask=retention_matched_random_mask(mask,obs,generator)
    check('random_mask_per_horizon_variable_counts',torch.equal(mask.sum(0),random_mask.sum(0)))
    loss_values=torch.ones_like(y,requires_grad=True);loss=weighted_auxiliary_loss(loss_values,obs,mask,True);loss.backward()
    check('soft_weight_retains_all_valid_gradients',bool((loss_values.grad[obs]>0).all()))
    checkpoint=out/'toy_state.pt';torch.save(dict(model=parameter.detach(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state()),checkpoint)
    loaded=load_checkpoint(checkpoint,map_location=d,weights_only=False)
    check('checkpoint_reload',torch.equal(loaded['model'],parameter.detach()))
    result=dict(status='PASS',checks=len(records),seed=seed,device=str(d),
        actual_gpu=torch.cuda.get_device_name(d) if d.type=='cuda' else None,
        torch=torch.__version__,cuda=torch.version.cuda,seconds=time.time()-started,
        optimizer_trajectories=1,optimizer_steps=2,
        scope='Component mathematics only. Actual Moirai parity, synthetic modes, labels and end-to-end gates remain separate.',
        records=records)
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
