"""One training-only coefficient calibration, common to all horizons.

Uses one already completed exploration warmup state, never validation/test
outcomes. Candidate lists and the gradient-scale selection rules are fixed.
"""
import math
from pathlib import Path
import torch
from .runtime import atomic_json,load_policy,optimizer_for,checked_step,seed_all
from .public_data import prepare_csv
from .data import datasets,batch,structured,forecast_dimensions
from .policy import joints,expert_log_prob
from .probability import nll
from .distribution_crps import moirai_full_crps
from .rl import make_rollout,objective
from .continuous_estimators import exact_reference_kl_surrogate


def gradient_norm(loss,policy,retain_graph=False):
    params=[p for p in policy.parameters() if p.requires_grad]
    gradients=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=retain_graph)
    if not any(g is not None for g in gradients):raise RuntimeError('Calibration has no gradient')
    if any(not torch.isfinite(g).all() for g in gradients if g is not None):raise FloatingPointError('Invalid calibration gradient')
    return math.sqrt(sum(float(g.double().square().sum()) for g in gradients if g is not None))


def calibrate(model_root,csv_file,parent,output,revision):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);scratch=out.parent.parent/'calibration_work';scratch.mkdir(exist_ok=True)
    rows=[];raw_moments=[];rng_seed=198359
    for h in [96,192,336,720]:
        protocol=prepare_csv(str(csv_file),'ETTh1',scratch/f'data_H{h}',horizon=h,source_revision=revision)
        config=dict(dataset='ETTh1',protocol=protocol,artifact=str(scratch/f'data_H{h}'),weights=str(model_root),route='A0',
            checkpoint=str(parent),checkpoint_kind='policy_state',local_files_only=True,force_fixed_kappa=True,arm='E16',seed=rng_seed,nll_anchor_coefficient=0.)
        seed_all(rng_seed);policy=load_policy(config,'cuda').eval();reference=load_policy(config,'cuda').eval().requires_grad_(False)
        data=datasets(policy,'ETTh1',protocol)['train'];d=forecast_dimensions(protocol)
        # Common prediction origins 5000..7920 are valid for all four H.
        origins=torch.linspace(0,2920,128).round().long().tolist()
        count=0;total=0.;square=0.
        for begin in range(0,len(origins),4):
            b=batch(data,origins[begin:begin+4],'cuda');record=make_rollout(policy,reference,b,config)
            with torch.no_grad():
                _,target,_,_,_=joints(reference(b),b)
                error=-(structured(record['actions']['forecast'],d,h).double()-structured(target,d,h)[:,None,None].double()).square()
                count+=error.numel();total+=float(error.sum());square+=float(error.square().sum())
        raw_moments.append(dict(horizon=h,mean=total/count,second_moment=square/count,windows=len(origins)))
        b=batch(data,[0,997],'cuda');experts,target,w,_,_=joints(policy(b),b)
        crps,_=moirai_full_crps(experts,w,target,per_component=4)
        likelihood=nll(experts,w,target)
        crps_norm=gradient_norm(crps,policy,retain_graph=True);nll_norm=gradient_norm(likelihood,policy)
        if crps_norm<=0 or nll_norm<=0:raise RuntimeError('Zero calibration treatment')
        # Two aggressive-LR updates check numerical execution, not performance.
        optimizer=optimizer_for(policy,1.5e-6);old=make_rollout(policy,reference,b,config);steps=[]
        for update in range(2):
            optimizer.zero_grad(set_to_none=True);loss,diagnostic=objective(policy,b,old,config)
            loss.backward();steps.append(dict(loss=float(loss.detach()),**checked_step(policy,optimizer)))
        current_experts,_,_,_,_=joints(policy(b),b);lp=expert_log_prob(current_experts,old['actions'])
        geometric=(lp.double()-old['old'].double()).mean(-1)
        deviations=(geometric.exp()-1).abs().detach().flatten()
        fractions={str(c):float((deviations>c).double().mean()) for c in [.001,.005,.02]}
        pg,_=objective(policy,b,old,config);pg_norm=gradient_norm(pg,policy)
        current_experts,_,_,_,_=joints(policy(b),b);lp=expert_log_prob(current_experts,old['actions'])
        kl_surrogate,estimate=exact_reference_kl_surrogate(lp,old['old'],old['reference_lp'])
        kl_norm=gradient_norm(kl_surrogate,policy)
        rows.append(dict(horizon=h,CRPS_gradient_norm=crps_norm,NLL_gradient_norm=nll_norm,
            geometric_clip_fractions=fractions,policy_gradient_norm=pg_norm,KL_gradient_norm=kl_norm,
            KL_estimate=float(estimate.detach()),aggressive_lr_updates=steps,training_windows_only=True))
        del policy,reference,optimizer;torch.cuda.empty_cache()
    mean=sum(r['mean'] for r in raw_moments)/4;second=sum(r['second_moment'] for r in raw_moments)/4
    scale=math.sqrt(max(second-mean*mean,1e-12))
    nll_candidates=[.01,.1,1.];kl_candidates=[.0001,.001,.01];clip_candidates=[.001,.005,.02]
    def pick(candidates,fn):return min(candidates,key=lambda c:(fn(c),c))
    nll_errors={str(c):sum(abs(c*r['NLL_gradient_norm']/r['CRPS_gradient_norm']-.1) for r in rows)/4 for c in nll_candidates}
    kl_errors={str(c):sum(abs(c*r['KL_gradient_norm']/max(r['policy_gradient_norm'],1e-12)-.1) for r in rows)/4 for c in kl_candidates}
    clip_errors={str(c):abs(sum(r['geometric_clip_fractions'][str(c)] for r in rows)/4-.1) for c in clip_candidates}
    frozen=dict(frozen_reward_scale=scale,probability_NLL_coefficient=pick(nll_candidates,lambda c:nll_errors[str(c)]),
        explicit_reference_KL_coefficient=pick(kl_candidates,lambda c:kl_errors[str(c)]),
        geometric_clip=pick(clip_candidates,lambda c:clip_errors[str(c)]),annealed_NLL_initial=.01)
    result=dict(status='PASS',qualified_recipes=['R01','R05','R06','R07','R10','R11','P03','T06'],
        calibration=frozen,candidates=dict(NLL=nll_candidates,KL=kl_candidates,geometric_clip=clip_candidates),
        selection_errors=dict(NLL=nll_errors,KL=kl_errors,geometric_clip=clip_errors),rows=rows,reward_moments=raw_moments,
        optimizer_updates=8,optimizer_trajectories=4,test_used=False,validation_used=False,
        scope='One common recipe calibration; all H equally weighted; same historical-free new-round C02 prepared state used for train-only native transfer probes; coefficients frozen before formal treatment runs')
    atomic_json(out/'result.json',result);return result
