"""Train-input-only feature checks, separate from optimizer/input-epoch accounting."""
import torch
from .policy import joints,rollout
from .data import structured
from .runtime import atomic_json


def check(policy,batch,config,output):
    assert not policy.fixed_kappa and policy.k==8
    assert all(p.requires_grad for p in policy.router_concentration.parameters())
    device=batch['target'].device;devices=[device.index if device.index is not None else torch.cuda.current_device()]
    prior_cpu=torch.get_rng_state();prior_gpu=torch.cuda.get_rng_state_all()
    with torch.no_grad(),torch.random.fork_rng(devices=devices):
        experts,target,w,alpha,kappa=joints(policy(batch),batch)
        assert (alpha>0).all() and torch.allclose(alpha.sum(-1,keepdim=True),kappa)
        assert torch.equal(w.argmax(-1),alpha.argmax(-1))
        altered=dict(batch);altered['target']=batch['target'].clone()
        selected=batch['prediction_mask'][...,None].expand_as(altered['target'])
        altered['target'][selected]+=17.
        changed,_,changed_w,changed_alpha,_=joints(policy(altered),altered)
        for a,b in zip(experts,changed):torch.testing.assert_close(a.mean,b.mean,atol=0,rtol=0)
        torch.testing.assert_close(w,changed_w,atol=0,rtol=0);torch.testing.assert_close(alpha,changed_alpha,atol=0,rtol=0)
        h=config['protocol']['horizon'];d=7
        means=torch.stack([structured(e.mean,d,h) for e in experts],1)
        index=w.argmax(-1);manual=means[torch.arange(len(w),device=device),index]
        assert torch.isfinite(manual).all()
        actions=rollout(experts,w,alpha,3,4)
        assert actions['pi'].shape==(len(w),3,8) and actions['mode'].shape==(len(w),3,4)
        assert not actions['pi'].requires_grad and not actions['forecast'].requires_grad
        evidence=dict(status='PASS',scope='Before formal training, two training histories only; no test or optimizer update',
            optimizer_updates=0,diagnostic_training_input_exposures=len(w),
            future_label_invariance='Exact mode means, expected weights and concentrations unchanged after altering future labels',
            MAP_mode='argmax expected outer probability; first index on ties; finite selected legal mean',
            positive_concentration_and_sum=True,total_concentration_trainable=True,
            sampling=dict(pi=actions['pi'].cpu().tolist(),mode_id=actions['mode'].cpu().tolist(),layout='pi[N,B,K];mode[N,B,G]',
                group_shared_draw=True,independent_group_sampling='Independent Dirichlet.sample leading dimension in audited rollout implementation'),
            component_family_scope='Augmented native per-coordinate family action, distinct from one outer mode per complete future trajectory')
    assert torch.equal(prior_cpu,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(prior_gpu,torch.cuda.get_rng_state_all()))
    evidence['evaluation_preserves_training_RNG']=True;atomic_json(output,evidence);return evidence
