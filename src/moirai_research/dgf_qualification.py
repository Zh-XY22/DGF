"""HOPE real-data DGF gate. Limited updates never replace scientific training."""
from pathlib import Path
import json,torch,numpy as np
from .runtime import seed_all,optimizer_for,checked_step,save_state,restore_state,atomic_json,load_policy
from .policy import Policy,joints
from .probability import nll
from .public_data import prepare_csv
from .data import datasets,batch,structured
from .dgf_reference_rl import make_rollout,objective,feature_scales,validate_contract
from .dgf_reference_math import mode_diversity
from .io_utils import load_checkpoint,buffered_torch_save

def qualify(weights,csv,output,parent_checkpoints,data_revision):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);cases=[]
    for horizon in [192,336]:
        seed_all(1103);work=out/('H'+str(horizon));work.mkdir()
        protocol=prepare_csv(str(csv),'ETTh1',work/'data',horizon,data_revision)
        values=np.load(work/'data/normalized.npy',allow_pickle=False)[:protocol['train_end']]
        fft=np.abs(np.fft.rfft(values[:len(values)//horizon*horizon].reshape(-1,horizon,7),axis=1));fft/=fft.sum(1,keepdims=True)+1e-6
        scales={k:np.maximum(np.mean(v*v,axis=0),1e-6).tolist() for k,v in [('level',values),('diff1',np.diff(values,axis=0)),('diff2',np.diff(values,n=2,axis=0))]}
        scales['fft']=np.maximum(np.mean(fft*fft,axis=(0,1)),1e-6).tolist();protocol['diversity_scales']=scales
        state=load_checkpoint(parent_checkpoints[str(horizon)],map_location='cpu',weights_only=False)
        converted=work/'initial_module.pt';buffered_torch_save({'state_dict':state['policy_state']},converted)
        policy=Policy(str(weights),'D-post',str(converted)).to('cuda').eval()
        data=datasets(policy,'ETTh1',protocol)['train'];b=batch(data,[0,1],'cuda')
        prepare_optimizer=optimizer_for(policy,1e-7);prepare_optimizer.zero_grad(set_to_none=True)
        experts,target,w,_,_=joints(policy(b),b);prepare_loss=nll(experts,w,target);prepare_loss.backward();prep=checked_step(policy,prepare_optimizer)
        checkpoint=work/'prepared.pt';buffered_torch_save(dict(policy_state=policy.state_dict(),fixed_kappa=True,routing='dirichlet'),checkpoint)
        settings=dict(gamma=1.,lambda_dyn=.01,lambda_div=.001,dynamic_coefficients=[1.,1.,1.,1.],diversity_cap=1.,diversity_rho=1.,fixed_total_concentration=False,enable_dirichlet_objective=True,enable_dynamic=True,enable_diversity=True)
        config=dict(weights=str(weights),route='D-post',checkpoint=str(checkpoint),checkpoint_kind='policy_state',phase='scientific',force_fixed_kappa=False,dgf_complete=settings,method_kind='FULL',protocol=protocol,dataset='ETTh1',artifact=str(work/'data'),J=8,G=4,K=8,epochs=4,updates_per_input_batch=1,qualification_only=True,global_input_batch=2,PPO_clip=None,reference_beta=0.,nll_anchor_coefficient=0.)
        validate_contract(config);policy=load_policy(config,'cuda').eval();assert not policy.fixed_kappa
        assert all(p.requires_grad for p in policy.router_concentration.parameters())
        optimizer=optimizer_for(policy,1e-7);stats=[]
        for update in range(2):
            optimizer.zero_grad(set_to_none=True);record=make_rollout(policy,None,b,config);loss,diag=objective(policy,b,record,config)
            experts,target,w,alpha,kappa=joints(policy(b),b)
            direction=-(torch.distributions.Dirichlet(alpha[:,None]).log_prob(record['actions']['pi'])*record['dir_advantage']).mean()
            dg=torch.autograd.grad(direction,tuple(policy.router_concentration.parameters()),retain_graph=True)
            means=torch.stack([structured(e.mean,7,horizon) for e in experts],1).double()
            diversity,distance=mode_diversity(means,record['observed'],feature_scales(config,b))
            head=next(policy.heads[0].parameters());div_grad=torch.autograd.grad(-settings['lambda_div']*diversity,head)[0]
            assert all(torch.isfinite(x).all() for x in dg) and torch.isfinite(div_grad).all()
            assert sum(x.abs().sum() for x in dg)>0 and div_grad.abs().sum()>0
            loss.backward();step=checked_step(policy,optimizer);stats.append(dict(**diag,**step,concentration_gradient_norm=sum(float(x.norm()) for x in dg),weighted_diversity_head_gradient_norm=float(div_grad.norm())))
        resume=work/'resume.pt';save_state(resume,policy,optimizer,config,dict(epoch=0,next_input=2,global_step=2))
        with torch.no_grad():before=joints(policy(b),b);means=[e.mean.clone() for e in before[0]];a=before[3].clone()
        cursor,_=restore_state(resume,policy,optimizer,config)
        with torch.no_grad():
            after=joints(policy(b),b)
            for x,y in zip(means,after[0]):torch.testing.assert_close(x,y.mean,atol=0,rtol=0)
            torch.testing.assert_close(a,after[3],atol=0,rtol=0)
        assert cursor['global_step']==2 and len(optimizer.state)>0
        cases.append(dict(horizon=horizon,preparation=prep,RL_updates=stats,optimizer_updates=3,optimizer_trajectories=2,full_features_enabled=True,total_concentration_unfrozen=True,exact_parameter_and_concentration_reload=True,optimizer_state_entries=len(optimizer.state),protocol=protocol))
    result=dict(status='PASS',scope='Two horizons, one real NLL preparation update and two fresh full-DGF RL updates each; no formal validation or test',cases=cases,optimizer_updates=6,optimizer_trajectories=4,scientific_paths_added=0,test_used=False)
    atomic_json(out/'result.json',result);return result
