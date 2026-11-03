"""Four declared core ablations, limited train-entry checks, no formal test."""
from pathlib import Path
import torch
from .runtime import atomic_json
from .dgf_initialization import training_protocol
from .train import train


def qualify(weights,csv,output,parents,data_revision):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);protocol=training_protocol(csv,out/'data',192,data_revision);cases=[]
    for method in ['A01','A02','A03','A04']:
        settings=dict(gamma=1.,lambda_dyn=.01,lambda_div=.001,dynamic_coefficients=[1.,1.,1.,1.],diversity_cap=1.,diversity_rho=1.,fixed_total_concentration=False,enable_dirichlet_objective=True,enable_dynamic=True,enable_diversity=True)
        if method=='A01':settings['fixed_total_concentration']=True
        if method=='A02':settings.update(gamma=0.,enable_dirichlet_objective=False)
        if method=='A03':settings.update(lambda_dyn=0.,enable_dynamic=False)
        if method=='A04':settings.update(lambda_div=0.,enable_diversity=False)
        work=out/method;work.mkdir();config=dict(dataset='ETTh1',weights=str(weights),route='D-post',seed=1103,checkpoint=parents['192'],checkpoint_kind='policy_state',phase='scientific',force_fixed_kappa=settings['fixed_total_concentration'],
            dgf_complete=settings,method_kind='ABLATION',protocol=protocol,artifact=str(out/'data'),J=8,G=100,K=8,arm='E03',algorithm_arm='E03',epochs=1,updates_per_input_batch=1,qualification_only=True,diagnostic_input_limit=2,diagnostic_validation_limit=2,
            global_input_batch=2,microbatch=1,lr=5e-7,PPO_clip=None,reference_beta=0.,nll_anchor_coefficient=0.,dgf_selection_only=True,record_probability_checkpoints=True,checkpoint_interval_batches=1,test_release=False)
        file=work/'config.json';atomic_json(file,config);result=train(file,work/'training','cuda',1103)
        assert result['status']=='DIAGNOSTIC_ONLY' and result['cursor']['global_step']==1
        assert result['first_step']['relative_parameter_change']>0
        cases.append(dict(method=method,source=str(work/'training/result.json'),optimizer_updates=1,enabled_features=settings,classification='ABLATION_NOT_FULL_METHOD',first_step=result['first_step']))
        torch.cuda.empty_cache()
    result=dict(status='PASS',cases=cases,optimizer_updates=4,optimizer_trajectories=4,scientific_paths_added=0,test_used=False,scope='Four deliberate ablations each one update on two H192 training inputs and two validation inputs; not scientific-budget results')
    atomic_json(out/'result.json',result);return result
