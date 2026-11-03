"""Actual training-entry and high-LR train-only probe; no scientific/test result."""
from pathlib import Path
import json,torch
from .runtime import atomic_json
from .dgf_initialization import training_protocol
from .train import train


def qualify(weights,csv,output,parents,data_revision):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);cases=[]
    for horizon in [192,336]:
        work=out/('H'+str(horizon));protocol=training_protocol(csv,work/'data',horizon,data_revision)
        settings=dict(gamma=1.,lambda_dyn=.01,lambda_div=.001,dynamic_coefficients=[1.,1.,1.,1.],diversity_cap=1.,diversity_rho=1.,fixed_total_concentration=False,enable_dirichlet_objective=True,enable_dynamic=True,enable_diversity=True)
        config=dict(dataset='ETTh1',weights=str(weights),route='D-post',seed=1103,checkpoint=parents[str(horizon)],checkpoint_kind='policy_state',phase='scientific',force_fixed_kappa=False,
            dgf_complete=settings,method_kind='FULL',protocol=protocol,artifact=str(work/'data'),J=8,G=100,K=8,
            arm='E03',algorithm_arm='E03',epochs=1,updates_per_input_batch=1,qualification_only=True,diagnostic_input_limit=2,diagnostic_validation_limit=2,
            global_input_batch=2,microbatch=1,lr=1.5e-6,PPO_clip=None,reference_beta=0.,nll_anchor_coefficient=0.,
            dgf_selection_only=True,record_probability_checkpoints=True,checkpoint_interval_batches=1,test_release=False)
        path=work/'config.json';atomic_json(path,config)
        result=train(str(path),work/'training','cuda',1103)
        assert result['status']=='DIAGNOSTIC_ONLY' and result['cursor']['global_step']==1 and result['cursor']['independent_input_exposures']==2
        assert result['point_selection']['P1_val_MSE_best']['status']=='VALID' and result['probability_selection']['status']=='VALID'
        assert result['first_step']['relative_parameter_change']>0
        cases.append(dict(horizon=horizon,source=str(work/'training/result.json'),updates=result['cursor']['global_step'],selection=result['point_selection'],probability_selection=result['probability_selection'],first_step=result['first_step'],high_LR_probe=1.5e-6,J=8,G=100,formal_test=False,full_validation=False))
        torch.cuda.empty_cache()
    result=dict(status='PASS',cases=cases,optimizer_updates=2,optimizer_trajectories=2,scientific_paths_added=0,test_used=False,
        scope='Two real training-entry invocations: two train inputs and two validation inputs per H, one high-LR full-DGF update each; not full-budget training')
    atomic_json(out/'result.json',result);return result
