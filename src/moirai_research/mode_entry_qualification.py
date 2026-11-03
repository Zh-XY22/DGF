"""Real public-weight D-pre lifecycle plus predictable/ambiguous Gaussian gate."""
from pathlib import Path
import json,sys,subprocess,gc
import torch
from .runtime import seed_all,atomic_json,load_policy
from .public_data import prepare_csv
from .data import datasets,batch
from .policy import Policy,joints
from .train import weights_checkpoint
from .synthetic import audit_modes

def qualify_modes(model_root,csv_file,output,revision):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);scratch=out.parent.parent/'mode_entry_work';scratch.mkdir(exist_ok=True)
    synthetic=audit_modes(scratch/'synthetic',device='cuda',seed=198409);cases=[]
    def execute(command,path):
        run=subprocess.run(command,capture_output=True,text=True,timeout=1800)
        if run.returncode:
            (out/'entry_error.txt').write_text((run.stdout+'\n'+run.stderr)[-14000:]);raise RuntimeError('D-pre entry qualification failed')
        return json.loads((path/'result.json').read_text())
    for h in [96,192,336,720]:
        artifact=scratch/f'data_H{h}';protocol=prepare_csv(str(csv_file),'ETTh1',artifact,horizon=h,source_revision=revision)
        common=dict(dataset='ETTh1',protocol=protocol,artifact=str(artifact),weights=str(model_root),route='D-pre',
            local_files_only=True,force_fixed_kappa=True,seed=198409,lr=5e-7,microbatch=1,global_input_batch=2,
            qualification_only=True,diagnostic_input_limit=2,diagnostic_validation_limit=2,four_horizon_campaign=True,
            warmup_campaign=True,record_probability_checkpoints=True,checkpoint_interval_batches=1,test_release=False,K=8,kappa=8.,J=8,G=4)
        seed_all(198409);policy=Policy(str(model_root),'D-pre',local_files_only=True).cuda().eval()
        b=batch(datasets(policy,'ETTh1',protocol)['train'],[0,1],'cuda')
        with torch.no_grad():
            result=policy(b);w=result[1];alpha=result[2];torch.testing.assert_close(alpha.sum(-1),alpha.new_full((2,),8.),rtol=1e-6,atol=1e-6)
            assert bool((w.std(-1)>1e-8).all()) and len({id(p) for head in policy.heads for p in head.parameters()})>0
            altered={k:v.clone() if torch.is_tensor(v) else v for k,v in b.items()};valid=altered['prediction_mask'][...,None]&altered['observed_mask']
            altered['target'][valid]+=1000.;after=policy(altered)
            torch.testing.assert_close(result[1],after[1],rtol=0,atol=0)
            experts,_,_,_,_=joints(result,b);differences=[float((experts[k].mean-experts[0].mean).abs().max()) for k in range(1,8)]
            assert all(v>0 for v in differences)
        params=sum(p.numel() for p in policy.parameters());trainable=sum(p.numel() for p in policy.parameters() if p.requires_grad)
        del result,experts,after,policy;gc.collect();torch.cuda.empty_cache()
        config=dict(common,phase='supervised',fixed_epochs=1,supervised_objective='legal_trajectory_mixture_NLL')
        file=out/f'SFT_H{h}.json';atomic_json(file,config);destination=scratch/f'SFT_H{h}'
        command=[sys.executable,'-c','from moirai_research.supervised import fit;fit('+repr(str(file))+','+repr(str(destination))+',device="cuda",seed=198409)']
        sft=execute(command,destination);parent=sft['readouts']['final']['checkpoint'];stages=[dict(stage='public_SFT',updates=sft['cursor']['global_step'],command=command)]
        for stage in ['ordinary_warmup','native_RL']:
            config=dict(common,phase='preparation' if stage=='ordinary_warmup' else 'scientific',preparation='ordinary',
                arm='E17',algorithm_arm='E17',checkpoint=parent,checkpoint_kind='policy_state',nll_anchor_coefficient=0.,
                epochs=2 if stage=='ordinary_warmup' else 1,updates_per_input_batch=1 if stage=='ordinary_warmup' else 2,
                recipe_id='D00')
            file=out/f'{stage}_H{h}.json';atomic_json(file,config);destination=scratch/f'{stage}_H{h}'
            command=[sys.executable,'-m','moirai_research.cli','train','--config',str(file),'--output',str(destination),'--device','cuda','--seed','198409']
            receipt=execute(command,destination);parent=receipt['readouts']['final']['checkpoint']
            assert receipt['cursor']['global_step']==2 and receipt['first_step']['relative_parameter_change']>0
            stages.append(dict(stage=stage,updates=receipt['cursor']['global_step'],command=command,first_step=receipt['first_step']))
        final=receipt['readouts']['final'];assert final['reloaded'] and final['validation']['windows']==2
        cases.append(dict(horizon=h,parameters=params,trainable_parameters=trainable,K=8,kappa=8.,head_mean_maximum_differences=differences,
            future_router_invariance=True,nonuniform_router_allowed=True,stages=stages,validation=final['validation']))
        atomic_json(out/'progress.json',dict(cases=cases))
    result=dict(status='PASS',qualified_recipes=['D00'],cases=cases,synthetic=synthetic,
        optimizer_updates=synthetic['optimizer_steps']+sum(s['updates'] for c in cases for s in c['stages']),optimizer_trajectories=1+3*len(cases),
        test_used=False,scope='Installed public entry with real D-pre SFT/preparation/native RL/reload, four H, limited updates; synthetic recovery is not proof of useful ETT modes')
    atomic_json(out/'result.json',result);return result
