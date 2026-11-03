"""Installed paired-credit entry and checkpoint continuation qualification."""
from pathlib import Path
import json,gc,shutil
import torch
from .runtime import seed_all,atomic_json
from .public_data import prepare_csv
from .policy import Policy
from .train import weights_checkpoint
from .paired_credit import fit_pair

def execute_pair(model_root,artifact,parent,output,seed,qualification=False,public_revision=None,path_ids=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);scratch=out.parent.parent/('paired_execution_'+out.name);scratch.mkdir(exist_ok=True)
    artifact=Path(artifact);protocol=json.loads((artifact/'data_protocol.json').read_text());protocol.update(builder_root=str(artifact/'processed'),artifact=str(artifact))
    common=dict(dataset='ETTh1',protocol=protocol,artifact=str(artifact),weights=str(model_root),revision=public_revision,
        route='A0',checkpoint=str(parent),checkpoint_kind='policy_state',local_files_only=True,force_fixed_kappa=True,
        seed=seed,lr=5e-7,microbatch=1 if qualification else 8,global_input_batch=2 if qualification else 512,
        four_horizon_campaign=True,warmup_campaign=True,record_probability_checkpoints=True,checkpoint_interval_batches=1,
        test_release=False,phase='scientific',arm='E16',algorithm_arm='E16',epochs=2 if qualification else 4,updates_per_input_batch=2,
        nll_anchor_coefficient=0.,J=8,G=4,difference_reward_first=.1,difference_reward_second=.1,
        credit_behavior='Frozen shared parent with stopped clipped trajectory IS weight; literal common actions',
        credit_importance_bounds=[.8,1.2],credit_normalizer='Common G-axis std of accuracy plus both differences; correction=0 epsilon=1e-6')
    if qualification:common.update(qualification_only=True,diagnostic_input_limit=2,diagnostic_validation_limit=2)
    configs={method:dict(common,recipe_id=method,continuous_recipe=method,scientific_path=(path_ids or {}).get(method,f'{method}--H{protocol["horizon"]}--seed{seed}')) for method in ['R12','R13']}
    outputs={method:str(scratch/method) for method in configs}
    result=fit_pair(configs,outputs,str(scratch/'common_actions'))
    for method,directory in outputs.items():
        source=Path(directory);destination=out/'stages'/method;destination.mkdir(parents=True,exist_ok=True)
        for file in source.glob('*.json*'):
            if file.is_file():shutil.copy2(file,destination/file.name)
    result.update(qualified_recipes=['R12','R13'],test_used=False,horizon=protocol['horizon'],seed=seed,
        shared_parent=str(parent),formal_budget=not qualification)
    atomic_json(out/'result.json',result);return result

def qualify_pair(model_root,csv_file,output,revision):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);scratch=out.parent.parent/'paired_qualification';scratch.mkdir(exist_ok=True);cases=[]
    for h in [96,192,336,720]:
        artifact=scratch/f'data_H{h}';prepare_csv(str(csv_file),'ETTh1',artifact,horizon=h,source_revision=revision)
        seed_all(198431);policy=Policy(str(model_root),'A0',local_files_only=True);parent=scratch/f'parent_H{h}.pt'
        weights_checkpoint(parent,policy,dict(route='A0',seed=198431),dict(global_step=0));del policy;gc.collect()
        case=execute_pair(model_root,artifact,parent,out/f'H{h}',198431,qualification=True)
        assert all(v['cursor']['global_step']==4 and v['cursor']['epoch']==2 for v in case['results'].values())
        assert case['optimizer_process_starts']==4 and len(case['paired_action_evidence'])==2
        cases.append(dict(horizon=h,optimizer_updates=case['optimizer_updates'],optimizer_trajectories=2,
            paired_action_evidence=case['paired_action_evidence'],results={k:dict(first_step=v['first_step'],validation=v['readouts']['final']['validation']) for k,v in case['results'].items()}))
    result=dict(status='PASS',qualified_recipes=['R12','R13'],cases=cases,optimizer_updates=sum(c['optimizer_updates'] for c in cases),
        optimizer_trajectories=8,optimizer_process_starts=16,test_used=False,
        scope='Four H, two independent policies, exact same actions per epoch, full optimizer/RNG/cursor continuation between epoch subprocesses, independent reload; not an on-policy unbiased GRPO claim')
    atomic_json(out/'result.json',result);return result
