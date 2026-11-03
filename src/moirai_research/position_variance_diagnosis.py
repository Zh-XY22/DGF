"""Position-credit group variance on frozen training states; no optimizer."""
from pathlib import Path
import json,gc
import torch
from .runtime import load_policy,seed_all,atomic_json
from .data import datasets,batch,structured,forecast_dimensions
from .policy import joints,rollout

@torch.no_grad()
def diagnose(model_root,cases,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;rows=[]
    for case in cases:
        p=json.loads((Path(case['data'])/'data_protocol.json').read_text());p.update(builder_root=str(Path(case['data'])/'processed'),artifact=case['data'])
        h=case['horizon'];seed=case['seed'];d=forecast_dimensions(p)
        for stage,key in [('RL_before','parent'),('RL_final','final')]:
            config=dict(weights=str(model_root),route='A0',checkpoint=case[key],checkpoint_kind='policy_state',local_files_only=True,force_fixed_kappa=True)
            seed_all(198481+h+seed+(0 if stage=='RL_before' else 100003));policy=load_policy(config,'cuda').eval();data=datasets(policy,'ETTh1',p)['train']
            origins=torch.linspace(0,2920,128).round().long().tolist();low=0;zero=0;groups=0;trajectory_low=0;trajectory_groups=0
            for start in range(0,len(origins),4):
                b=batch(data,origins[start:start+4],'cuda');experts,target,w,alpha,_=joints(policy(b),b)
                actions=rollout(experts,w,alpha,8,4);prediction=structured(actions['forecast'],d,h).double();truth=structured(target,d,h).double()
                error=-(prediction-truth[:,None,None]).square();std=error.std(2,correction=0)
                low+=int((std<1e-8).sum());zero+=int((std==0).sum());groups+=std.numel()
                whole=error.mean((-2,-1)).std(2,correction=0);trajectory_low+=int((whole<1e-8).sum());trajectory_groups+=whole.numel()
            rows.append(dict(horizon=h,seed=seed,stage=stage,training_origins=128,position_groups=groups,
                low_position_std_fraction=low/groups,zero_position_std_fraction=zero/groups,trajectory_low_std_fraction=trajectory_low/trajectory_groups,
                position_group_axes='history,subgroup_J,horizon_position,variable; std over G with correction=0',checkpoint=case[key]))
            atomic_json(out/'progress.json',dict(rows=rows));del policy,experts;gc.collect();torch.cuda.empty_cache()
    macro=sum(v['low_position_std_fraction'] for v in rows)/len(rows)
    result=dict(status='PASS',rows=rows,macro_low_position_std_fraction=macro,Q01_Q02_trigger=macro>=.1,
        threshold=.1,group_std_threshold=1e-8,optimizer_updates=0,optimizer_trajectories=0,test_used=False,validation_used=False,
        sampling='128 fixed common training origins, both exploration seeds, four H, before and after RL; each H/seed/stage has equal trigger weight',
        scope='Position groups used by actual credit; separate from retained whole-trajectory training trace statistics; fixed-state diagnostic, not a replay of every update')
    atomic_json(out/'result.json',result);return result
