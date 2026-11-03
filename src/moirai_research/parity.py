"""Private research-entry parity audit; optional path arguments are not public defaults."""
from .io_utils import load_checkpoint
import importlib.util
import json
from pathlib import Path
import sys
import time
import torch
from .policy import Policy,joints,mixture_nll
from .data import datasets,batch,assert_complete_future

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def audit_parity(config_path,output,device='cuda',seed=1):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(config_path).read_text());legacy_root=Path(config['historical_code'])
    # Actual existing implementation, not a hand-written mock of its computation.
    sys.path.insert(0,str(legacy_root))
    load_module('legacy_objectives',config['historical_actions'])
    old=load_module('legacy_policy_for_parity',legacy_root/'model_v1.py')
    reports=[];began=time.time()
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    for case in config['cases']:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        payload=load_checkpoint(case['checkpoint'],map_location='cpu',weights_only=False)
        official=case['route']=='A0'
        from .hdfs_checkpoints import local_checkpoint
        with local_checkpoint(case['checkpoint']) as local:
            old_policy=old.Policy(case['weights'],case['route'],local if official else None).to(device).eval()
        new_policy=Policy(case['weights'],case['route'],case['checkpoint'] if official else None).to(device).eval()
        if not official:
            old_policy.load_state_dict(payload['policy_state'],strict=True)
            new_policy.load_state_dict(payload['policy_state'],strict=True)
            for p in [old_policy,new_policy]:p.fixed_kappa=payload['fixed_kappa'];p.routing=payload['routing']
        exact_state=all(torch.equal(v,new_policy.state_dict()[k]) for k,v in old_policy.state_dict().items())
        assert exact_state
        ds=datasets(new_policy,case['dataset'],case['protocol'])
        b=batch(ds['val'],[0,1],device);dim=7 if case['protocol']['mode']=='M' else 1
        mask=assert_complete_future(b,96*dim)
        with torch.no_grad():
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);o=old_policy(b);oe,oy,ow,oa,ok=old.joints(o,b)
            old_nll=old.mixture_nll(o,b)
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);n=new_policy(b);ne,ny,nw,na,nk=joints(n,b)
            new_nll=mixture_nll(n,b)
            mean_error=max(float((a.mean-z.mean).abs().max()) for a,z in zip(oe,ne))
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);old_actions=old.rollout(oe,ow,oa,2,4)
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            from .policy import rollout,expert_log_prob
            new_actions=rollout(ne,nw,na,2,4)
            action_equal=all(torch.equal(old_actions[k],new_actions[k]) for k in old_actions)
            old_lp=old.expert_log_prob(oe,old_actions);new_lp=expert_log_prob(ne,old_actions)
            lp_error=float((old_lp-new_lp).abs().max())
        # Save/reload independently; does not perform an optimizer update.
        path=out/(case['dataset']+'-'+case['route']+'-reload.pt')
        torch.save(dict(policy_state=new_policy.state_dict(),route=case['route']),path)
        reload=load_checkpoint(path,map_location=device,weights_only=False)
        new_policy.load_state_dict(reload['policy_state'],strict=True)
        reload_exact=all(torch.equal(v,reload['policy_state'][k]) for k,v in new_policy.state_dict().items())
        row=dict(dataset=case['dataset'],route=case['route'],source_checkpoint=case['checkpoint'],
            old_entry=str(legacy_root/'model_v1.py'),new_entry=__import__('inspect').getfile(Policy),
            state_equal=exact_state,mean_max_abs_error=mean_error,logp_max_abs_error=lp_error,
            old_NLL=float(old_nll),new_NLL=float(new_nll),action_equal=action_equal,
            reload_exact=reload_exact,windows={k:len(v) for k,v in ds.items()},valid_targets=int(mask.sum()),
            parameters=sum(p.numel() for p in new_policy.parameters()),
            trainable_parameters=sum(p.numel() for p in new_policy.parameters() if p.requires_grad),
            tolerance=1e-6,
            scope='Current historical entry vs independent package on two fixed validation inputs; not full formal reevaluation')
        assert mean_error<=1e-6 and lp_error<=1e-6 and action_equal and reload_exact and abs(float(old_nll-new_nll))<=1e-6
        reports.append(row);(out/'cases.json').write_text(json.dumps(reports,indent=2));print(json.dumps(row),flush=True)
        del old_policy,new_policy,ds,b,o,n,oe,ne;torch.cuda.empty_cache()
    result=dict(status='PASS',cases=reports,seed=seed,seconds=time.time()-began,actual_gpu=torch.cuda.get_device_name(),optimizer_trajectories=0)
    (out/'result.json').write_text(json.dumps(result,indent=2));return result
