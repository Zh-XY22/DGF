"""HOPE-only microbatch-matched forward/backward timing, without optimizer steps."""
from pathlib import Path
import time,statistics,torch
from .runtime import load_policy,seed_all,atomic_json
from .data import datasets,batch
from .dgf_reference_rl import make_rollout,objective
from .probe import to_cpu,to_device

def diagnose(cases,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);results=[]
    for case in cases:
        config=case['config'];policy=load_policy(config,'cuda').eval();data=datasets(policy,'ETTh1',config['protocol'])['train'];ids=[0,127,1023]
        batches=[to_cpu(batch(data,[i],'cuda')) for i in ids];measurements=[]
        for repeat in [-1,0,1,2]:
            for G in [100,32,16]:
                c=dict(config,G=G,_optimizer_step=1);c.pop('adaptive_dgf',None)
                policy.zero_grad(set_to_none=True);seed_all(271310+max(0,repeat));torch.cuda.synchronize();began=time.perf_counter()
                records=[to_cpu(make_rollout(policy,None,to_device(b,'cuda'),c)) for b in batches]
                for b,r in zip(batches,records):
                    loss,diag=objective(policy,to_device(b,'cuda'),to_device(r,'cuda'),c);assert torch.isfinite(loss);(loss/len(batches)).backward()
                norm=torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.,error_if_nonfinite=True);torch.cuda.synchronize();seconds=time.perf_counter()-began
                if repeat>=0:measurements.append(dict(G=G,repeat=repeat,seconds=seconds,gradient_norm=float(norm),zero_total_gradient=float(norm)==0))
        summary={}
        for G in [100,32,16]:
            rows=[r for r in measurements if r['G']==G];mean=statistics.mean(r['gradient_norm'] for r in rows)
            summary[str(G)]=dict(median_seconds=statistics.median(r['seconds'] for r in rows),gradient_norm_mean=mean,gradient_norm_CV=statistics.stdev(r['gradient_norm'] for r in rows)/mean if mean else None)
        result=dict(case_id=case['case_id'],horizon=config['protocol']['horizon'],seed=config['seed'],checkpoint=case['source_checkpoint'],training_input_indices=ids,microbatch=1,effective_probe_batch=3,full_training_effective_batch=512,measurements=measurements,summary=summary,scope='Fresh rollout, CPU staging, actual objective backward and clipping; optimizer.step intentionally omitted. Microbatch shape matched, N=3 is not full-N throughput or an unbiased variance estimate.',optimizer_updates=0,optimizer_trajectories=0,formal_test_used=False)
        atomic_json(out/(case['case_id']+'.json'),result);results.append(result)
    result=dict(status='PASS',cases=results,optimizer_updates=0,optimizer_trajectories=0,formal_test_used=False);atomic_json(out/'result.json',result);return result
