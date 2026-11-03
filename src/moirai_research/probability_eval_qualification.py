"""Independent scalar definitions plus installed 1000-draw four-H entry proof."""
import json,math,subprocess,sys
from pathlib import Path
import torch
from .runtime import seed_all,atomic_json


def qualify_probability_evaluator(model_root,csv_file,output,revision):
    from .crps_math import probability_metrics
    from .public_data import prepare_csv
    from .policy import Policy
    from .train import weights_checkpoint
    from .runtime import load_policy
    from .evaluation import evaluation
    from .probability_evaluation import evaluate_distribution
    out=Path(output);out.mkdir(parents=True,exist_ok=True);scratch=out.parent.parent/'probability_eval_work';scratch.mkdir(exist_ok=True)
    x=[[-2.,0.,1.],[-.5,.2,-1.],[.2,.5,-2.],[.9,1.,.3],[2.3,1.7,.8]];y=[0.,.5,-1.];g=len(x)
    metrics=probability_metrics(torch.tensor(x,device='cuda',dtype=torch.float64),torch.tensor(y,device='cuda',dtype=torch.float64))
    fair=sum(sum(abs(row[k]-y[k]) for row in x)/g-sum(abs(x[i][k]-x[j][k]) for i in range(g) for j in range(g) if i!=j)/(2*g*(g-1)) for k in range(3))
    def quantile(k,q):
        values=sorted(row[k] for row in x);index=(g-1)*q;lo=math.floor(index);hi=math.ceil(index)
        return values[lo]+(index-lo)*(values[hi]-values[lo])
    pinball=sum(2*max(q*(y[k]-quantile(k,q)),(q-1)*(y[k]-quantile(k,q))) for q in [.1,.2,.3,.4,.5,.6,.7,.8,.9] for k in range(3))
    assert abs(metrics['fair_CRPS_sum']-fair)<1e-12
    assert abs(metrics['wQL9_pinball_twice_sum']-pinball)<1e-12
    for level in [.8,.9,.95]:
        key=str(round(level*100));intervals=[(quantile(k,(1-level)/2),quantile(k,(1+level)/2)) for k in range(3)]
        covered=sum(lo<=target<=hi for (lo,hi),target in zip(intervals,y));width=sum(hi-lo for lo,hi in intervals)
        assert metrics['covered_'+key+'_sum']==covered and abs(metrics['width_'+key+'_sum']-width)<1e-12
    cases=[]
    for h in [96,192,336,720]:
        artifact=scratch/f'data_H{h}';protocol=prepare_csv(str(csv_file),'ETTh1',artifact,horizon=h,source_revision=revision)
        seed_all(198373);policy=Policy(str(model_root),'A0',local_files_only=True);parent=scratch/f'parent_H{h}.pt'
        weights_checkpoint(parent,policy,dict(route='A0'),dict(global_step=0));del policy
        config=dict(dataset='ETTh1',protocol=protocol,artifact=str(artifact),weights=str(model_root),route='A0',checkpoint=str(parent),
            checkpoint_kind='policy_state',local_files_only=True,seed=198373,force_fixed_kappa=True,splits=['val'],
            four_horizon_campaign=True,distribution_samples=1000,distribution_eval_seed=198367,distribution_input_block=4,diagnostic_limit=6)
        file=out/f'evaluation_H{h}.json';atomic_json(file,config);directory=scratch/f'eval_H{h}'
        command=[sys.executable,'-m','moirai_research.cli','evaluate-distribution','--config',str(file),'--output',str(directory),'--device','cuda','--seed','198373']
        process=subprocess.run(command,capture_output=True,text=True,timeout=900)
        if process.returncode:
            (out/f'error_H{h}.txt').write_text((process.stdout+'\n'+process.stderr)[-10000:]);raise RuntimeError('Distribution evaluator failed at H'+str(h))
        receipt=json.loads((directory/'result.json').read_text());assert receipt['status']=='DIAGNOSTIC_ONLY'
        metric=receipt['results']['val'];assert metric['samples']==1000 and metric['windows']==6 and metric['distribution_evidence_multiplicity']==1
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        policy=load_policy(config,'cuda');aligned=dict(protocol,eval_input_block=4)
        legacy=evaluation(policy,aligned,'val',None,198373,str(artifact),limit=6)
        check=evaluate_distribution(policy,aligned,'val',str(artifact),samples=100,evaluation_seed=198373+65537,input_block=4,limit=6)
        differences={key:check['results'][scale]['metrics']['fair_empirical_CRPS']-legacy[key] for key,scale in [('CRPS','standardized'),('raw_CRPS','original')]}
        assert all(abs(value)<1e-8 for value in differences.values())
        cases.append(dict(horizon=h,command=command,result=metric,matched_100_draw_legacy_CRPS_differences=differences,
            matched_input_and_RNG=True,precision_matched=True,processed_stream_batches=2,windows_per_H=6))
        del policy;torch.cuda.empty_cache()
        atomic_json(out/'progress.json',dict(cases=cases))
    result=dict(status='PASS',optimizer_updates=0,optimizer_trajectories=0,cases=cases,test_used=False,
        independent_formula_check=dict(fair_CRPS_sum=fair,pinball_twice_sum=pinball,passed=True),
        scope='Installed 1000-independent-draw CLI on six validation windows across two streaming batches for all four H; scalar reference implemented separately; not full checkpoint evaluation')
    atomic_json(out/'result.json',result);return result
