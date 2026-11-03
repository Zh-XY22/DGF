"""HOPE-only numerical gates; no formal test prediction or optimizer update."""
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import torch
from .runtime import atomic_json,load_policy,seed_all


def qualify_normal(output):
    from .crps_math import normal_crps,empirical_crps,iid_crps_credit
    seed_all(198331);device='cuda';dtype=torch.float64;rows=[]
    groups=100000;g=8
    for mu0,sigma0,y0 in [(0.,1.,0.),(.4,1.2,-.3),(-.5,.7,1.1)]:
        mu=torch.tensor(mu0,device=device,dtype=dtype,requires_grad=True)
        logs=torch.tensor(math.log(sigma0),device=device,dtype=dtype,requires_grad=True)
        sigma=logs.exp();target=mu.new_tensor(y0)
        analytic=normal_crps(mu,sigma,target)
        exact=torch.autograd.grad(analytic,(mu,logs));eps=1e-5
        fd_mu=(normal_crps(mu.detach()+eps,sigma.detach(),target)-normal_crps(mu.detach()-eps,sigma.detach(),target))/(2*eps)
        fd_log=(normal_crps(mu.detach(),(logs.detach()+eps).exp(),target)-normal_crps(mu.detach(),(logs.detach()-eps).exp(),target))/(2*eps)
        assert abs(float(fd_mu-exact[0]))<1e-7 and abs(float(fd_log-exact[1]))<1e-7
        noise=torch.randn((g,groups),device=device,dtype=dtype);samples=mu.detach()+sigma.detach()*noise
        credit=iid_crps_credit(samples,target,iid=True,behavior_is_current=True)
        # Log-density score, then negate the reward gradient to minimize CRPS.
        estimates=[-(credit*noise/sigma.detach()).mean(0),-(credit*(noise.square()-1)).mean(0)]
        case=dict(mu=mu0,sigma=sigma0,target=y0,groups=groups,G=g,analytic_value=float(analytic.detach()),
                  finite_difference=[float(fd_mu),float(fd_log)],analytic_gradient=[float(t) for t in exact],estimates=[])
        for estimate,truth in zip(estimates,exact):
            mean=estimate.mean();se=estimate.std(correction=1)/math.sqrt(groups)
            assert abs(float(mean-truth))<max(.003,6*float(se)),(float(mean),float(truth),float(se))
            case['estimates'].append(dict(mean=float(mean),standard_error=float(se)))
        path_mu=mu.detach().clone().requires_grad_();path_log=logs.detach().clone().requires_grad_()
        path_samples=path_mu+path_log.exp()*noise
        path_loss=empirical_crps(path_samples,target,fair=True).mean()
        gradients=torch.autograd.grad(path_loss,(path_mu,path_log))
        assert all(abs(float(a-b))<.008 for a,b in zip(gradients,exact))
        case['pathwise_gradient']=[float(t) for t in gradients]
        if mu0==y0==0:
            bad=(-((-(samples-target).abs()+.5*(samples[:,None]-samples[None,:]).abs().sum(1)/(g-1))*(noise.square()-1))).mean()
            assert abs(float(bad-exact[1]))>.1
            case['wrong_half_pair_coefficient_rejected']=True
        rows.append(case)
    for flags in [(False,True),(True,False)]:
        try:iid_crps_credit(samples,target,iid=flags[0],behavior_is_current=flags[1])
        except ValueError:pass
        else:raise AssertionError('Invalid iid/on-policy premise was accepted')
    result=dict(status='PASS',optimizer_updates=0,test_used=False,cases=rows,
        scope='Normal analytic and finite-difference gradients, iid score-function sign/coefficient, pathwise comparison; no Moirai probability-training qualification yet')
    atomic_json(Path(output)/'result.json',result);return result


def qualify_data(model_root,csv_file,output,revision):
    from .public_data import prepare_csv
    from .data import datasets,batch,forecast_dimensions,assert_complete_future
    from .policy import Policy,joints
    out=Path(output);out.mkdir(parents=True,exist_ok=True);seed_all(198331)
    policy=Policy(str(model_root),'A0',local_files_only=True).to('cuda').eval();rows=[]
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    for h in [96,192,336,720]:
        protocol=prepare_csv(str(csv_file),'ETTh1',out/f'data_H{h}',horizon=h,source_revision=revision)
        data=datasets(policy,'ETTh1',protocol);checks=[]
        for split in ['train','val','test']:
            for index in sorted({0,len(data[split])-1}):
                b=batch(data[split],[index],'cuda');valid=assert_complete_future(b,h*7)
                # Test labels may be checked for structural validity, but no test
                # model prediction, metric or candidate selection is performed.
                checks.append(dict(split=split,index=index,effective_future_elements=int(valid.sum()),future_padding_excluded=True))
                if split=='test':continue
                with torch.no_grad():
                    experts,truth,w,alpha,kappa=joints(policy(b),b)
                    assert truth.shape==(1,h*7) and experts[0].mean.shape==truth.shape
                    changed={k:(v.clone() if torch.is_tensor(v) else v) for k,v in b.items()}
                    changed['target'][valid]=19.
                    other,_,ow,_,_=joints(policy(changed),changed)
                    assert torch.equal(experts[0].mean,other[0].mean) and torch.equal(w,ow)
        rows.append(dict(horizon=h,windows=protocol['windows'],protocol=protocol,checks=checks,
            native_interface='One native direct forecast; no repeated short forecast or future truth prefix',
            common_validation_origin_count=2161))
    result=dict(status='PASS',optimizer_updates=0,test_prediction_used=False,cases=rows,device=torch.cuda.get_device_name(),
        scope='Public-weight native four-H shape, valid padding, split indices and history-only prediction checks; not trained-checkpoint transfer')
    atomic_json(out/'result.json',result);return result


def qualify_transfer(checkpoint,model_root,csv_file,output,revision,seed,method):
    from .public_data import prepare_csv
    from .evaluation import evaluation
    out=Path(output);out.mkdir(parents=True,exist_ok=True);cases=[]
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    for h in [96,192,336,720]:
        directory=out/f'data_H{h}'
        protocol=prepare_csv(str(csv_file),'ETTh1',directory,horizon=h,source_revision=revision)
        config=dict(weights=str(model_root),route='A0',checkpoint=str(checkpoint),checkpoint_kind='policy_state',force_fixed_kappa=True,local_files_only=True)
        policy=load_policy(config,'cuda').eval()
        metrics=evaluation(policy,protocol,'val',None,seed,str(directory),None)
        common=evaluation(policy,protocol,'val',None,seed,str(directory),2161) if h!=720 else metrics
        cases.append(dict(horizon=h,validation=metrics,common_origin_validation=common,common_origin_rule='Original origins8640..10800 inclusive; no future targets enter history'))
        del policy;torch.cuda.empty_cache()
        atomic_json(out/'progress.json',dict(method=method,cases=cases,test_used=False))
    result=dict(status='PASS',method=method,seed=seed,optimizer_updates=0,test_used=False,cases=cases,
        one_common_checkpoint=True,training_mode='Historical H96 checkpoint transferred without training; not a jointly trained model')
    atomic_json(out/'result.json',result);return result
