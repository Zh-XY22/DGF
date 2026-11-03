"""HOPE-only mixed-distribution CRPS gradient and native-head checks."""
import math
from pathlib import Path
import torch
from .runtime import atomic_json,seed_all


def exact_normal_count_crps(theta,y):
    mu,logs,logits,mix=theta.unbind();sigma=logs.exp();w=mix.sigmoid()
    k=torch.arange(200,device=theta.device,dtype=theta.dtype)
    nb=torch.distributions.NegativeBinomial(total_count=theta.new_tensor(2.),logits=logits)
    probabilities=nb.log_prob(k).exp()
    if abs(float(probabilities.sum().detach())-1)>1e-12:raise ValueError('Discrete numerical support insufficient')
    def absolute_normal(value):
        z=(mu-value)/sigma
        return sigma*(2*torch.exp(-z*z/2)/math.sqrt(2*math.pi)+z*torch.erf(z/math.sqrt(2)))
    observed=w*absolute_normal(y)+(1-w)*(probabilities*(k-y).abs()).sum()
    cross=(probabilities*absolute_normal(k)).sum()
    count_pair=(probabilities[:,None]*probabilities[None,:]*(k[:,None]-k[None,:]).abs()).sum()
    pair=w*w*2*sigma/math.sqrt(math.pi)+2*w*(1-w)*cross+(1-w).square()*count_pair
    return observed-.5*pair


def qualify_distribution(model_root,csv_file,output,revision):
    from .distribution_crps import full_distribution_crps,moirai_full_crps,fresh_iid_crps_score
    from .crps_math import iid_crps_credit
    from .policy import Policy,joints
    from .public_data import prepare_csv
    from .data import datasets,batch
    out=Path(output);out.mkdir(parents=True,exist_ok=True);seed_all(198337)
    base=torch.tensor([.3,math.log(1.2),-.5,.7],device='cuda',dtype=torch.float64)
    theta=base.clone().requires_grad_();exact=exact_normal_count_crps(theta,theta.new_tensor(.2));truth=torch.autograd.grad(exact,theta)[0]
    finite=[]
    for i in range(4):
        delta=torch.zeros_like(base);delta[i]=1e-5
        value=(exact_normal_count_crps(base+delta,base.new_tensor(.2))-exact_normal_count_crps(base-delta,base.new_tensor(.2)))/(2e-5)
        assert abs(float(value-truth[i]))<1e-7;finite.append(float(value))
    gradients={'full_family':[],'iid_score':[]}
    for repeat in range(16):
        n=2000
        for estimator in gradients:
            theta=base.clone().requires_grad_();mu,logs,logits,mix=theta.unbind();sigma=logs.exp();w=mix.sigmoid()
            normal=torch.distributions.Normal(mu.expand(n,1),sigma.expand(n,1))
            nb=torch.distributions.NegativeBinomial(total_count=torch.full((n,1),2.,device='cuda',dtype=torch.float64),logits=logits.expand(n,1))
            weights=torch.stack([w,1-w]).expand(n,1,2);target=torch.full((n,1),.2,device='cuda',dtype=torch.float64)
            if estimator=='full_family':
                loss,_=full_distribution_crps([normal,nb],weights,torch.zeros_like(target),torch.ones_like(target),target,per_component=4)
            else:
                with torch.no_grad():
                    atomic=torch.rand((32,n,1),device='cuda')>=w
                    values=torch.where(atomic,nb.sample((32,)),normal.sample((32,))).detach()
                credit=iid_crps_credit(values,target,iid=True,behavior_is_current=True)
                safe=torch.where(atomic,values,torch.zeros_like(values))
                lp=torch.where(atomic,(1-w).log()+nb.log_prob(safe),w.log()+normal.log_prob(values))
                loss=-(credit*lp).mean()
            gradient=torch.autograd.grad(loss,theta)[0];assert torch.isfinite(gradient).all()
            gradients[estimator].append(gradient.detach())
    toy=[]
    for estimator,records in gradients.items():
        stack=torch.stack(records);mean=stack.mean(0);se=stack.std(0,correction=1)/math.sqrt(len(records))
        assert torch.all((mean-truth).abs()<torch.maximum(6*se,torch.full_like(se,.01))),estimator
        toy.append(dict(estimator=estimator,mean_gradient=mean.tolist(),standard_error=se.tolist(),reference_gradient=truth.tolist(),finite_difference=finite))
    policy=Policy(str(model_root),'A0',local_files_only=True).to('cuda').eval();native=[]
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    for h in [96,192,336,720]:
        protocol=prepare_csv(str(csv_file),'ETTh1',out/f'data_H{h}',horizon=h,source_revision=revision)
        data=datasets(policy,'ETTh1',protocol)['train'];packed=batch(data,[0,1],'cuda')
        for method in ['P01','P02','P06']:
            policy.zero_grad(set_to_none=True);experts,target,w,_,_=joints(policy(packed),packed)
            if method=='P02':loss,diagnostic=fresh_iid_crps_score(experts,w,target,samples=32)
            else:loss,diagnostic=moirai_full_crps(experts,w,target,per_component=4,almost_fair_alpha=.95 if method=='P06' else None)
            loss.backward();named=[(n,p) for n,p in policy.named_parameters() if p.grad is not None]
            assert named and all(torch.isfinite(p.grad).all() for _,p in named)
            squared=sum(p.grad.double().square().sum() for _,p in named);assert squared>0
            native.append(dict(method=method,horizon=h,loss=float(loss.detach()),gradient_norm=float(squared.sqrt()),
                parameters_with_gradient=len(named),nonzero_gradient_tensors=sum(bool((p.grad!=0).any()) for _,p in named),diagnostic=diagnostic,
                input_indices=[0,1],split='train',optimizer_updates=0))
            atomic_json(out/'progress.json',dict(toy=toy,native=native,test_used=False))
    result=dict(status='PASS',toy=toy,native=native,optimizer_updates=0,test_used=False,
        scope='Normal+count mixture reference/finite differences and full actual Moirai family gradient checks at four H; no prepared-state coefficient/stability calibration yet')
    atomic_json(out/'result.json',result);return result
