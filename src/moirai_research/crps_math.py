"""Finite-ensemble scores and fresh iid score-function credit.

All operations use one declared sample axis. A shared latent correlated ensemble
must not call the iid gradient entry. Forecast protocol selection is irrelevant
to these distribution scores. No PPO/clipping or advantage normalization here.
"""
import math
import torch


def empirical_crps(samples, target, sample_axis=0, fair=True, almost_fair_alpha=None):
    x=samples.movedim(sample_axis,0).double();y=target.double();g=x.shape[0]
    if g<2:raise ValueError('At least two ensemble members are required')
    ordered=x.sort(dim=0).values
    ranks=torch.arange(1,g+1,device=x.device,dtype=x.dtype).reshape((g,)+(1,)*(x.ndim-1))
    pair=((2*ranks-g-1)*ordered).sum(0)
    correction=pair/(g*(g-1) if fair else g*g)
    if almost_fair_alpha is not None:
        if not fair or not 0<almost_fair_alpha<=1:raise ValueError('Invalid almost-fair mixing level')
        correction=correction*(1-(1-almost_fair_alpha)/g)
    return (x-y).abs().mean(0)-correction


def iid_crps_credit(samples,target,sample_axis=0,*,iid,behavior_is_current):
    """Reward credit for maximizing negative CRPS; pair coefficient is ONE.

    Samples and target are detached, no centering, clipping, stale actions or
    random group score broadcast. Pointwise credit requires conditional position
    factors; shared mode/router scores need the corresponding joint consequence.
    """
    if iid is not True or behavior_is_current is not True:
        raise ValueError('This estimator requires iid fresh on-policy draws')
    x=samples.detach().movedim(sample_axis,0).double();y=target.detach().double();g=x.shape[0]
    if g<2:raise ValueError('At least two iid members are required')
    pair=(x[:,None]-x[None,:]).abs().sum(1)/(g-1)
    return (-(x-y).abs()+pair).movedim(0,sample_axis).detach()


def normal_crps(mu,sigma,target):
    if not (sigma>0).all():raise ValueError('Normal scale must be positive')
    z=(target-mu)/sigma
    phi=torch.exp(-z*z/2)/math.sqrt(2*math.pi)
    return sigma*(z*torch.erf(z/math.sqrt(2))+2*phi-1/math.sqrt(math.pi))


def probability_metrics(samples,target,sample_axis=0):
    """Only scalar reductions; caller streams sums, never stores whole forecasts."""
    x=samples.movedim(sample_axis,0).double();y=target.double()
    q=torch.arange(.1,1.,.1,device=x.device,dtype=x.dtype)
    quantiles=torch.quantile(x,q,dim=0);error=y[None]-quantiles
    shape=(len(q),)+(1,)*y.ndim
    pinball=torch.maximum(q.reshape(shape)*error,(q.reshape(shape)-1)*error)
    result=dict(fair_CRPS_sum=float(empirical_crps(x,y,fair=True).sum()),
        biased_empirical_CRPS_sum=float(empirical_crps(x,y,fair=False).sum()),
        wQL9_pinball_twice_sum=float(2*pinball.sum()),wQL9_abs_target_denominator=float(y.abs().sum()),
        effective_elements=y.numel(),num_samples=len(x),wQL9_quantiles=9)
    for level in [.8,.9,.95]:
        lo,hi=torch.quantile(x,torch.tensor([(1-level)/2,(1+level)/2],device=x.device,dtype=x.dtype),dim=0)
        key=str(round(level*100))
        result['covered_'+key+'_sum']=int(((y>=lo)&(y<=hi)).sum())
        result['width_'+key+'_sum']=float((hi-lo).sum())
    return result
