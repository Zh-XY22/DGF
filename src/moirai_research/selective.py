"""Independent implementation of the published AND/absolute-residual definition.

No upstream callback code is copied. Its repository currently has no located license.
Indexing uses actual target timestamps and variable IDs, never packed sample offsets.
"""
import torch

def absolute_time_statistics(absolute_error,target_time,variable_id,observed,num_times,num_variables):
    key=(target_time*num_variables+variable_id).reshape(-1)
    active=observed.reshape(-1)
    key=key[active];value=absolute_error.reshape(-1)[active].double()
    if key.numel() and (key.min()<0 or key.max()>=num_times*num_variables):raise ValueError('Time index outside training series')
    count=torch.zeros(num_times*num_variables,device=value.device,dtype=torch.float64)
    total=torch.zeros_like(count);square=torch.zeros_like(count)
    count.scatter_add_(0,key,torch.ones_like(value));total.scatter_add_(0,key,value);square.scatter_add_(0,key,value.square())
    # Explicit initialized sufficient statistics; safe to all_reduce before finalization.
    return dict(count=count.reshape(num_times,num_variables),sum=total.reshape(num_times,num_variables),
                sum_square=square.reshape(num_times,num_variables))

def finalize_uncertainty(stats,rejection=.1,min_observations=2):
    n=stats['count'];mean=stats['sum']/n.clamp_min(1)
    variance=(stats['sum_square']/n.clamp_min(1)-mean.square()).clamp_min(0)
    keep=torch.ones_like(n,dtype=torch.bool)
    thresholds=[]
    for v in range(n.shape[1]):
        sufficient=n[:,v]>=min_observations
        if sufficient.any():
            threshold=torch.quantile(variance[sufficient,v],1-rejection)
            keep[sufficient,v]=variance[sufficient,v]<threshold
            # Fully constant uncertainty is not evidence for rejecting every point.
            if not keep[sufficient,v].any() and variance[sufficient,v].max()==variance[sufficient,v].min():keep[sufficient,v]=True
            thresholds.append(float(threshold))
        else:thresholds.append(None)
    return keep,dict(variance=variance,thresholds=thresholds,insufficient=n<min_observations)

def dual_mask(prediction,target,estimator_prediction,observed,uncertainty_mask,rejection=.1,min_coverage=.5):
    score=(prediction-target).abs()-(estimator_prediction-target).abs()
    anomaly=torch.ones_like(observed)
    # Code-consistent quantile is within each window across horizon, per variable.
    for n in range(target.shape[0]):
        for v in range(target.shape[-1]):
            good=observed[n,:,v]
            if good.any():
                q=torch.quantile(score[n,good,v],rejection)
                anomaly[n,good,v]=score[n,good,v]>q
    selected=observed & uncertainty_mask & anomaly
    coverage=selected.sum(1)/observed.sum(1).clamp_min(1)
    fallback=coverage<min_coverage
    selected=torch.where(fallback[:,None],observed,selected)
    return selected.detach(),dict(anomaly=anomaly.detach(),uncertainty=uncertainty_mask.detach(),
                                 pre_fallback_coverage=coverage.detach(),fallback=fallback.detach())

def weighted_auxiliary_loss(position_loss,observed,selected,soft=False):
    weight=(.2+.8*selected.to(position_loss.dtype)) if soft else selected.to(position_loss.dtype)
    weight=(weight*observed).detach()
    if weight.sum()==0:weight=observed.to(position_loss.dtype)
    return (torch.where(observed,position_loss,torch.zeros_like(position_loss))*weight).sum()/weight.sum().clamp_min(1)

def retention_matched_random_mask(selected,observed,generator):
    """Exact selected count per horizon/variable over histories; no test labels."""
    result=torch.zeros_like(observed)
    for h in range(observed.shape[1]):
        for v in range(observed.shape[2]):
            ids=observed[:,h,v].nonzero().flatten();k=int(selected[:,h,v].sum())
            order=torch.randperm(len(ids),device=ids.device,generator=generator)
            result[ids[order[:k]],h,v]=True
    return result
