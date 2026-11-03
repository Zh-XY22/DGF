"""Temporal admissibility is applied before ranking historical neighbors."""
import torch

def admissible(query_origin,candidate_origins,horizon,train_end,gap=None):
    gap=horizon if gap is None else gap
    # A full-horizon gap follows the neighbor's label end. Self/overlap cannot pass.
    return ((candidate_origins+horizon+gap<=query_origin)&
            (candidate_origins>=0)&(candidate_origins+horizon<=train_end))

def temporal_unique(sorted_candidates,horizon):
    chosen=[]
    for origin in sorted_candidates.tolist():
        if all(abs(origin-other)>=2*horizon for other in chosen):chosen.append(origin)
        if len(chosen)==16:break
    return torch.tensor(chosen,device=sorted_candidates.device,dtype=sorted_candidates.dtype)

def route_from_logits(logits,kappa=8.,floor=1e-4):
    k=logits.shape[-1]
    if not 0<=floor<1/k:raise ValueError('Floor must allow a nonuniform router')
    weights=floor+(1-k*floor)*logits.softmax(-1)
    return weights,weights*kappa
