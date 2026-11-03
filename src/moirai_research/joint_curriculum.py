"""Configuration-only short-to-long weighting with matched total H shares.

Each epoch retains every H. Only dominance changes; integrated optimizer
weights remain exactly one quarter per H, even with uneven epoch budgets.
"""
import math

def balanced_epoch_weights(optimizer_updates_by_epoch):
    counts=list(optimizer_updates_by_epoch)
    if not counts or any(not isinstance(v,int) or isinstance(v,bool) or v<1 for v in counts):raise ValueError('Positive exact optimizer-update counts required')
    total=sum(counts);weights=[];start=0
    for count in counts:
        end=start+count;row=[]
        for h in range(4):
            overlap=max(0.,min(end,(h+1)*total/4)-max(start,h*total/4))
            row.append(.1+.6*overlap/count)
        if not math.isclose(sum(row),1.,abs_tol=1e-12) or min(row)<.1-1e-12:raise ValueError('Invalid curriculum weights')
        weights.append(row);start=end
    for h in range(4):
        actual=sum(n*w[h] for n,w in zip(counts,weights))
        if not math.isclose(actual,total/4,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Curriculum changed total horizon optimizer weight')
    return dict(rule='balanced-short-to-long-v1',scope='scientific_phase_only',horizons=[96,192,336,720],
        horizon_weights_by_epoch=weights,optimizer_updates_by_epoch=counts,
        total_optimizer_weight_per_H=total/4,minimum_H_weight=.1,
        interpretation='Uniform shared SFT and preparation are unchanged; scientific-phase horizon dominance moves from short to long with equal integrated update weights')
