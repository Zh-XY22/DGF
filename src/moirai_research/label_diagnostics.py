"""Past-fold weak-label predictability diagnostics; no optimizer or test set."""
import json,time
from pathlib import Path
import numpy as np
import torch
from .runtime import atomic_json


def diagnose(config_path,output,device='cuda',seed=1):
    config=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True);began=time.time()
    archive=np.load(config['neighbors'],allow_pickle=False)
    keys=['q','q_random','confidence','n_eff','entropy','true_cluster','fold','variable_ids','strict_nonoverlap_candidate_count']
    x={k:torch.as_tensor(archive[k],device=device) for k in keys}
    eligible=(x['confidence']>0)&(x['true_cluster']>=0)
    strict=eligible&(x['strict_nonoverlap_candidate_count']>=2)
    targets=torch.nn.functional.one_hot(x['true_cluster'].clamp_min(0).long(),8).double()
    def table(mask):
        count=int(mask.sum())
        if not count:return dict(windows=0,reason='No eligible past-only neighbors in this diagnostic subset',metrics=None)
        result={}
        for name,prob in [('neighbors',x['q']),('shuffled_control',x['q_random']),('uniform',torch.full_like(x['q'],1/8))]:
            p=prob[mask].double();y=targets[mask];assert torch.isfinite(p).all() and torch.allclose(p.sum(-1),torch.ones(count,device=device,dtype=torch.float64),atol=1e-5)
            logloss=-(y*p.clamp_min(1e-6).log()).sum(-1);brier=(p-y).square().sum(-1)
            certainty,pred=p.max(-1);correct=pred==x['true_cluster'][mask];ece=p.new_zeros(())
            for bin in range(10):
                selected=(certainty>=bin/10)&(certainty<(bin+1)/10) if bin<9 else certainty>=.9
                if selected.any():ece+=selected.double().mean()*(correct[selected].double().mean()-certainty[selected].mean()).abs()
            result[name]=dict(mode_log_loss=float(logloss.mean()),Brier_sum_over_modes=float(brier.mean()),top1_accuracy=float(correct.double().mean()),ECE_10_bins=float(ece))
        return dict(windows=count,metrics=result,mean_confidence=float(x['confidence'][mask].double().mean()),
            effective_neighbors_quantiles=torch.quantile(x['n_eff'][mask].double(),torch.tensor([0,.25,.5,.75,1],device=device,dtype=torch.float64)).cpu().tolist(),
            mean_label_entropy=float(x['entropy'][mask].double().mean()),true_mode_counts=torch.bincount(x['true_cluster'][mask].long(),minlength=8).cpu().tolist())
    result=dict(status='PASS',optimizer_trajectories=0,optimizer_steps=0,config=config,seed=seed,
        past_only_weak_label=table(eligible),strict_candidate_availability_subset=table(strict),
        total_windows=len(eligible),fallback_real_label_only_windows=int((~eligible).sum()),
        by_variable={str(int(v)):table(eligible&(x['variable_ids']==v)) for v in x['variable_ids'].unique()},
        by_fold={str(int(v)):table(eligible&(x['fold']==v)) for v in x['fold'].unique() if int(v)>=0},
        log_probability_floor=1e-6,source_fit='Each query future cluster uses past-fold centers; future cluster is used only as diagnostic outcome',
        scope='Rolling held-out training queries, not validation/test; descriptive dependent-window statistics, not significance or router-training evidence',
        limitation='Past-only neighbors can share history. The separate subset only has at least two possible fully disjoint candidates; its actual retrieved neighbors may still share history. This is an availability diagnostic, not an independent-neighbor experiment.',
        actual_gpu=torch.cuda.get_device_name(),seconds=time.time()-began)
    atomic_json(out/'result.json',result);return result
