"""Frozen, train-only E15 reliability and exact within-stratum R15 permutation."""
import json,time
from pathlib import Path
import numpy as np
from .runtime import atomic_json
from .selective_statistics import build as build_original,save_array

def build(config_path,output,device='cuda',seed=1):
    config=json.loads(Path(config_path).read_text());out=Path(output)
    result=build_original(config_path,output,device,seed)
    weights=np.load(out/'soft_weight.npy',allow_pickle=False)
    indices=np.load(out/'indices.npy',allow_pickle=False)
    assert weights.shape[0]==indices.shape[0]
    rng=np.random.default_rng(int(config['permutation_seed']))
    randomized=np.empty_like(weights);same=0;total=0
    mapping_identity=[];strata=[]
    multivariate=config['protocol']['mode']=='M'
    for variable in range(len(config['protocol']['variables'])):
        rows=np.arange(len(indices)) if multivariate else np.flatnonzero(indices[:,1]==variable)
        column=variable if multivariate else 0
        for horizon in range(config['protocol']['horizon']):
            permutation=rng.permutation(len(rows))
            source=weights[rows,horizon,column]
            selected=source[permutation]
            randomized[rows,horizon,column]=selected
            assert np.array_equal(np.sort(source),np.sort(selected))
            assert np.count_nonzero(source)==np.count_nonzero(selected)
            mapping_identity.append(dict(variable=variable,horizon=horizon,inputs=len(rows),same=int(np.count_nonzero(source==selected))))
            same+=int(np.count_nonzero(source==selected));total+=len(rows)
            strata.append(dict(variable=variable,horizon=horizon,inputs=len(rows),mean=float(source.mean()),
                minimum=float(source.min()),maximum=float(source.max()),exact_weight_distribution=True))
    assert np.isfinite(randomized).all() and np.all(randomized>0)
    save_array(out/'random_weight.npy',randomized)
    proof=dict(status='PASS',permutation_seed=config['permutation_seed'],mapping_summary=mapping_identity,
        strata=strata,same_value_fraction=same/total,nonuniform=bool(weights.min()!=weights.max()),
        observed_constraint='Complete ETT future labels only; zeros remain observed; no padding included',
        permutation_scope='training window axis independently within true variable and forecast position',
        exact_sorted_values_equal=True,independent_of_candidate_forecast=True,test_used=False,optimizer_trajectories=0)
    atomic_json(out/'random_weight_proof.json',proof)
    result.update(random_weight_artifact=str(out/'random_weight.npy'),random_weight_proof=str(out/'random_weight_proof.json'),
        R15_has_nonuniform_treatment=proof['nonuniform'])
    atomic_json(out/'result.json',result);return result
