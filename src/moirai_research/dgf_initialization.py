"""Portable DGF construction and train-only feature statistics."""
from pathlib import Path
import numpy as np
from .runtime import seed_all,atomic_json
from .policy import Policy
from .public_data import prepare_csv
from .io_utils import load_checkpoint,buffered_torch_save


def training_protocol(csv,output,horizon,data_revision):
    protocol=prepare_csv(str(csv),'ETTh1',output,horizon,data_revision)
    scales=diversity_statistics(output,horizon)
    protocol['diversity_scales']=scales
    atomic_json(Path(output)/'data_protocol.json',protocol)
    return protocol


def diversity_statistics(artifact,horizon):
    values=np.load(Path(artifact)/'normalized.npy',allow_pickle=False)[:8640]
    fft=np.abs(np.fft.rfft(values[:len(values)//horizon*horizon].reshape(-1,horizon,7),axis=1))
    fft/=fft.sum(1,keepdims=True)+1e-6
    scales={key:np.maximum(np.mean(x*x,axis=0),1e-6).tolist() for key,x in [('level',values),('diff1',np.diff(values,axis=0)),('diff2',np.diff(values,n=2,axis=0))]}
    scales['fft']=np.maximum(np.mean(fft*fft,axis=(0,1)),1e-6).tolist()
    return scales


def initialize(weights,parent,output,seed,revision=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);seed_all(seed)
    source=load_checkpoint(parent,map_location='cpu',weights_only=False)
    if source.get('route',source.get('config',{}).get('route'))!='A0':raise ValueError('Expected a declared A0 SFT parent')
    if revision and source.get('config',{}).get('revision')!=revision:raise ValueError('Public revision differs from the parent')
    converted=out/'initial_module.pt';buffered_torch_save(dict(state_dict=source['policy_state']),converted)
    policy=Policy(str(weights),'D-post',str(converted),revision=revision)
    destination=out/'constructed.pt';buffered_torch_save(dict(policy_state=policy.state_dict(),route='D-post',routing='dirichlet',fixed_kappa=True,parent_checkpoint=str(parent),seed=seed),destination)
    result=dict(status='PASS',checkpoint=str(destination),optimizer_updates=0,optimizer_trajectories=0,
        operation='Weights-only A0 SFT to eight initialized D-post heads; not optimizer continuation',seed=seed)
    atomic_json(out/'result.json',result);return result
