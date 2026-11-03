"""Atomic remote writes using a completed in-memory archive, no numerical changes."""
import io,os,json
from pathlib import Path
import numpy as np
import torch


def write_buffer(path,buffer):
    from .hdfs_checkpoints import write_if_enabled
    path=Path(path)
    remote=write_if_enabled(path,buffer)
    if not remote:
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
        with tmp.open('wb') as stream:
            view=buffer.getbuffer();written=stream.write(view);assert written==len(view);stream.flush()
        os.replace(tmp,path)
    if path.suffix.lower() in {'.pt','.pth','.ckpt','.safetensors'}:
        record_identity(path,buffer.getbuffer(),'completed_serialization_buffer')


def record_identity(path,view,operation):
    """Identity journaling is disabled for this campaign by user instruction."""
    return None


def buffered_torch_save(value,path):
    buffer=io.BytesIO();torch.save(value,buffer);write_buffer(path,buffer)


def buffered_npz(path,**arrays):
    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);write_buffer(path,buffer)


def equal_state(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b) and a.dtype==b.dtype
    elif isinstance(a,np.ndarray):assert a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:equal_state(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):equal_state(x,y)
    else:assert a==b


def audit_io(config_path,output,device='cuda',seed=1):
    import json,time
    from .runtime import atomic_json
    config=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    # Real optimizer/RNG/cursor state from a completed resource profile, not an invented checkpoint.
    state=load_checkpoint(config['checkpoint'],map_location='cpu',weights_only=False)
    assert all(k in state for k in ['policy_state','optimizer_state','rng','cursor','config'])
    t=time.time();buffered_torch_save(state,out/'buffered_resume.pt');seconds=time.time()-t
    loaded=load_checkpoint(out/'buffered_resume.pt',map_location='cpu',weights_only=False);equal_state(state,loaded)
    torch.manual_seed(seed);samples=torch.randn(8,4,96,7,device=device)
    arrays=dict(samples=samples.cpu().numpy(),index=np.arange(8),mask=np.ones((8,96,7),dtype=bool),scalars=np.array([0.,1.,-1.]))
    t=time.time();buffered_npz(out/'block.npz',**arrays);npz_seconds=time.time()-t
    reloaded=dict(np.load(out/'block.npz',allow_pickle=False));equal_state(arrays,reloaded)
    result=dict(status='PASS',optimizer_trajectories=0,optimizer_steps=0,config=config,
       complete_optimizer_rng_cursor_state_equal=True,numeric_arrays_equal=True,
       checkpoint_seconds=seconds,npz_seconds=npz_seconds,checkpoint_bytes=checkpoint_size(out/'buffered_resume.pt'),
       actual_gpu=torch.cuda.get_device_name(),format='ordinary torch.save and numpy compressed npz; byte layout not used as scientific evidence')
    atomic_json(out/'result.json',result);return result


def load_checkpoint(path,**kwargs):
    from .hdfs_checkpoints import local_checkpoint
    with local_checkpoint(path) as local:
        return torch.load(local,**kwargs)

def checkpoint_exists(path):
    from .hdfs_checkpoints import checkpoint_exists as exists
    return exists(path)

def checkpoint_size(path):
    from .hdfs_checkpoints import local_checkpoint
    with local_checkpoint(path) as local:return Path(local).stat().st_size
