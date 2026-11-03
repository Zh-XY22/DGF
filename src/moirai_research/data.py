"""Official LSF transform seam, with explicit paths supplied by configuration."""
import torch
from torch.utils.data import default_collate

def forecast_dimensions(protocol):
    if protocol['mode'] not in ('M','S'):
        raise ValueError('Unsupported variable organization')
    variables=protocol['variables']
    if not variables or len(set(variables))!=len(variables):
        raise ValueError('Protocol requires distinct declared variables')
    return len(variables) if protocol['mode']=='M' else 1

def datasets(policy,name,protocol):
    from uni2ts.model.moirai import MoiraiFinetune
    from uni2ts.data.builder.simple import generate_finetune_builder,generate_eval_builder
    p=protocol
    ft=MoiraiFinetune(module=policy.module,min_patches=2,min_mask_ratio=.15,max_mask_ratio=.5,
        max_dim=max(128,len(p['variables'])),num_training_steps=None,num_warmup_steps=0,context_length=p['context'],
        prediction_length=p.get('horizon',96),patch_size=p['patch'],lr=1e-7,weight_decay=.1)
    ft.seq_fields=ft.seq_fields+('sample_id',)
    shared=dict(prediction_length=p.get('horizon',96),context_length=p['context'],patch_size=p['patch'],
                mode=p['mode'],storage_path=p['builder_root'],distance=1)
    result={
        'train':generate_finetune_builder(name,train_length=p['train_end'],**shared).load_dataset(ft.train_transform_map),
        'val':generate_eval_builder(name+'_eval',offset=p['train_end'],eval_length=p['val_end']-p['train_end'],**shared).load_dataset(ft.val_transform_map),
        'test':generate_eval_builder(name+'_eval',offset=p['val_end'],eval_length=p['test_end']-p['val_end'],**shared).load_dataset(ft.val_transform_map),
    }
    origin_start = p.get('train_origin_start', p['context'])
    if origin_start < p['context']:raise ValueError('Training origin cannot precede available context')
    if origin_start > p['context']:
        from torch.utils.data import Subset
        offset = origin_start-p['context']
        result['train'] = Subset(result['train'], range(offset, len(result['train'])))
    for split,dataset in result.items():
        if len(dataset)!=p['windows'][split]:raise ValueError(f'{name}/{split}: unexpected window count')
    return result

def batch(dataset,indices,device='cuda'):
    result=default_collate([dataset[int(i)] for i in indices])
    result={k:v.to(device) for k,v in result.items()}
    result['_index']=torch.tensor(indices,device=device,dtype=torch.long)
    return result

def structured(flat,variables,horizon=96):
    """Complete variable-major packing only; not a missing-axis compression adapter."""
    return flat.reshape(flat.shape[:-1]+(variables,horizon)).transpose(-1,-2)

def assert_complete_future(batch,expected):
    valid=batch['prediction_mask'][...,None]&batch['observed_mask']
    if not (valid.flatten(1).sum(-1)==expected).all():
        raise ValueError('Incomplete packed future: explicit original-position adapter required')
    return valid
