"""Explicit portable configurations. IO locations are supplied by the caller."""
import json
from pathlib import Path
from .configuration import write_config_set

RECIPES=['F00','F01','F02','F03','F04','F05','C01','C02','C03','A01','A02','A03','A04']

def make_configs(data_dir,weights,revision,parent_sft,work_dir,seed,recipe='F00'):
    if recipe not in RECIPES:raise ValueError('Unregistered DGF recipe')
    data=Path(data_dir).expanduser().resolve();work=Path(work_dir).expanduser().resolve();work.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((data/'data_protocol.json').read_text())
    if (protocol['dataset'],protocol['mode'],protocol['context'],protocol['patch'],protocol['ddof'])!=('ETTh1','M',5000,64,1) or protocol['horizon'] not in [192,336]:raise ValueError('Only registered ETTh1 protocols are supported')
    if protocol.get('variables')!=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'] or [protocol.get(k) for k in ['train_end','val_end','test_end']]!=[8640,11520,14400]:raise ValueError('Variable order or split differs')
    if 'diversity_scales' not in protocol:raise ValueError('Use prepare-dgf-data for frozen training-only statistics')
    full=recipe.startswith(('F','A'));route='D-post' if full or recipe=='C01' else 'A0'
    settings=dict(gamma=1.,lambda_dyn=.01,lambda_div=.001,dynamic_coefficients=[1.,1.,1.,1.],diversity_cap=1.,diversity_rho=1.,fixed_total_concentration=False,enable_dirichlet_objective=True,enable_dynamic=True,enable_diversity=True)
    lr=5e-7
    if recipe=='F01':lr*=.3
    if recipe=='F02':lr*=3
    if recipe=='F03':settings['gamma']*=.3
    if recipe=='F04':settings['lambda_dyn']*=.3
    if recipe=='F05':settings['lambda_div']*=.3
    if recipe=='A01':settings['fixed_total_concentration']=True
    if recipe=='A02':settings.update(enable_dirichlet_objective=False,gamma=0.)
    if recipe=='A03':settings.update(enable_dynamic=False,lambda_dyn=0.)
    if recipe=='A04':settings.update(enable_diversity=False,lambda_div=0.)
    common=dict(dataset='ETTh1',model='moirai-1.0-R-small',weights=str(Path(weights).expanduser().resolve()),revision=revision,local_files_only=True,
        route=route,protocol=protocol,artifact=str(data),seed=seed,recipe_id=recipe,scientific_path=f'{recipe}-H{protocol["horizon"]}-seed{seed}',
        global_input_batch=512,microbatch=1,J=8,G=100,K=8 if route=='D-post' else 1,updates_per_input_batch=1,
        optimizer='AdamW',weight_decay=.1,betas=[.9,.98],optimizer_eps=1e-6,gradient_norm_clip=1.,schedule='constant',
        precision='float32_tf32; float64 reward and metric accumulation',world_size=1,checkpoint_interval_batches=1,
        dgf_selection_only=True,record_probability_checkpoints=True,four_horizon_campaign=True,test_release=False,preparation='ordinary',checkpoint_kind='policy_state')
    prep=dict(common,phase='preparation',epochs=1,lr=1e-7,force_fixed_kappa=True,matched_dgf_baseline=True,
        arm='E03' if route=='D-post' else 'E01',algorithm_arm='E03' if route=='D-post' else 'E01',
        checkpoint=str(work/'construction/constructed.pt') if route=='D-post' else str(Path(parent_sft).expanduser().resolve()))
    nll=recipe in ['C01','C03'];arm='E03' if full else 'E00' if nll else 'E16'
    scientific=dict(common,phase='scientific',epochs=4,lr=lr,force_fixed_kappa=not full or settings['fixed_total_concentration'],matched_dgf_baseline=not full,
        arm=arm,algorithm_arm=arm,method_kind='ABLATION' if recipe.startswith('A') else 'FULL' if full else 'CONTROL',
        PPO_clip=None if full or nll else .2,reference_beta=0. if full or nll else .01,nll_anchor_coefficient=0.,
        checkpoint_from_receipt=dict(path=str(work/'preparation/result.json'),key=['readouts','final','checkpoint']))
    if full:scientific['dgf_complete']=settings
    construction=dict(weights=common['weights'],revision=revision,parent_sft=str(Path(parent_sft).expanduser().resolve()),output=str(work/'construction'),seed=seed,required=route=='D-post')
    write_config_set(work,{'construction.json':construction,'preparation.json':prep,'scientific.json':scientific})
    return dict(status='CONFIGURATIONS_WRITTEN_NOT_EXECUTED',recipe=recipe,route=route,construction_required=route=='D-post',files=[str(work/name) for name in ['construction.json','preparation.json','scientific.json']])
