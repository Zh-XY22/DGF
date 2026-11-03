"""Portable configuration assembly. Reads metadata only; no model/data execution."""
from .hdfs_checkpoints import checkpoint_exists
import json
import math
from pathlib import Path


def write_config_set(work,values):
    """Preflight the complete metadata set before any immutable file is created."""
    serialized={name:json.dumps(value,indent=2,allow_nan=False)+'\n' for name,value in values.items()}
    for name,value in values.items():
        path=work/name
        if path.is_file() and json.loads(path.read_text())!=value:
            raise ValueError('Existing configuration differs; use a new work directory: '+name)
    for name,text in serialized.items():
        path=work/name
        if not path.exists():
            temporary=path.with_suffix(path.suffix+'.partial');temporary.write_text(text);temporary.replace(path)


def resolve_checkpoint(config):
    c=dict(config)
    if 'checkpoint_from_receipt' in c:
        binding=c.pop('checkpoint_from_receipt');path=Path(binding['path']).expanduser().resolve()
        receipt=json.loads(path.read_text())
        if receipt.get('status') not in ['PASS','DIAGNOSTIC_ONLY']:
            raise ValueError('Upstream receipt is not successful')
        if receipt.get('status')=='DIAGNOSTIC_ONLY' and not c.get('qualification_only'):
            raise ValueError('A diagnostic start cannot be used as a formal experiment')
        value=receipt
        for key in binding['key']:value=value[key]
        if not isinstance(value,str) or not checkpoint_exists(value):raise ValueError('Selected checkpoint is unavailable')
        c.update(checkpoint=value,checkpoint_kind='policy_state',checkpoint_receipt=str(path),checkpoint_selection_key=binding['key'])
    return c


def make_configs(data_dir,work_dir,arm,model_id,revision,seed,auxiliary_metadata=None,recipe_path=None,
                 supervised_receipt=None,ordinary_preparation_receipt=None,microbatch=None):
    """Existing CLI stages, explicit public model revision, no cluster defaults."""
    data_dir=Path(data_dir).expanduser().resolve();work=Path(work_dir).expanduser().resolve();work.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((data_dir/'data_protocol.json').read_text());ds=protocol['dataset']
    solar=ds=='SolarAL_hourly_mean'
    if ds not in ['ETTh1','ETTm1','SolarAL_hourly_mean'] or protocol['horizon']!=96:raise ValueError('Unsupported registered H96 comparison')
    if solar and (len(protocol.get('variables',[]))!=137 or protocol.get('mode')!='M' or
                  (protocol.get('context'),protocol.get('patch'))!=(512,64)):
        raise ValueError('Solar requires the declared full137-variable M/context512/patch64 protocol')
    recipe=json.loads(Path(recipe_path).read_text()) if recipe_path else None
    if solar and ((recipe is None and arm not in ['E02','E03','E14']) or
                  (recipe is not None and recipe.get('components')!=['E14'])):
        raise ValueError('Conditional Solar scope is D-pre NLL/R and the frozen E14 diagnostic only')
    if recipe is None and arm not in [f'E{i:02d}' for i in range(18)]:raise ValueError('Unknown registered arm')
    if recipe is not None:
        from .selection import CANDIDATES,PREPARATION,REWARD,compatible
        components=recipe['components']
        if not 1<=len(components)<=2 or len(set(components))!=len(components) or any(a not in CANDIDATES for a in components):
            raise ValueError('Recipe must contain one or two registered candidate components')
        if len(components)==2 and not compatible(components):raise ValueError('Incompatible recipe components')
        expected_preparation=[a for a in components if a in PREPARATION]
        expected_algorithm=next((a for a in components if a in REWARD),'E03')
        if recipe['preparation_components']!=expected_preparation or recipe['algorithm_arm']!=expected_algorithm:
            raise ValueError('Recipe stages differ from the registered component meanings')
        if arm not in ['NEW1','NEW2','COMBO1','COMBO2','COMBO3','COMBO4']:
            raise ValueError('An explicit recipe requires a finite recipe slot')
    if not revision:raise ValueError('An explicit public weight revision is required')
    route='A0' if recipe is None and arm in ['E00','E01','E16'] else 'D-pre';micro=1 if solar else 16 if ds=='ETTh1' else 64
    if microbatch is not None:
        if not isinstance(microbatch,int) or isinstance(microbatch,bool) or microbatch<1 or microbatch>512:
            raise ValueError('Microbatch must be a positive integer no greater than independent input batch512')
        micro=microbatch
    common=dict(dataset=ds,model=Path(model_id).name,model_id=model_id,route=route,weights=model_id,revision=revision,local_files_only=False,protocol=protocol,
        artifact=str(data_dir),seed=seed,force_fixed_kappa=True,microbatch=micro,checkpoint_interval_batches=64,
        objective_version='extended-v1-measure-v1',optimizer_version='historical-AdamW-full-v1',test_release=False)
    supervised=dict(common,max_epochs=200,patience=3,global_input_batch=64 if ds=='ETTh1' or solar else 512,
        lr=1e-7 if route=='A0' or ds=='ETTh1' or solar else 5e-7,
        supervised_objective='official_PackedNLLLoss' if route=='A0' else 'legal_trajectory_mixture_NLL')
    prep=dict(common,phase='preparation',arm='E01' if route=='A0' else 'E03',scientific_arm=arm,preparation='ordinary',lr=5e-7,nll_anchor_coefficient=1.,
        checkpoint_from_receipt=dict(path=str(work/'supervised/result.json'),key=['best_checkpoint']))
    train=dict(common,phase='scientific',arm=arm,lr=5e-7,nll_anchor_coefficient=0. if arm in ['E16','E17'] else 1.,
        checkpoint_from_receipt=dict(path=str(work/'preparation/result.json'),key=['readouts','final','checkpoint']))
    supplied=json.loads(Path(auxiliary_metadata).read_text()) if auxiliary_metadata else {}
    if recipe is not None:
        for config in [prep,train]:config.update(recipe=recipe)
        train['algorithm_arm']=recipe['algorithm_arm']
        if recipe['preparation_components']:
            coefficients=supplied.get('preparation_coefficients',{})
            if any(a not in coefficients or not isinstance(coefficients[a],(int,float)) or not math.isfinite(coefficients[a]) or not 0<coefficients[a]<=.1 for a in recipe['preparation_components']):
                raise ValueError('Explicit frozen coefficients in (0, 0.1] required for every qualified preparation component')
            prep.update(preparation='composite',auxiliary_coefficient=1.,distillation_samples=8,
                preparation_components=[dict(arm=a,coefficient=coefficients[a]) for a in recipe['preparation_components']])
        for group,key in [({'E04'},'neighbor_artifact'),({'E06','E07','E08'},'teacher_config'),({'E09','E10','E15'},'selective_artifact')]:
            if set(recipe['components'])&group:
                if key not in supplied:raise ValueError('Qualified training-only auxiliary metadata required: '+key)
                if key!='selective_artifact' or set(recipe['preparation_components'])&{'E09','E10'}:
                    prep[key]=supplied[key]
        if recipe['algorithm_arm']=='E15':train['selective_artifact']=supplied['selective_artifact']
        if recipe['algorithm_arm']=='E14':
            if 'difference_coefficients' not in supplied:raise ValueError('Explicit frozen difference coefficients required')
            values=supplied['difference_coefficients']
            if set(values)!={'1','2'} or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in values.values()):
                raise ValueError('Both finite nonnegative difference coefficients are required')
            train['difference_coefficients']=supplied['difference_coefficients']
    if arm in [f'E{i:02d}' for i in range(4,12)]:
        required=['auxiliary_coefficient']+(['neighbor_artifact'] if arm in ['E04','E05'] else ['teacher_config'] if arm in ['E06','E07','E08'] else ['selective_artifact'])
        if any(k not in supplied for k in required):raise ValueError('Qualified training-only auxiliary metadata required: '+', '.join(required))
        coefficient=supplied['auxiliary_coefficient']
        if not isinstance(coefficient,(int,float)) or not math.isfinite(coefficient) or coefficient<0:
            raise ValueError('A finite nonnegative auxiliary coefficient is required')
        prep.update(preparation='auxiliary',preparation_arm=arm,distillation_samples=8,**{k:supplied[k] for k in required})
    if arm=='E14':train['difference_coefficients']={'1':.1,'2':.1}
    if arm=='E15':
        if 'selective_artifact' not in supplied:raise ValueError('Frozen training selective statistics required')
        train['selective_artifact']=supplied['selective_artifact']
    shared={}
    if supervised_receipt:
        from .auxiliary_configuration import checked_receipt
        selected,checkpoint=checked_receipt(supervised_receipt,protocol,'supervised',seed)
        cfg=selected['config']
        if any(cfg.get(k)!=common[k] for k in ['route','weights','revision','protocol']) or Path(cfg['artifact']).resolve()!=data_dir:
            raise ValueError('Shared supervised start differs from this model/data/route contract')
        path=str(Path(supervised_receipt).expanduser().resolve())
        prep['checkpoint_from_receipt']['path']=path;shared['supervised']=path
    if ordinary_preparation_receipt:
        if not supervised_receipt:raise ValueError('Shared ordinary preparation also requires its supervised receipt')
        if prep['preparation']!='ordinary':raise ValueError('Auxiliary preparation cannot be bypassed with an ordinary checkpoint')
        path=Path(ordinary_preparation_receipt).expanduser().resolve();r=json.loads(path.read_text());cfg=r['config']
        if r.get('status')!='PASS' or r.get('test_used') is not False or r.get('epochs')!=1:
            raise ValueError('A complete one-epoch ordinary preparation receipt is required')
        if cfg.get('phase')!='preparation' or cfg.get('preparation')!='ordinary' or cfg.get('checkpoint')!=checkpoint:
            raise ValueError('Ordinary preparation did not start from the supplied supervised checkpoint')
        if any(cfg.get(k)!=prep[k] for k in ['route','weights','revision','protocol','seed','lr','nll_anchor_coefficient']) or Path(cfg['artifact']).resolve()!=data_dir:
            raise ValueError('Shared preparation differs from the requested scientific contract')
        final=r.get('readouts',{}).get('final',{})
        if not final.get('reloaded') or not final.get('validation',{}).get('complete') or not checkpoint_exists(final['checkpoint']):
            raise ValueError('Shared preparation has no complete reloaded final readout')
        train['checkpoint_from_receipt']['path']=str(path);shared['preparation']=str(path)
    validation=dict(common,splits=['val'],checkpoint_from_receipt=dict(path=str(work/'training/result.json'),key=['readouts','val_MSE_best','checkpoint']))
    configs={'supervised':supervised,'preparation':prep,'training':train,'validation':validation}
    commands=[]
    for stage,command in [('supervised','fit-supervised'),('preparation','train'),('training','train'),('validation','evaluate')]:
        if stage in shared:continue
        commands.append(['moirai-research',command,'--config',str(work/(stage+'.json')),'--output',str(work/stage),'--device','cuda','--seed',str(seed)])
    values={name+'.json':value for name,value in configs.items()}
    values['commands.json']=dict(status='GENERATED_NOT_EXECUTED',commands=commands,
        shared_receipts=shared,formal_test_included=False,limits='Four full RL epochs; global input batch512; two updates. Supervised <=200 epochs with patience3.')
    write_config_set(work,values)
    return dict(status='GENERATED_NOT_EXECUTED',configs=list(configs),commands=commands,shared_receipts=shared)
