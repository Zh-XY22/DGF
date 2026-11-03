"""Portable entry configuration checks, independent of the training backend."""
from pathlib import Path
import json,hashlib,math,re

ALLOWED={'lr','G','adaptive_dgf','dgf_complete','recipe_id','retain_all_validation_checkpoints','epochs','registered_budget_extension'}

def checked_overrides(value):
    def positive(x):return not isinstance(x,bool) and isinstance(x,(float,int)) and math.isfinite(x) and x>0
    if not isinstance(value,dict) or set(value)-ALLOWED:raise ValueError('Overrides may only change declared scientific treatment fields')
    if 'epochs' in value or 'registered_budget_extension' in value:
        extension=value.get('registered_budget_extension',{})
        if type(value.get('epochs')) is not int or type(extension.get('prefix_epochs')) is not int or not 4<=extension['prefix_epochs']<extension.get('target_epochs',0)==value['epochs']:
            raise ValueError('Public duration treatment requires its exact registered budget')
    if 'lr' in value and (isinstance(value['lr'],bool) or not isinstance(value['lr'],(int,float)) or not math.isfinite(value['lr']) or value['lr']<=0):raise ValueError('Invalid positive LR')
    if 'G' in value and (type(value['G']) is not int or value['G']<2):raise ValueError('Complete hierarchical training requires at least two trajectories')
    if 'dgf_complete' in value:
        full=value['dgf_complete']
        if full.get('fixed_total_concentration') is not False or not all(full.get(k) is True for k in ['enable_dirichlet_objective','enable_dynamic','enable_diversity']):raise ValueError('Public performance overrides must retain all core features')
        if any(not positive(full.get(k)) for k in ['gamma','lambda_dyn','lambda_div','diversity_cap','diversity_rho']):raise ValueError('Core weights and bounds must stay finite and positive')
        if len(full.get('dynamic_coefficients',[]))!=4 or not all(positive(v) for v in full['dynamic_coefficients']):raise ValueError('All four dynamic components must remain active')
    if 'recipe_id' in value and not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]{0,100}',value['recipe_id']):raise ValueError('Invalid public recipe ID')
    if 'adaptive_dgf' in value:
        if 'lr' not in value:raise ValueError('Adaptive overrides must declare the exact initial LR')
        from .adaptive_dgf import learning_rate
        options=value['adaptive_dgf']
        if set(options)-{'learning_rate','component_gradient_steps','calibrate_NLL_gradient','nll_coefficient','nll_start_step'}:raise ValueError('Unknown adaptive field; no silently inactive treatment')
        coefficient=options.get('nll_coefficient',0.)
        if isinstance(coefficient,bool) or not isinstance(coefficient,(float,int)) or not math.isfinite(coefficient) or coefficient<0:raise ValueError('Invalid explicit NLL extension')
        total=7*value.get('epochs',4)
        for step in range(total):learning_rate(dict(lr=value.get('lr',5e-7),adaptive_dgf=options),step,total)
        if not 0<=options.get('nll_start_step',0)<total:raise ValueError('NLL transition outside the full budget')
    return dict(value)

def content_tag(path):
    state=hashlib.blake2b()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):state.update(block)
    return state.hexdigest()

def input_binding(csv,weights,source,settings):
    weights=Path(weights);source=Path(source)
    model=[f for f in sorted(weights.rglob('*')) if f.is_file() and f.suffix in ['.json','.safetensors']]
    if not model:raise ValueError('Public base weights are missing')
    return dict(settings=settings,csv=content_tag(csv),model={str(f.relative_to(weights)):content_tag(f) for f in model},implementation={str(f.relative_to(source)):content_tag(f) for f in sorted(source.rglob('*.py')) if '__pycache__' not in f.parts})
