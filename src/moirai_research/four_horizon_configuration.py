"""Materialize a frozen public stage template using only caller-owned IO."""
from pathlib import Path
import copy,json,shlex
from .configuration import write_config_set

def make_configs(template_path,data_dir,work_dir,weights,seed):
    template=json.loads(Path(template_path).read_text());data=Path(data_dir).expanduser().resolve()
    work=Path(work_dir).expanduser().resolve();work.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((data/'data_protocol.json').read_text())
    if template.get('version')!=1 or template['dataset']!='ETTh1':raise ValueError('Unsupported public study template')
    if protocol['dataset']!='ETTh1' or protocol['horizon'] not in [96,192,336,720]:raise ValueError('Only the four registered ETTh1 horizons are supported')
    contract=template['data_contract']
    for key,value in contract.items():
        if protocol.get(key)!=value:raise ValueError('Public data protocol differs from the frozen template: '+key)
    if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:raise ValueError('An explicit nonnegative train seed is required')
    files={};commands=[];stages=[];names={s['name'] for s in template['stages']}
    if len(names)!=len(template['stages']):raise ValueError('Duplicate stage in public template')
    for stage in template['stages']:
        name=stage['name'];config=copy.deepcopy(stage['configuration'])
        config.update(weights=str(Path(weights).expanduser().resolve()),artifact=str(data),protocol=protocol,seed=seed,
            scientific_path=f'public-{template["recipe_id"]}-H{protocol["horizon"]}-seed{seed}-{name}',
            local_files_only=True,test_release=False)
        if stage.get('parent'):
            if stage['parent'] not in [s['name'] for s in stages]:raise ValueError('Stage parent must precede its child')
            config['checkpoint_from_receipt']=dict(path=str(work/stage['parent']/'result.json'),key=['readouts',stage['parent_selection'],'checkpoint'])
        files[name+'.json']=config
        command=['python','-m','moirai_research.cli',stage['entrypoint'],'--config',str(work/(name+'.json')),
            '--output',str(work/name),'--device','cuda','--seed',str(seed)]
        commands.append(shlex.join(command));stages.append(dict(name=name,entrypoint=stage['entrypoint'],configuration=str(work/(name+'.json')),output=str(work/name),command=command))
    files['stage_manifest.json']=dict(status='CONFIGURATIONS_ONLY_NOT_EXECUTED',recipe_id=template['recipe_id'],seed=seed,
        horizon=protocol['horizon'],stages=stages,full_budget=True,test_used=False,
        interpretation='Frozen scientific settings; paths are caller-supplied. Creating configurations does not execute training or establish numerical reproduction.')
    write_config_set(work,files)
    script=work/'run_stages.sh';text='#!/usr/bin/env bash\nset -euo pipefail\n'+'\n'.join(commands)+'\n'
    if script.exists() and script.read_text()!=text:raise ValueError('Existing execution script differs; use a new output directory')
    if not script.exists():script.write_text(text)
    return files['stage_manifest.json']
