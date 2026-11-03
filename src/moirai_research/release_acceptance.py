"""Final local-release qualification in fresh environments, inside one GPU job.

Each dataset owns two tiny optimizer trajectories. Full formal reproduction
reloads the separate preselected research checkpoint; it does not evaluate the
tiny qualification model as a scientific result.
"""
import copy,json,math,os,subprocess,time
from pathlib import Path
from .clean_install import run as clean_install
from .runtime import atomic_json


def run(config_path,output,device='cuda',seed=1):
    config=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    assert config['exemplar_seed']==17 and seed==1
    if config.get('scope')=='conditional_solar':
        assert {c['dataset'] for c in config['cases']}=={'SolarAL_hourly_mean'} and len(config['cases'])==1
    else:assert {c['dataset'] for c in config['cases']}=={'ETTh1','ETTm1'} and len(config['cases'])==2
    began=time.time();records=[]
    for case in config['cases']:
        ds=case['dataset'];directory=out/ds;directory.mkdir(exist_ok=True)
        qualification=case['qualification'];assert qualification['dataset']==ds
        qpath=directory/'qualification.json';atomic_json(qpath,qualification)
        installed=clean_install(qpath,directory/'clean',device,seed)
        assert installed['status']=='PASS' and installed['optimizer_trajectories']==2 and not installed['system_site_packages']
        entry=json.loads(Path(installed['e2e_receipt']).read_text());environment=Path(entry['environment']['prefix'])
        assert entry['public_data_matches_reference'] and entry['exact_reload']
        reference=json.loads(Path(case['reference_receipt']).read_text());assert reference['status']=='PASS'
        expected=reference['results']['test'];original=reference['config']
        assert expected['complete'] and expected['seed']==17 and expected['num_samples']==100 and expected['independent_marginal_samples']
        assert original['checkpoint']==case['checkpoint'] and original['seed']==17
        public_data=directory/'clean/e2e/data';protocol=json.loads((public_data/'data_protocol.json').read_text())
        for key in ['dataset','mode','context','patch','horizon','variables','train_end','val_end','test_end','ddof','windows']:
            assert protocol[key]==original['protocol'][key], 'Public data contract differs: '+key
        evaluation=copy.deepcopy(original)
        evaluation.update(artifact=str(public_data.resolve()),protocol=protocol,weights=qualification['public_model_cache'],
            revision=qualification['model_revision'],local_files_only=True,splits=['test'])
        evaluation.pop('diagnostic_limit',None)
        # Carry the already frozen choices as explicit local files, not a switch
        # that permits arbitrary test selection in the new environment.
        for field in ['test_release_receipt','readout_selection_receipt']:
            frozen=json.loads(Path(original[field]).read_text())
            assert frozen['status']=='PASS' and not frozen['test_used']
            destination=directory/(field+'.json');atomic_json(destination,frozen);evaluation[field]=str(destination.resolve())
        epath=directory/'full_evaluation.json';atomic_json(epath,evaluation)
        env=os.environ.copy()
        for key in ['PYTHONPATH','PYTHONHOME','PIP_EXTRA_INDEX_URL','PIP_INDEX_URL','PIP_TRUSTED_HOST']:env.pop(key,None)
        env.update(VIRTUAL_ENV=str(environment),PATH=str(environment/'bin')+os.pathsep+env['PATH'],PYTHONNOUSERSITE='1',
            PIP_CONFIG_FILE=os.devnull,HF_HUB_DISABLE_TELEMETRY='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        bindings=case['public_metadata_bindings'];student=json.loads(Path(bindings['supervised_receipt']).read_text())
        data_dir=student['config']['artifact'];metadata_checks=[]
        calls=[] if ds=='SolarAL_hourly_mean' else [('selective',['make-auxiliary-configs','--data-dir',data_dir,'--work-dir',str(directory/'metadata-selective'),
            '--stage','selective','--coefficient','.1','--seed','17','--supervised-receipt',bindings['supervised_receipt'],
            '--estimator-receipt',bindings['estimator_receipt']]),
            ('distillation',['make-auxiliary-configs','--data-dir',data_dir,'--work-dir',str(directory/'metadata-teacher'),
            '--stage','distillation','--coefficient','.01','--seed','17','--supervised-receipt',bindings['supervised_receipt'],
            '--teacher-receipt',bindings['teacher_receipt']])]
        calls += [('shared-start',['make-run-configs','--data-dir',data_dir,'--work-dir',str(directory/'metadata-shared'),
            '--arm','E03' if ds=='SolarAL_hourly_mean' else 'E13','--model-id',student['config']['weights'],'--revision',student['config']['revision'],'--seed','17',
            '--microbatch',str(student['config']['microbatch']),
            '--supervised-receipt',bindings['supervised_receipt'],'--ordinary-preparation-receipt',bindings['ordinary_preparation_receipt']]),
            ('readouts',['make-readout-configs','--training-receipt',bindings['readout_training_receipt'],
            '--preparation-receipt',bindings['readout_preparation_receipt'],'--selection-receipt',bindings['selection_receipt'],
            '--scientific-path',bindings['scientific_path'],'--work-dir',str(directory/'metadata-readouts')])]
        for name,arguments in calls:
            with (directory/('metadata-'+name+'.log')).open('w') as stream:
                subprocess.run([str(environment/'bin/python'),'-m','moirai_research.cli',*arguments],env=env,cwd=directory,
                    stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=120)
            metadata_checks.append(dict(interface=name,status='CONFIGURATION_ASSEMBLY_ONLY',optimizer_trajectories=0))
        assembled=json.loads((directory/'metadata-readouts/readout_selection.json').read_text())
        prior_selection=json.loads(Path(original['readout_selection_receipt']).read_text())
        assert assembled['selected']==prior_selection['selected']
        destination=directory/'formal_reproduction'
        command=[str(environment/'bin/python'),'-m','moirai_research.cli','evaluate','--config',str(epath.resolve()),
            '--output',str(destination.resolve()),'--device',device,'--seed','17']
        with (directory/'formal_reproduction.log').open('w') as stream:
            subprocess.run(command,env=env,cwd=directory,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=14400)
        reproduced=json.loads((destination/'result.json').read_text());actual=reproduced['results']['test']
        assert reproduced['status']=='PASS' and actual['complete'] and actual['windows']==expected['windows']
        assert actual['seed']==17 and actual['num_samples']==100 and actual['units']==expected['units']
        assert actual['independent_marginal_samples']
        comparisons=[]
        def compare(left,right,group):
            for name,old in left.items():
                if name=='seconds' or not isinstance(old,(int,float)) or isinstance(old,bool):continue
                value=right[name]
                if not math.isfinite(old) or not math.isfinite(value):raise ValueError('Nonfinite formal reproduction metric '+name)
                tolerance=max(1e-5,abs(old)*1e-4)
                comparisons.append(dict(group=group,metric=name,reference=old,reproduced=value,delta=value-old,
                    tolerance=tolerance,within_tolerance=abs(value-old)<=tolerance))
        compare(expected,actual,'ALL')
        assert set(expected['per_variable'])==set(actual['per_variable'])
        for variable,values in expected['per_variable'].items():compare(values,actual['per_variable'][variable],variable)
        record=dict(dataset=ds,checkpoint=case['checkpoint'],exemplar_seed=17,method=case['method'],
            reference_receipt=case['reference_receipt'],reproduction_receipt=str(destination/'result.json'),
            installation_receipt=str(directory/'clean/result.json'),comparisons=comparisons,
            public_metadata_interface_checks=metadata_checks,public_metadata_scope=bindings['scope'],
            actual_gpu=reproduced['actual_gpu'],optimizer_trajectories=2,
            meaning='The interface uses seed1 tiny training; scientific reproduction reloads the independent, validation-frozen seed17 research checkpoint')
        records.append(record);atomic_json(out/'progress.json',records)
        if not all(c['within_tolerance'] for c in comparisons):
            raise ValueError('Formal public-entry reproduction differs beyond the predeclared absolute/relative tolerance; preserve evidence and diagnose without changing the threshold')
    result=dict(status='PASS',scope='Final frozen source installed in fresh environments, public CSV equivalence, tiny SFT/RL interfaces and full selected-checkpoint test reproduction',
        cases=records,optimizer_trajectories=2*len(records),actual_gpu=records[0]['actual_gpu'],seconds=time.time()-began,
        test_selection=False,exemplar_seed=17,code_release_uploaded=False,
        limitations=['This verifies a local release candidate, not a public upload','Tiny qualification training is not the full recommended method training',
            'Formal scores belong to the separately reloaded research checkpoints','A passed reproduction is not an improvement verdict'])
    atomic_json(out/'result.json',result);return result
