"""Compare uninterrupted training with an explicit optimizer/RNG extension."""
import json,shutil,subprocess,sys
from pathlib import Path
import torch
from .runtime import atomic_json,seed_all
from .io_utils import load_checkpoint
from .resume_extension import extend_resume


def qualify_extension(model_root,csv_file,output,revision):
    from .public_data import prepare_csv
    from .policy import Policy
    from .train import weights_checkpoint
    out=Path(output);out.mkdir(parents=True,exist_ok=True);scratch=out.parent.parent/'extension_work';scratch.mkdir(exist_ok=True);cases=[]
    def copy(source,destination):
        destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    for h in [96,192,336,720]:
        data=scratch/f'data_H{h}';protocol=prepare_csv(str(csv_file),'ETTh1',data,horizon=h,source_revision=revision)
        seed_all(198383);policy=Policy(str(model_root),'A0',local_files_only=True);parent=scratch/f'initial_H{h}.pt'
        weights_checkpoint(parent,policy,dict(route='A0'),dict(global_step=0));del policy
        base=dict(dataset='ETTh1',protocol=protocol,artifact=str(data),weights=str(model_root),route='A0',checkpoint=str(parent),checkpoint_kind='policy_state',
            local_files_only=True,force_fixed_kappa=True,seed=198383,phase='scientific',arm='E16',algorithm_arm='E16',lr=5e-7,
            global_input_batch=2,microbatch=1,updates_per_input_batch=2,nll_anchor_coefficient=0.,
            qualification_only=True,diagnostic_input_limit=2,diagnostic_validation_limit=2,warmup_campaign=True,four_horizon_campaign=True,
            record_probability_checkpoints=True,checkpoint_interval_batches=1,test_release=False)
        def run(config,name):
            file=out/f'{name}_H{h}.json';atomic_json(file,config);directory=scratch/f'{name}_H{h}'
            command=[sys.executable,'-m','moirai_research.cli','train','--config',str(file),'--output',str(directory),'--device','cuda','--seed','198383']
            result=subprocess.run(command,capture_output=True,text=True,timeout=900)
            if result.returncode:
                (out/f'{name}_H{h}_error.txt').write_text((result.stdout+'\n'+result.stderr)[-10000:]);raise RuntimeError('Extension gate failed: '+name)
            return directory,json.loads((directory/'result.json').read_text()),command
        prefix,pr,_=run(dict(base,recipe_id='B02',scientific_path='gate-prefix',epochs=1),'prefix')
        target=dict(base,recipe_id='T01',scientific_path='gate-extension',epochs=2,fixed_epoch_checkpoints=[1,2])
        directory=scratch/f'extended_H{h}';mapping={v['checkpoint']:v['checkpoint'] for v in pr['readouts'].values()}
        resume,provenance=extend_resume(prefix/'resume.pt',target,mapping,copy,directory)
        extended,er,command=run(dict(target,resume=resume),'extended')
        uninterrupted,ur,_=run(dict(target,scientific_path='gate-uninterrupted'),'uninterrupted')
        final=load_checkpoint(extended/'checkpoints/final.pt',map_location='cpu',weights_only=False)['policy_state']
        golden=load_checkpoint(uninterrupted/'checkpoints/final.pt',map_location='cpu',weights_only=False)['policy_state']
        maximum=max(float((final[k].double()-golden[k].double()).abs().max()) for k in final)
        assert maximum<1e-6 and er['optimizer_steps_this_attempt']==2 and er['cursor']['global_step']==4
        assert er['first_step_this_attempt']['relative_parameter_change']>0
        assert {'budget_epoch_1','budget_epoch_2'}<=set(er['readouts'])
        assert er['readouts']['budget_epoch_2']['checkpoint']==er['readouts']['final']['checkpoint']
        cases.append(dict(horizon=h,status='PASS',optimizer_updates=8,optimizer_trajectories=3,
            maximum_parameter_difference=maximum,prefix_updates=2,extension_new_updates=2,uninterrupted_updates=4,
            provenance=provenance,installed_command=command,budget_checkpoints_retained=True))
        atomic_json(out/'progress.json',dict(cases=cases))
    result=dict(status='PASS',qualified_recipes=['T01'],optimizer_updates=32,optimizer_trajectories=12,cases=cases,test_used=False,
        scope='Four-H two-window entry proof of optimizer/RNG/cursor extension and fixed-budget aliases; not full 16-epoch performance reproduction')
    atomic_json(out/'result.json',result);return result
