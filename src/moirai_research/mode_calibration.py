"""One common training-only mode-specialization coefficient and real-entry check."""
from pathlib import Path
import json,sys,subprocess,gc
import torch
from .runtime import atomic_json,load_policy,seed_all
from .public_data import prepare_csv
from .data import datasets,batch
from .policy import joints
from .probability import nll
from .mode_specialization import ModeSpecialization
from .operator_calibration import gradient_norm

def calibrate_modes(model_root,csv_file,parent,labels_root,output,revision):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);scratch=out.parent.parent/'mode_calibration_work';scratch.mkdir(exist_ok=True)
    rows=[];configs={}
    for h in [96,192,336,720]:
        artifact=scratch/f'data_H{h}';protocol=prepare_csv(str(csv_file),'ETTh1',artifact,horizon=h,source_revision=revision)
        config=dict(dataset='ETTh1',protocol=protocol,artifact=str(artifact),weights=str(model_root),route='D-pre',checkpoint=str(parent),
            checkpoint_kind='policy_state',local_files_only=True,force_fixed_kappa=True,seed=198419,lr=5e-7,microbatch=1,global_input_batch=2,
            qualification_only=True,diagnostic_input_limit=2,diagnostic_validation_limit=2,four_horizon_campaign=True,warmup_campaign=True,
            record_probability_checkpoints=True,checkpoint_interval_batches=1,test_release=False,K=8,kappa=8.,J=8,G=4,
            phase='preparation',preparation='ordinary',preparation_arm='D01',arm='E17',algorithm_arm='E17',epochs=1,updates_per_input_batch=2,
            mode_labels_artifact=str(Path(labels_root)/f'H{h}'),recipe_id='D01')
        seed_all(198419);policy=load_policy(config,'cuda').eval();b=batch(datasets(policy,'ETTh1',protocol)['train'],[0,997],'cuda')
        plugin=ModeSpecialization(config,'cuda');experts,target,w,_,_=joints(policy(b),b)
        likelihood=nll(experts,w,target);auxiliary,diag=plugin.auxiliary(b,experts,target,w)
        base_norm=gradient_norm(likelihood,policy,retain_graph=True);aux_norm=gradient_norm(auxiliary,policy)
        assert base_norm>0 and aux_norm>0
        rows.append(dict(horizon=h,NLL_gradient_norm=base_norm,auxiliary_gradient_norm=aux_norm,diagnostic=diag))
        configs[h]=config;del policy,experts,plugin;gc.collect();torch.cuda.empty_cache()
    candidates=[.01,.03,.1];errors={str(c):sum(abs(c*r['auxiliary_gradient_norm']/r['NLL_gradient_norm']-.1) for r in rows)/4 for c in candidates}
    coefficient=min(candidates,key=lambda c:(errors[str(c)],c));cases=[]
    atomic_json(out/'coefficient_freeze.json',dict(candidates=candidates,errors=errors,auxiliary_coefficient=coefficient,training_only=True))
    for h,config in configs.items():
        config['auxiliary_coefficient']=coefficient;file=out/f'D01_H{h}.json';atomic_json(file,config);destination=scratch/f'D01_H{h}'
        command=[sys.executable,'-m','moirai_research.cli','train','--config',str(file),'--output',str(destination),'--device','cuda','--seed','198419']
        run=subprocess.run(command,capture_output=True,text=True,timeout=1800)
        if run.returncode:
            (out/f'D01_H{h}_error.txt').write_text((run.stdout+'\n'+run.stderr)[-14000:]);raise RuntimeError('Mode-specialization installed entry failed')
        receipt=json.loads((destination/'result.json').read_text());assert receipt['cursor']['global_step']==2
        assert receipt['first_step']['relative_parameter_change']>0 and receipt['preparation_alignment']['maximum_target_difference']<=1e-6
        diagnostics=[json.loads(v) for v in (destination/'micro_diagnostics.jsonl').read_text().splitlines()]
        assert all(abs(x['auxiliary'])>0 for v in diagnostics for x in v['diagnostics'])
        cases.append(dict(horizon=h,optimizer_updates=2,first_step=receipt['first_step'],validation=receipt['readouts']['final']['validation'],
            command=command,preparation_alignment=receipt['preparation_alignment'],treatment_evidence=diagnostics))
    result=dict(status='PASS',qualified_recipes=['D01'],calibration=dict(auxiliary_coefficient=coefficient),candidates=candidates,errors=errors,rows=rows,cases=cases,
        optimizer_updates=8,optimizer_trajectories=4,validation_used_for_calibration=False,test_used=False,
        scope='Training-only common coefficient, then installed preparation/reload check; combined prototype/router/expert treatment, not an isolated mechanism')
    atomic_json(out/'result.json',result);return result
