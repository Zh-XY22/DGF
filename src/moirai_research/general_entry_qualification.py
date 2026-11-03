"""Small installed-CLI integration runs; never scientific result substitutes."""
import json
from pathlib import Path
import subprocess
import sys
from .runtime import atomic_json,seed_all


def qualify_general(model_root,csv_file,output,revision,group,only_method=None):
    from .public_data import prepare_csv
    from .policy import Policy
    from .train import weights_checkpoint
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    scratch=out.parent.parent/('entry_'+group);scratch.mkdir(exist_ok=True)
    assert not scratch.is_relative_to(out.parent)
    if group=='direct':
        methods={'B04':dict(direct_objective='MAP_MAE'),'B05':dict(direct_objective='MAP_MSE')}
    elif group=='continuous':
        methods={f'R{i:02d}':dict(continuous_recipe=f'R{i:02d}') for i in range(1,14)}
    elif group=='composed':
        methods={'X01_R02_R03':dict(continuous_recipe='R02',clipping_recipe='R03'),
            'X01_R07_R04':dict(continuous_recipe='R07',clipping_recipe='R04')}
    elif group=='point-setting-composition':
        methods={'X03_R02_LR3':dict(continuous_recipe='R02',lr=1.5e-6)}
    elif group=='local-credit-regeneration':
        methods={'R13_WRITE':dict(continuous_recipe='R13',paired_action_role='WRITE'),
            'R13_REGENERATE':dict(continuous_recipe='R13',paired_action_role='REGENERATE')}
    elif group=='point-probability-mix':
        mixture=dict(probability_objective='fair_full_CRPS',coefficient=.01,crps_per_component=4,measure_component_gradients=True)
        methods={'X06_MAP_MSE':dict(arm='E00',algorithm_arm='E00',direct_objective='MAP_MSE',point_probability_mix=mixture),
            'X06_R02':dict(continuous_recipe='R02',point_probability_mix=mixture)}
    elif group=='budgets':
        methods={'B06':dict(epochs=2,updates_by_epoch=[1,2]),'B07':{},
            'T02':dict(arm='E00',algorithm_arm='E00',epochs=2,updates_by_epoch=[1,2]),'T05':dict(lr=1.5e-7)}
    elif group=='contexts':
        methods={'T03':dict(context_override=512),'T04':dict(context_override=2048)}
    elif group=='SAM':
        common=dict(phase='preparation',preparation='ordinary',preparation_arm='S06',warmup_arm='S02',auxiliary_coefficient=.1,repair_version='C02-S02-S06-TIME-v2-v1')
        methods={'T07':dict(common,warmup_optimizer='SAM',sam_rho=.01),'T08':dict(common)}
    elif group=='NLLCRPS':
        methods={'P03':dict(arm='E00',algorithm_arm='E00',direct_objective='fair_full_CRPS_NLL',probability_NLL_coefficient=.01)}
    elif group=='transports':
        methods={'P04':dict(arm='E00',algorithm_arm='E00',transport_mode='P04_LOCATION',direct_objective='frozen_residual_location_MSE'),
            'P05':dict(arm='E00',algorithm_arm='E00',transport_mode='P05_SCALE',direct_objective='frozen_mean_scale_CRPS',lr=.01)}
    else:raise ValueError('Unknown integration gate')
    if only_method is not None:
        if only_method not in methods:raise ValueError('Unregistered method in this gate')
        methods={only_method:methods[only_method]}
    cases=[]
    for h in [96,192,336,720]:
        protocol=prepare_csv(str(csv_file),'ETTh1',scratch/f'data_H{h}',horizon=h,source_revision=revision)
        seed_all(198347);policy=Policy(str(model_root),'A0',local_files_only=True)
        parent=scratch/f'initial_H{h}.pt';weights_checkpoint(parent,policy,dict(seed=198347,route='A0'),dict(global_step=0));del policy
        regeneration_reference=None
        for method,change in methods.items():
            data_directory=scratch/f'data_H{h}'
            active_protocol=protocol
            if group=='contexts':
                data_directory=out/f'data_C{change["context_override"]}_H{h}'
                active_protocol=prepare_csv(str(csv_file),'ETTh1',data_directory,horizon=h,source_revision=revision,
                    context_override=change['context_override'],train_origin_start=5000)
            config=dict(dataset='ETTh1',protocol=protocol,artifact=str(scratch/f'data_H{h}'),weights=str(model_root),route='A0',
                checkpoint=str(parent),checkpoint_kind='policy_state',local_files_only=True,force_fixed_kappa=True,
                phase='scientific',arm='E00' if group=='direct' else 'E16',algorithm_arm='E00' if group=='direct' else 'E16',
                recipe_id=method,seed=198347,lr=5e-7,global_input_batch=2,microbatch=1,epochs=1,updates_per_input_batch=2,
                qualification_only=True,diagnostic_input_limit=2,diagnostic_validation_limit=2,warmup_campaign=True,
                four_horizon_campaign=True,record_probability_checkpoints=True,checkpoint_interval_batches=1,test_release=False,
                J=8,G=8 if method=='R09' else 4,nll_anchor_coefficient=0.,frozen_reward_scale=1.,
                clip_low=.2,clip_high=.28,cispo_upper=1.2,geometric_clip=.005,geometric_block_length=32,
                difference_reward_first=.1,difference_reward_second=.1,explicit_reference_KL_coefficient=.01,
                annealed_NLL_initial=.01)
            config.update(change)
            if group=='local-credit-regeneration' and method=='R13_WRITE':config['paired_action_cache']=str(scratch/f'common_actions_H{h}')
            config.update(protocol=active_protocol,artifact=str(data_directory))
            config.pop('context_override',None)
            file=out/f'{method}_H{h}.json';atomic_json(file,config);directory=scratch/f'{method}_H{h}'
            command=[sys.executable,'-m','moirai_research.cli','train','--config',str(file),'--output',str(directory),'--device','cuda','--seed','198347']
            result=subprocess.run(command,capture_output=True,text=True,timeout=900)
            if result.returncode:
                (out/f'{method}_H{h}_error.txt').write_text((result.stdout+'\n'+result.stderr)[-10000:])
                raise RuntimeError('Installed '+group+' entry failed: '+method+'/H'+str(h))
            receipt=json.loads((directory/'result.json').read_text())
            atomic_json(out/f'{method}_H{h}_observed_training.json',receipt)
            expected=sum(config.get('updates_by_epoch',[config['updates_per_input_batch']]*config['epochs']))
            assert receipt['status']=='DIAGNOSTIC_ONLY' and receipt['cursor']['global_step']==expected
            assert receipt['first_step']['relative_parameter_change']>0
            if group=='SAM' and method=='T07':assert receipt['first_step']['SAM_gradient_evaluations_per_optimizer_update']==2
            if group=='SAM' and method=='T08':assert 'SAM_gradient_evaluations_per_optimizer_update' not in receipt['first_step']
            transport_proof=None
            if group=='transports':
                from .runtime import load_policy
                from .policy import joints
                from .data import datasets,batch
                import torch
                selected=load_policy(dict(config,checkpoint=receipt['readouts']['final']['checkpoint']),'cuda').eval()
                b=batch(datasets(selected,'ETTh1',active_protocol)['train'],[0,1],'cuda')
                with torch.no_grad():
                    experts,_,_,_,_=joints(selected(b),b);joint=experts[0];ref=joint.reference
                    if method=='P04':
                        assert torch.equal(joint.scale,ref.scale) and joint.mixture is ref.mixture
                        change=float((joint.mean-ref.mean).abs().max());assert change>0
                        transport_proof=dict(frozen_residual_family_and_scale=True,maximum_location_change=change)
                    else:
                        assert receipt['first_step']['trainable_parameters']==1 and torch.equal(joint.mean,ref.mean)
                        factor=float(selected.log_scale.exp());assert factor!=1 and abs(float(selected.log_scale))<=.03
                        assert abs(receipt['readouts']['initial']['validation']['MAP_MSE']-receipt['readouts']['final']['validation']['MAP_MSE'])<1e-10
                        transport_proof=dict(reference_mean_exactly_fixed=True,positive_scale_factor=factor,trainable_parameters=1)
                    rng=torch.cuda.get_rng_state();base_samples=ref.sample(8)['forecast'];torch.cuda.set_rng_state(rng)
                    samples=joint.sample(8)['forecast']
                    expected_samples=base_samples+(joint.mean-ref.mean) if method=='P04' else ref.mean+selected.log_scale.exp()*(base_samples-ref.mean)
                    torch.testing.assert_close(samples,expected_samples,atol=2e-6,rtol=2e-6)
                assert all(not p.requires_grad and p.grad is None for p in selected.reference.parameters())
                del selected;torch.cuda.empty_cache()
            final=receipt['readouts']['final'];assert final['reloaded'] and final['validation']['windows']==2
            diagnostics=[json.loads(line) for line in (directory/'micro_diagnostics.jsonl').read_text().splitlines()]
            assert len(diagnostics)==expected
            regeneration_proof=None
            if group=='local-credit-regeneration':
                import torch
                state=torch.load(final['checkpoint'],map_location='cpu',weights_only=False)
                # Compare the exact policy state, not rounded output metrics.
                parameters=state['policy_state']
                observed=[d for row in diagnostics for d in row['diagnostics']]
                seed_stream=[d['paired_action_seed'] for d in observed]
                assert observed and all(isinstance(v,int) for v in seed_stream)
                if method=='R13_WRITE':regeneration_reference=(parameters,seed_stream)
                else:
                    before,expected_seeds=regeneration_reference
                    assert set(before)==set(parameters) and expected_seeds==seed_stream
                    maximum=max(float((before[k]-parameters[k]).abs().max()) for k in before if torch.is_floating_point(before[k]))
                    for key in before:torch.testing.assert_close(before[key],parameters[key],rtol=1e-6,atol=1e-7)
                    regeneration_proof=dict(policy_state_maximum_difference=maximum,identical_action_seed_stream=True,optimizer_updates=expected,scope='Same reference, seed, input and full update budget; R13 cache WRITE versus deterministic regeneration')
            if group=='point-probability-mix':
                observed=[d['point_probability_mix'] for row in diagnostics for d in row['diagnostics']]
                assert observed and all(x['coefficient']==.01 and abs(x['probability_loss'])>0 and x['base_gradient_norm']>0 and x['probability_gradient_norm']>0 and x['fresh_current_policy_probability_term'] and not x['strictly_proper_objective'] for x in observed)
            if group=='composed':
                observed=[d for update in diagnostics for d in update['diagnostics']]
                assert observed and all(d['advantage_recipe']==change['continuous_recipe'] and d['clipping_recipe']==change['clipping_recipe'] for d in observed)
                if change['continuous_recipe']=='R02':assert all(d['global_complete_batch_scale']>0 for d in observed)
                # Reject cross-granularity/optimizer stacking rather than silently ignore it.
                from .four_horizon_rl import clipping_operator
                for bad in ['R05','R06','R10']:
                    try:clipping_operator(dict(continuous_recipe='R02',clipping_recipe=bad))
                    except ValueError:pass
                    else:raise AssertionError('Unqualified composition was accepted')
            cases.append(dict(method=method,horizon=h,status='PASS',optimizer_updates=expected,input_exposure=2*config['epochs'],
                first_step=receipt['first_step'],validation=final['validation'],public_command=command,
                transport_proof=transport_proof,regeneration_proof=regeneration_proof,
                micro_diagnostics=diagnostics,full_budget_training=False,test_used=False))
            atomic_json(out/'progress.json',dict(cases=cases,optimizer_updates=sum(c['optimizer_updates'] for c in cases)))
    result=dict(status='PASS',qualified_recipes=['X03'] if group=='point-setting-composition' else ['R13'] if group=='local-credit-regeneration' else ['X01'] if group=='composed' else ['X06'] if group=='point-probability-mix' else list(methods),cases=cases,optimizer_updates=sum(c['optimizer_updates'] for c in cases),
        optimizer_trajectories=len(cases),test_used=False,
        scope='Installed public CLI; four H; declared finite update schedule per method/H; independent reload. Coefficients here qualify integration only, not prepared-state calibration.')
    atomic_json(out/'result.json',result);return result
