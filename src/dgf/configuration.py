"""Explicit public settings; no backend, model or array imports."""
import json,copy,math
from pathlib import Path


def load_v1(path):
    c=json.loads(Path(path).read_text())
    if c.get('format_version')!=1:raise ValueError('Unsupported configuration version')
    data=c['data'];stages=c['stages'];rl=stages['rl'];init=stages['initialization']
    if data['dataset']!='ETTh1' or data['horizon']not in [192,336]:raise ValueError('Unsupported example data/horizon')
    if (data['mode'],data['context'],data['patch'],data['split_ends'],data['variables'])!=('M',5000,64,[8640,11520,14400],['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']):raise ValueError('Data contract differs from this implementation')
    if c['evaluation']['selection']!='validation_MAP_MSE_best' or c['evaluation']['probability_samples']!=1000:raise ValueError('Examples use validation MAP MSE selection and 1000 independent marginal samples')
    if c['validation']['selection']!='MAP_MSE_min' or c['validation']['early_stopping']:raise ValueError('Fixed-budget MAP selection required')
    if rl['fixed_total_concentration'] or min(rl[k]for k in ['dirichlet_objective_weight','dynamic_weight','diversity_weight'])<=0:raise ValueError('All complete-method branches must remain active')
    if init['modes']!=8 or rl['groups']<2 or rl['trajectories_per_group']<2:raise ValueError('Nondegenerate hierarchical groups and modes required')
    if rl['epochs']!=4 or stages['preparation']['epochs']!=1 or stages['sft']['epochs']!=10:raise ValueError('Unsupported formal training budget')
    if any(not math.isfinite(float(x))or float(x)<=0 for x in [rl['lr'],stages['sft']['lr'],stages['preparation']['lr']]):raise ValueError('Finite positive learning rates required')
    constants={('optimizer','name'):'AdamW',('optimizer','betas'):[.9,.98],('optimizer','epsilon'):1e-6,('optimizer','gradient_clip_norm'):1.,('optimizer','scheduler'):'constant',('optimizer','reset_each_stage'):True,('numerics','model_dtype'):'float32',('numerics','tf32_matmul'):True,('numerics','tf32_cudnn'):True,('numerics','amp'):False,('numerics','world_size'):1}
    for (section,key),value in constants.items():
        if c[section][key]!=value:raise ValueError('Backend does not implement a change to '+section+'.'+key)
    for key,value in dict(head_perturbation=1e-4,router_hidden_width=128,router_mean_weight_std=.001,concentration_initial=8.,weights_floor=1e-4,concentration_floor=.001).items():
        if init[key]!=value:raise ValueError('Unsupported initialization change: '+key)
    if rl['ppo_clip']is not None or rl['reference_kl']!=0 or rl['nll_anchor']!=0 or rl['updates_per_fresh_rollout']!=1 or rl['rollout_reuse']!=1:raise ValueError('This interface implements the complete on-policy objective')
    fixed={
      'data.stride':{'train':1,'validation':1,'test':1},
      'data.scaler':{'fit':'train_only','ddof':1,'arrow_denominator_epsilon':1e-10,'forecast_denominator_epsilon':0.0},
      'data.internal_scaling':True,'data.missing_values':'reject','data.padding':'right_pad_to_patch_mask_then_crop','data.zeros_observed':True,
      'optimizer.weight_decay':{'linear_projection_weights':0.1,'other_parameters':0.0},
      'optimizer.accumulation':'microbatch_loss_weighted_by_actual_inputs_in_logical_batch','numerics.reward_and_metrics_dtype':'float64',
      'stages.sft.objective':'official_PackedNLLLoss','stages.sft.parent':'public_pretrained_weights','stages.sft.output_for_next_stage':'validation_MAP_MSE_best',
      'stages.initialization.route':'D-post','stages.initialization.operation':'copy_SFT_heads_and_seeded_perturbation',
      'stages.initialization.concentration_parameterization':'floored_softmax_weights_times_softplus_total','stages.initialization.trainable':'backbone_heads_and_router','stages.initialization.lora':False,
      'stages.preparation.objective':'legal_trajectory_mixture_NLL','stages.preparation.fixed_total_concentration':True,'stages.preparation.parent':'initialized_SFT_weights','stages.preparation.output_for_next_stage':'final',
      'stages.rl.accuracy_reward':'negative_mean_squared_error','stages.rl.dynamic_terms':['first_difference_L1','second_difference_L1','normalized_rfft_amplitude_L1','first_forecast_to_last_history_L1'],
      'stages.rl.diversity_scales':'fit_train_only_level_diff1_diff2_fft','stages.rl.advantage':'within_group_G_and_across_group_means_B','stages.rl.advantage_std_correction':0,'stages.rl.advantage_epsilon':1e-6,
      'stages.rl.trajectory_log_probability':'augmented_native_family_joint_trajectory_plus_mode_given_pi','stages.rl.dirichlet_log_probability':'log_Dirichlet_pi_given_alpha','stages.rl.gradient_estimator':'detached_score_function','stages.rl.objective_reduction':'mean_NBG_sample_mean_NB_dirichlet_mean_N_diversity',
      'validation':{'split':'validation','first_epoch_fractions':16,'later_epoch_fractions':4,'include_initial':True,'repeated_update_states':'reuse_evaluation','selection':'MAP_MSE_min','tie_break':'first_strict_improvement','fixed_budget':True,'early_stopping':False,'samples':100,'rng':'stage_seed+65537+global_input_block_index','input_block':64},
      'evaluation':{'selection':'validation_MAP_MSE_best','map':'argmax_expected_Dirichlet_weight_then_selected_mode_mean','mode_tie_break':'first_index','point_samples':100,'point_rng':'training_seed+65537+104729+global_input_block_index','probability_samples':1000,'probability_seeds':[1107001,1107003],'probability_input_block':4,'probability_sampling':'iid_marginal_outer_mode_per_sample','crps':'fair_empirical','scales':['standardized','original'],'save_predictions':True}}
    for path,expected in fixed.items():
        value=c
        for part in path.split('.'):value=value[part]
        if value!=expected:raise ValueError('Backend does not implement a change to '+path)
    if len(rl['dynamic_coefficients'])!=4 or any(v<=0 for v in rl['dynamic_coefficients']):raise ValueError('Four positive dynamic coefficients required')
    return c


def native_v1(c,phase,protocol,model,data,work):
    stage=c['stages'][phase];sft=phase=='sft';rl=phase=='rl'
    v=dict(dataset='ETTh1',model='moirai-1.0-R-small',model_id=c['model']['id'],weights=str(model),revision=c['model']['revision'],local_files_only=True,protocol=copy.deepcopy(protocol),artifact=str(data),route='A0'if sft else'D-post',seed=c['random_seeds'][phase],phase='supervised'if sft else'scientific'if rl else'preparation',global_input_batch=stage['effective_batch'],microbatch=stage['microbatch'],lr=stage['lr'],checkpoint_interval_batches=stage['checkpoint_interval_batches'],world_size=1,test_release=False,four_horizon_campaign=True,record_probability_checkpoints=True,precision='float32 with TF32; float64 metric accumulation',selection_objective='Full validation P1-MSE, fixed budget; no NLL early stopping')
    if sft:v.update(fixed_epochs=stage['epochs'],max_epochs=stage['epochs'],supervised_objective=stage['objective']);return v
    v.update(epochs=stage['epochs'],J=c['stages']['rl']['groups'],G=c['stages']['rl']['trajectories_per_group'],K=c['stages']['initialization']['modes'],arm='E03',algorithm_arm='E03',preparation='ordinary',checkpoint_kind='policy_state',updates_per_input_batch=1,nll_anchor_coefficient=0.,dgf_selection_only=True,force_fixed_kappa=not rl,matched_dgf_baseline=not rl,scientific_path=c['name']+'-'+phase)
    if not rl:v['checkpoint']=str(work/'initialization/constructed.pt');return v
    r=stage;v.update(method_kind='FULL',recipe_id='complete_dgf',PPO_clip=None,reference_beta=0.,checkpoint_from_receipt=dict(path=str(work/'preparation/result.json'),key=['readouts','final','checkpoint']),dgf_complete=dict(gamma=r['dirichlet_objective_weight'],lambda_dyn=r['dynamic_weight'],lambda_div=r['diversity_weight'],dynamic_coefficients=r['dynamic_coefficients'],diversity_cap=r['diversity_cap'],diversity_rho=r['diversity_rho'],fixed_total_concentration=False,enable_dirichlet_objective=True,enable_dynamic=True,enable_diversity=True))
    return v


def load(path):
    return load_v1(path)


def native(c,phase,protocol,model,data,work):
    if c.get('format_version')!=1:
        raise ValueError('Unsupported configuration version')
    return native_v1(c,phase,protocol,model,data,work)
