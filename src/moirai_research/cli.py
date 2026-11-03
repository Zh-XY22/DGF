import argparse

def main():
    parser=argparse.ArgumentParser(prog='moirai-research')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare-dgf-data',help='ETTh1 H192/336 data and train-only diversity statistics')
    p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--horizon',type=int,choices=[192,336],required=True);p.add_argument('--data-revision',required=True)
    p=sub.add_parser('initialize-dgf',help='Eight outer mode heads from a declared A0 SFT checkpoint; no optimizer continuation')
    p.add_argument('--config',required=True)
    p=sub.add_parser('make-dgf-configs',help='Full DGF reference by default; explicit controls and ablations')
    p.add_argument('--data-dir',required=True);p.add_argument('--work-dir',required=True);p.add_argument('--weights',required=True)
    p.add_argument('--parent-sft',required=True);p.add_argument('--revision',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--recipe',default='F00')
    p=sub.add_parser('make-four-horizon-configs',help='Materialize an explicit accepted recipe template using caller-owned model/data/output paths')
    p.add_argument('--template',required=True);p.add_argument('--data-dir',required=True);p.add_argument('--work-dir',required=True)
    p.add_argument('--weights',required=True);p.add_argument('--seed',type=int,required=True)
    p=sub.add_parser('train-joint',help='One policy and one common checkpoint across four horizons')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,required=True)
    p=sub.add_parser('evaluate-frozen-manifest',help='Portable frozen checkpoint/readout aliases, point100 and optional distribution1000')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    p=sub.add_parser('evaluate-distribution',help='One distribution record per checkpoint; streamed fair CRPS, nine-quantile wQL and intervals')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,required=True)
    p=sub.add_parser('evaluate-warmup',help='Complete validation and frozen test with explicit common precision; metrics only')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,required=True)
    for name,description in [('audit-hdfs','Zero-update full checkpoint HDFS publication and reload qualification'),('audit-rollout-diagnostics','Actual-checkpoint group reward variance and fixed training-window qualification'),('audit-pretrained','Compare research base weights with a fixed public pretrained snapshot'),('probe-diagnostic-parity','Fixed-input same-device interface and cross-device mean attribution'),('diagnose-modes','Training-frozen mode naming and complete stored-validation calibration'),('audit-analysis-diagnostics','Actual-checkpoint mode diagnostics and retained-validation statistical qualification')]:
        p=sub.add_parser(name,help=description)
        p.add_argument('--config',required=True);p.add_argument('--output',required=True)
        p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('final-clean-release',help='Fresh package installation and full validation-frozen checkpoint reproduction')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('repair-neighbor-fallback',help='Training-only strict-coverage fallback and actual auxiliary-gradient equality proof')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('analyze-finite',help='Recompute retained test errors and report paired seed/time-block uncertainty')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('audit-math',help='Component numerical checks; CUDA or small CPU checks')
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('audit-modes',help='Known predictable/ambiguous two-mode recovery')
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('audit-parity',help='Compare the existing research entry with the public package')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('probe-lr',help='Train-only real-Moirai LR and probability-measure checks')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('evaluate',help='Complete validation/test with fixed public decision protocols')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('evaluate-readouts',help='Formal test of validation-frozen readouts, with exact-state aliasing')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('replay-historical',help='Deduplicated finite historical checkpoint protocol replay')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('train-estimator',help='Training-only frozen DLinear reference, selected by validation MSE')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('build-neighbors',help='Past-only fold-frozen weak future labels and matched shuffled control')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('audit-rl-entry',help='Real-policy gradient qualification and baseline objective parity')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('train',help='Qualified ordinary preparation and finite baseline training')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('build-selective-statistics',help='Frozen full-training residual statistics and matched masks')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('profile-entry',help='Fixed-global-batch real updates and full-validation resource admission')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('audit-preparation',help='Train-only auxiliary gradient and cache alignment qualification')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('audit-io',help='Check complete state and prediction-array buffered write equivalence')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('audit-public-data',help='Rebuild public CSV and verify actual native transform parity')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('fit-supervised',help='Public-weight official SFT or legal trajectory-mixture warmup')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    for name,description in [('clean-install','Fresh isolated installation and public-data end-to-end qualification'),('qualification-e2e','Explicitly diagnostic two-stage public-data numerical qualification')]:
        p=sub.add_parser(name,help=description)
        p.add_argument('--config',required=True);p.add_argument('--output',required=True)
        p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('diagnose-labels',help='Past-fold weak-label log-loss/Brier and strict nonoverlap coverage')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('prepare-csv',help='Public ETT CSV to official LSF storage and explicit split indices')
    p.add_argument('--source',required=True);p.add_argument('--dataset',choices=['ETTh1','ETTm1'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--horizon',type=int,default=96);p.add_argument('--source-revision')
    p=sub.add_parser('prepare-solar',help='Worker-only declared all-variable hourly Solar engineering protocol; qualification pending')
    p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--source-revision',required=True)
    p=sub.add_parser('audit-extension',help='Worker-only full-variable packing and future-label invariance qualification')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('analyze-extension',help='Worker-only five-seed fixed-method transfer analysis')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('audit-composite',help='Train-only proof that two simultaneous auxiliaries retain one NLL anchor')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('make-run-configs',help='Assemble portable finite-arm configs from completed data metadata')
    p.add_argument('--data-dir',required=True);p.add_argument('--work-dir',required=True)
    p.add_argument('--arm',required=True);p.add_argument('--model-id',default='Salesforce/moirai-1.0-R-small')
    p.add_argument('--revision',required=True);p.add_argument('--seed',type=int,default=1)
    p.add_argument('--auxiliary-metadata')
    p.add_argument('--recipe',help='Validation-frozen finite recipe JSON; requires explicit auxiliary metadata')
    p.add_argument('--supervised-receipt',help='Reuse this completed same-seed/model/data supervised start')
    p.add_argument('--ordinary-preparation-receipt',help='Reuse an ordinary preparation from that exact supervised start')
    p.add_argument('--microbatch',type=int,help='Explicit measured input microbatch; does not change global input exposure or candidate groups')
    p=sub.add_parser('make-auxiliary-configs',help='Assemble existing neighbor, estimator, selective or teacher stages from completed receipts')
    p.add_argument('--data-dir',required=True);p.add_argument('--work-dir',required=True)
    p.add_argument('--stage',choices=['neighbors','estimator','selective','distillation'],required=True)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--coefficient',type=float)
    p.add_argument('--supervised-receipt');p.add_argument('--estimator-receipt');p.add_argument('--teacher-receipt')
    p=sub.add_parser('make-readout-configs',help='Bind retained checkpoints using complete validation after explicit method-selection release')
    p.add_argument('--training-receipt',required=True);p.add_argument('--preparation-receipt',required=True)
    p.add_argument('--selection-receipt',required=True);p.add_argument('--work-dir',required=True)
    p.add_argument('--scientific-path',required=True)
    for name in ['audit-point-entry','audit-point-readouts','audit-opd','profile-opd','build-point-selective','reload-point-historical','evaluate-point-readouts','analyze-point-study','accept-point-release']:
        p=sub.add_parser(name)
        p.add_argument('--config',required=True);p.add_argument('--output',required=True)
        p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1)
    p=sub.add_parser('make-point-configs',help='Portable new point-target campaign configuration builder')
    p.add_argument('--data-dir',required=True);p.add_argument('--work-dir',required=True)
    p.add_argument('--arm',required=True,help='Nine mandatory IDs, or validation-selected C15 / CO-N-O01 / CO-N-O02 / CO-S-O01 / CO-S-O02')
    p.add_argument('--model-id',default='Salesforce/moirai-1.0-R-small')
    p.add_argument('--revision',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--auxiliary-metadata',help='Frozen auxiliary metadata; conditional IDs require an existing validation-only recipe and selection receipt')
    p.add_argument('--supervised-receipt')
    p.add_argument('--preparation-receipt');p.add_argument('--teacher-receipt');p.add_argument('--microbatch',type=int)
    args=parser.parse_args()
    if args.command=='make-four-horizon-configs':
        from .four_horizon_configuration import make_configs
        import json
        print(json.dumps(make_configs(args.template,args.data_dir,args.work_dir,args.weights,args.seed),indent=2))
    elif args.command=='evaluate-frozen-manifest':
        from .frozen_manifest_evaluation import execute
        execute(args.config,args.output,args.device)
    elif args.command=='evaluate-distribution':
        from .probability_evaluation import evaluate_config
        evaluate_config(args.config,args.output,args.device,args.seed)
    elif args.command=='evaluate-warmup':
        from .warmup_readout import evaluate_frozen
        evaluate_frozen(args.config,args.output,args.device,args.seed)
    elif args.command=='make-point-configs':
        from .point_configuration import make_configs
        import json
        print(json.dumps(make_configs(args.data_dir,args.work_dir,args.arm,args.model_id,args.revision,args.seed,
            args.auxiliary_metadata,args.supervised_receipt,args.preparation_receipt,args.teacher_receipt,args.microbatch),indent=2))
    elif args.command=='reload-point-historical':
        from .historical_point_reload import reload
        reload(args.config,args.output,args.device,args.seed)
    elif args.command=='evaluate-point-readouts':
        from .point_readouts import evaluate_readouts
        evaluate_readouts(args.config,args.output,args.device,args.seed)
    elif args.command=='analyze-point-study':
        from .point_analysis import analyze
        analyze(args.config,args.output,args.device,args.seed)
    elif args.command=='accept-point-release':
        from .point_release_acceptance import run
        run(args.config,args.output,args.device,args.seed)
    elif args.command=='audit-point-entry':
        from .point_qualification import audit
        audit(args.config,args.output,args.device,args.seed)
    elif args.command=='audit-point-readouts':
        from .readout_isolation_qualification import audit
        audit(args.config,args.output,args.device,args.seed)
    elif args.command=='audit-opd':
        from .opd_qualification import audit
        audit(args.config,args.output,args.device,args.seed)
    elif args.command=='profile-opd':
        from .opd_profile import profile
        profile(args.config,args.output,args.device,args.seed)
    elif args.command=='build-point-selective':
        from .point_selective import build
        build(args.config,args.output,args.device,args.seed)
    elif args.command=='analyze-extension':
        from .extension_analysis import analyze
        analyze(args.config,args.output,args.device,args.seed)
    elif args.command=='audit-extension':
        from .extension_audit import audit
        audit(args.config,args.output,args.device,args.seed)
    elif args.command=='prepare-solar':
        from .solar_data import prepare_solar
        prepare_solar(args.source,args.output,args.source_revision)
    elif args.command=='audit-hdfs':
        from .hdfs_checkpoints import audit
        audit(args.config,args.output,args.device,args.seed)
    elif args.command=='audit-rollout-diagnostics':
        from .rollout_diagnostics import audit
        audit(args.config,args.output,args.device,args.seed)
    elif args.command=='audit-pretrained':
        from .pretrained_audit import audit
        audit(args.config,args.output,args.device,args.seed)
    elif args.command in ['diagnose-modes','audit-analysis-diagnostics','probe-diagnostic-parity']:
        from .mode_diagnostics import diagnose,audit,probe_parity
        {'diagnose-modes':diagnose,'audit-analysis-diagnostics':audit,'probe-diagnostic-parity':probe_parity}[args.command](args.config,args.output,args.device,args.seed)
    elif args.command=='final-clean-release':
        from .release_acceptance import run
        run(args.config,args.output,args.device,args.seed)
    elif args.command=='repair-neighbor-fallback':
        from .neighbor_fallback import apply_fallback
        apply_fallback(args.config,args.output,args.device,args.seed)
    elif args.command=='analyze-finite':
        from .analysis import analyze
        analyze(args.config,args.output,args.device,args.seed)
    elif args.command=='audit-composite':
        from .composite import audit
        audit(args.config,args.output,args.device,args.seed)
    elif args.command=='make-run-configs':
        from .configuration import make_configs
        import json
        print(json.dumps(make_configs(args.data_dir,args.work_dir,args.arm,args.model_id,args.revision,args.seed,args.auxiliary_metadata,args.recipe,
            args.supervised_receipt,args.ordinary_preparation_receipt,args.microbatch),indent=2))
    elif args.command=='make-auxiliary-configs':
        from .auxiliary_configuration import make_auxiliary_configs
        import json
        print(json.dumps(make_auxiliary_configs(args.data_dir,args.work_dir,args.stage,args.seed,args.coefficient,
            args.supervised_receipt,args.estimator_receipt,args.teacher_receipt),indent=2))
    elif args.command=='make-readout-configs':
        from .readout_configuration import make_readout_configs
        import json
        print(json.dumps(make_readout_configs(args.training_receipt,args.preparation_receipt,args.selection_receipt,
            args.work_dir,args.scientific_path),indent=2))
    elif args.command=='prepare-csv':
        from .public_data import prepare_csv
        prepare_csv(args.source,args.dataset,args.output,args.horizon,args.source_revision)
    elif args.command=='audit-math':
        from .audit import core_audit
        core_audit(args.output,args.device,args.seed)
    elif args.command=='audit-modes':
        from .synthetic import audit_modes
        audit_modes(args.output,args.device,args.seed)
    elif args.command=='audit-public-data':
        from .public_data_audit import audit
        audit(args.config,args.output,args.device,args.seed)
    elif args.command=='fit-supervised':
        from .supervised import fit
        fit(args.config,args.output,args.device,args.seed)
    elif args.command=='clean-install':
        from .clean_install import run
        run(args.config,args.output,args.device,args.seed)
    elif args.command=='qualification-e2e':
        from .qualification import run
        run(args.config,args.output,args.device,args.seed)
    elif args.command=='diagnose-labels':
        from .label_diagnostics import diagnose
        diagnose(args.config,args.output,args.device,args.seed)
    elif args.command=='audit-io':
        from .io_utils import audit_io
        audit_io(args.config,args.output,args.device,args.seed)
    elif args.command=='audit-preparation':
        from .preparation_audit import audit
        audit(args.config,args.output,args.device,args.seed)
    elif args.command=='profile-entry':
        from .performance import profile
        profile(args.config,args.output,args.device,args.seed)
    elif args.command=='build-selective-statistics':
        from .selective_statistics import build
        build(args.config,args.output,args.device,args.seed)
    elif args.command=='train-joint':
        import json
        from pathlib import Path
        if json.loads(Path(args.config).read_text())['seed']!=args.seed:raise ValueError('CLI seed differs from frozen joint configuration')
        from .joint_train import fit_joint
        fit_joint(args.config,args.output,args.device)
    elif args.command=='prepare-dgf-data':
        from .dgf_initialization import training_protocol
        training_protocol(args.source,args.output,args.horizon,args.data_revision)
    elif args.command=='initialize-dgf':
        import json
        from pathlib import Path
        from .dgf_initialization import initialize
        c=json.loads(Path(args.config).read_text())
        if c['required']:initialize(c['weights'],c['parent_sft'],c['output'],c['seed'],c['revision'])
    elif args.command=='make-dgf-configs':
        from .dgf_configuration import make_configs
        print(make_configs(args.data_dir,args.weights,args.revision,args.parent_sft,args.work_dir,args.seed,args.recipe))
    elif args.command=='train':
        from .train import train
        train(args.config,args.output,args.device,args.seed)
    elif args.command=='audit-rl-entry':
        from .rl_audit import audit
        audit(args.config,args.output,args.device,args.seed)
    elif args.command=='build-neighbors':
        from .neighbors import build_config
        build_config(args.config,args.output,args.device,args.seed)
    elif args.command=='train-estimator':
        from .estimator import train_estimator
        train_estimator(args.config,args.output,args.device,args.seed)
    elif args.command=='replay-historical':
        from .evaluation import historical_replay
        historical_replay(args.config,args.output,args.device,args.seed)
    elif args.command=='evaluate':
        from .evaluation import evaluate_config
        evaluate_config(args.config,args.output,args.device,args.seed)
    elif args.command=='evaluate-readouts':
        from .readouts import evaluate_readouts
        evaluate_readouts(args.config,args.output,args.device,args.seed)
    elif args.command=='probe-lr':
        from .probe import probe
        probe(args.config,args.output,args.device,args.seed)
    elif args.command=='audit-parity':
        from .parity import audit_parity
        audit_parity(args.config,args.output,args.device,args.seed)

if __name__=='__main__':main()
