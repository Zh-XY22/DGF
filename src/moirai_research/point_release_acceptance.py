"""Fresh-package qualification of rolling preparation and native-H96 readouts.

The tiny OPD/RL paths and optional final SFT-console stage are interface qualification, not scientific
performance. Complete recommended-checkpoint evaluations are a separate part
of the receipt. This module never authorizes test release or selects a method.
"""
import argparse
import copy
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import venv


def _read(path):
    return json.loads(Path(path).read_text())


def _public_sft_cli_enabled(config):
    value=config.get('exercise_public_sft_cli',False)
    if not isinstance(value,bool):raise ValueError('Public SFT CLI qualification must be explicitly boolean')
    return value


def _public_only_minimal_e2e(config):
    value=config.get('public_only_minimal_e2e',False)
    if not isinstance(value,bool):raise ValueError('Public-only interface qualification must be explicitly boolean')
    if value:
        if not _public_sft_cli_enabled(config):raise ValueError('Public-only interface requires a fresh public-weight supervised CLI update')
        for representative in config.get('representative_e2e_configs',[]):
            student=representative.get('student_config') or {}
            if student.get('checkpoint') or student.get('resume') or representative.get('teacher_config'):
                raise ValueError('Public-only tiny qualification cannot consume trained student/teacher artifacts')
    return value


def _supervised_cli_config(native,dataset):
    if dataset not in ('ETTh1','ETTm1'):raise ValueError('Qualification dataset is outside this campaign')
    config=dict(native,fixed_epochs=1,qualification_only=True,test_release=False,
        diagnostic_input_limit=2,diagnostic_validation_limit=2,global_input_batch=2,microbatch=1,
        supervised_objective='legal_trajectory_mixture_NLL',lr=1e-7 if dataset=='ETTh1' else 5e-7)
    for name in ('checkpoint','resume'):config.pop(name,None)
    return config


def _write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.tmp-{os.getpid()}')
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value
    temporary.write_text(json.dumps(clean(record), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _interface_only(config):
    """Early clean-install proof cannot release or consume formal test data."""
    value = config.get('interface_only', False)
    if not isinstance(value, bool):
        raise ValueError('interface_only must be an explicit boolean')
    if value:
        forbidden = ('frozen_selection', 'official_reproduction_cases',
                     'formal_test_receipts', 'scientific_paths')
        if config.get('test_release') is not False or any(config.get(key) for key in forbidden):
            raise ValueError('Interface qualification cannot bind frozen or formal-test readouts')
        representatives = config.get('representative_e2e_configs', [])
        if any(row.get('qualification_inputs') != 2 for row in representatives):
            raise ValueError('Early interface qualification is limited to two training inputs per dataset')
    if value and _public_sft_cli_enabled(config):
        if config.get('qualification_contract')!='public-weight-six-trajectory-interface-v1' or not _public_only_minimal_e2e(config):
            raise ValueError('Extra supervised CLI updates require the explicit six-trajectory public-only contract')
    return value


def _retain_tiny_artifacts(source, destination, writer):
    """Retain receipts and only explicitly bounded qualification states.

    The installed child writes to worker-local scratch. The caller's optional
    storage adapter publishes the completed states; original child receipts
    remain unchanged, with a separate location map. No historical tree scan.
    """
    source, destination = Path(source), Path(destination)
    names = ['preparation_resume.pt', 'native_rl_resume.pt']
    supervised=source/'supervised_cli/result.json'
    if supervised.is_file():
        record=_read(supervised)
        if record.get('status')!='DIAGNOSTIC_ONLY' or record.get('optimizer_steps')!=1:
            raise ValueError('Public SFT qualification exceeds its one-update contract')
        required=['supervised_cli/resume.pt','supervised_cli/checkpoints/initial.pt','supervised_cli/checkpoints/final.pt']
        if not all((source/name).is_file() for name in required):
            raise ValueError('Public SFT qualification states are incomplete')
        names.extend(required)
        optional='supervised_cli/checkpoints/point-best-step1.pt'
        if (source/optional).is_file():names.append(optional)
    unexpected = [p for p in source.rglob('*') if p.suffix in ('.pt', '.pth', '.ckpt', '.safetensors')
                  and str(p.relative_to(source)) not in names]
    if unexpected or any(not (source/name).is_file() or (source/name).is_symlink() for name in names):
        raise ValueError('Tiny qualification checkpoint set differs from the explicitly bounded generated states')
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('*.pt','*.pth','*.ckpt','*.safetensors'))
    mapping = []
    for name in names:
        local = source/name
        retained = destination/name
        writer(retained, io.BytesIO(local.read_bytes()))
        mapping.append(dict(execution_path=str(local), retained_path=str(retained),
                            publication_completed=True))
    # Every destination is completed before removing any local execution state.
    for name in names:
        (source/name).unlink()
    _write(destination/'artifact_locations.json', dict(execution_root=str(source),
        retained_root=str(destination), checkpoint_paths=mapping,
        raw_child_receipts_rewritten=False,
        scope='Child local writes followed by outer publication; not an in-child remote reload claim'))
    return mapping


def _archive_preflight(config, out, scratch):
    """Use the same qualified native-CLI backend as completed state writes."""
    if not config.get('require_external_checkpoint_archive'):
        return None
    required = ('RESEARCH_WORKER_DIR', 'MOIRAI_HDFS_CHECKPOINT_CATEGORY',
                'MOIRAI_HDFS_ARCHIVE_URI', 'MOIRAI_HDFS_ARCHIVE_MOUNT')
    if any(not os.environ.get(key) for key in required):
        raise ValueError('Required external checkpoint archive is not configured; no optimizer run started')
    from .hdfs_checkpoints import publish_cli, destination as archive_destination
    import uuid
    out, scratch = Path(out), Path(scratch)
    marker = scratch/'archive-preflight.txt'
    marker.write_text('Clean-package qualification storage preflight; no model state.\n')
    relative = archive_destination(out/'archive-preflight.txt', os.environ['RESEARCH_WORKER_DIR'],
                                   os.environ['MOIRAI_HDFS_CHECKPOINT_CATEGORY'])
    uri = os.environ['MOIRAI_HDFS_ARCHIVE_URI'].rstrip('/')+'/'+relative+'.'+uuid.uuid4().hex
    size = publish_cli(marker, uri)
    record = dict(status='PASS', uri=uri, bytes=size, before_optimizer_execution=True,
        backend='native-cli-immutable-generations-v1', scope='New qualification destination only')
    _write(out/'archive_preflight.json', record)
    return record


def _official_cases(config, frozen):
    """Bind every protocol's own frozen recommendations to canonical receipts."""
    seeds = set(config.get('seeds', [101, 211, 307, 401, 503]))
    supplied = config.get('official_reproduction_cases', [])
    wrappers = dict(config.get('formal_test_receipts', {}))
    for path in config.get('scientific_paths', []):
        if path.get('formal_test_receipt'):
            wrappers[path['experiment_id']] = path['formal_test_receipt']
    result = []
    for recipe in frozen.get('deliveries', frozen.get('recipes', [])):
        protocol=recipe.get('protocol')
        if protocol not in ('P1','P2','P3') or recipe.get('status') not in ('FROZEN', 'VALID'):
            continue
        objective = recipe.get('selection_objective', recipe.get('objective'))
        method = recipe.get('method', recipe.get('arm'))
        checkpoints = recipe.get('checkpoints', [])
        if {row['seed'] for row in checkpoints} != seeds or len(checkpoints) != 5:
            raise ValueError('A public frozen recommendation lacks five distinct checkpoint seeds')
        for row in checkpoints:
            key = (recipe['dataset'], method, row['seed'], protocol, objective, row['checkpoint'])
            case = next((dict(case) for case in supplied if (
                case['dataset'], case['method'], case['seed'], case['protocol'],
                case['selection_objective'], case['checkpoint']) == key), None)
            identifier = row.get('experiment_id')
            if case is None:
                if identifier not in wrappers:
                    raise ValueError('Missing formal-test receipt for a frozen recommended seed: '+str(identifier))
                wrapper = _read(wrappers[identifier])
                chosen = next((item for item in wrapper.get('readouts', [])
                    if item.get('split') == 'test' and item.get('protocol') == protocol
                    and item.get('selection_objective') == objective and item.get('checkpoint') == row['checkpoint']), None)
                if chosen is None:
                    raise ValueError('Frozen recommendation has no exact formal canonical readout')
                alias = next((item for item in wrapper.get('cases', []) if item.get('case_id') == chosen['readout']), {})
                case = dict(dataset=recipe['dataset'], method=method, seed=row['seed'], protocol=protocol,
                    selection_objective=objective, checkpoint=row['checkpoint'], scientific_path=identifier,
                    reference_receipt=chosen['receipt'], formal_wrapper_receipt=wrappers[identifier],
                    formal_readout=chosen['readout'], exact_alias_evidence=alias.get('exact_parameter_equality'))
            result.append(case)
    if len(result) > 60:
        raise ValueError('Final public recommendation scope exceeds sixty protocol-specific slots')
    recipe_gaps = frozen.get('recipe_gaps', [])
    if not result and not recipe_gaps:
        raise ValueError('No frozen recommendation and no declared scientific-terminal recipe gap')
    return result


def minimal_e2e(config_path, output, device='cuda', seed=101):
    """Two real trajectories: online rolling OPD preparation, then native RL."""
    import importlib.metadata
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from .data import datasets, batch
    from .evaluation import evaluation
    from .opd import OPD_ARMS
    from .preparation import Preparation
    from .policy import Policy, joints
    from .probability import nll
    from .probe import relative_update
    from .public_data import prepare_csv
    from .rl import make_rollout, objective
    from .runtime import seed_all, load_policy, optimizer_for, checked_step, save_state, restore_state
    config = _read(config_path)
    if config['dataset'] not in ('ETTh1', 'ETTm1') or config['arm'] not in ('O01', 'O02', 'O03'):
        raise ValueError('Tiny point release qualification requires an actual registered OPD arm and ETT dataset')
    if config.get('seed', seed) != seed:
        raise ValueError('Qualification seed differs from explicit config')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    began = time.time()
    seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    protocol = prepare_csv(config.get('public_csv_cache') or config['public_csv_url'],
        config['dataset'], out/'data', 96, config['data_revision'])
    data_match = None
    if config.get('reference_data_artifact'):
        for name in ('normalized.npy', 'scaler_mean.npy', 'scaler_std.npy', 'train_indices.npy', 'val_indices.npy', 'test_indices.npy'):
            np.testing.assert_array_equal(np.load(out/'data'/name, allow_pickle=False),
                np.load(Path(config['reference_data_artifact'])/name, allow_pickle=False))
        data_match = True
    weights = config.get('public_model_cache') or snapshot_download(repo_id=config['model_id'],
        revision=config['model_revision'], allow_patterns=['config.json', 'model.safetensors'])
    native = dict(config.get('student_config', {}), dataset=config['dataset'], weights=str(weights),
        revision=config['model_revision'], local_files_only=True, route='D-pre', seed=seed,
        protocol=protocol, artifact=str(out/'data'), force_fixed_kappa=True, qualification_only=True)
    supervised_cli=None
    if _public_sft_cli_enabled(config):
        if int(config.get('qualification_inputs',2)) != 2:
            raise ValueError('Final public SFT CLI qualification uses exactly two inputs')
        supervised_config=_supervised_cli_config(native,config['dataset'])
        file=out/'supervised_cli_config.json';_write(file,supervised_config)
        command=[str(Path(sys.executable).with_name('moirai-research')),'fit-supervised',
            '--config',str(file),'--output',str(out/'supervised_cli'),'--device',device,'--seed',str(seed)]
        log=out/'supervised_cli.log'
        with log.open('w') as stream:
            completed=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=900,check=False)
        if completed.returncode:raise RuntimeError('Actual installed public SFT CLI failed; see supervised_cli.log')
        supervised_cli=_read(out/'supervised_cli/result.json')
        if supervised_cli.get('status')!='DIAGNOSTIC_ONLY' or supervised_cli.get('optimizer_steps')!=1 or not supervised_cli.get('public_weight_start'):
            raise ValueError('Public SFT CLI did not execute exactly one public-weight update')
        if supervised_cli['first_step']['relative_parameter_update']<=0 or supervised_cli['first_step']['frozen_max_change']!=0:
            raise ValueError('Public SFT CLI lacks a valid parameter update/freeze proof')
        native.update(checkpoint=supervised_cli['readouts']['final']['checkpoint'],checkpoint_kind='policy_state')
        _write(out/'supervised_cli_command.json',dict(command=command,returncode=completed.returncode,
            log=str(log),qualification_only=True,optimizer_trajectories=1,optimizer_steps=1))
    if native.get('checkpoint'):
        policy = load_policy(native, device).eval()
        start_scope = 'Existing independently trained shared supervised start, loaded in the fresh package'
        if supervised_cli is not None:start_scope='Fresh public-weight SFT through the installed console CLI, one two-input update; not full scientific SFT'
    else:
        policy = Policy(str(weights), 'D-pre', revision=config['model_revision'], local_files_only=True).to(device).eval()
        policy.configure_update(fixed_kappa=True)
        start_scope = 'Public pretrained backbone with native eight-mode construction; not a full supervised reproduction'
    teacher = copy.deepcopy(config.get('teacher_config') or dict(native, route='A0', checkpoint=None,
        checkpoint_kind='legacy', provenance='Public pretrained single-head interface teacher; not a scientific teacher-selection claim'))
    teacher.update(protocol=protocol, artifact=str(out/'data'), dataset=config['dataset'])
    preparation_config = dict(native, phase='preparation', arm=config['arm'], preparation_arm=config['arm'],
        teacher_config=teacher, distillation_samples=8, opd_blocks=3, opd_block_length=32,
        auxiliary_coefficient=config['opd_coefficient'], lr=5e-7, nll_anchor_coefficient=1.)
    plugin = Preparation(preparation_config, device)
    data = datasets(policy, config['dataset'], protocol)['train']
    count = int(config.get('qualification_inputs', 2))
    if not 1 <= count <= 8:
        raise ValueError('Tiny qualification input count must be explicitly bounded in [1,8]')
    item = batch(data, range(count), device)
    wrapper_checks = []
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    for arm in OPD_ARMS:
        candidate = plugin if arm == config['arm'] else Preparation(dict(preparation_config, arm=arm, preparation_arm=arm), device)
        experts, target, weights_now, _, _ = joints(policy(item), item)
        auxiliary, diagnostic = candidate.auxiliary(item, experts, target, weights_now,
            student_policy=policy, policy_version=0)
        gradients = torch.autograd.grad(auxiliary, trainable, allow_unused=True)
        valid = [gradient for gradient in gradients if gradient is not None]
        if not valid or not all(torch.isfinite(gradient).all() for gradient in valid):
            raise ValueError('Actual preparation wrapper produced missing/nonfinite policy gradients')
        gradient_norm = float(torch.stack([gradient.square().sum() for gradient in valid]).sum().sqrt())
        if gradient_norm <= 0 or diagnostic['preparation_arm'] != arm or diagnostic['native_calls'] != 3:
            raise ValueError('Actual preparation wrapper did not retain its OPD arm, rolling states or effective gradient')
        wrapper_checks.append(dict(arm=arm, preparation_wrapper=True, gradient_norm=gradient_norm,
            gradient_tensors=len(valid), native_calls=diagnostic['native_calls'], optimizer_updates=0))
        del candidate, auxiliary, gradients, valid, experts, target, weights_now
    initial = {name: parameter.detach().cpu().clone() for name, parameter in policy.named_parameters()}
    optimizer = optimizer_for(policy, 5e-7)
    prep_records, prefix_records = [], []
    for version in range(2):
        optimizer.zero_grad(set_to_none=True)
        experts, target, weights_now, _, _ = joints(policy(item), item)
        anchor = nll(experts, weights_now, target)
        auxiliary, diagnostic = plugin.auxiliary(item, experts, target, weights_now,
            student_policy=policy, policy_version=version, capture=True)
        prefixes = diagnostic.pop('_prefix_traces')
        prefix_records.append([value.detach().cpu() for value in prefixes])
        loss = anchor + config['opd_coefficient']*auxiliary
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite tiny OPD training loss; not a diagnostic-score veto')
        loss.backward()
        update = checked_step(policy, optimizer)
        if diagnostic['native_calls'] != 3 or len(diagnostic['events']) != 3:
            raise ValueError('The installed package did not execute three rolling states')
        prep_records.append(dict(step=version+1, loss=float(loss.detach()), anchor=float(anchor.detach()),
            **update, rolling=diagnostic))
    prep_delta, prep_frozen = relative_update(policy, initial)
    if not prep_delta > 0 or prep_frozen != 0:
        raise ValueError('Tiny OPD preparation lacks a finite effective parameter update or changed frozen parameters')
    if not any(not torch.equal(a, b) for a, b in zip(prefix_records[0], prefix_records[1])):
        raise ValueError('Online prefix refresh produced identical retained trajectories in both updates')
    treatment = max(event['max_abs_log_density_difference'] for record in prep_records for event in record['rolling']['events'])
    if not treatment > 1e-7:
        raise ValueError('Teacher/student block distributions are indistinguishable; qualification has no distillation treatment')
    prep_state = out/'preparation_resume.pt'
    save_state(prep_state, policy, optimizer, preparation_config,
        dict(epoch=0, next_input=count, global_step=2), extra=dict(qualification_only=True))
    _write(out/'preparation_receipt.json', dict(status='PASS_DIAGNOSTIC_ONLY', optimizer_trajectories=1,
        optimizer_steps=2, checkpoint=str(prep_state), updates=prep_records, relative_parameter_change=prep_delta,
        frozen_max_change=prep_frozen, test_used=False))
    rl_config = dict(native, checkpoint=str(prep_state), checkpoint_kind='policy_state', arm='E03',
        phase='scientific', nll_anchor_coefficient=1., lr=5e-7)
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    optimizer = optimizer_for(policy, 5e-7)
    old = make_rollout(policy, reference, item, rl_config)
    rl_initial = {name: parameter.detach().cpu().clone() for name, parameter in policy.named_parameters()}
    rl_records = []
    for update_index in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostic = objective(policy, item, old, rl_config)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite tiny native RL training loss')
        loss.backward()
        gradient = checked_step(policy, optimizer)
        rl_records.append(dict(step=update_index+1, loss=float(loss.detach()), **gradient, diagnostic=diagnostic))
    rl_delta, rl_frozen = relative_update(policy, rl_initial)
    if not rl_delta > 0 or rl_frozen != 0:
        raise ValueError('Native RL failed to update parameters or changed frozen parameters')
    state = out/'native_rl_resume.pt'
    save_state(state, policy, optimizer, rl_config, dict(epoch=0, next_input=count, global_step=2))
    with torch.no_grad():
        before = joints(policy(item), item)
        saved_means = [value.mean.detach().clone() for value in before[0]]
        saved_weights = before[2].detach().clone()
    reloaded_config = dict(rl_config, checkpoint=str(state))
    reloaded = load_policy(reloaded_config, device).eval()
    reloaded_optimizer = optimizer_for(reloaded, 5e-7)
    cursor, _ = restore_state(state, reloaded, reloaded_optimizer, rl_config)
    if cursor['global_step'] != 2:
        raise ValueError('Optimizer/RNG/cursor checkpoint did not restore')
    with torch.no_grad():
        after = joints(reloaded(item), item)
        for previous, current in zip(saved_means, after[0]):
            torch.testing.assert_close(previous, current.mean, atol=0, rtol=0)
        torch.testing.assert_close(saved_weights, after[2], atol=0, rtol=0)
    metrics = evaluation(reloaded, protocol, 'val', out/'diagnostic_validation', seed, str(out/'data'), limit=2)
    if metrics['complete'] or metrics['windows'] != 2:
        raise ValueError('Tiny qualification must remain explicitly diagnostic')
    import moirai_research
    result = dict(status='PASS', qualification_only=True, optimizer_trajectories=2+int(supervised_cli is not None), optimizer_steps=4+int(supervised_cli is not None),
        public_sft_cli_checked=supervised_cli is not None,
        public_weights_only_interface=supervised_cli is not None and not config.get('teacher_config')
            and not (config.get('student_config') or {}).get('checkpoint') and not (config.get('student_config') or {}).get('resume'),
        teacher_checkpoint_supplied=bool(teacher.get('checkpoint')),
        teacher_source_scope='Supplied trained teacher' if teacher.get('checkpoint') else 'Public pretrained single-head interface teacher; not the scientific teacher',
        supervised_cli_receipt=str(out/'supervised_cli/result.json') if supervised_cli is not None else None,
        formal_preparation_wrapper_exercised=True, preparation_wrapper_gradient_checks=wrapper_checks,
        dataset=config['dataset'], arm=config['arm'], start_scope=start_scope,
        public_data_matches_reference=data_match, prefix_policy_versions=[0, 1], prefix_refresh_observed=True,
        teacher_student_max_abs_block_log_density_difference=treatment,
        rolling_states_per_update=3, native_downstream_horizon=96, native_RL_rollout_updates=2,
        preparation_receipt=str(out/'preparation_receipt.json'), preparation_checkpoint=str(prep_state),
        native_RL_checkpoint=str(state), native_RL_updates=rl_records, exact_reload=True,
        optimizer_rng_cursor_restored=True, RL_parameter_change=rl_delta, frozen_max_change=rl_frozen,
        diagnostic_validation=metrics, diagnostic_metrics_do_not_gate_points=True, test_used=False,
        environment=dict(prefix=sys.prefix, package_file=moirai_research.__file__, python=sys.version,
            torch=torch.__version__, cuda=torch.version.cuda,
            dependencies={key: importlib.metadata.version(key) for key in ('numpy', 'pandas', 'gluonts', 'lightning', 'safetensors')}),
        actual_gpu=torch.cuda.get_device_name(), seconds=time.time()-began,
        limitation='Tiny OPD-to-native-RL execution proves interfaces and updates, not full scientific performance')
    _write(out/'result.json', result)
    return result


def run(config_path, output, device='cuda', seed=1):
    """Install the frozen public archive and reproduce explicit frozen cases."""
    from .clean_install import dependency_lock, extract_public_package
    from .point_selection import POINT_PROTOCOLS
    from .point_test_gate import test_release
    config = _read(config_path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    began = time.time()
    interface_only = _interface_only(config)
    public_sft=_public_sft_cli_enabled(config)
    public_only=_public_only_minimal_e2e(config)
    representatives = config['representative_e2e_configs']
    if len(representatives) != 2 or {row['dataset'] for row in representatives} != {'ETTh1', 'ETTm1'}:
        raise ValueError('Exactly two dataset-specific public qualification paths are required')
    frozen = {} if interface_only else _read(config['frozen_selection'])
    official = [] if interface_only else _official_cases(config, frozen)
    scratch = Path(tempfile.mkdtemp(prefix='moirai-point-release-'))
    _archive_preflight(config, out, scratch)
    project = extract_public_package(config['package_path'], scratch/'package')
    lock = out/'requirements.lock'
    lock.write_text(dependency_lock(project))
    if config.get('pinned_requirements') and Path(config['pinned_requirements']).read_text() != lock.read_text():
        raise ValueError('Frozen public dependency closure differs from the supplied lock')
    environment = scratch/'venv'
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment)
    python = str(environment/'bin/python')
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PIP_INDEX_URL', 'PIP_EXTRA_INDEX_URL', 'PIP_TRUSTED_HOST'):
        env.pop(key, None)
    for key in list(env):
        if key.startswith('MOIRAI_HDFS_') or key in ('RESEARCH_WORKER_DIR', 'MOIRAI_HADOOP_CLI'):
            env.pop(key, None)
    env.update(VIRTUAL_ENV=str(environment), PATH=str(environment/'bin')+os.pathsep+env['PATH'],
        PYTHONNOUSERSITE='1', PIP_CONFIG_FILE=os.devnull, HF_HUB_DISABLE_TELEMETRY='1',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PIP_CACHE_DIR=str(scratch/'pip-cache'))
    commands = []
    def execute(arguments, name, timeout):
        command = [python, *arguments]
        with (out/(name+'.log')).open('w') as stream:
            completed = subprocess.run(command, env=env, cwd=scratch, stdout=stream,
                stderr=subprocess.STDOUT, timeout=timeout, check=False)
        record = dict(name=name, command=command, returncode=completed.returncode, log=str(out/(name+'.log')))
        commands.append(record)
        _write(out/'command_records.json', commands)
        if completed.returncode:
            raise RuntimeError(f'Clean package command failed: {name}; see retained command record')
    index = ['--no-index', '--find-links', config['wheelhouse']] if config.get('wheelhouse') else ['--index-url', 'https://pypi.org/simple']
    execute(['-m', 'pip', 'install', *index, '--disable-pip-version-check', '-r', str(lock)], 'install_dependencies', 5400)
    execute(['-m', 'pip', 'install', *index, '--no-deps', '--no-build-isolation', str(project)], 'install_package', 600)
    execute(['-m', 'pip', 'check'], 'pip_check', 120)
    execute(['-c', 'import json;from moirai_research.point_analysis import validate_qualification;print(json.dumps(validate_qualification()))'],
            'scalar_record_decision_check', 120)
    tiny = []
    public_data = {}
    for original_representative in representatives:
        representative = copy.deepcopy(original_representative)
        representative['exercise_public_sft_cli']=public_sft
        dataset = representative['dataset']
        for name in ('student_config', 'teacher_config'):
            binding = representative.get(name, {})
            if binding.get('checkpoint'):
                import torch
                from .io_utils import load_checkpoint
                # An explicit read cache, not archive migration or source mutation.
                state = load_checkpoint(binding['checkpoint'], map_location='cpu', weights_only=False)
                local = scratch/(dataset+'-'+name+'.pt')
                torch.save(state, local)
                binding['checkpoint'] = str(local)
        file = out/(dataset+'-e2e.json')
        _write(file, representative)
        execution_output = scratch/(dataset+'-e2e')
        execute(['-m', 'moirai_research.point_release_acceptance', '--stage', 'minimal-e2e',
            '--config', str(file), '--output', str(execution_output), '--device', device,
            '--seed', str(representative['seed'])], dataset+'-new_OPD_to_native_RL', 7200)
        receipt = _read(execution_output/'result.json')
        if receipt.get('status') != 'PASS' or receipt['optimizer_trajectories'] != 2+int(public_sft) or not receipt['exact_reload']:
            raise ValueError('A real rolling-prefix-to-native-RL qualification is missing')
        if public_sft and not receipt.get('public_sft_cli_checked'):
            raise ValueError('Final qualification did not execute the real installed supervised CLI')
        if public_only and (not receipt.get('public_weights_only_interface') or receipt.get('teacher_checkpoint_supplied') is not False):
            raise ValueError('Tiny public-data pipeline still depends on unpublished training artifacts')
        if not receipt.get('formal_preparation_wrapper_exercised') or {r['arm'] for r in receipt.get('preparation_wrapper_gradient_checks', [])} != {'O01','O02','O03'}:
            raise ValueError('Clean qualification bypassed a formal OPD Preparation wrapper')
        if receipt['environment']['prefix'] != str(environment) or str(environment) not in receipt['environment']['package_file']:
            raise ValueError('Qualification imported code from outside the fresh installation')
        from .io_utils import write_buffer
        from .hdfs_checkpoints import remote_uri
        def publish_state(path, buffer):
            write_buffer(path, buffer)
            if config.get('require_external_checkpoint_archive') and remote_uri(path) is None:
                raise ValueError('A qualification state was not published to the required external archive')
        destination = out/(dataset+'-e2e')
        receipt['retained_checkpoint_locations'] = _retain_tiny_artifacts(execution_output, destination, publish_state)
        receipt['raw_execution_receipt'] = str(destination/'result.json')
        receipt['storage_interpretation'] = 'Installed child reloaded local state; outer retained publication is separately recorded'
        tiny.append(receipt)
        public_data[dataset] = destination/'data'
    reproduced = []
    evaluated_states = []
    for index_number, case in enumerate(official):
        dataset = case['dataset']
        reference = _read(case['reference_receipt'])
        original = copy.deepcopy(reference['config'])
        release = test_release(config['frozen_selection'], case['scientific_path'])
        if original['seed'] != case['seed'] or (original.get('checkpoint') != case['checkpoint'] and case.get('exact_alias_evidence') is not True):
            raise ValueError('The recommended checkpoint differs from its original formal receipt')
        recipes = frozen.get('deliveries', frozen.get('recipes', []))
        if not any(row.get('dataset') == dataset and row.get('protocol') == case['protocol']
            and row.get('selection_objective', row.get('objective')) == case['selection_objective']
            and row.get('method', row.get('arm')) == case['method'] for row in recipes):
            raise ValueError('Public recommendation was not frozen by validation before formal test')
        protocol = _read(public_data[dataset]/'data_protocol.json')
        for key in ('mode', 'context', 'patch', 'horizon', 'variables', 'train_end', 'val_end', 'test_end', 'ddof', 'windows', 'scaler_mean', 'scaler_std_ddof1'):
            if protocol[key] != original['protocol'][key]:
                raise ValueError('Public CSV protocol differs from the formal reference: '+key)
        original.update(protocol=protocol, artifact=str(public_data[dataset]), splits=['val', 'test'],
            checkpoint=case['checkpoint'], test_release_receipt=config['frozen_selection'])
        original.pop('diagnostic_limit', None)
        directory = out/f'official-{index_number:03d}'
        directory.mkdir()
        import torch
        from .io_utils import load_checkpoint
        checkpoint_state = load_checkpoint(case['checkpoint'], map_location='cpu', weights_only=False)
        parameters = checkpoint_state['policy_state']
        architecture = tuple(checkpoint_state.get(key) for key in ('route', 'routing', 'fixed_kappa'))
        prior = next((item for item in evaluated_states if item['dataset'] == dataset and item['seed'] == case['seed']
            and item['route'] == original['route'] and item['architecture'] == architecture
            and item['parameters'].keys() == parameters.keys()
            and all(item['parameters'][name].dtype == value.dtype and torch.equal(item['parameters'][name], value)
                    for name, value in parameters.items())), None)
        if prior is None:
            local = scratch/f'official-{index_number:03d}-read-cache.pt'
            torch.save(checkpoint_state, local)
            original['checkpoint'] = str(local)
            file = directory/'config.json'
            _write(file, original)
            execute(['-m', 'moirai_research.cli', 'evaluate', '--config', str(file), '--output', str(directory/'evaluation'),
                '--device', device, '--seed', str(case['seed'])], f'official_{index_number:03d}', 14400)
            actual_receipt = str(directory/'evaluation/result.json')
            evaluated_states.append(dict(dataset=dataset, seed=case['seed'], route=original['route'],
                architecture=architecture, parameters=parameters, receipt=actual_receipt))
        else:
            actual_receipt = prior['receipt']
        actual = _read(actual_receipt)
        comparisons, diagnostic_gaps = [], []
        for split in ('val', 'test'):
            expected = reference['results'][split]
            observed = actual['results'][split]
            if not observed['complete'] or observed['windows'] != expected['windows'] or observed['seed'] != case['seed']:
                raise ValueError('Clean reproduction did not complete the exact full-window protocol')
            for protocol_key, fields in POINT_PROTOCOLS.items():
                for metric in ('MSE', 'MAE'):
                    key = fields[metric]
                    before, after = expected.get(key), observed.get(key)
                    if not _finite(before):
                        comparisons.append(dict(split=split, protocol=protocol_key, metric=metric,
                            status='INVALID_IN_REFERENCE', reference=before, actual=after))
                        continue
                    tolerance = max(1e-5, abs(before)*1e-4)
                    okay = _finite(after) and abs(after-before) <= tolerance
                    comparisons.append(dict(split=split, protocol=protocol_key, metric=metric,
                        reference=before, actual=after, tolerance=tolerance, within_tolerance=okay))
                    if not okay:
                        _write(directory/'point_parity_failure.json', comparisons)
                        raise ValueError('Clean point prediction reproduction exceeds the frozen tolerance; not a probability-quality veto')
            for metric in ('CRPS', 'marginal_NLL', 'joint_NLL_per_position', 'coverage_80', 'width_80'):
                before, after = expected.get(metric), observed.get(metric)
                diagnostic_gaps.append(dict(split=split, metric=metric, reference=before, actual=after,
                    delta=after-before if _finite(before) and _finite(after) else None,
                    decision='RECORDED_ONLY_NO_POINT_VETO'))
        reproduced.append(dict(dataset=dataset, method=case['method'], seed=case['seed'], protocol=case['protocol'],
            selection_objective=case['selection_objective'], checkpoint=case['checkpoint'],
            reference_receipt=case['reference_receipt'], reproduction_receipt=actual_receipt,
            exact_parameter_alias_of=None if prior is None else prior['receipt'],
            comparisons=comparisons, diagnostics=diagnostic_gaps, frozen_epoch=release['frozen_epoch']))
        _write(out/'official_reproduction_progress.json', reproduced)
    result = dict(status='PASS', optimizer_trajectories=sum(r['optimizer_trajectories'] for r in tiny),
        optimizer_steps=sum(r['optimizer_steps'] for r in tiny), public_sft_cli_checked=public_sft,
        public_weights_only_minimal_e2e=public_only,
        minimal_teacher_scope='Public pretrained interface teacher; original scientific teachers unchanged' if public_only else 'Supplied or default interface teacher, recorded per dataset',
        full_checkpoint_reproduction_requires_supplied_trained_artifacts=not interface_only,
        interface_only=interface_only,
        package_path=config['package_path'], system_site_packages=False, installation_prefix=str(environment),
        cluster_storage_environment_removed=True,
        scope=('Fresh public package: two bounded training inputs per dataset, '+('public-weight supervised CLI, ' if public_sft else '')+'rolling OPD then native RL and reload; no formal test or scientific performance reproduction'
            if interface_only else 'Fresh public package: public-weight supervised console command, rolling OPD and native RL per dataset, plus complete explicitly recommended-checkpoint validation/test reproduction'),
        minimal_e2e_receipts=tiny, official_reproduction_cases=reproduced,
        official_checkpoint_cases=len(reproduced), frozen_selection=config.get('frozen_selection'),
        independently_reproduced_parameter_states=len(evaluated_states),
        official_reproduction_scope=('NOT_EXECUTED_INTERFACE_ONLY' if interface_only else
            'All frozen P1/P2/P3 MSE/MAE recommendations, five seeds per available dataset/choice, at most sixty slots; identical parameter states share full validation/test execution'),
        unavailable_recipe_evidence=frozen.get('recipe_gaps', []),
        command_records=str(out/'command_records.json'), record_check_scope='Scalar OR/rounding/seed consistency only',
        model_numerical_reload_scope=('Actual tiny optimizer/RNG reload only; full recommended-checkpoint prediction evaluation remains a separate mandatory final gate'
            if interface_only else 'Actual tiny optimizer/RNG reload plus independent full recommended-checkpoint prediction evaluation'),
        test_used=bool(official),
        diagnostic_metrics_do_not_gate_points=True, test_selection=False, public_upload_performed=False,
        seconds=time.time()-began)
    _write(out/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['minimal-e2e', 'accept'], required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=101)
    args = parser.parse_args()
    function = minimal_e2e if args.stage == 'minimal-e2e' else run
    function(args.config, args.output, args.device, args.seed)
