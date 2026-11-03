"""Fresh isolated environment acceptance, executed inside a new HOPE worker.

No system site packages, no inherited PYTHONPATH, public package index only.
The dependency closure is pinned before install and kept with the raw installer
log. A successful installation alone never produces an acceptance receipt.
"""
import importlib.metadata as metadata
import json,os,subprocess,sys,tempfile,time,tomllib,venv
import zipfile
from pathlib import Path


def extract_public_package(archive,destination):
    """Extract a bounded, regular-file-only public package into worker scratch."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as package:
        members=package.infolist()
        if sum(m.file_size for m in members)>25_000_000:raise ValueError('Unexpectedly large code package')
        for member in members:
            path=Path(member.filename)
            if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0]!='moirai-research':
                raise ValueError('Unsafe public package member')
            if member.is_dir() or ((member.external_attr>>16)&0o170000)==0o120000:
                raise ValueError('Public package must contain only regular files')
        if len({m.filename for m in members})!=len(members):raise ValueError('Duplicate public package member')
        package.extractall(destination)
    root=destination/'moirai-research'
    if not (root/'pyproject.toml').is_file() or not (root/'src/moirai_research/cli.py').is_file():
        raise ValueError('Incomplete public package')
    return root


def dependency_lock(project):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    roots=tomllib.loads((project/'pyproject.toml').read_text())['project']['dependencies']+['setuptools>=68','wheel']
    pending=[Requirement(x) for x in roots];seen={};versions={}
    while pending:
        req=pending.pop();name=canonicalize_name(req.name);extras=set(req.extras)|{''}
        if req.url:raise ValueError('Direct or private dependency URLs are not accepted')
        fresh=extras-seen.get(name,set())
        if not fresh:continue
        seen.setdefault(name,set()).update(extras);dist=metadata.distribution(req.name);version=dist.version
        if req.specifier and version not in req.specifier:raise ValueError(f'Installed reference {name} does not satisfy declared package range')
        versions[name]=version
        for item in dist.requires or []:
            child=Requirement(item)
            if child.marker is None or any(child.marker.evaluate({'extra':x}) for x in fresh):pending.append(child)
    return '\n'.join(f'{name}=={version}' for name,version in sorted(versions.items()))+'\n'


def run(config_path,output,device='cuda',seed=1):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);config=json.loads(Path(config_path).read_text())
    scratch=Path(tempfile.mkdtemp(prefix='moirai-clean-'))
    project=extract_public_package(config['package_archive'],scratch/'package') if config.get('package_archive') else Path(__file__).resolve().parents[2]
    began=time.time();lock=out/'requirements.lock'
    reference_lock=dependency_lock(project)
    if config.get('pinned_requirements'):
        pinned=Path(config['pinned_requirements']).read_text()
        if pinned!=reference_lock:raise ValueError('Offline installation lock differs from frozen reference dependency closure')
    lock.write_text(reference_lock)
    environment=scratch/'venv';venv.EnvBuilder(with_pip=True,system_site_packages=False).create(environment)
    python=str(environment/'bin/python');env=os.environ.copy()
    for key in ['PYTHONPATH','PYTHONHOME','PIP_EXTRA_INDEX_URL','PIP_INDEX_URL','PIP_TRUSTED_HOST']:env.pop(key,None)
    env.update(VIRTUAL_ENV=str(environment),PATH=str(environment/'bin')+os.pathsep+env['PATH'],PYTHONNOUSERSITE='1',PIP_CONFIG_FILE=os.devnull,
        HF_HOME=str(scratch/'cache/huggingface'),HF_DATASETS_CACHE=str(scratch/'cache/datasets'),PIP_CACHE_DIR=str(scratch/'cache/pip'),HF_HUB_DISABLE_TELEMETRY='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    def execute(args,logname,timeout):
        with (out/logname).open('w') as stream:
            subprocess.run(args,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=timeout,cwd=scratch)
    index=['--no-index','--find-links',config['wheelhouse'],'--disable-pip-version-check'] if config.get('wheelhouse') else ['--index-url','https://pypi.org/simple','--disable-pip-version-check']
    execute([python,'-m','pip','install',*index,'-r',str(lock)],'dependency_install.log',5400)
    execute([python,'-m','pip','install',*index,'--no-deps','--no-build-isolation',str(project)],'package_install.log',600)
    execute([python,'-m','pip','check'],'pip_check.log',120)
    configuration_test_logs=[]
    for filename in ['test_public_configuration_metadata.py','test_auxiliary_configuration_metadata.py','test_readout_configuration_metadata.py']:
        if (project/'tests'/filename).is_file():
            logname=filename.removesuffix('.py')+'.log'
            execute([python,'-m','unittest','discover','-s',str(project/'tests'),'-p',filename],logname,120)
            configuration_test_logs.append(str(out/logname))
    cpu_receipt=out/'cpu_group_interface.json'
    execute([python,'-c',
        'import sys; from moirai_research.rollout_diagnostics import numerical_checks; from moirai_research.runtime import atomic_json; '
        'atomic_json(sys.argv[1],dict(status="PASS",checks=numerical_checks("cpu"),optimizer_trajectories=0,optimizer_steps=0,'
        'scope="CPU population-variance, constant-group and dtype/reduction interfaces only; no model or optimizer"))',
        str(cpu_receipt)],'cpu_group_interface.log',120)
    execute([python,'-m','moirai_research.cli','qualification-e2e','--config',str(Path(config_path).resolve()),'--output',str(out/'e2e'),
             '--device',device,'--seed',str(seed)],'e2e.log',7200)
    receipt=json.loads((out/'e2e/result.json').read_text());assert receipt['status']=='PASS'
    assert str(environment) in receipt['environment']['prefix'] and str(environment) in receipt['environment']['package_file']
    result=dict(status='PASS',scope='Fresh isolated package installation and small-budget public-data SFT/RL/reload interface acceptance; not formal research performance',
        cache_scope='worker-local scratch; shared public input cache is read-only',config=config,dependency_lock=str(lock),e2e_receipt=str(out/'e2e/result.json'),optimizer_trajectories=2,
        package_archive=config.get('package_archive'),installation_source='explicit public code archive' if config.get('package_archive') else 'frozen worker source',
        configuration_test_logs=configuration_test_logs,
        cpu_group_interface_receipt=str(cpu_receipt),
        actual_gpu=receipt['actual_gpu'],seconds=time.time()-began,system_site_packages=False,installation_mode='offline public wheels' if config.get('wheelhouse') else 'public index',external_public_index='https://pypi.org/simple')
    path=out/'result.json';tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(result,indent=2));tmp.replace(path);return result
