"""Regression proof: four concurrent builds have disjoint writable roots."""
from concurrent.futures import ThreadPoolExecutor
import json,shutil,subprocess,sys
from pathlib import Path
from .runtime import atomic_json


def qualify_install(package_source,wheels,output):
    source=Path(package_source);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    scratch=out.parent.parent/'parallel_install_work';scratch.mkdir(exist_ok=True)
    def install(index):
        directory=scratch/str(index);directory.mkdir();package=directory/'private_build_source'
        shutil.copytree(source,package,ignore=shutil.ignore_patterns('build','*.egg-info','__pycache__','*.pyc'))
        python=directory/'venv/bin/python';commands=[
            [sys.executable,'-m','venv','--system-site-packages',str(directory/'venv')],
            [str(python),'-m','pip','install','--no-index','--find-links',str(wheels),'--no-build-isolation',str(package)]]
        for command in commands:
            result=subprocess.run(command,capture_output=True,text=True,timeout=600)
            if result.returncode:
                (out/f'install_{index}_error.txt').write_text((result.stdout+'\n'+result.stderr)[-10000:]);raise RuntimeError('Parallel private install failed')
        program="import pathlib,json,moirai_research;source=pathlib.Path("+repr(str(package/'src'))+");installed=pathlib.Path(moirai_research.__file__).parent.parent;files=list(source.rglob('*.py'));bad=[str(p.relative_to(source)) for p in files if not (installed/p.relative_to(source)).is_file() or p.read_bytes()!=(installed/p.relative_to(source)).read_bytes()];assert not bad,bad;print(json.dumps(dict(files_checked=len(files),status='EXACT_SOURCE_BYTES_MATCH')))"
        result=subprocess.run([str(python),'-c',program],capture_output=True,text=True,timeout=120)
        if result.returncode:
            (out/f'identity_{index}_error.txt').write_text(result.stderr[-5000:]);raise RuntimeError('Installed source differs from the frozen input')
        return dict(index=index,private_build_root=str(package),commands=commands,identity=json.loads(result.stdout),status='PASS')
    with ThreadPoolExecutor(max_workers=4) as pool:cases=list(pool.map(install,range(4)))
    assert len({c['private_build_root'] for c in cases})==4
    result=dict(status='PASS',cases=cases,optimizer_updates=0,optimizer_trajectories=0,
        scope='Concurrent offline installation and complete research/vendor Python source equality; no scientific training',test_used=False)
    atomic_json(out/'result.json',result);return result
