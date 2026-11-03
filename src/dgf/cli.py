"""Public CLI; numerical modules are imported only by execution commands."""
import argparse,json,os
from pathlib import Path
from .configuration import load


def main():
    p=argparse.ArgumentParser(prog='dgf',description='Complete Dirichlet-guided forecasting')
    p.add_argument('command',choices=['export-config','prepare','sft','initialize','prepare-modes','rl','train','evaluate','plan'])
    p.add_argument('--config',required=True);p.add_argument('--output-dir',default=os.environ.get('OUTPUT_DIR'))
    p.add_argument('--csv',default=os.environ.get('ETTH1_CSV'));p.add_argument('--model-dir',default=os.environ.get('MOIRAI_MODEL_DIR'))
    p.add_argument('--checkpoint',default=os.environ.get('CHECKPOINT'));p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true',help='Explicit short qualification only: one update per phase, two validation/test windows')
    a=p.parse_args();config=load(a.config)
    if a.command=='plan':
        from .stage_contract import plan
        print(json.dumps(plan(config),indent=2));return
    if a.command=='export-config':
        text=json.dumps(config,indent=2)+'\n'
        if a.output_dir:
            output=Path(a.output_dir).expanduser().resolve();output.mkdir(parents=True,exist_ok=True);(output/'resolved_config.json').write_text(text)
        print(text);return
    if not a.output_dir or not a.csv or not a.model_dir:p.error('Set OUTPUT_DIR, ETTH1_CSV and MOIRAI_MODEL_DIR, or provide their CLI flags')
    from .pipeline import execute
    execute(a,config)
