"""Download the public inputs to caller-owned directories outside this repository."""
import argparse,json,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--data-dir',required=True);p.add_argument('--model-dir',required=True);a=p.parse_args()
c=json.loads(Path(a.config).read_text());data=Path(a.data_dir).expanduser().resolve();model=Path(a.model_dir).expanduser().resolve();repo=Path(__file__).resolve().parents[1]
if data.is_relative_to(repo)or model.is_relative_to(repo):raise ValueError('Keep inputs outside the repository')
data.mkdir(parents=True,exist_ok=True);target=data/'ETTh1.csv';temporary=data/'ETTh1.csv.partial'
if not target.exists():
    urllib.request.urlretrieve(c['data']['source_url'],temporary);temporary.replace(target)
from huggingface_hub import snapshot_download
snapshot_download(repo_id=c['model']['id'],revision=c['model']['revision'],local_dir=str(model),allow_patterns=['config.json','*.safetensors','LICENSE*'])
print(json.dumps(dict(csv=str(target),model_dir=str(model))))
