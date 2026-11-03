"""Local IO and phase orchestration around the existing numerical implementation."""
import copy,json,datetime
from pathlib import Path
from .configuration import native
from .stage_runner import run_stage
from .evaluation_contract import materialize
from moirai_research.dgf_initialization import training_protocol,initialize
from moirai_research.supervised import fit
from moirai_research.train import train
from moirai_research.runtime import atomic_json
from moirai_research.public_chain_metadata import input_binding,content_tag


def execute(args,c):
    work=Path(args.output_dir).expanduser().resolve();csv=Path(args.csv).expanduser().resolve();model=Path(args.model_dir).expanduser().resolve()
    if not csv.is_file()or not(model/'config.json').is_file():raise FileNotFoundError('Provide the public CSV and complete local pretrained snapshot')
    source=Path(__file__).resolve().parent
    repo=source.parents[1]if source.parent.name=='src'else None
    if repo and any(x.is_relative_to(repo)for x in [work,csv,model]):raise ValueError('Data, model and outputs must be outside the source repository')
    binding=input_binding(csv,model,source,dict(configuration=c,qualification_only=args.smoke))
    import moirai_research,uni2ts
    binding['backend']={}
    for package in [moirai_research,uni2ts]:
        folder=Path(package.__file__).resolve().parent
        binding['backend'][package.__name__]={str(f.relative_to(folder)):content_tag(f)for f in sorted(folder.rglob('*.py'))}
    work.mkdir(parents=True,exist_ok=True);saved=work/'.entry_binding.json'
    if saved.exists():
        if json.loads(saved.read_text())!=binding:raise ValueError('Inputs, source or settings differ from this working directory')
        if args.command=='train'and not args.resume:raise FileExistsError('Use --resume to continue an existing chain')
    else:
        if any(work.iterdir()):raise FileExistsError('Use a fresh output directory')
        atomic_json(saved,binding)
    atomic_json(work/'resolved_config.json',dict(c,execution=dict(qualification_only=args.smoke,model_dir=str(model),csv=str(csv),output_dir=str(work))))
    data=work/'data'
    if not(data/'data_protocol.json').is_file():
        if args.command not in ['prepare','train']:raise FileNotFoundError('Run prepare first')
        protocol=training_protocol(csv,data,c['data']['horizon'],c['data']['revision'])
    else:protocol=json.loads((data/'data_protocol.json').read_text())
    expected=dict(dataset=c['data']['dataset'],mode=c['data']['mode'],context=c['data']['context'],patch=c['data']['patch'],horizon=c['data']['horizon'],variables=c['data']['variables'],ddof=c['data']['scaler']['ddof'])
    if any(protocol.get(k)!=v for k,v in expected.items()):raise ValueError('Prepared data contract differs')
    prepared={str(f.relative_to(data)):content_tag(f)for f in sorted(data.rglob('*'))if f.is_file()and f.suffix in ['.json','.npy','.arrow','.csv']}
    marker=work/'.prepared_data_binding.json'
    if marker.exists()and json.loads(marker.read_text())!=prepared:raise ValueError('Prepared data changed; recovery rejected')
    atomic_json(marker,prepared)
    if args.command=='prepare':return
    def stage(name):
        config=native(c,name,protocol,model,data,work)
        if args.smoke:
            config.update(qualification_only=True,global_input_batch=2,microbatch=1,diagnostic_input_limit=2,diagnostic_validation_limit=2)
            if name=='sft':config.update(fixed_epochs=1,max_epochs=1)
            else:config['epochs']=1
        folder=work/('scientific'if name=='rl'else name)
        return run_stage(config,work/(name+'.json'),folder,fit if name=='sft'else train,c['random_seeds'][name],args.resume)
    if args.command in ['train','sft']:stage('sft')
    if args.command=='sft':return
    if args.command in ['train','initialize']:
        result=json.loads((work/'sft/result.json').read_text());parent=result['readouts']['P1_val_MSE_best']['checkpoint']
        target=work/'initialization/constructed.pt'
        if not target.is_file():initialize(str(model),parent,str(target.parent),c['random_seeds']['initialization'],c['model']['revision'])
    if args.command=='initialize':return
    if args.command in ['train','prepare-modes']:stage('preparation')
    if args.command=='prepare-modes':return
    if args.command in ['train','rl']:stage('rl')
    if args.command in ['train','rl']:
        results={name:json.loads((work/folder/'result.json').read_text())for name,folder in [('sft','sft'),('preparation','preparation'),('rl','scientific')]}
        summary=dict(status='QUALIFICATION_ONLY'if args.smoke else'TRAINING_COMPLETE_TEST_SEALED',from_public_weights=True,test_used=False,phase_updates={name:r['optimizer_steps']if name=='sft'else r['cursor']['global_step']for name,r in results.items()},selection='validation_MAP_MSE_best',qualification_only=args.smoke)
        atomic_json(work/'training_receipt.json',summary);print(json.dumps(summary));return
    if args.command=='evaluate':evaluate(args,c,work,data,model)


def evaluate(args,c,work,data,model):
    from moirai_research.evaluation import evaluate_config
    from moirai_research.probability_evaluation import evaluate_config as evaluate_distribution
    training=work/'scientific/result.json'
    if not training.is_file():raise FileNotFoundError('A completed scientific training result is required for validation selection')
    result=json.loads(training.read_text());chosen=result['readouts']['P1_val_MSE_best'];checkpoint=args.checkpoint or chosen['checkpoint']
    if args.checkpoint and (not Path(chosen['checkpoint']).is_file() or content_tag(checkpoint)!=content_tag(chosen['checkpoint'])):raise ValueError('CHECKPOINT must be the validation-selected state or an identical local copy')
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');destination=work/'evaluation'/stamp
    destination.mkdir(parents=True)
    for rng in c['evaluation']['probability_seeds']:
        folder=destination/str(rng)
        materialize(training,data,model,folder,'P1_val_MSE_best',checkpoint,rng,qualification=args.smoke)
        if args.smoke:
            # Qualification is explicitly limited; never publish a complete-window receipt.
            for name in ['point.json','probability.json']:
                file=folder/name;value=json.loads(file.read_text());value['diagnostic_limit']=2;atomic_json(file,value)
        if rng==c['evaluation']['probability_seeds'][0]:evaluate_config(folder/'point.json',destination/'point','cuda',c['random_seeds']['rl'])
        evaluate_distribution(folder/'probability.json',folder/'probability','cuda',c['random_seeds']['rl'])
    receipt=dict(status='QUALIFICATION_ONLY'if args.smoke else'EVALUATION_COMPLETE',checkpoint=str(checkpoint),selection='validation_MAP_MSE_best',selected_initial=Path(chosen['checkpoint']).name=='initial.pt',point_directory=str(destination/'point'),probability_samples=1000,probability_seeds=c['evaluation']['probability_seeds'],all_results=str(destination))
    atomic_json(destination/'evaluation_receipt.json',receipt);print(json.dumps(receipt))
