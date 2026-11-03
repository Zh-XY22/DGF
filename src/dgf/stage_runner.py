import json,copy,datetime
from pathlib import Path
from moirai_research.configuration import resolve_checkpoint
from moirai_research.runtime import atomic_json

def run_stage(config, file, directory, runner, seed, resume_allowed):
    expected = resolve_checkpoint(config) if config['phase'] != 'supervised' else config
    result_file = directory / 'result.json'
    if result_file.is_file():
        prior = json.loads(result_file.read_text())
        if prior.get('status') in ['PASS', 'DIAGNOSTIC_ONLY']:
            if prior['config'] != expected:
                raise ValueError('Completed stage configuration differs; never overwrite it')
            required = {prior[key] for key in ['best_checkpoint', 'continuation_checkpoint'] if prior.get(key)}
            for group in ['readouts', 'point_selection', 'probability_selection']:
                required.update((value['checkpoint'] for value in prior.get(group, {}).values() if isinstance(value, dict) and value.get('checkpoint')))
            if any((not Path(path).is_file() for path in required)):
                raise FileNotFoundError('Completed stage references missing original checkpoint artifacts')
            return prior
    continuation = directory / 'resume.pt'
    if continuation.is_file():
        if not resume_allowed:
            raise ValueError('Existing state requires explicit --resume')
        config = dict(config, resume=str(continuation))
    elif any(directory.glob('*curve.json')):
        raise ValueError('Partial training exists without complete recovery state')
    atomic_json(file, config)
    runner(file, directory, 'cuda', seed)
    result = json.loads(result_file.read_text())
    if result['status'] == 'INTERRUPTED_WITH_STATE':
        raise SystemExit('Interrupted with complete state; restore the working directory and use --resume')
    if result['status'] not in ['PASS', 'DIAGNOSTIC_ONLY']:
        raise ValueError('Stage did not complete')
    return result
