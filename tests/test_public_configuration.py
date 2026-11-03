"""Model-free checks for the two published forecasting configurations."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dgf.configuration import load, native
from dgf.stage_contract import plan

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ['etth1_h192_crps.json', 'etth1_h336_mse.json']


class PublicConfiguration(unittest.TestCase):
    def test_two_explicit_complete_configs(self):
        files = sorted((ROOT / 'config').glob('*.json'))
        self.assertEqual([file.name for file in files], CONFIGS)
        for file in files:
            with self.subTest(config=file.name):
                c = load(file)
                self.assertEqual(c['format_version'], 1)
                self.assertEqual(set(c['random_seeds'].values()), {1117})
                r = native(c, 'rl', {'horizon': c['data']['horizon']},
                           Path('/model'), Path('/data'), Path('/work'))
                self.assertEqual(r['lr'], 1.5e-6)
                self.assertEqual(r['G'], 100)
                self.assertEqual(r['J'], 8)
                self.assertFalse(r['dgf_complete']['fixed_total_concentration'])
                self.assertTrue(all(r['dgf_complete'][key] for key in
                                    ['enable_dirichlet_objective', 'enable_dynamic', 'enable_diversity']))
                self.assertNotIn('adaptive_dgf', r)
                self.assertEqual(c['evaluation']['selection'], 'validation_MAP_MSE_best')

    def test_source_is_parseable(self):
        for file in (ROOT / 'src/dgf').glob('*.py'):
            ast.parse(file.read_text(), filename=str(file))

    def test_metadata_cli_is_model_free(self):
        # A fresh process isolates this check from numerical tests imported by discovery.
        code = '''import runpy, sys
sys.argv = sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as error:
    if error.code not in (None, 0):
        raise
assert not any(name.split('.')[0] in {'torch', 'lightning'} for name in sys.modules)
'''
        commands = [(['--help'], None)]
        for name in CONFIGS:
            commands.extend((([command, '--config', str(ROOT / 'config' / name)], command)
                             for command in ['export-config', 'plan']))
        for arguments, command in commands:
            with self.subTest(arguments=arguments):
                result = subprocess.run([sys.executable, '-c', code,
                                         str(ROOT / 'scripts/run.py'), *arguments],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                if command == 'export-config':
                    self.assertEqual(json.loads(result.stdout)['format_version'], 1)
                elif command == 'plan':
                    self.assertEqual([row['stage'] for row in json.loads(result.stdout)],
                                     ['sft', 'preparation', 'rl'])
                else:
                    self.assertIn('export-config', result.stdout)

    def test_unsupported_profile_is_rejected(self):
        c = load(ROOT / 'config' / CONFIGS[0])
        for change in [{'format_version': 2}, {'data': dict(c['data'], horizon=720)}]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                edited = copy.deepcopy(c)
                edited.update(change)
                path = Path(folder) / 'unsupported.json'
                path.write_text(json.dumps(edited))
                with self.assertRaises(ValueError):
                    load(path)

    def test_plan_preserves_stage_transfers(self):
        for name in CONFIGS:
            rows = plan(load(ROOT / 'config' / name))
            self.assertEqual([row['epochs'] for row in rows], [10, 1, 4])
            self.assertEqual([row['selection'] for row in rows],
                             ['validation_MAP_MSE_best', 'final', 'validation_MAP_MSE_best'])
            self.assertEqual(rows[2]['parent'], 'preparation.final')


if __name__ == '__main__':
    unittest.main()
