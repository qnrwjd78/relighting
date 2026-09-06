from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TrainingLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT / 'scripts/lib', self.root / 'scripts/lib')
        for path in (ROOT / 'scripts').glob('launch_exp*.sh'):
            shutil.copy2(path, self.root / 'scripts' / path.name)
        self.log = self.root / 'calls.jsonl'
        for name in ('python-stub', 'accelerate-stub'):
            stub = self.root / name
            stub.write_text('''#!/usr/bin/python3
import json, os, sys
with open(os.environ['CALL_LOG'], 'a') as handle:
    handle.write(json.dumps({'tool': os.path.basename(sys.argv[0]), 'args': sys.argv[1:],
        'cwd': os.getcwd(), 'gpus': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'allocator': os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}) + '\\n')
if os.path.basename(sys.argv[0]) == 'python-stub':
    sys.exit(int(os.environ.get('PREFLIGHT_EXIT', '0')))
''')
            stub.chmod(0o755)
        self.env = {**os.environ, 'CALL_LOG': str(self.log),
                    'PYTHON_BIN': str(self.root / 'python-stub'),
                    'ACCELERATE_BIN': str(self.root / 'accelerate-stub')}
        self.env.pop('PYTORCH_CUDA_ALLOC_CONF', None)

    def run_launcher(self, name, **env):
        return subprocess.run(['bash', str(self.root / 'scripts' / name)],
                              cwd='/', env={**self.env, **env}, capture_output=True, text=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_exp1_variants_keep_config_and_batch_contract(self):
        for suffix, batch, nogc in [('', 5, False), ('_nogc', 5, True), ('_b4_nogc', 4, True)]:
            with self.subTest(suffix=suffix):
                self.log.write_text('')
                result = self.run_launcher(f'launch_exp1_shadow_mask_8gpu{suffix}.sh')
                self.assertEqual(result.returncode, 0, result.stderr)
                preflight, train = self.calls()
                config = f'configs/train_480/exp1_7x7x5_power06_rgb_shadow_mask_vae_scene64_15ep_b{batch}x8_ga1_gb{batch*8}' + ('_nogc' if nogc else '') + '.json'
                extra = ['--expected-global-batch', '32'] if batch == 4 else []
                self.assertEqual(preflight['args'], ['scripts/preflight_exp1_shadow_mask_training.py', '--config', config, *extra])
                self.assertEqual(train['args'], ['launch', '--config_file', 'configs/accelerate_8gpu_ddp.yaml', 'model/train_tokenlight_scene_cache_shadow_safe_retained.py', '--config', config])
                self.assertEqual(train['cwd'], str(self.root))
                self.assertEqual(train['allocator'], 'expandable_segments:True' if batch == 4 else None)

    def test_exp2_variants_keep_device_and_preflight_contract(self):
        for size, devices in [(4, '4,5,6,7'), (8, '0,1,2,3,4,5,6,7')]:
            with self.subTest(size=size):
                self.log.write_text('')
                result = self.run_launcher(f'launch_exp2_joint_rgb_shadow_mask_{size}gpu.sh')
                self.assertEqual(result.returncode, 0, result.stderr)
                preflight, train = self.calls()
                config = f'configs/train_480/exp2_7x7x5_power06_rgb_joint_shadow_mask_clean_20ep_b5x{size}_ga{2 if size == 4 else 1}_gb40.json'
                self.assertEqual(preflight['args'], ['model/train_tokenlight_joint_mask.py', '--config', config, '--preflight', '--preflight_max_samples', '128'])
                self.assertEqual(train['args'], ['launch', '--config_file', f'configs/accelerate_{size}gpu_ddp.yaml', 'model/train_tokenlight_joint_mask.py', '--config', config])
                self.assertEqual(train['gpus'], devices)

    def test_activated_environment_executables_are_used_by_default(self):
        for source, target in [('python-stub', 'python'), ('accelerate-stub', 'accelerate')]:
            shutil.copy2(self.root / source, self.root / target)
        env = dict(self.env)
        env.pop('PYTHON_BIN')
        env.pop('ACCELERATE_BIN')
        env['PATH'] = str(self.root) + os.pathsep + os.environ['PATH']
        for name in ['launch_exp1_shadow_mask_8gpu.sh', 'launch_exp2_joint_rgb_shadow_mask_4gpu.sh']:
            with self.subTest(name=name):
                self.log.write_text('')
                result = subprocess.run(['bash', str(self.root / 'scripts' / name)],
                                        cwd='/', env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual([call['tool'] for call in self.calls()], ['python', 'accelerate'])

    def test_failed_preflight_does_not_start_training(self):
        for name in ['launch_exp1_shadow_mask_8gpu.sh', 'launch_exp2_joint_rgb_shadow_mask_4gpu.sh']:
            with self.subTest(name=name):
                self.log.write_text('')
                result = self.run_launcher(name, PREFLIGHT_EXIT='7')
                self.assertEqual(result.returncode, 7)
                self.assertEqual(len(self.calls()), 1)

    def test_resume_requires_checkpoint_before_preflight(self):
        name = 'launch_exp2_joint_rgb_shadow_mask_resume_e3_gpu4567.sh'
        result = self.run_launcher(name)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Missing continuation checkpoint', result.stderr)
        self.assertFalse(self.log.exists())
        checkpoint = self.root / 'outputs/train/exp_2/rgb_joint_shadow_mask_cleancond_7x7x5_power06_scene64_fresh_20ep_b5x8_ga1_gb40_20260831_161445/epoch-3.safetensors'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        result = self.run_launcher(name)
        self.assertEqual(result.returncode, 0, result.stderr)
        preflight, train = self.calls()
        self.assertEqual(preflight['args'][2], 'configs/train_480/exp2_7x7x5_power06_rgb_joint_shadow_mask_clean_resume_e3_to20_b5x4_ga2_gb40.json')
        self.assertEqual(train['args'][-1], preflight['args'][2])
        self.assertEqual(train['gpus'], '4,5,6,7')
