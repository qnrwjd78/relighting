from pathlib import Path
import os
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/download_weights.sh'


class DownloadWeightsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'destination'

    def run_script(self, *args, env=None):
        return subprocess.run(['bash', str(SCRIPT), '--root', str(self.root), *args],
                              capture_output=True, text=True, env=env, timeout=30)

    def test_dry_run_lists_every_group_without_writing(self):
        result = self.run_script('--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ['Wan-AI/', 'Ruicheng/', 'geshang/', 'SAM ViT-B', 'EfficientNet', 'CLIP', 'SBU']:
            self.assertIn(name, result.stdout)
        self.assertFalse(self.root.exists())

    def test_missing_weights_fail_offline_without_creating_files(self):
        result = self.run_script('--only', 'wan', '--check')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('INCOMPLETE', result.stderr)
        self.assertFalse(self.root.exists())

    def test_pointer_file_is_not_accepted_as_a_checkpoint(self):
        file = self.root / 'weights/moge-3-vitl/model.pt'
        file.parent.mkdir(parents=True)
        file.write_text('version https://git-lfs.github.com/spec/v1\n' + 'x' * 2048)
        result = self.run_script('--only', 'moge', '--check')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Not a model binary', result.stderr)

    def test_corrupted_direct_weights_and_missing_sbu_are_reported(self):
        file = self.root / 'external/AdapterShadow/checkpoint/sam/sam_vit_b_01ec64.pth'
        file.parent.mkdir(parents=True)
        file.write_bytes(b'x' * 2048)
        result = self.run_script('--only', 'adapter', '--check')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SAM ViT-B', result.stderr)
        self.assertIn('SBU unavailable', result.stderr)
        self.assertEqual(file.read_bytes(), b'x' * 2048)

    def test_hub_download_checks_size_and_lfs_hash(self):
        modules = Path(self.temp.name) / 'modules'
        modules.mkdir()
        (modules / 'huggingface_hub.py').write_text('''
from types import SimpleNamespace
from pathlib import Path
import hashlib, os
DATA = b'mock-weight' * 256
class HfApi:
    def model_info(self, repo, revision, files_metadata):
        assert repo == 'Ruicheng/moge-3-vitl'
        assert revision == '184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be'
        return SimpleNamespace(siblings=[SimpleNamespace(rfilename='model.pt', size=len(DATA),
            lfs=SimpleNamespace(sha256=hashlib.sha256(DATA).hexdigest()))])
def snapshot_download(repo, revision, local_dir, allow_patterns):
    assert allow_patterns == ['model.pt']
    path = Path(local_dir); path.mkdir(parents=True, exist_ok=True)
    data = b'bad-weights' * 200 if os.environ.get('BAD_DOWNLOAD') else DATA
    (path / 'model.pt').write_bytes(data)
''')
        env = {**os.environ, 'PYTHONPATH': str(modules)}
        result = self.run_script('--only', 'moge', env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.run_script('--only', 'moge', env={**env, 'BAD_DOWNLOAD': '1'})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Size mismatch', result.stderr)
