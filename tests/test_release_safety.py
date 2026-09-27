"""Small release safety regressions; no dataset, checkpoint, or GPU is used."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts.confirmatory.resume_integrity import (
    expected_identity, raw_metadata, validate_cached_raw,
    verify_or_save_npz, verify_or_write_json,
)


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


class ReleaseSafetyTests(unittest.TestCase):
    def test_analyze_help_has_no_output_or_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder) / 'new-run'
            run_dir.mkdir()
            sentinel = run_dir / 'unchanged.txt'
            sentinel.write_text('existing output sentinel\n', encoding='utf-8')
            before = sentinel.read_bytes()
            env = {**os.environ, 'ORBIT_RUN_DIR': str(run_dir), 'PYTHONDONTWRITEBYTECODE': '1'}
            cmd = [sys.executable, str(ROOT / 'scripts/confirmatory/analyze_confirmatory.py'), '--help']
            completed = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('usage:', completed.stdout)
            self.assertEqual(sentinel.read_bytes(), before)
            self.assertEqual([p.name for p in run_dir.iterdir()], ['unchanged.txt'])

    def test_cached_raw_rejects_identity_and_metadata_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            split = folder / 'split.csv'
            split.write_text('sample_id,source_split,original_index,label,role\n'
                             'cifar10/train/2,train,2,4,cal\n', encoding='utf-8')
            identity = expected_identity('cal', {}, split)
            meta = raw_metadata('cal', identity, 'resnet18', 3, 'checkpoint', 'config', False)
            path, side = folder / 'cal.npz', folder / 'cal.json'
            save_npz(path, label=np.array([4]), original_index=np.array([2]),
                     features=np.zeros((1, 2)), raw_output=np.zeros((1, 2)),
                     prediction=np.array([1]), d_gop=np.array([0.2]),
                     evidence=np.ones((1, 2)), probability=np.full((1, 2), 0.5))
            side.write_text(json.dumps({**meta, 'sha256': sha(path)}), encoding='utf-8')
            self.assertEqual(len(validate_cached_raw(path, side, meta, identity, sha)), 8)
            with self.assertRaisesRegex(RuntimeError, 'seed'):
                validate_cached_raw(path, side, {**meta, 'seed': 4}, identity, sha)
            changed = {**identity, 'label': np.array([5])}
            with self.assertRaisesRegex(RuntimeError, 'label'):
                validate_cached_raw(path, side, raw_metadata('cal', changed, 'resnet18', 3, 'checkpoint', 'config', False), changed, sha)
            save_npz(path, label=np.array([9]), original_index=np.array([2]),
                     features=np.zeros((1, 2)), raw_output=np.zeros((1, 2)),
                     prediction=np.array([1]), d_gop=np.array([0.2]),
                     evidence=np.ones((1, 2)), probability=np.full((1, 2), 0.5))
            with self.assertRaisesRegex(RuntimeError, 'hash'):
                validate_cached_raw(path, side, meta, identity, sha)

    def test_reference_and_score_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            reference, side = folder / 'references.npz', folder / 'references.json'
            meta = {'dataset': 'CIFAR10/cal+train', 'backbone': 'wrn28_10',
                    'seed': 5, 'checkpoint_hash': 'a', 'config_hash': 'b',
                    'calibration_hash': 'c', 'train_hash': 'd',
                    'reference_fields': ['class_0']}
            arrays = {'class_0': np.array([0.1, 0.2])}
            self.assertEqual(verify_or_save_npz(reference, arrays, save_npz, sha, side, meta), 'saved')
            self.assertEqual(verify_or_save_npz(reference, arrays, save_npz, sha, side, meta), 'verified')
            with self.assertRaisesRegex(RuntimeError, 'checkpoint_hash'):
                verify_or_save_npz(reference, arrays, save_npz, sha, side, {**meta, 'checkpoint_hash': 'wrong'})
            with self.assertRaisesRegex(RuntimeError, 'class_0'):
                verify_or_save_npz(reference, {'class_0': np.array([0.1, 0.3])}, save_npz, sha, side, meta)
            score = folder / 'score.npz'
            record = {'sample_id': np.array(['id/0']), 'ground_truth': np.array([2]),
                      'dataset': np.array('CIFAR10'), 'backbone': np.array('resnet18'),
                      'seed': np.array(3), 'method': np.array('Orbit_GOP'),
                      'prediction_field': np.array('base_prediction'),
                      'score_field': np.array('score'), 'base_prediction': np.array([2]),
                      'score': np.array([0.25]), 'checkpoint_hash': np.array('a'),
                      'config_hash': np.array('b')}
            verify_or_save_npz(score, record, save_npz)
            for key, replacement in [('sample_id', np.array(['id/1'])),
                                     ('ground_truth', np.array([3])),
                                     ('dataset', np.array('CIFAR100')),
                                     ('seed', np.array(4)),
                                     ('method', np.array('Global_EDL')),
                                     ('score', np.array([0.30]))]:
                with self.assertRaisesRegex(RuntimeError, key):
                    verify_or_save_npz(score, {**record, key: replacement}, save_npz)

    def test_saved_json_is_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'calibration.json'
            self.assertEqual(verify_or_write_json(path, {'fit': 1}), 'saved')
            self.assertEqual(verify_or_write_json(path, {'fit': 1}), 'verified')
            with self.assertRaisesRegex(RuntimeError, 'mismatch'):
                verify_or_write_json(path, {'fit': 2})


if __name__ == '__main__':
    unittest.main()
