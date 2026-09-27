"""Fail-closed checks for artefacts reused by a new confirmatory run."""

import csv
import hashlib
import json

import numpy as np


def _digest(values):
    payload = json.dumps(np.asarray(values).tolist(), ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def expected_identity(role, manifest, split_csv):
    if role in ('train', 'cal'):
        with open(split_csv, newline='', encoding='utf-8') as stream:
            rows = [row for row in csv.DictReader(stream) if row['role'] == role]
        return {
            'dataset': 'CIFAR10/train',
            'sample_id': np.asarray([row['sample_id'] for row in rows]),
            'original_index': np.asarray([int(row['original_index']) for row in rows]),
            'label': np.asarray([int(row['label']) for row in rows]),
        }
    if role == 'Tiny':
        rows = manifest['Tiny']['rows']
        return {
            'dataset': 'Tiny',
            'sample_id': np.asarray([row['sample_id'] for row in rows]),
            'original_index': np.arange(len(rows)),
            'label': np.asarray([int(row['label']) for row in rows]),
        }
    if role in ('ID', 'CIFAR100'):
        record = manifest[role]
        return {
            'dataset': record['dataset'],
            'sample_id': np.asarray(record['sample_ids']),
            'original_index': np.asarray(record['indices']),
            'label': np.asarray(record['labels']),
        }
    raise ValueError('Unknown dataset role: ' + role)


def raw_metadata(role, identity, backbone, seed, checkpoint_hash, config_hash, full):
    return {
        'role': role,
        'dataset': identity['dataset'],
        'backbone': backbone,
        'seed': seed,
        'checkpoint_hash': checkpoint_hash,
        'config_hash': config_hash,
        'full': full,
        'sample_id_sha256': _digest(identity['sample_id']),
        'original_index_sha256': _digest(identity['original_index']),
        'label_sha256': _digest(identity['label']),
    }


def validate_cached_raw(path, side, metadata, identity, sha):
    if not path.is_file() or not side.is_file():
        raise RuntimeError('Incomplete raw cache; refusing resume: ' + str(path))
    saved = json.loads(side.read_text(encoding='utf-8'))
    for key, value in metadata.items():
        if saved.get(key) != value:
            raise RuntimeError('Raw cache metadata mismatch (' + key + '): ' + str(path))
    if saved.get('sha256') != sha(path):
        raise RuntimeError('Raw cache hash mismatch: ' + str(path))
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = {'label', 'original_index', 'features', 'raw_output'}
    if metadata['role'] != 'train':
        required.update(('prediction', 'd_gop', 'evidence', 'probability'))
    if metadata['full']:
        required.update(('gaia_z', 'gradnorm', 'cedl_views', 'transform_trace'))
    if not required.issubset(data):
        raise RuntimeError('Raw cache missing fields: ' + str(path))
    n = len(identity['label'])
    if any(value.ndim == 0 or value.shape[0] != n for value in data.values()):
        raise RuntimeError('Raw cache row count mismatch: ' + str(path))
    for key in ('label', 'original_index'):
        if not np.array_equal(data[key], identity[key]):
            raise RuntimeError('Raw cache identity mismatch (' + key + '): ' + str(path))
    return data


def verify_or_save_npz(path, arrays, save_npz, sha=None, side=None, metadata=None):
    if path.exists():
        if side is not None:
            if not side.is_file():
                raise RuntimeError('Reference metadata missing: ' + str(side))
            recorded = json.loads(side.read_text(encoding='utf-8'))
            for key, value in metadata.items():
                if recorded.get(key) != value:
                    raise RuntimeError('Reference metadata mismatch (' + key + '): ' + str(path))
            if recorded.get('sha256') != sha(path):
                raise RuntimeError('Reference hash mismatch: ' + str(path))
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise RuntimeError('Saved artefact fields mismatch: ' + str(path))
            for key, value in arrays.items():
                if not np.array_equal(archive[key], value):
                    raise RuntimeError('Saved artefact value mismatch (' + key + '): ' + str(path))
        return 'verified'
    if side is not None and side.exists():
        raise RuntimeError('Reference missing but metadata exists: ' + str(path))
    save_npz(path, **arrays)
    if side is not None:
        side.write_text(json.dumps({**metadata, 'sha256': sha(path)}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return 'saved'


def verify_or_write_json(path, payload):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != payload:
            raise RuntimeError('Saved metadata mismatch: ' + str(path))
        return 'verified'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return 'saved'
