"""Bake exact registry foot descriptors once for CUDA motion matching."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(asset_dir: Path, executable: Path, output: Path) -> Path:
    manifest_path = asset_dir / 'semantic_duel_assets_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    model_name = 'model.two_fighter_arena.xml'
    model = asset_dir / model_name
    if digest(model) != manifest['files'][model_name]['sha256']:
        raise ValueError('model SHA-256 mismatch')
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for clip in manifest['clips']:
        source = asset_dir / clip['files']['mujoco_joint_order']
        if digest(source) != manifest['files'][source.name]['sha256']:
            raise ValueError(f'clip SHA-256 mismatch: {source.name}')
        destination = output / f"motion_{clip['npz_path_id']}_foot_features.f32le"
        result = subprocess.run([str(executable), str(model), str(source),
            str(destination)], check=True, text=True, capture_output=True)
        measured = json.loads(result.stdout)
        values = np.fromfile(destination, dtype='<f4')
        if measured['frames'] != clip['frames'] or values.size != clip['frames'] * 6:
            raise ValueError('baked feature dimensions disagree with clip')
        if not np.isfinite(values).all():
            raise ValueError('baked feature contains nonfinite values')
        records.append({
            'npz_path_id': clip['npz_path_id'], 'file': destination.name,
            'frames': clip['frames'], 'sha256': digest(destination),
            'bytes': destination.stat().st_size, **measured,
        })
    result_path = output / 'foot_features_manifest.json'
    result_path.write_text(json.dumps({
        'schema': 'rek.g1_cuda_foot_features.v1', 'host': platform.node(),
        'asset_manifest_sha256': digest(manifest_path),
        'model_sha256': digest(model), 'bake_executable_sha256': digest(executable),
        'sampler': 'rek_g1_mujoco_feature_registry_sample', 'clips': records,
    }, indent=2) + '\n')
    return result_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('asset_dir', type=Path)
    parser.add_argument('executable', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(export(args.asset_dir, args.executable, args.output))
