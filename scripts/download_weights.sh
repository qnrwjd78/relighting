#!/usr/bin/env bash
# Download the public weights for exp_0/1/2 and both shadow_c2f paths.
set -euo pipefail
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON_BIN:-python}" - "$SCRIPT_ROOT" "$@" <<'PY'
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request

parser = argparse.ArgumentParser(description="Download pretrained weights; does not install external model code or train checkpoints.")
parser.add_argument('--root', type=Path, default=Path(sys.argv[1]), help='Repository root / destination layout')
parser.add_argument('--only', choices=['all', 'wan', 'moge', 'focus', 'adapter'], default='all')
parser.add_argument('--dry-run', action='store_true', help='Print destinations without network access or writes')
parser.add_argument('--check', action='store_true', help='Offline presence checks and SHA-256 checks for direct-download weights')
parser.add_argument('--sbu-file', type=Path, help='Existing official AdapterShadow SBU checkpoint to copy')
parser.add_argument('--sbu-url', default=os.environ.get('ADAPTERSHADOW_SBU_URL'), help='Direct HTTPS download URL for the official SBU checkpoint')
args = parser.parse_args(sys.argv[2:])
if args.check and args.dry_run:
    parser.error('--check and --dry-run are mutually exclusive')
if args.sbu_file and args.sbu_url:
    parser.error('Choose --sbu-file or --sbu-url')
root = args.root.expanduser().resolve()

# Revisions verified against the official Hugging Face model repositories.
snapshots = [
    ('wan', 'Wan-AI/Wan2.2-TI2V-5B', '921dbaf3f1674a56f47e83fb80a34bac8a8f203e',
     'weights/Wan2.2-TI2V-5B', [
        'diffusion_pytorch_model-00001-of-00003.safetensors',
        'diffusion_pytorch_model-00002-of-00003.safetensors',
        'diffusion_pytorch_model-00003-of-00003.safetensors',
        'diffusion_pytorch_model.safetensors.index.json',
        'Wan2.2_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth',
        'config.json', 'configuration.json',
        'google/umt5-xxl/tokenizer.json', 'google/umt5-xxl/spiece.model',
        'google/umt5-xxl/tokenizer_config.json', 'google/umt5-xxl/special_tokens_map.json']),
    ('moge', 'Ruicheng/moge-3-vitl', '184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be',
     'weights/moge-3-vitl', ['model.pt']),
    ('focus', 'geshang/focus_large_sd', 'f0037d462ccbadc52a33bd49d3eb2ca0f5f35339',
     'weights/FOCUS', ['focus_large_sd.pth']),
]
# SHA-256: SAM/EfficientNet verified against local official downloads;
# CLIP's full hash is also embedded in its upstream URL.
direct = [
    ('adapter', 'SAM ViT-B',
     'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
     root / 'external/AdapterShadow/checkpoint/sam/sam_vit_b_01ec64.pth',
     'ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912'),
    ('adapter', 'EfficientNet-B1',
     'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_ap-44ef0a3d.pth',
     root / 'external/AdapterShadow/efficientnet/hub/checkpoints/tf_efficientnet_b1_ap-44ef0a3d.pth',
     '44ef0a3dea0ea4e2e44825f5b19a08f2ed6f4221897e5f5058ff009bf7560239'),
    ('focus', 'FOCUS CLIP ViT-B/16',
     'https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt',
     Path.home() / '.cache/clip/ViT-B-16.pt',
     '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'),
]
sbu = root / 'external/AdapterShadow/checkpoint/sbu.ckpt'
selected = lambda group: args.only in ('all', group)
errors = []
verified = []


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_file(path, expected_hash=None):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'Missing or empty file: {path}')
    if path.suffix in ('.pt', '.pth', '.ckpt', '.safetensors'):
        with path.open('rb') as handle:
            prefix = handle.read(512).lstrip().lower()
        if path.stat().st_size < 1024 or prefix.startswith((b'<!doctype', b'<html', b'version https://git-lfs')):
            raise ValueError(f'Not a model binary (HTML/LFS pointer/truncated file): {path}')
    if expected_hash and sha256(path) != expected_hash:
        raise ValueError(f'SHA-256 mismatch: {path}')


def download_url(url, path, digest=None):
    if path.exists():
        # Never silently replace a potentially user-owned file with bad contents.
        check_file(path, digest)
        print(f'Already present: {path}', flush=True)
        return
    if not url.startswith('https://'):
        raise ValueError('Direct-download URLs must use HTTPS')
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'TokenLight-weight-downloader'})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        check_file(temporary, digest)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


for group, repo, revision, directory, files in snapshots:
    if not selected(group):
        continue
    destination = root / directory
    print(f'{repo}@{revision[:12]} -> {destination}', flush=True)
    if args.dry_run:
        continue
    try:
        if not args.check:
            from huggingface_hub import HfApi, snapshot_download
            info = HfApi().model_info(repo, revision=revision, files_metadata=True)
            metadata = {entry.rfilename: entry for entry in info.siblings}
            snapshot_download(repo, revision=revision, local_dir=str(destination), allow_patterns=files)
        for filename in files:
            path = destination / filename
            check_file(path)
            if not args.check:
                remote = metadata[filename]
                if remote.size is not None and path.stat().st_size != remote.size:
                    raise ValueError(f'Size mismatch: {path}')
                if remote.lfs:
                    check_file(path, remote.lfs.sha256)
            verified.append(str(path))
    except Exception as exc:
        errors.append(f'{repo}: {type(exc).__name__}: {exc}')

for group, name, url, path, digest in direct:
    if not selected(group):
        continue
    print(f'{name} -> {path}', flush=True)
    if args.dry_run:
        continue
    try:
        if args.check:
            check_file(path, digest)
        else:
            download_url(url, path, digest)
        verified.append(str(path))
    except Exception as exc:
        errors.append(f'{name}: {type(exc).__name__} (download or checksum failed)')

if selected('adapter'):
    print(f'AdapterShadow SBU -> {sbu}', flush=True)
    if args.dry_run:
        print('  Requires an existing checkpoint, --sbu-file, or --sbu-url / ADAPTERSHADOW_SBU_URL.')
    else:
        try:
            if not args.check and not sbu.exists():
                if args.sbu_file:
                    source = args.sbu_file.expanduser().resolve()
                    check_file(source)
                    sbu.parent.mkdir(parents=True, exist_ok=True)
                    descriptor, name = tempfile.mkstemp(prefix='sbu.', suffix='.ckpt', dir=sbu.parent)
                    os.close(descriptor)
                    temporary = Path(name)
                    try:
                        shutil.copyfile(source, temporary)
                        if sha256(source) != sha256(temporary):
                            raise ValueError('SBU copy checksum mismatch')
                        os.replace(temporary, sbu)
                    finally:
                        temporary.unlink(missing_ok=True)
                elif args.sbu_url:
                    download_url(args.sbu_url, sbu)
            check_file(sbu)
            verified.append(str(sbu))
        except Exception:
            errors.append('AdapterShadow SBU unavailable. Its official distribution ZIP contains no sbu.ckpt. '
                          'Supply --sbu-file PATH or --sbu-url HTTPS_URL, or place the official SBU checkpoint at ' + str(sbu))

print('Your trained TokenLight and C2F checkpoints are NOT public base weights; restore them or train again.')
print('External MoGe / AdapterShadow / FOCUS code and compiled dependencies are installed separately.')
if args.dry_run:
    print('Dry-run only: no files downloaded or modified.')
    raise SystemExit(0)
if not args.check:
    report = root / 'weights/download_report.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({'selection': args.only, 'verified_files': verified, 'errors': errors}, indent=2) + '\n')
if errors:
    print('\nINCOMPLETE:', file=sys.stderr)
    for error in errors:
        print(' - ' + error, file=sys.stderr)
    raise SystemExit(1)
print(f'OK: {len(verified)} files checked for selection {args.only}.')
PY
