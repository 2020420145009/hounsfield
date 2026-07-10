#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from med3d_image_utils import resolve_med_image_path, save_slice_cache


_VALID_IMAGE_SUFFIXES = ('.npy', '.nii', '.nii.gz')


def iter_records(path: Path):
    if path.suffix == '.jsonl':
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                yield item
        else:
            yield data


def main():
    parser = argparse.ArgumentParser(description='Precompute slice-cache folders for 3D medical images.')
    parser.add_argument('--inputs', nargs='+', required=True, help='Dataset json/jsonl paths')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--limit', type=int, default=-1)
    args = parser.parse_args()

    image_paths = []
    seen = set()
    skipped_unresolved = []
    for input_path in args.inputs:
        for rec in iter_records(Path(input_path)):
            img = rec.get('image')
            if isinstance(img, str) and img.lower().endswith(_VALID_IMAGE_SUFFIXES):
                resolved = resolve_med_image_path(img)
                if (
                    isinstance(resolved, str)
                    and resolved.lower().endswith(_VALID_IMAGE_SUFFIXES)
                    and os.path.isfile(resolved)
                    and resolved not in seen
                ):
                    seen.add(resolved)
                    image_paths.append(resolved)
                elif isinstance(resolved, str):
                    skipped_unresolved.append({'image': img, 'resolved': resolved, 'error': 'unresolved_or_missing_file'})
    if args.limit > 0:
        image_paths = image_paths[:args.limit]

    created = 0
    reused = 0
    failed = []
    for img_path in tqdm(image_paths, desc='precompute_slice_cache'):
        try:
            existing_before = False
            cache_dir = Path(img_path).parent / (Path(img_path).name.replace('.nii.gz', '').replace('.nii', '').replace('.npy', '') + '_slices')
            if cache_dir.exists() and any(cache_dir.iterdir()) and not args.overwrite:
                existing_before = True
            save_slice_cache(img_path, overwrite=args.overwrite)
            if existing_before:
                reused += 1
            else:
                created += 1
        except Exception as e:
            failed.append({'image': img_path, 'error': str(e)})

    print({
        'total': len(image_paths),
        'created': created,
        'reused': reused,
        'skipped_unresolved': len(skipped_unresolved),
        'failed': len(failed),
    })
    if skipped_unresolved or failed:
        fail_path = Path('slice_cache_failures.json')
        fail_path.write_text(
            json.dumps(
                {
                    'skipped_unresolved': skipped_unresolved,
                    'failed_during_precompute': failed,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )
        print(f'failed list written to {fail_path.resolve()}')


if __name__ == '__main__':
    main()
