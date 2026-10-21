#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageDraw

from med3d_image_utils import load_med_volume, get_slice_indices

OUT_ROOT = Path('/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/tmp_slice_sampling_demo')
CTRATE_ROOT = Path('/volume/med-train/data/CT-RATE-DATA')
SAMPLES = {
    'ctrate': str(CTRATE_ROOT / 'dataset' / 'train_fixed' / 'train_10003' / 'train_10003_a' / 'train_10003_a_1.nii.gz'),
    'm3d': '/volume/med-train/users/shuyan/flemingvlcode/lzs/3d-med-vqa/image/ct_quizze_npy1/003264/Axial_bone_window.npy',
    'lidc': '/volume/med-train/users/shuyan/flemingvlcode/fzy/3d/datasets/LIDC-IDRI/Processed_3D_CT_NPY/LIDC-IDRI-0001.npy',
}
NUM_SLICES = 8


def slice_to_image(sl):
    sl = np.squeeze(sl)
    while sl.ndim > 2:
        sl = sl[0]
    vmin, vmax = float(sl.min()), float(sl.max())
    if vmax > vmin:
        arr = ((sl - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    else:
        arr = np.zeros_like(sl, dtype=np.uint8)
    return Image.fromarray(arr, mode='L').convert('RGB')


def make_montage(vol, indices, out_path):
    thumbs = []
    for idx in indices:
        img = slice_to_image(vol[idx]).resize((256, 256))
        canvas = ImageOps.expand(img, border=4, fill='white')
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f'{idx}', fill='red')
        thumbs.append(canvas)
    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    w, h = thumbs[0].size
    montage = Image.new('RGB', (cols * w, rows * h), 'black')
    for i, img in enumerate(thumbs):
        x = (i % cols) * w
        y = (i // cols) * h
        montage.paste(img, (x, y))
    montage.save(out_path)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, path in SAMPLES.items():
        vol, resolved = load_med_volume(path)
        indices = get_slice_indices(vol.shape[0], NUM_SLICES).tolist()
        sample_dir = OUT_ROOT / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        out_path = sample_dir / f'{name}_montage_8slices.png'
        make_montage(vol, indices, out_path)
        summary[name] = {
            'requested_path': path,
            'resolved_path': resolved,
            'processed_shape': list(vol.shape),
            'indices': indices,
            'montage': str(out_path),
        }
    (OUT_ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
