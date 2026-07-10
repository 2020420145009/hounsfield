import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image


_RESOLVED_IMAGE_PATH_CACHE = {}
_SLICE_CACHE_EXTS = ('.png', '.jpg', '.jpeg')
_MED_IMAGE_EXTS = ('.npy', '.nii', '.nii.gz', '.png', '.jpg', '.jpeg')
_M3D_IMAGE_ROOTS = (
    '/volume/med-train/users/shuyan/flemingvlcode/lzs/3d-med-vqa/image/ct_quizze_npy1',
    '/volume/med-train/users/shuyan/flemingvlcode/lzs/3d-med-vqa/image/ct_quizze_npy',
    '/volume/med-train/users/shuyan/flemingvlcode/zyq/3d-med-vqa/image/ct_quizze_npy1',
    '/volume/med-train/users/shuyan/flemingvlcode/zyq/3d-med-vqa/image/ct_quizze_npy',
    '/volume/med-train/users/shuyan/flemingvlcode/fzy/3d-med-vqa/image/ct_quizze_npy1',
    '/volume/med-train/users/shuyan/flemingvlcode/fzy/3d-med-vqa/image/ct_quizze_npy',
)
_ROOT_REWRITES = (
    (
        '/volume/med-train/users/shuyan/lzs/3d-med-vqa/image/',
        '/volume/med-train/users/shuyan/flemingvlcode/lzs/3d-med-vqa/image/',
    ),
    (
        '/volume/med-train/users/shuyan/zyq/3d-med-vqa/image/',
        '/volume/med-train/users/shuyan/flemingvlcode/zyq/3d-med-vqa/image/',
    ),
    (
        '/volume/med-train/users/shuyan/fzy/3d-med-vqa/image/',
        '/volume/med-train/users/shuyan/flemingvlcode/fzy/3d-med-vqa/image/',
    ),
    (
        '/volume/med-train/users/shuyan/fzy/',
        '/volume/med-train/users/shuyan/flemingvlcode/fzy/',
    ),
)


def _strip_known_volume_suffix(name: str) -> str:
    if name.endswith('.nii.gz'):
        return name[:-7]
    for suffix in ('.npy', '.nii'):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return Path(name).stem


def _slice_sort_key(path: str):
    name = os.path.basename(path)
    m = re.search(r'(\d+)(?=\.[^.]+$)', name)
    if m:
        return (0, int(m.group(1)), name)
    return (1, name)


def _is_med_image_file(path: str) -> bool:
    if not path or not os.path.isfile(path):
        return False
    lower = path.lower()
    return lower.endswith(_MED_IMAGE_EXTS)


def resolve_med_image_path(img_path):
    if not isinstance(img_path, str):
        return img_path
    if os.path.exists(img_path):
        return img_path
    cached = _RESOLVED_IMAGE_PATH_CACHE.get(img_path)
    if cached is not None:
        return cached

    candidates = []
    basename = os.path.basename(img_path.rstrip('/'))
    lower_basename = basename.lower()

    # Fast-fail malformed directory-like paths that are not concrete image files.
    # These otherwise trigger extremely expensive recursive glob searches.
    if lower_basename in {'ct_quizze_npy', 'ct_quizze_npy1', 'ct_case'}:
        _RESOLVED_IMAGE_PATH_CACHE[img_path] = img_path
        return img_path

    for old_root, new_root in _ROOT_REWRITES:
        if img_path.startswith(old_root):
            rewritten = img_path.replace(old_root, new_root, 1)
            if os.path.exists(rewritten):
                _RESOLVED_IMAGE_PATH_CACHE[img_path] = rewritten
                return rewritten
            candidates.append(rewritten)
    lidc_basename = re.fullmatch(r'(LIDC-IDRI-\d+)\.nii\.gz', os.path.basename(img_path))
    if lidc_basename:
        case_id = lidc_basename.group(1)
        lidc_roots = [
            '/volume/med-train/users/shuyan/flemingvlcode/fzy/3d/datasets/LIDC-IDRI/Cleaned_3D_CoT_Dataset',
            '/volume/med-train/users/shuyan/fzy/3d/datasets/LIDC-IDRI/Cleaned_3D_CoT_Dataset',
        ]
        for root in lidc_roots:
            candidates.append(os.path.join(root, case_id, f'{case_id}_full_volume.nii.gz'))
            candidates.append(os.path.join(root, case_id, f'{case_id}.nii.gz'))
    m3d_match = re.search(r'ct_quizze_npy1?/(.+)$', img_path)
    if m3d_match:
        rel = m3d_match.group(1)
        rel_basename = os.path.basename(rel.rstrip('/'))
        if not rel_basename or '.' not in rel_basename:
            _RESOLVED_IMAGE_PATH_CACHE[img_path] = img_path
            return img_path
        for base in _M3D_IMAGE_ROOTS:
            candidates.append(os.path.join(base, rel))

        for candidate in candidates:
            if _is_med_image_file(candidate):
                _RESOLVED_IMAGE_PATH_CACHE[img_path] = candidate
                return candidate

    # Some M3D-derived paths may contain intermediate case folders such as:
    #   .../ct_case/005529/Sagittal_CTU/ct_quizze_npy1/<file>
    # or may point to the `ct_quizze_npy1` directory itself instead of a file.
    # Try to recover by locating the `ct_quizze_npy1` segment and resolving from there.
    parts = Path(img_path).parts
    if 'ct_quizze_npy1' in parts or 'ct_quizze_npy' in parts:
        seg = 'ct_quizze_npy1' if 'ct_quizze_npy1' in parts else 'ct_quizze_npy'
        idx = parts.index(seg)
        suffix_parts = parts[idx + 1 :]
        if not suffix_parts:
            _RESOLVED_IMAGE_PATH_CACHE[img_path] = img_path
            return img_path
        suffix_basename = suffix_parts[-1].rstrip('/')
        if not suffix_basename or '.' not in suffix_basename:
            _RESOLVED_IMAGE_PATH_CACHE[img_path] = img_path
            return img_path
        for base in _M3D_IMAGE_ROOTS:
            candidate = os.path.join(base, *suffix_parts)
            if _is_med_image_file(candidate):
                _RESOLVED_IMAGE_PATH_CACHE[img_path] = candidate
                return candidate

    _RESOLVED_IMAGE_PATH_CACHE[img_path] = img_path
    return img_path


def get_slice_cache_dir(img_path):
    resolved = resolve_med_image_path(img_path)
    directory = os.path.dirname(resolved)
    stem = _strip_known_volume_suffix(os.path.basename(resolved))
    return os.path.join(directory, f'{stem}_slices')


def list_cached_slice_paths(img_path):
    cache_dir = get_slice_cache_dir(img_path)
    if not os.path.isdir(cache_dir):
        return []
    files = []
    for entry in os.listdir(cache_dir):
        lower = entry.lower()
        if lower.endswith(_SLICE_CACHE_EXTS):
            files.append(os.path.join(cache_dir, entry))
    return sorted(files, key=_slice_sort_key)


def _validate_cached_slice_paths(slice_paths):
    if not slice_paths:
        return False
    for path in slice_paths:
        try:
            if os.path.getsize(path) < 1024:
                return False
            with Image.open(path) as img:
                img.verify()
        except Exception:
            return False
    return True


def _load_slice_cache_metadata(cache_dir):
    meta_path = os.path.join(cache_dir, 'metadata.json')
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _write_slice_cache_metadata(cache_dir, resolved, vol_shape, num_slices):
    meta = {
        'source_image': resolved,
        'num_slices': num_slices,
        'shape': list(vol_shape),
    }
    with open(os.path.join(cache_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _is_slice_cache_complete(resolved):
    cache_dir = get_slice_cache_dir(resolved)
    slice_paths = list_cached_slice_paths(resolved)
    if not slice_paths:
        return False
    meta = _load_slice_cache_metadata(cache_dir)
    if meta is not None:
        expected = meta.get('num_slices')
        if isinstance(expected, int) and expected != len(slice_paths):
            return False
    if not _validate_cached_slice_paths(slice_paths):
        return False
    if meta is None:
        _write_slice_cache_metadata(cache_dir, resolved, vol_shape=[], num_slices=len(slice_paths))
    return True


def load_med_volume(img_path):
    img_path = resolve_med_image_path(img_path)
    if str(img_path).endswith('.npy'):
        vol = np.load(img_path)
    else:
        try:
            import nibabel as nib
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                'nibabel is required to load .nii/.nii.gz volumes, but it is not installed in the current environment.'
            ) from e
        vol = nib.load(img_path).get_fdata()

    vol = np.squeeze(vol)
    while vol.ndim > 3:
        vol = np.squeeze(vol[0])
    if vol.ndim == 2:
        vol = np.expand_dims(vol, axis=0)

    if vol.ndim == 3 and vol.shape[2] < vol.shape[0] and vol.shape[2] < vol.shape[1]:
        vol = vol.transpose(2, 0, 1)
    elif vol.ndim == 3 and vol.shape[1] < vol.shape[0] and vol.shape[1] < vol.shape[2]:
        vol = vol.transpose(1, 0, 2)
    return vol, img_path


def get_slice_indices(depth, num_slices):
    if depth <= 0:
        raise ValueError(f'invalid depth: {depth}')
    return np.linspace(0, depth - 1, num_slices).astype(int)


def _slice_to_uint8(sl):
    sl = np.squeeze(sl)
    while sl.ndim > 2:
        sl = sl[0]
    v_min, v_max = sl.min(), sl.max()
    if v_max > v_min:
        return ((sl - v_min) / (v_max - v_min) * 255.0).astype(np.uint8)
    return np.zeros_like(sl, dtype=np.uint8)


def save_slice_cache(img_path, overwrite=False, metadata=True):
    vol, resolved = load_med_volume(img_path)
    cache_dir = get_slice_cache_dir(resolved)
    os.makedirs(cache_dir, exist_ok=True)

    existing = list_cached_slice_paths(resolved)
    if existing and not overwrite and _is_slice_cache_complete(resolved):
        return existing, cache_dir
    if existing and not overwrite:
        overwrite = True

    if overwrite:
        for old in existing:
            os.remove(old)
        meta_path = os.path.join(cache_dir, 'metadata.json')
        if os.path.exists(meta_path):
            os.remove(meta_path)

    vol = np.clip(vol, -1000, 400)
    saved = []
    for i in range(vol.shape[0]):
        sl = _slice_to_uint8(vol[i])
        out_path = os.path.join(cache_dir, f'slice_{i:04d}.png')
        tmp_path = out_path + '.tmp'
        Image.fromarray(sl, mode='L').save(tmp_path, format='PNG')
        os.replace(tmp_path, out_path)
        saved.append(out_path)

    if metadata:
        _write_slice_cache_metadata(cache_dir, resolved, vol.shape, len(saved))

    return saved, cache_dir


def get_pixel_values(img_data, num_slices, image_processor, slice_cache_mode='auto'):
    if slice_cache_mode not in {'auto', 'require', 'disable'}:
        raise ValueError(f'unsupported slice_cache_mode: {slice_cache_mode}')

    if isinstance(img_data, list):
        indices = get_slice_indices(len(img_data), num_slices)
        pv_list = []
        for i in indices:
            slice_path = resolve_med_image_path(img_data[i])
            img = Image.open(slice_path).convert('RGB').resize((448, 448))
            pv = image_processor(images=img, return_tensors='pt').pixel_values[0]
            pv_list.append(pv)
        return torch.stack(pv_list, dim=0)

    cached_slices = []
    if slice_cache_mode != 'disable':
        cached_slices = list_cached_slice_paths(img_data)
        if cached_slices and not _is_slice_cache_complete(img_data):
            cached_slices = []

    if not cached_slices and slice_cache_mode == 'auto' and isinstance(img_data, str) and (
        img_data.endswith('.npy') or img_data.endswith('.nii') or img_data.endswith('.nii.gz')
    ):
        try:
            save_slice_cache(img_data, overwrite=False)
            cached_slices = list_cached_slice_paths(img_data)
        except Exception:
            cached_slices = []

    if cached_slices:
        indices = get_slice_indices(len(cached_slices), num_slices)
        pv_list = []
        for i in indices:
            img = Image.open(cached_slices[i]).convert('RGB').resize((448, 448))
            pv = image_processor(images=img, return_tensors='pt').pixel_values[0]
            pv_list.append(pv)
        return torch.stack(pv_list, dim=0)

    if slice_cache_mode == 'require':
        cache_dir = get_slice_cache_dir(img_data)
        raise FileNotFoundError(f'slice cache missing or incomplete for {img_data}: {cache_dir}')

    vol, _ = load_med_volume(img_data)
    depth = vol.shape[0]
    vol = np.clip(vol, -1000, 400)
    indices = get_slice_indices(depth, num_slices)
    pv_list = []
    for i in indices:
        sl = _slice_to_uint8(vol[i])
        img = Image.fromarray(np.stack([sl] * 3, axis=-1), mode='RGB').resize((448, 448))
        pv_list.append(image_processor(images=img, return_tensors='pt').pixel_values[0])
    return torch.stack(pv_list, dim=0)
