# 3D Image Format And Slice Examples

This folder summarizes the current 3D image formats for `M3D`, `CT-RATE`, and `LIDC`, and shows example 8-slice montages using the shared loader in `med3d_image_utils.py`.

| Dataset | Source format | Example path pattern | Raw typical shape | Postprocessed shape | Main issue |
|---|---|---|---|---|---|
| M3D | `.npy` | `.../ct_quizze_npy1/.../*.npy` | `(1, 32, 256, 256)` | `(32, 256, 256)` | Has leading singleton dimension; must `squeeze` first |
| CT-RATE | `.nii.gz` | `.../train_fixed/.../*.nii.gz` | `(512, 512, 290)` | `(290, 512, 512)` | Depth axis is last; must transpose to `D,H,W` |
| LIDC | `.nii.gz` | `.../LIDC-IDRI-xxxx_full_volume.nii.gz` | `(512, 512, 261)` | `(261, 512, 512)` | Depth axis is last; must transpose to `D,H,W` |

## M3D

- Example file: `/volume/med-train/users/shuyan/flemingvlcode/lzs/3d-med-vqa/image/ct_quizze_npy1/000008/Axial_C__arterial_phase.npy`
- Raw shape: `(1, 32, 256, 256)`
- Postprocessed shape: `(32, 256, 256)`
- Value range: `0.0` to `1.0`
- Sampled slice indices: `[0, 4, 8, 13, 17, 22, 26, 31]`
- Montage: `m3d_montage_8slices.png`

## CT-RATE

- Example file: `/volume/med-train/data/CT-RATE-DATA/dataset/train_fixed/train_10000/train_10000_a/train_10000_a_1.nii.gz`
- Raw shape: `(512, 512, 290)`
- Postprocessed shape: `(290, 512, 512)`
- Value range: `-1180.0` to `2214.0`
- Sampled slice indices: `[0, 41, 82, 123, 165, 206, 247, 289]`
- Montage: `ct_rate_montage_8slices.png`

## LIDC

- Example file: `/volume/med-train/users/shuyan/flemingvlcode/fzy/3d/datasets/LIDC-IDRI/Cleaned_3D_CoT_Dataset/LIDC-IDRI-0002/LIDC-IDRI-0002_full_volume.nii.gz`
- Raw shape: `(512, 512, 261)`
- Postprocessed shape: `(261, 512, 512)`
- Value range: `-3024.0` to `1752.0`
- Sampled slice indices: `[0, 37, 74, 111, 148, 185, 222, 260]`
- Montage: `lidc_montage_8slices.png`
