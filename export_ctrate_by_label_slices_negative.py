#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from med3d_image_utils import get_slice_indices, load_med_volume


DEFAULT_INPUT_DIR = Path("/volume/med-train/data/CT-RATE-DATA/by_label")
DEFAULT_AGGREGATED_INFO = Path("/volume/med-train/data/CT-RATE-DATA/aggregated_image_info.json")
DEFAULT_DATASET_ROOT = Path("/volume/med-train/data/CT-RATE-DATA/dataset")
DEFAULT_OUTPUT_DIR = Path("/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/by_label_img_negative")

CATEGORY_TO_LABEL_NAME = {
    "arterial_wall_calcification": "Arterial wall calcification",
    "cardiomegaly": "Cardiomegaly",
    "coronary_artery_wall_calcification": "Coronary artery wall calcification",
    "emphysema": "Emphysema",
    "interlobular_septal_thickening": "Interlobular septal thickening",
    "lung_opacity": "Lung opacity",
    "medical_material": "Medical material",
    "pericardial_effusion": "Pericardial effusion",
}


def is_export_complete(output_dir: Path, num_slices: int) -> bool:
    if not output_dir.is_dir():
        return False
    meta_path = output_dir / "metadata.json"
    if not meta_path.is_file():
        return False
    png_files = sorted(output_dir.glob("slice_*.png"))
    if len(png_files) != num_slices:
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if meta.get("num_slices") != num_slices:
        return False
    exported = meta.get("exported_slices")
    return isinstance(exported, list) and len(exported) == num_slices


def strip_volume_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    if name.endswith(".npy"):
        return name[:-4]
    return Path(name).stem


def resolve_ctrate_image_path(image_name: str, dataset_root: Path) -> Path:
    image_name = Path(image_name).name
    match = re.fullmatch(r"(?P<split>train|valid|test)_(?P<caseid>\d+)_(?P<study>[a-z])_(?P<series>\d+)\.nii\.gz", image_name)
    if not match:
        raise ValueError(f"unsupported CT-RATE image name: {image_name}")

    split = match.group("split")
    case_id = f"{split}_{match.group('caseid')}"
    study_id = f"{case_id}_{match.group('study')}"
    split_dir = f"{split}_fixed"
    return dataset_root / split_dir / case_id / study_id / image_name


def slice_to_uint8(slice_array: np.ndarray) -> np.ndarray:
    slice_array = np.squeeze(slice_array)
    while slice_array.ndim > 2:
        slice_array = slice_array[0]
    v_min = float(slice_array.min())
    v_max = float(slice_array.max())
    if v_max > v_min:
        scaled = (slice_array - v_min) / (v_max - v_min) * 255.0
        return scaled.astype(np.uint8)
    return np.zeros_like(slice_array, dtype=np.uint8)


def export_volume_slices(image_path: Path, output_dir: Path, num_slices: int):
    volume, resolved_path = load_med_volume(str(image_path))
    volume = np.clip(volume, -1000, 400)
    indices = get_slice_indices(volume.shape[0], num_slices)

    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for out_idx, slice_idx in enumerate(indices):
        slice_u8 = slice_to_uint8(volume[slice_idx])
        out_path = output_dir / f"slice_{out_idx:02d}_z{int(slice_idx):04d}.png"
        Image.fromarray(slice_u8, mode="L").save(out_path)
        exported.append(
            {
                "slice_rank": out_idx,
                "source_z_index": int(slice_idx),
                "file": out_path.name,
            }
        )

    meta = {
        "source_image": str(resolved_path),
        "num_slices": num_slices,
        "volume_shape": list(volume.shape),
        "exported_slices": exported,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def read_aggregated_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path} is not a JSON list")
    return [row for row in rows if isinstance(row, dict)]


def get_label_name(category_name: str) -> str:
    return CATEGORY_TO_LABEL_NAME.get(category_name, category_name.replace("_", " ").title())


def parse_label_value(labels_raw: str, label_name: str) -> int | None:
    pattern = rf"{re.escape(label_name)}:\s*([01])"
    match = re.search(pattern, labels_raw)
    return int(match.group(1)) if match else None


def iter_negative_images(rows: list[dict], category_name: str, max_images: int):
    label_name = get_label_name(category_name)
    seen = set()
    selected = []
    failures = []
    for row_idx, row in enumerate(rows, start=1):
        image_name = row.get("image")
        labels_raw = str(row.get("labels", ""))
        if not isinstance(image_name, str) or not image_name:
            failures.append({"row": row_idx, "image": image_name, "error": "missing_image_field"})
            continue

        label_value = parse_label_value(labels_raw, label_name)
        if label_value is None:
            failures.append({"row": row_idx, "image": image_name, "error": f"missing_label:{label_name}"})
            continue
        if label_value != 0:
            continue

        image_name = Path(image_name).name
        if image_name in seen:
            continue
        seen.add(image_name)
        selected.append(image_name)
        if max_images > 0 and len(selected) >= max_images:
            break
    return selected, failures


def process_category(
    category_name: str,
    aggregated_rows: list[dict],
    dataset_root: Path,
    output_root: Path,
    num_slices: int,
    max_images: int,
):
    category_dir = output_root / category_name
    selected_images, selection_failures = iter_negative_images(aggregated_rows, category_name, max_images)
    exported = 0
    skipped_existing = 0
    failures = list(selection_failures)

    for image_name in selected_images:
        try:
            image_path = resolve_ctrate_image_path(image_name, dataset_root)
        except ValueError as exc:
            failures.append({"image": image_name, "error": str(exc)})
            continue

        if not image_path.is_file():
            failures.append({"image": image_name, "resolved": str(image_path), "error": "file_not_found"})
            continue

        image_dir = category_dir / strip_volume_suffix(image_name)
        if is_export_complete(image_dir, num_slices):
            skipped_existing += 1
            continue
        export_volume_slices(image_path, image_dir, num_slices=num_slices)
        exported += 1

    summary = {
        "category": category_name,
        "selection": "label_0",
        "total_selected_images": len(selected_images),
        "exported_images": exported,
        "skipped_existing": skipped_existing,
        "failed_images": len(failures),
        "failures": failures,
    }
    category_dir.mkdir(parents=True, exist_ok=True)
    with (category_dir / "_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def get_category_names(input_dir: Path, wanted: str, limit_jsonls: int) -> list[str]:
    jsonl_paths = sorted(input_dir.glob("*.jsonl"))
    if wanted:
        wanted_stem = Path(wanted).stem
        jsonl_paths = [p for p in jsonl_paths if p.stem == wanted_stem or p.name == wanted]
    if limit_jsonls > 0:
        jsonl_paths = jsonl_paths[:limit_jsonls]
    if not jsonl_paths:
        raise FileNotFoundError(f"no jsonl files found in {input_dir}")
    return [p.stem for p in jsonl_paths]


def main():
    parser = argparse.ArgumentParser(
        description="Export evenly sampled CT slices for the first label=0 images of each CT-RATE by_label category."
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing by_label jsonl files")
    parser.add_argument("--aggregated_info", type=Path, default=DEFAULT_AGGREGATED_INFO, help="Aggregated metadata with label strings")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT, help="CT-RATE dataset root")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output root for exported negative slice folders")
    parser.add_argument("--num_slices", type=int, default=8, help="Number of evenly sampled slices to export per image")
    parser.add_argument("--limit_jsonls", type=int, default=-1, help="Only process the first N category jsonl files")
    parser.add_argument("--max_images_per_jsonl", type=int, default=200, help="Only export the first N unique label=0 images from each category")
    parser.add_argument("--category", type=str, default="", help="Only process one category, e.g. emphysema or emphysema.jsonl")
    args = parser.parse_args()

    category_names = get_category_names(args.input_dir, args.category, args.limit_jsonls)
    aggregated_rows = read_aggregated_rows(args.aggregated_info)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for category_name in category_names:
        print(f"[process] {category_name} (label=0)")
        summary = process_category(
            category_name=category_name,
            aggregated_rows=aggregated_rows,
            dataset_root=args.dataset_root,
            output_root=args.output_dir,
            num_slices=args.num_slices,
            max_images=args.max_images_per_jsonl,
        )
        print(f"[done] {category_name}: exported={summary['exported_images']} skipped_existing={summary['skipped_existing']} failed={summary['failed_images']}")
        summaries.append(summary)

    aggregate = {
        "input_dir": str(args.input_dir),
        "aggregated_info": str(args.aggregated_info),
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "num_slices": args.num_slices,
        "max_images_per_jsonl": args.max_images_per_jsonl,
        "selection": "label_0",
        "num_categories": len(category_names),
        "categories": summaries,
    }
    with (args.output_dir / "_aggregate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
