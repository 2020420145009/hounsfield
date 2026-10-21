#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from med3d_image_utils import get_slice_indices, load_med_volume


DEFAULT_INPUT_DIR = Path("/volume/med-train/data/CT-RATE-DATA/by_label")
DEFAULT_DATASET_ROOT = Path("/volume/med-train/data/CT-RATE-DATA/dataset")
DEFAULT_OUTPUT_DIR = Path("/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/by_label_img")


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


def iter_jsonl_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_no} is not valid JSON") from exc


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


def process_jsonl(
    jsonl_path: Path,
    dataset_root: Path,
    output_root: Path,
    num_slices: int,
    max_images_per_jsonl: int,
):
    category = jsonl_path.stem
    category_dir = output_root / category
    seen = set()
    exported = 0
    skipped_existing = 0
    failures = []

    for line_no, record in iter_jsonl_records(jsonl_path):
        image_name = record.get("image")
        if not isinstance(image_name, str) or not image_name:
            failures.append({"line": line_no, "image": image_name, "error": "missing_image_field"})
            continue

        image_name = Path(image_name).name
        if image_name in seen:
            continue
        if max_images_per_jsonl > 0 and len(seen) >= max_images_per_jsonl:
            break
        seen.add(image_name)

        try:
            image_path = resolve_ctrate_image_path(image_name, dataset_root)
        except ValueError as exc:
            failures.append({"line": line_no, "image": image_name, "error": str(exc)})
            continue

        if not image_path.is_file():
            failures.append(
                {
                    "line": line_no,
                    "image": image_name,
                    "resolved": str(image_path),
                    "error": "file_not_found",
                }
            )
            continue

        image_dir = category_dir / strip_volume_suffix(image_name)
        if is_export_complete(image_dir, num_slices):
            skipped_existing += 1
            continue
        export_volume_slices(image_path, image_dir, num_slices=num_slices)
        exported += 1

    summary = {
        "category": category,
        "jsonl": str(jsonl_path),
        "total_unique_images": len(seen),
        "exported_images": exported,
        "skipped_existing": skipped_existing,
        "failed_images": len(failures),
        "failures": failures,
    }
    category_dir.mkdir(parents=True, exist_ok=True)
    with (category_dir / "_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Export evenly sampled CT slices for each image referenced by CT-RATE by_label jsonl files."
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing by_label jsonl files")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT, help="CT-RATE dataset root")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output root for exported slice folders")
    parser.add_argument("--num_slices", type=int, default=8, help="Number of evenly sampled slices to export per image")
    parser.add_argument("--limit_jsonls", type=int, default=-1, help="Only process the first N jsonl files")
    parser.add_argument("--max_images_per_jsonl", type=int, default=-1, help="Only export the first N unique images from each jsonl")
    parser.add_argument("--category", type=str, default="", help="Only process one category, e.g. emphysema or emphysema.jsonl")
    args = parser.parse_args()

    jsonl_paths = sorted(args.input_dir.glob("*.jsonl"))
    if args.category:
        wanted = args.category
        wanted_stem = Path(wanted).stem
        jsonl_paths = [p for p in jsonl_paths if p.stem == wanted_stem or p.name == wanted]
    if args.limit_jsonls > 0:
        jsonl_paths = jsonl_paths[: args.limit_jsonls]
    if not jsonl_paths:
        raise FileNotFoundError(f"no jsonl files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for jsonl_path in jsonl_paths:
        print(f"[process] {jsonl_path.name}")
        summary = process_jsonl(
            jsonl_path=jsonl_path,
            dataset_root=args.dataset_root,
            output_root=args.output_dir,
            num_slices=args.num_slices,
            max_images_per_jsonl=args.max_images_per_jsonl,
        )
        print(f"[done] {jsonl_path.stem}: exported={summary['exported_images']} skipped_existing={summary['skipped_existing']} failed={summary['failed_images']}")
        summaries.append(summary)

    aggregate = {
        "input_dir": str(args.input_dir),
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "num_slices": args.num_slices,
        "num_jsonls": len(jsonl_paths),
        "categories": summaries,
    }
    with (args.output_dir / "_aggregate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
