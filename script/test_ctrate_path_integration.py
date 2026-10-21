#!/usr/bin/env python3
import json
from pathlib import Path

import build_m3dtest_like_subset_v2 as subset_builder
import demo_slice_sampling as slice_demo


EXPECTED_CTRATE_ROOT = Path('/volume/med-train/data/CT-RATE-DATA')
OLD_CTRATE_ROOT = Path('/volume/med-train/users/shuyan/flemingvlcode/fzy/3d/datasets/CT-RATE-Data')
MIXED_JSONL = Path(
    '/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/'
    'mixed_datasets_rebalanced/m3dtest_like_v2/m3dtest_like_mcq_plus_open_v2.jsonl'
)


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f'{message}: expected {expected!r}, got {actual!r}')


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def check_module_constants():
    assert_equal(subset_builder.CTRATE_ROOT, EXPECTED_CTRATE_ROOT, 'subset builder CTRATE_ROOT mismatch')
    assert_equal(slice_demo.CTRATE_ROOT, EXPECTED_CTRATE_ROOT, 'demo CTRATE_ROOT mismatch')
    assert_equal(
        subset_builder.CTRATE_MCQ_PATH,
        EXPECTED_CTRATE_ROOT / 'Synthesized_CoT_Dataset_ctrate' / 'ct_rate_mcq_clean.jsonl',
        'CTRATE_MCQ_PATH mismatch',
    )
    assert_equal(
        subset_builder.CTRATE_OPEN_PATH,
        EXPECTED_CTRATE_ROOT / 'Synthesized_CoT_Dataset_ctrate' / 'ct_rate_open_vqa_three_tiers_5000_15000_clean.jsonl',
        'CTRATE_OPEN_PATH mismatch',
    )
    assert_equal(
        Path(slice_demo.SAMPLES['ctrate']),
        EXPECTED_CTRATE_ROOT / 'dataset' / 'train_fixed' / 'train_10003' / 'train_10003_a' / 'train_10003_a_1.nii.gz',
        'demo CTRATE sample path mismatch',
    )


def check_resolve_ctrate_image_path():
    train_name = 'train_10003_a_1.nii.gz'
    valid_name = 'valid_10003_a_1.nii.gz'

    train_path = subset_builder.resolve_ctrate_image_path(train_name)
    valid_path = subset_builder.resolve_ctrate_image_path(valid_name)

    assert_equal(
        train_path,
        str(EXPECTED_CTRATE_ROOT / 'dataset' / 'train_fixed' / 'train_10003' / 'train_10003_a' / train_name),
        'train image path resolution mismatch',
    )
    assert_equal(
        valid_path,
        str(EXPECTED_CTRATE_ROOT / 'dataset' / 'valid_fixed' / 'valid_10003' / 'valid_10003_a' / valid_name),
        'valid image path resolution mismatch',
    )


def check_filesystem():
    assert_true(EXPECTED_CTRATE_ROOT.is_dir(), f'missing CTRATE root: {EXPECTED_CTRATE_ROOT}')
    assert_true(not OLD_CTRATE_ROOT.exists(), f'old CTRATE root should not exist: {OLD_CTRATE_ROOT}')
    assert_true(subset_builder.CTRATE_MCQ_PATH.is_file(), f'missing file: {subset_builder.CTRATE_MCQ_PATH}')
    assert_true(subset_builder.CTRATE_OPEN_PATH.is_file(), f'missing file: {subset_builder.CTRATE_OPEN_PATH}')
    assert_true(Path(slice_demo.SAMPLES['ctrate']).is_file(), f'missing demo sample: {slice_demo.SAMPLES["ctrate"]}')
    assert_true(MIXED_JSONL.is_file(), f'missing mixed dataset jsonl: {MIXED_JSONL}')


def check_mixed_jsonl():
    total_lines = 0
    ctrate_entries = 0
    sampled_existing_paths = 0

    with MIXED_JSONL.open('r', encoding='utf-8') as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            total_lines += 1
            assert_true(str(OLD_CTRATE_ROOT) not in line, f'old CTRATE path still present on line {line_no}')
            record = json.loads(line)
            image_path = str(record.get('image', ''))
            source = str(record.get('_dataset_source', ''))

            if source.startswith('ctrate:'):
                ctrate_entries += 1
                assert_true(
                    image_path.startswith(str(EXPECTED_CTRATE_ROOT)),
                    f'CTRATE record does not use new root on line {line_no}: {image_path}',
                )
                if sampled_existing_paths < 5:
                    assert_true(Path(image_path).is_file(), f'CTRATE image missing on line {line_no}: {image_path}')
                    sampled_existing_paths += 1

    assert_true(total_lines > 0, 'mixed dataset jsonl is empty')
    assert_true(ctrate_entries > 0, 'mixed dataset jsonl contains no CTRATE records')
    assert_true(sampled_existing_paths == 5, 'did not validate the expected number of CTRATE sample files')


def main():
    check_module_constants()
    check_resolve_ctrate_image_path()
    check_filesystem()
    check_mixed_jsonl()
    print('CT-RATE integration test passed')


if __name__ == '__main__':
    main()
