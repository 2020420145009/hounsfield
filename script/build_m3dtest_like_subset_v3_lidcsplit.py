#!/usr/bin/env python3
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import build_m3dtest_like_subset_v2 as base


ROOT = Path('/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d')
OUT_DIR = ROOT / 'mixed_datasets_rebalanced' / 'm3dtest_like_v3_lidcsplit'
OUT_JSONL = OUT_DIR / 'm3dtest_like_mcq_plus_open_v3_lidcsplit.jsonl'
OUT_SUMMARY = OUT_DIR / 'summary.json'

LIDC_MCQ_PATH = Path('/volume/med-train/users/shuyan/flemingvlcode/fzy/3d/datasets/LIDC-IDRI/Cleaned_3D_CoT_Dataset/medical_lidc_mcq_vqa_dataset4.json')
LIDC_OPEN_PATH = Path('/volume/med-train/users/shuyan/flemingvlcode/fzy/3d/datasets/LIDC-IDRI/Cleaned_3D_CoT_Dataset/medical_lidc_open_vqa_dataset4.json')


def main():
    random.seed(base.SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reject_stats = Counter()

    m3d_mcq = base.prepare_rows(base.read_jsonl(base.M3D_MCQ_PATH), 'm3d:mcq', 'mcq', reject_stats)
    ctrate_mcq = base.prepare_rows(base.read_jsonl(base.CTRATE_MCQ_PATH), 'ctrate:mcq', 'mcq', reject_stats)
    lidc_mcq = base.prepare_rows(base.read_jsonl(LIDC_MCQ_PATH), 'lidc:mcq', 'mcq', reject_stats)

    mcq_pool = []
    for r in m3d_mcq + ctrate_mcq + lidc_mcq:
        mcq_pool.append((base.score_mcq(r, r['_dataset_source']), r))
    mcq_pool.sort(key=lambda x: x[0], reverse=True)

    selected_mcq = []
    global_seen_q = set()
    by_letter = defaultdict(list)
    for score, rec in mcq_pool:
        by_letter[rec['_subset_answer_letter']].append((score, rec))
    achievable_per_answer = min(len(by_letter[letter]) for letter in ['A', 'B', 'C', 'D'])
    target_per_answer = min(base.TARGET_PER_ANSWER, achievable_per_answer)
    for letter in ['A', 'B', 'C', 'D']:
        picked = 0
        for score, rec in by_letter[letter]:
            q = rec['conversations'][0]['value']
            if q in global_seen_q:
                continue
            selected_mcq.append(rec)
            global_seen_q.add(q)
            picked += 1
            if picked >= target_per_answer:
                break
    selected_by_letter = defaultdict(list)
    for rec in selected_mcq:
        selected_by_letter[rec['_subset_answer_letter']].append(rec)
    actual_balanced_per_answer = min(len(selected_by_letter[letter]) for letter in ['A', 'B', 'C', 'D'])
    if any(len(selected_by_letter[letter]) != actual_balanced_per_answer for letter in ['A', 'B', 'C', 'D']):
        rebalanced_mcq = []
        for letter in ['A', 'B', 'C', 'D']:
            rebalanced_mcq.extend(selected_by_letter[letter][:actual_balanced_per_answer])
        selected_mcq = rebalanced_mcq

    m3d_open = base.prepare_rows(base.read_jsonl(base.M3D_OPEN_PATH), 'm3d:open', 'open', reject_stats)
    ctrate_open = base.prepare_rows(base.read_jsonl(base.CTRATE_OPEN_PATH), 'ctrate:open', 'open', reject_stats)
    lidc_open = base.prepare_rows(base.read_jsonl(LIDC_OPEN_PATH), 'lidc:open', 'open', reject_stats)

    open_pool = []
    for r in m3d_open + ctrate_open + lidc_open:
        open_pool.append((base.score_open(r, r['_dataset_source']), r))
    open_pool.sort(key=lambda x: x[0], reverse=True)

    selected_open = []
    seen_open = set(global_seen_q)
    open_by_polarity = defaultdict(list)
    for score, rec in open_pool:
        open_by_polarity[base.classify_record_polarity(rec)].append((score, rec))
    target_open_pos = min(base.TARGET_OPEN_TOTAL // 2, len(open_by_polarity['positive']))
    target_open_neg = min(base.TARGET_OPEN_TOTAL // 2, len(open_by_polarity['negative']))
    for polarity, target_total in [('positive', target_open_pos), ('negative', target_open_neg)]:
        picked = 0
        for score, rec in open_by_polarity[polarity]:
            q = rec['conversations'][0]['value']
            if q in seen_open:
                continue
            selected_open.append(rec)
            seen_open.add(q)
            picked += 1
            if picked >= target_total:
                break
    remaining_open = base.TARGET_OPEN_TOTAL - len(selected_open)
    if remaining_open > 0:
        for bucket_name in ['unknown', 'positive', 'negative']:
            for score, rec in open_by_polarity[bucket_name]:
                q = rec['conversations'][0]['value']
                if q in seen_open:
                    continue
                selected_open.append(rec)
                seen_open.add(q)
                remaining_open -= 1
                if remaining_open <= 0:
                    break
            if remaining_open <= 0:
                break

    dataset = selected_mcq + selected_open
    random.Random(base.SEED).shuffle(dataset)

    with open(OUT_JSONL, 'w', encoding='utf-8') as f:
        for rec in dataset:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    summary = {
        'output_jsonl': str(OUT_JSONL),
        'total': len(dataset),
        'task_type_counts': dict(Counter(r['task_type'] for r in dataset)),
        'dataset_source_counts': dict(Counter(r['_dataset_source'] for r in dataset)),
        'mcq_answer_counts': dict(Counter(r.get('_subset_answer_letter') for r in dataset if r['task_type'] == 'mcq_vqa')),
        'config': {
            'target_mcq_total': base.TARGET_MCQ_TOTAL,
            'target_open_total': base.TARGET_OPEN_TOTAL,
            'target_per_answer': base.TARGET_PER_ANSWER,
            'attempted_target_per_answer': target_per_answer,
            'actual_target_per_answer': actual_balanced_per_answer,
            'actual_target_mcq_total': actual_balanced_per_answer * 4,
            'actual_target_open_positive': target_open_pos,
            'actual_target_open_negative': target_open_neg,
            'seed': base.SEED,
        },
        'paths': {
            'm3dtest': str(base.M3DTEST_PATH),
            'm3d_mcq': str(base.M3D_MCQ_PATH),
            'm3d_open': str(base.M3D_OPEN_PATH),
            'ctrate_mcq': str(base.CTRATE_MCQ_PATH),
            'ctrate_open': str(base.CTRATE_OPEN_PATH),
            'lidc_mcq': str(LIDC_MCQ_PATH),
            'lidc_open': str(LIDC_OPEN_PATH),
        },
        'rejection_stats': dict(reject_stats),
        'open_polarity_counts': dict(Counter(base.classify_record_polarity(r) for r in dataset if r['task_type'] == 'open_vqa')),
    }
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
