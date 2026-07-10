#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

MCQ_TASK_TYPES = {'multiple_choice', 'mcq_vqa', 'mcq', 'general_qa'}


def read_jsonl(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def normalize_text(text):
    text = str(text or '').strip()
    return re.sub(r'\s+', ' ', text)


def normalize_free_text(text):
    text = normalize_text(text).lower()
    text = re.sub(r'</?think>|</?answer>', ' ', text)
    text = re.sub(r'[^a-z0-9\u4e00-\u9fff\s]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_choice_letter(text):
    text = str(text or '').strip()
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, flags=re.I | re.S)
    if answer_match:
        text = answer_match.group(1).strip()
    else:
        think_match = re.search(r'<think>.*?</think>', text, flags=re.I | re.S)
        if think_match:
            text = text[:think_match.start()] + text[think_match.end():]
    text = normalize_text(text)
    m = re.search(r'\b([A-D])\b', text, flags=re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r'\(([A-Da-d])\)', text)
    if m:
        return m.group(1).upper()
    m = re.match(r'\s*([A-Da-d])[\).:\s]', text)
    if m:
        return m.group(1).upper()
    txt = text.lower()
    if txt in {'yes', 'no'}:
        return txt.upper()
    return None


def infer_task_type(row):
    for key in ('task_type', 'type', '_task_type'):
        if key in row and row[key]:
            return str(row[key]).lower()
    q = ''
    if 'question' in row:
        q = str(row['question'])
    elif 'conversations' in row and row['conversations']:
        q = str(row['conversations'][0].get('value', ''))
    ql = q.lower()
    if 'choice a:' in ql or '<multiple_choice>' in ql or re.search(r'\bA\)', q):
        return 'mcq_vqa'
    return 'open_vqa'


def extract_gold(row):
    for key in ('answer', 'gt', 'label', 'gold', 'target', 'ground_truth', 'prediction_gt', 'reference'):
        if key in row and row[key] not in (None, ''):
            return row[key]
    if 'conversations' in row and len(row['conversations']) >= 2:
        return row['conversations'][1].get('value', '')
    return ''


def extract_pred(row):
    for key in ('prediction_clean', 'prediction', 'pred', 'response', 'output', 'prediction_raw'):
        if key in row and row[key] not in (None, ''):
            return row[key]
    return ''


def bleu_score(pred, ref):
    pred_tokens = normalize_free_text(pred).split()
    ref_tokens = normalize_free_text(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    score = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)
    return score * 1000.0


def _lcs_len(a_tokens, b_tokens):
    if not a_tokens or not b_tokens:
        return 0
    dp = [0] * (len(b_tokens) + 1)
    for i in range(1, len(a_tokens) + 1):
        prev = 0
        for j in range(1, len(b_tokens) + 1):
            temp = dp[j]
            if a_tokens[i - 1] == b_tokens[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def rouge_l_score(pred, ref):
    pred_tokens = normalize_free_text(pred).split()
    ref_tokens = normalize_free_text(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1 * 1000.0


def evaluate_rows(rows):
    mcq_total = mcq_correct = 0
    open_total = 0
    bleu_vals = []
    rouge_vals = []
    task_counter = Counter()
    source_counter = Counter()
    details = []
    errors = []

    for idx, row in enumerate(rows):
        task_type = infer_task_type(row)
        task_counter[task_type] += 1
        if '_dataset_source' in row:
            source_counter[str(row['_dataset_source'])] += 1
        gold = extract_gold(row)
        pred = extract_pred(row)

        detail = dict(row)
        detail['_eval_index'] = idx
        detail['_eval_task_type'] = task_type
        detail['_gold'] = gold
        detail['_pred'] = pred

        if task_type in MCQ_TASK_TYPES:
            gold_letter = extract_choice_letter(gold)
            pred_letter = extract_choice_letter(pred)
            detail['_gold_choice'] = gold_letter
            detail['_pred_choice'] = pred_letter
            if gold_letter is not None:
                mcq_total += 1
                is_correct = pred_letter == gold_letter
                detail['_is_correct'] = is_correct
                if is_correct:
                    mcq_correct += 1
                else:
                    detail['_error_type'] = 'mcq_mismatch'
                    errors.append(detail)
            else:
                detail['_is_correct'] = None
        else:
            open_total += 1
            bleu = bleu_score(pred, gold)
            rouge_l = rouge_l_score(pred, gold)
            detail['_bleu'] = bleu
            detail['_rougeL'] = rouge_l
            bleu_vals.append(bleu)
            rouge_vals.append(rouge_l)
        details.append(detail)

    summary = {
        'num_rows': len(rows),
        'task_type_counts': dict(task_counter),
        'dataset_source_counts': dict(source_counter),
        'mcq_total': mcq_total,
        'mcq_correct': mcq_correct,
        'mcq_accuracy': (mcq_correct / mcq_total) if mcq_total else 0.0,
        'open_total': open_total,
        'open_bleu': (sum(bleu_vals) / len(bleu_vals)) if bleu_vals else 0.0,
        'open_rougeL': (sum(rouge_vals) / len(rouge_vals)) if rouge_vals else 0.0,
        'num_errors': len(errors),
    }
    return summary, details, errors


def summarize(rows):
    summary, _, _ = evaluate_rows(rows)
    return summary

def default_output_paths(input_path: Path):
    stem = str(input_path)[:-6] if str(input_path).endswith('.jsonl') else str(input_path)
    return {
        'summary_json': stem + '_metrics.json',
        'details_jsonl': stem + '_eval_details.jsonl',
        'errors_jsonl': stem + '_errors.jsonl',
        'append_summary_jsonl': str(input_path.parent / 'metrics_summary.jsonl'),
    }


def generate_report_bundle(input_path, experiment_name=None, summary_json=None, append_summary_jsonl=None, details_jsonl=None, errors_jsonl=None):
    input_path = Path(input_path)
    rows = read_jsonl(input_path)
    summary, details, errors = evaluate_rows(rows)
    summary['input'] = str(input_path)
    if experiment_name:
        summary['experiment_name'] = experiment_name

    defaults = default_output_paths(input_path)
    summary_json = summary_json or defaults['summary_json']
    append_summary_jsonl = append_summary_jsonl or defaults['append_summary_jsonl']
    details_jsonl = details_jsonl or defaults['details_jsonl']
    errors_jsonl = errors_jsonl or defaults['errors_jsonl']

    summary_json_path = Path(summary_json)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    append_path = Path(append_summary_jsonl)
    append_path.parent.mkdir(parents=True, exist_ok=True)
    with append_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(summary, ensure_ascii=False) + '\n')

    write_jsonl(Path(details_jsonl), details)
    write_jsonl(Path(errors_jsonl), errors)

    return {
        'summary': summary,
        'summary_json': str(summary_json_path),
        'append_summary_jsonl': str(append_path),
        'details_jsonl': str(Path(details_jsonl)),
        'errors_jsonl': str(Path(errors_jsonl)),
    }


def main():
    parser = argparse.ArgumentParser(description='Compute MCQ accuracy and open-answer BLEU / ROUGE-L from jsonl results')
    parser.add_argument('--input', required=True, help='Input result jsonl file')
    parser.add_argument('--summary_json', default=None, help='Optional path to save summary json')
    parser.add_argument('--append_summary_jsonl', default=None, help='Optional path to append one-line summary jsonl')
    parser.add_argument('--details_jsonl', default=None, help='Optional path to save per-row evaluation details jsonl')
    parser.add_argument('--errors_jsonl', default=None, help='Optional path to save error rows jsonl')
    parser.add_argument('--experiment_name', default=None, help='Optional experiment name to include in saved summary')
    parser.add_argument('--long_text_mode', default='metrics', choices=['metrics'], help='Compatibility flag; kept for older commands')
    args = parser.parse_args()
    bundle = generate_report_bundle(
        args.input,
        experiment_name=args.experiment_name,
        summary_json=args.summary_json,
        append_summary_jsonl=args.append_summary_jsonl,
        details_jsonl=args.details_jsonl,
        errors_jsonl=args.errors_jsonl,
    )
    print(json.dumps(bundle['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
