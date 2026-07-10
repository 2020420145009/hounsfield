#!/usr/bin/env python3
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d')
OUT_DIR = ROOT / 'mixed_datasets_rebalanced' / 'm3dtest_like_v2'
OUT_JSONL = OUT_DIR / 'm3dtest_like_mcq_plus_open_v2.jsonl'
OUT_SUMMARY = OUT_DIR / 'summary.json'

M3DTEST_PATH = Path('/volume/med-train/users/shuyan/flemingvlcode/zyq/M3Dtest1.jsonl')
M3D_MCQ_PATH = Path('/volume/med-train/users/shuyan/flemingvlcode/zyq/M3DCOT/detailed_type_medical_m3d_mcq_vqa_dataset_adjusted.jsonl')
M3D_OPEN_PATH = Path('/volume/med-train/users/shuyan/flemingvlcode/zyq/M3DCOT/detail_type_medical_m3d_open_vqa_dataset3_adjusted.jsonl')
CTRATE_ROOT = Path('/volume/med-train/data/CT-RATE-DATA')
CTRATE_MCQ_PATH = CTRATE_ROOT / 'Synthesized_CoT_Dataset_ctrate' / 'ct_rate_mcq_clean.jsonl'
CTRATE_OPEN_PATH = CTRATE_ROOT / 'Synthesized_CoT_Dataset_ctrate' / 'ct_rate_open_vqa_three_tiers_5000_15000_clean.jsonl'
LIDC_MIXED_PATH = Path('/volume/med-train/users/shuyan/flemingvlcode/fzy/3d/datasets/LIDC-IDRI/Cleaned_3D_CoT_Dataset/medical_lidc_all_mixed_datasetall.jsonl')

TARGET_MCQ_TOTAL = 1352
TARGET_OPEN_TOTAL = 400
TARGET_PER_ANSWER = TARGET_MCQ_TOTAL // 4
SEED = 42

LIDC_TEMPLATE_LINES = {
    'Identify and describe the primary anatomy involved.',
    'Identify and describe the primary anatomical region/structures relevant to the task.',
    'Identify and describe the primary anatomical region/structures.',
    'Compare the mass lesion to normal adrenal gland appearance.',
    'Compare with normal thoracic anatomy and describe the abnormal evidence.',
    'Describe the abnormality type and characteristics.',
    'Determine the primary abnormality type or structured descriptive summary.',
    'Determine the primary abnormality type.',
    'Conclude the findings based on the description.',
    'Conclusion.',
}

TEMPLATE_ONLY_PATTERNS = [
    r'^Identify and describe the primary anatomy involved\.?$',
    r'^Identify and describe the primary anatomical region/structures(?: relevant to the task)?\.?$',
    r'^Compare .* to normal .*\.?$',
    r'^Compare with normal .*\.?$',
    r'^Describe the abnormality type and characteristics\.?$',
    r'^Determine the primary abnormality type.*$',
    r'^Conclude the findings.*$',
    r'^Conclusion\.?$',
]

REPORT_LEAK_PATTERNS = [
    r'\breport (states|said|says|describes|described|mentions|notes|noted)\b',
    r'\baccording to the report\b',
    r'\bthe report\b',
    r'\bpatient summary\b',
    r'\bimpression section\b',
    r'\bfindings are consistent with\b',
    r'\blabels? (are|is) marked\b',
    r'\bthe (question|prompt) states\b',
    r'\bdescribed in the report\b',
    r'\bthe report states the\b',
    r'\bmetadata\b',
    r'\bjson\b',
    r'\bintegrated_meta\b',
    r'\bclinical diagnosis\b',
    r'\bscan context\b',
    r'\bfile name\b',
    r'\bpath\b',
    r'\bpatient data\b',
    r'\bclinical history\b',
    r'\bhistory of\b',
    r'\bknown diagnosis\b',
    r'\bbiopsy proven\b',
    r'\bpathology\b',
    r'\bfrom the labels\b',
    r'\bfrom the summary\b',
    r'\bfrom the metadata\b',
    r'\bfrom the json\b',
    r'\bderived from the json\b',
    r'\bextracted from the json\b',
    r'\bbased on the (report|summary|metadata|json|clinical history|provided information)\b',
    r'\bthe provided information\b',
    r'\bthe provided data\b',
]

INSTRUCTION_LEAK_PATTERNS = [
    r'\bchoose exactly one\b',
    r'\breturn only\b',
    r'\boutput format\b',
    r'\bline [1-7]\b',
    r'\bfirst line\b',
    r'\bnon-empty lines?\b',
    r'\bpure json\b',
    r'\bjson object\b',
    r'\bslot_id\b',
    r'\bquestion_type\b',
    r'\bdimension must\b',
    r'\byou must generate\b',
    r'\bidentify and describe the primary anatomy involved\b',
    r'\bidentify and describe the primary anatomical region\b',
    r'\bformally observe and report\b',
    r'\bdetermine the primary abnormality type\b',
    r'\bevaluate the severity\b',
    r'\boutput the definitive final clinical judgment\b',
    r'\bnext-step recommendation\b',
    r'\bcompare with normal anatomy\b',
]

NEGATIVE_ANSWER_PATTERNS = [
    r'\bno\b',
    r'\bnone\b',
    r'\bnormal\b',
    r'\babsent\b',
    r'\bnegative\b',
    r'\bwithout\b',
    r'\bnot present\b',
]

POSITIVE_ANSWER_PATTERNS = [
    r'\byes\b',
    r'\bpresent\b',
    r'\babnormal\b',
    r'\blesion\b',
    r'\bmass\b',
    r'\bnodule\b',
    r'\bconsolidation\b',
    r'\bcontrast\b',
    r'\barterial\b',
    r'\bportal venous\b',
]


def read_jsonl(path):
    data = []
    if str(path).endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return raw if isinstance(raw, list) else [raw]
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def resolve_ctrate_image_path(image_value):
    image_value = str(image_value or '').strip()
    if image_value.startswith('/'):
        return image_value
    if image_value.endswith('.nii.gz') and image_value.startswith('train_'):
        stem = image_value[:-7]
        parts = stem.split('_')
        if len(parts) >= 3:
            study = '_'.join(parts[:2])
            series = '_'.join(parts[:3])
            return str(CTRATE_ROOT / 'dataset' / 'train_fixed' / study / series / image_value)
    if image_value.endswith('.nii.gz') and image_value.startswith('valid_'):
        stem = image_value[:-7]
        parts = stem.split('_')
        if len(parts) >= 3:
            study = '_'.join(parts[:2])
            series = '_'.join(parts[:3])
            return str(CTRATE_ROOT / 'dataset' / 'valid_fixed' / study / series / image_value)
    return image_value


def normalize_question(text):
    text = str(text or '')
    text = text.replace('<image>\n', ' ').replace('<image>', ' ')
    text = re.sub(r'\[Perform .*?\]\s*', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_answer_letter(text):
    s = str(text or '').strip()
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', s, flags=re.I | re.S)
    if answer_match:
        s = answer_match.group(1).strip()
    else:
        think_match = re.search(r'<think>.*?</think>', s, flags=re.I | re.S)
        if think_match:
            s = s[:think_match.start()] + s[think_match.end():]
    m = re.search(r'\b([A-D])\b', s)
    if m:
        return m.group(1)
    m = re.search(r'\(([A-Da-d])\)', s)
    if m:
        return m.group(1).upper()
    m = re.match(r'\s*([A-Da-d])[\).:\s]', s)
    if m:
        return m.group(1).upper()
    return None


def clean_choice_text(text):
    text = re.sub(r'\s+', ' ', str(text or '')).strip()
    text = re.sub(r',\s*,+', ',', text)
    text = re.sub(r',+$', '', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r',\s*', ', ', text)
    return text.strip(' .,;\n\t')

def normalize_first_reasoning_line(line):
    line = str(line or '').strip()
    first_sentence = re.split(r'(?<=[.!?])\s+', line, maxsplit=1)[0].strip()
    m = re.match(r'^This is a .*? question about (.+?)\.?$', first_sentence, flags=re.I)
    if m:
        qtype = m.group(1).strip().rstrip('.')
        return f'This is a {qtype} question.'
    m = re.match(r'^This is a (.+?) question\.?$', first_sentence, flags=re.I)
    if m:
        qtype = m.group(1).strip().rstrip('.')
        return f'This is a {qtype} question.'
    return first_sentence


def sanitize_reasoning_line(line):
    line = str(line or '').strip()
    if not line:
        return None
    if line in LIDC_TEMPLATE_LINES:
        return None
    for pat in TEMPLATE_ONLY_PATTERNS:
        if re.match(pat, line, flags=re.I):
            return None
    line = re.sub(r'^(Conclusion|Final answer|Final conclusion)\s*:?\s*', '', line, flags=re.I)
    line = re.sub(r'\s+', ' ', line).strip()
    return line if line else None


def sanitize_cot_answer(answer_text, question_text=None, force_mcq=False):
    raw = str(answer_text or '').strip()
    think_match = re.search(r'<think>(.*?)</think>', raw, flags=re.S | re.I)
    answer_match = re.search(r'<answer>(.*?)</answer>', raw, flags=re.S | re.I)

    if think_match:
        body = think_match.group(1)
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        cleaned = []
        for idx, line in enumerate(lines):
            if idx == 0:
                nl = normalize_first_reasoning_line(line)
                if nl:
                    cleaned.append(nl)
            else:
                nl = sanitize_reasoning_line(line)
                if nl:
                    cleaned.append(nl)
        if not cleaned and lines:
            cleaned = [normalize_first_reasoning_line(lines[0])]
        think_block = '<think>\n' + '\n'.join(cleaned).strip() + '\n</think>'
    else:
        think_block = None

    final_answer = answer_match.group(1).strip() if answer_match else raw
    if force_mcq and question_text is not None:
        letter = extract_answer_letter(final_answer)
        choice_map = extract_choice_map(question_text)
        if letter and letter in choice_map:
            final_answer = f'{letter}. {clean_choice_text(choice_map[letter])}'
        elif letter:
            final_answer = letter
    final_answer = re.sub(r'\s+', ' ', final_answer).strip()
    answer_block = '<answer>\n' + final_answer + '\n</answer>'

    if think_block:
        return think_block + '\n' + answer_block
    return answer_block if '<answer>' in raw or force_mcq else raw


def split_think_answer_blocks(text):
    raw = str(text or '').strip()
    think_match = re.search(r'<think>(.*?)</think>', raw, flags=re.S | re.I)
    answer_match = re.search(r'<answer>(.*?)</answer>', raw, flags=re.S | re.I)
    think_lines = []
    if think_match:
        think_lines = [ln.strip() for ln in think_match.group(1).splitlines() if ln.strip()]
    answer_text = answer_match.group(1).strip() if answer_match else raw
    return think_lines, answer_text


def contains_report_leak(text):
    s = normalize_question(text).lower()
    return any(re.search(pat, s, flags=re.I) for pat in REPORT_LEAK_PATTERNS)


def contains_instruction_leak(text):
    s = normalize_question(text).lower()
    return any(re.search(pat, s, flags=re.I) for pat in INSTRUCTION_LEAK_PATTERNS)


def normalize_for_match(text):
    s = clean_choice_text(text).lower()
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def mentions_choice_text(line, choice_text):
    hay = normalize_for_match(line)
    needle = normalize_for_match(choice_text)
    return bool(needle and hay and needle in hay)


def extract_reasoning_option_labels(think_lines):
    labels = set()
    for line in think_lines[1:]:
        for match in re.finditer(r'(?:\boption\s*|\bchoice\s*)?([A-D])[\):.]', line, flags=re.I):
            labels.add(match.group(1).upper())
    return labels


def find_option_traversal_line_index(think_lines):
    for idx, line in enumerate(think_lines[1:], start=1):
        labels = {m.group(1).upper() for m in re.finditer(r'(?:\boption\s*|\bchoice\s*)?([A-D])[\):.]', line, flags=re.I)}
        if labels == {'A', 'B', 'C', 'D'}:
            return idx
    return None


def validate_reasoning_quality(question_text, answer_text, task_type):
    think_lines, final_answer = split_think_answer_blocks(answer_text)
    raw_text = str(answer_text or '')

    if contains_report_leak(raw_text) or contains_report_leak(final_answer):
        return False, 'report_leak'
    if contains_instruction_leak(raw_text):
        return False, 'instruction_leak'

    if task_type == 'mcq_vqa':
        if len(think_lines) < 4:
            return False, 'cot_too_short'
        choice_map = extract_choice_map(question_text)
        answer_letter = extract_answer_letter(final_answer)
        chosen_text = choice_map.get(answer_letter) if answer_letter else None
        traversal_line_idx = find_option_traversal_line_index(think_lines)
        if set(choice_map.keys()) == {'A', 'B', 'C', 'D'} and traversal_line_idx is None:
            return False, 'mcq_no_option_traversal'
        if chosen_text:
            cutoff = traversal_line_idx if traversal_line_idx is not None else len(think_lines) - 1
            for line in think_lines[1:cutoff]:
                if mentions_choice_text(line, chosen_text):
                    return False, 'answer_first_cheating'
    else:
        if think_lines and len(think_lines) < 4:
            return False, 'cot_too_short'

    return True, None


def classify_answer_polarity(text):
    s = normalize_question(text).lower()
    if any(re.search(pat, s, flags=re.I) for pat in NEGATIVE_ANSWER_PATTERNS):
        return 'negative'
    if any(re.search(pat, s, flags=re.I) for pat in POSITIVE_ANSWER_PATTERNS):
        return 'positive'
    return 'unknown'


def classify_record_polarity(record):
    answer_text = str(record['conversations'][1]['value'])
    if record.get('task_type') == 'mcq_vqa':
        letter = record.get('_subset_answer_letter')
        choice_map = extract_choice_map(record['conversations'][0]['value'])
        if letter and letter in choice_map:
            return classify_answer_polarity(choice_map[letter])
    return classify_answer_polarity(answer_text)


def extract_choice_map(question):
    q = normalize_question(question)
    patterns = [
        r'Choice\s*([A-D]):\s*(.*?)(?=Choice\s*[A-D]:|$)',
        r'\b([A-D])\)\s*(.*?)(?=\b[A-D]\)|$)',
        r'\b([A-D])\.\s*(.*?)(?=\b[A-D]\.\s|$)',
    ]
    for pat in patterns:
        matches = re.findall(pat, q, flags=re.I)
        if matches:
            out = {}
            for letter, content in matches:
                cleaned = clean_choice_text(content)
                out[letter.upper()] = cleaned
            if len(out) >= 2:
                return out
    return {}


def format_mcq_question(question):
    q = normalize_question(question)
    choice_map = extract_choice_map(q)
    if not choice_map:
        return q
    stem = q
    first_choice = re.search(r'(Choice\s*A:|\bA\)|\bA\.)', q, flags=re.I)
    if first_choice:
        stem = q[:first_choice.start()].strip()
    ordered = [f'Choice {k}: {clean_choice_text(choice_map[k])}' for k in sorted(choice_map)]
    if stem.endswith('?'):
        return stem + ' ' + ', '.join(ordered)
    return stem.rstrip(' .') + '? ' + ', '.join(ordered)


def format_mcq_answer(question, answer):
    raw = str(answer).strip()
    sanitized = sanitize_cot_answer(raw, question_text=question, force_mcq=True)
    if '<answer>' in sanitized or '<think>' in sanitized:
        return sanitized
    letter = extract_answer_letter(sanitized)
    choice_map = extract_choice_map(question)
    if letter and letter in choice_map:
        return f"{letter}. {clean_choice_text(choice_map[letter])}"
    if letter:
        return letter
    return sanitized


def build_reference_patterns():
    return {
        'which phase': 4.0,
        'what phase': 4.0,
        'current phase': 4.0,
        'which plane': 4.0,
        'what plane': 4.0,
        'in which plane': 4.0,
        'which organ': 3.0,
        'which structure': 3.0,
        'where does': 3.0,
        'where is': 3.0,
        'in which': 2.5,
        'what is the shape': 3.0,
        'what is the content': 3.0,
        'is there any': 2.5,
        'are there any': 2.5,
        'which segment': 2.5,
        'which zone': 2.5,
        'axial': 2.0,
        'coronal': 2.0,
        'sagittal': 2.0,
        'arterial phase': 2.0,
        'portal venous phase': 2.0,
        'non-contrast': 2.0,
        'delayed phase': 2.0,
        'lesion': 1.5,
        'mass': 1.5,
        'laceration': 1.5,
        'calcification': 1.5,
        'appendix': 1.5,
        'renal': 1.5,
        'colon': 1.5,
        'thorax': 1.5,
    }


REFERENCE_NGRAMS = build_reference_patterns()


def text_score(question):
    q = normalize_question(question).lower()
    score = 0.0
    for term, weight in REFERENCE_NGRAMS.items():
        if term in q:
            score += weight
    return score


def score_mcq(record, source):
    q = normalize_question(record['conversations'][0]['value'])
    score = text_score(q)
    choice_map = extract_choice_map(q)
    if len(choice_map) == 4:
        score += 1.5
    if source == 'm3d:mcq':
        score += 2.0
    elif source == 'lidc:mcq':
        score += 0.8
    if 'choice a:' in q.lower():
        score += 0.5
    if any(term in q.lower() for term in ['phase', 'plane', 'which organ', 'where', 'which structure', 'which segment']):
        score += 1.0
    return score


def score_open(record, source):
    q = normalize_question(record['conversations'][0]['value'])
    a = str(record['conversations'][1]['value'])
    score = text_score(q)
    if source == 'm3d:open':
        score += 2.0
    elif source == 'lidc:open':
        score += 0.8
    if any(term in q.lower() for term in ['what', 'which', 'where', 'describe', 'identify', 'measure', 'abnormality']):
        score += 0.5
    if len(a.split()) < 180:
        score += 0.5
    return score


def prepare_mcq(record, source):
    q = format_mcq_question(record['conversations'][0]['value'])
    a = format_mcq_answer(q, record['conversations'][1]['value'])
    letter = extract_answer_letter(a)
    if letter is None:
        return None, 'missing_answer_letter'
    ok, reason = validate_reasoning_quality(q, a, 'mcq_vqa')
    if not ok:
        return None, reason
    out = dict(record)
    if source.startswith('ctrate:'):
        out['image'] = resolve_ctrate_image_path(out.get('image'))
    out['conversations'] = [
        {'from': 'human', 'value': q},
        {'from': 'gpt', 'value': a},
    ]
    out['task_type'] = 'mcq_vqa'
    out['_dataset_source'] = source
    out['_subset_answer_letter'] = letter
    return out, None


def prepare_open(record, source):
    q = normalize_question(record['conversations'][0]['value'])
    a = sanitize_cot_answer(str(record['conversations'][1]['value']).strip(), question_text=q, force_mcq=False)
    ok, reason = validate_reasoning_quality(q, a, 'open_vqa')
    if not ok:
        return None, reason
    out = dict(record)
    if source.startswith('ctrate:'):
        out['image'] = resolve_ctrate_image_path(out.get('image'))
    out['conversations'] = [
        {'from': 'human', 'value': q},
        {'from': 'gpt', 'value': a},
    ]
    out['task_type'] = 'open_vqa'
    out['_dataset_source'] = source
    return out, None


def prepare_rows(rows, source, mode, reject_stats):
    prepared = []
    for row in rows:
        rec, reason = (prepare_mcq(row, source) if mode == 'mcq' else prepare_open(row, source))
        if rec is None:
            reject_stats[f'{source}|{mode}|{reason or "rejected"}'] += 1
            continue
        prepared.append(rec)
    return prepared



def split_lidc_mixed(rows):
    mcq_rows, open_rows = [], []
    for r in rows:
        task = str(r.get('task_type', '')).lower()
        if task in {'mcq_vqa', 'multiple_choice'}:
            mcq_rows.append(r)
        elif task in {'open_vqa', 'general_qa', 'free_response'}:
            open_rows.append(r)
    return mcq_rows, open_rows

def main():
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reject_stats = Counter()

    m3d_mcq = prepare_rows(read_jsonl(M3D_MCQ_PATH), 'm3d:mcq', 'mcq', reject_stats)
    ctrate_mcq = prepare_rows(read_jsonl(CTRATE_MCQ_PATH), 'ctrate:mcq', 'mcq', reject_stats)
    lidc_mixed = read_jsonl(LIDC_MIXED_PATH)
    lidc_mcq_rows, lidc_open_rows = split_lidc_mixed(lidc_mixed)
    lidc_mcq = prepare_rows(lidc_mcq_rows, 'lidc:mcq', 'mcq', reject_stats)

    mcq_pool = []
    for r in m3d_mcq + ctrate_mcq + lidc_mcq:
        mcq_pool.append((score_mcq(r, r['_dataset_source']), r))
    mcq_pool.sort(key=lambda x: x[0], reverse=True)

    selected_mcq = []
    global_seen_q = set()
    by_letter = defaultdict(list)
    for score, rec in mcq_pool:
        by_letter[rec['_subset_answer_letter']].append((score, rec))
    achievable_per_answer = min(len(by_letter[letter]) for letter in ['A', 'B', 'C', 'D'])
    target_per_answer = min(TARGET_PER_ANSWER, achievable_per_answer)
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

    m3d_open = prepare_rows(read_jsonl(M3D_OPEN_PATH), 'm3d:open', 'open', reject_stats)
    ctrate_open = prepare_rows(read_jsonl(CTRATE_OPEN_PATH), 'ctrate:open', 'open', reject_stats)
    lidc_open = prepare_rows(lidc_open_rows, 'lidc:open', 'open', reject_stats)
    open_pool = []
    for r in m3d_open + ctrate_open + lidc_open:
        open_pool.append((score_open(r, r['_dataset_source']), r))
    open_pool.sort(key=lambda x: x[0], reverse=True)

    selected_open = []
    seen_open = set(global_seen_q)
    open_by_polarity = defaultdict(list)
    for score, rec in open_pool:
        open_by_polarity[classify_record_polarity(rec)].append((score, rec))
    target_open_pos = min(TARGET_OPEN_TOTAL // 2, len(open_by_polarity['positive']))
    target_open_neg = min(TARGET_OPEN_TOTAL // 2, len(open_by_polarity['negative']))
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
    remaining_open = TARGET_OPEN_TOTAL - len(selected_open)
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
    random.Random(SEED).shuffle(dataset)

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
            'target_mcq_total': TARGET_MCQ_TOTAL,
            'target_open_total': TARGET_OPEN_TOTAL,
            'target_per_answer': TARGET_PER_ANSWER,
            'attempted_target_per_answer': target_per_answer,
            'actual_target_per_answer': actual_balanced_per_answer,
            'actual_target_mcq_total': actual_balanced_per_answer * 4,
            'actual_target_open_positive': target_open_pos,
            'actual_target_open_negative': target_open_neg,
            'seed': SEED,
        },
        'paths': {
            'm3dtest': str(M3DTEST_PATH),
            'm3d_mcq': str(M3D_MCQ_PATH),
            'm3d_open': str(M3D_OPEN_PATH),
            'ctrate_mcq': str(CTRATE_MCQ_PATH),
            'ctrate_open': str(CTRATE_OPEN_PATH),
            'lidc_mixed': str(LIDC_MIXED_PATH),
        },
        'rejection_stats': dict(reject_stats),
        'open_polarity_counts': dict(Counter(classify_record_polarity(r) for r in dataset if r['task_type'] == 'open_vqa')),
    }
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
