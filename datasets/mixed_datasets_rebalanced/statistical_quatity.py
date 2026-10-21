import json
from collections import defaultdict

file_path = "/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/mixed_datasets_rebalanced/m3d_ctrate_lidc_mix_14.4k/m3d_ctrate_lidc_mix_14.4k.jsonl"

# 统计结构：count[dimension][question_type] = 数量
count = defaultdict(lambda: defaultdict(int))

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        dim = data.get('dimension')
        qtype = data.get('question_type')
        if dim and qtype:
            count[dim][qtype] += 1

# 输出顺序与期望的问题类型（用于匹配长名称）
dimensions_order = ['Rapid Triage', 'Standard Assessment', 'Comprehensive Diagnosis']
expected_qtypes = {
    'Rapid Triage': [
        'Single-Point Measurement',
        'Existence Judgment',
        'Single-Point Localization',
        'Binary State'
    ],
    'Standard Assessment': [
        'Feature Description',
        'Multidimensional Comprehensive Description',
        'Anomaly Classification',
        'Regional Comparison'
    ],
    'Comprehensive Diagnosis': [
        'Next-Step Recommendation',
        'Differential Diagnosis',
        'Primary Diagnosis Integration',
        'Clinical Significance Integration',
        'Severity/Risk Assessment'
    ]
}

for dim in dimensions_order:
    print(f"\n{'='*60}")
    print(f"Dimension / Difficulty: {dim}")
    print(f"{'Question Type':<55} {'Count':>6}")
    print('-'*60)
    
    qtypes_in_data = count[dim]
    seen = set()
    
    # 按期望顺序匹配实际长名称（前缀匹配）
    for short_name in expected_qtypes.get(dim, []):
        matched = None
        for full_name in qtypes_in_data:
            if full_name.startswith(short_name):
                matched = full_name
                break
        if matched:
            cnt = qtypes_in_data[matched]
            print(f"{matched:<55} {cnt:>6}")
            seen.add(matched)
        else:
            print(f"{short_name:<55} {0:>6}")
    
    # 输出未在预期列表中的类型
    for full_name in sorted(qtypes_in_data):
        if full_name not in seen:
            cnt = qtypes_in_data[full_name]
            print(f"{full_name:<55} {cnt:>6}")
    
    dim_total = sum(qtypes_in_data.values())
    print('-'*60)
    print(f"{'TOTAL':<55} {dim_total:>6}")

# 全局总计
total_all = sum(sum(qtypes.values()) for qtypes in count.values())
print(f"\nOverall Total: {total_all}")