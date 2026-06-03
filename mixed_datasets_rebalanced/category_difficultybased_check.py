# import json
# from collections import defaultdict

# file_path = "/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/mixed_datasets_rebalanced/m3d_ctrate_lidc_mix_14.4k/m3d_ctrate_lidc_mix_14.4k.jsonl"

# # 统计：count[task_type][dimension] = 数量
# stats = defaultdict(lambda: defaultdict(int))
# task_types_set = set()
# dimensions_set = set()

# with open(file_path, 'r', encoding='utf-8') as f:
#     for line in f:
#         data = json.loads(line)
#         task = data.get('task_type')
#         dim = data.get('dimension')
#         if task and dim:
#             stats[task][dim] += 1
#             task_types_set.add(task)
#             dimensions_set.add(dim)

# # 确定输出顺序
# dim_order = ['Rapid Triage', 'Standard Assessment', 'Comprehensive Diagnosis']
# # 只保留实际出现的维度，并按顺序
# dim_cols = [d for d in dim_order if d in dimensions_set]
# # 添加可能出现的其他维度（如果有）
# other_dims = sorted(dimensions_set - set(dim_order))
# dim_cols.extend(other_dims)

# task_order = sorted(task_types_set)

# # 打印表格
# header = f"{'task_type':<20}" + "".join(f"{dim:>25}" for dim in dim_cols) + f"{'Total':>8}"
# print(header)
# print('-' * len(header))

# grand_total = 0
# for task in task_order:
#     row = f"{task:<20}"
#     task_total = 0
#     for dim in dim_cols:
#         cnt = stats[task].get(dim, 0)
#         row += f"{cnt:>25}"
#         task_total += cnt
#     row += f"{task_total:>8}"
#     print(row)
#     grand_total += task_total

# print('-' * len(header))
# # 打印各难度总计
# total_row = f"{'TOTAL':<20}"
# for dim in dim_cols:
#     col_total = sum(stats[t].get(dim, 0) for t in task_order)
#     total_row += f"{col_total:>25}"
# total_row += f"{grand_total:>8}"
# print(total_row)
import json

file_path = "/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/mixed_datasets_rebalanced/m3d_ctrate_lidc_mix_v1/m3d_ctrate_lidc_mix_v1.jsonl"

target_task = "mcq_vqa"
target_dim = "Standard Assessment"
max_examples = 5  # 需要展示的例数

found = 0
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data.get('task_type') == target_task and data.get('dimension') == target_dim:
            found += 1
            print(f"--- Example {found} ---")
            print(f"ID: {data.get('id')}")
            # 打印对话中的问题（human部分）
            for conv in data.get('conversations', []):
                if conv.get('from') == 'human':
                    print(f"Question: {conv.get('value')}")
                elif conv.get('from') == 'gpt':
                    # 可选：只显示简短答案，避免冗长
                    answer = conv.get('value', '')
                    # 提取 <answer> 标签内容作为简要答案
                    import re
                    match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL)
                    brief = match.group(1).strip() if match else answer[:200]
                    print(f"Answer (brief): {brief}")
            print()
            if found >= max_examples:
                break

print(f"Total matching records found (truncated to first {max_examples}): {found}")