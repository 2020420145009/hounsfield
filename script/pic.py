import json
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 自动解析与统计模块 (Data Extraction)
# ==========================================
file_path = "/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/mixed_datasets_rebalanced/m3d_ctrate_lidc_mix_v1/m3d_ctrate_lidc_mix_v1.jsonl"

source_counts = Counter()
q_len = defaultdict(list)
cot_len = defaultdict(list)
ans_len = defaultdict(list)
q_types = Counter()
keywords = Counter()

# 定义要提取的核心医学词汇（可自行添加）
med_words = ['nodule', 'lesion', 'density', 'spiculation', 'liver', 'kidney', 'pleural', 'effusion', 'artery', 'aorta', 'calcification', 'margin', 'mass', 'cyst']

def map_source(src):
    if not src: return 'Unknown'
    src_lower = src.lower()
    if 'm3d' in src_lower: return 'M3D'
    if 'ctrate' in src_lower: return 'CT-RATE'
    if 'lidc' in src_lower: return 'LIDC'
    return 'Other'

print("正在解析数据集文件...")
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        
        # 统计数据源
        src = map_source(data.get('_dataset_source', ''))
        source_counts[src] += 1
        
        # 统计题型 (截断括号后的冗余描述以保持图表整洁)
        qt = data.get('question_type', 'Unknown')
        if qt != 'Unknown':
            qt = qt.split('(')[0].strip()
            # 缩短一些过长的题型名称
            qt = qt.replace('Single-Point', 'S.P.')
            q_types[qt] += 1
        
        # 提取对话
        conv = data.get('conversations', [])
        q_text, gpt_text = "", ""
        for turn in conv:
            if turn['from'] == 'human': q_text = turn['value']
            if turn['from'] == 'gpt': gpt_text = turn['value']
        
        # 正则提取 <think> 和 <answer>
        think_match = re.search(r'<think>(.*?)</think>', gpt_text, re.DOTALL)
        ans_match = re.search(r'<answer>(.*?)</answer>', gpt_text, re.DOTALL)
        cot_text = think_match.group(1) if think_match else gpt_text
        ans_text = ans_match.group(1) if ans_match else ""
        
        # 记录长度 (以单词数粗略近似 Token)
        q_len[src].append(len(q_text.split()))
        cot_len[src].append(len(cot_text.split()))
        ans_len[src].append(len(ans_text.split()))
        
        # 统计医学高频词 (基于 CoT 内容)
        words = re.findall(r'\b[a-z]{4,}\b', cot_text.lower())
        for w in words:
            if w in med_words:
                keywords[w.capitalize()] += 1

# 提取排序后的数据给画图模块用
sources = ['CT-RATE', 'M3D', 'LIDC']
samples = [source_counts[s] for s in sources]
total_samples = sum(samples)

avg_q = [np.mean(q_len[s]) for s in sources]
avg_cot = [np.mean(cot_len[s]) for s in sources]
avg_ans = [np.mean(ans_len[s]) for s in sources]

top_q_types = q_types.most_common(5)
top_keywords = keywords.most_common(8)

# ==========================================
# 2. 论文级高质量图表生成模块 (Visualization)
# ==========================================
print("正在生成高清图表...")

colors_sources = ['#4C72B0', '#55A868', '#DD8452']
colors_tokens = ['#8DA0CB', '#FC8D62', '#66C2A5']

fig = plt.figure(figsize=(16, 12))
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# 创建不规则的网格布局: 左上放图+表，右上放柱状图，下面放横条图
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1])

# --- 图 (a): 数据源占比 & 精确统计表格 ---
ax1 = plt.subplot(gs[0, 0])
ax1.axis('off')
ax1.set_title('(a) Curated Dataset Composition & Statistics', fontweight='bold', pad=20)

# 上半部分：带具体数量的环形图
ax_pie = fig.add_axes([0.15, 0.65, 0.22, 0.25])
labels_with_counts = [f"{s}\n{c} ({c/total_samples*100:.1f}%)" for s, c in zip(sources, samples)]
wedges, texts = ax_pie.pie(samples, labels=labels_with_counts, colors=colors_sources, startangle=140, 
                           wedgeprops=dict(width=0.4, edgecolor='w'))
ax_pie.text(0, 0, f"Total\n{total_samples}", ha='center', va='center', fontweight='bold', fontsize=14)

# 下半部分：嵌入一个统计表格 (完美替代你的 LaTeX 表格需求)
ax_tab = fig.add_axes([0.08, 0.50, 0.35, 0.15])
ax_tab.axis('off')
table_data = [['Source', 'Samples', 'Q', 'CoT', 'Ans']]
for s, c, q, cot, ans in zip(sources, samples, avg_q, avg_cot, avg_ans):
    table_data.append([s, f"{c}", f"{q:.1f}", f"{cot:.1f}", f"{ans:.1f}"])
table_data.append(['Total', f"{total_samples}", f"{np.mean([np.mean(q_len[s]) for s in sources]):.1f}", 
                   f"{np.mean([np.mean(cot_len[s]) for s in sources]):.1f}", 
                   f"{np.mean([np.mean(ans_len[s]) for s in sources]):.1f}"])

table = ax_tab.table(cellText=table_data, colWidths=[0.25, 0.2, 0.15, 0.15, 0.15], loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.5)
# 给表头加粗变色
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#40466e')
    elif i == len(table_data) - 1:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#f0f0f0')

# --- 图 (b): Token 长度对比 ---
ax2 = plt.subplot(gs[0, 1])
x = np.arange(len(sources))
width = 0.25

ax2.bar(x - width, avg_q, width, label='Question (Q)', color=colors_tokens[0], edgecolor='white')
rects2 = ax2.bar(x, avg_cot, width, label='Reasoning Chain (CoT)', color=colors_tokens[1], edgecolor='white', hatch='//')
ax2.bar(x + width, avg_ans, width, label='Answer (Ans)', color=colors_tokens[2], edgecolor='white')

ax2.set_ylabel('Average Token Count (Words)')
ax2.set_title('(b) High-Quality Reasoning: Token Counts', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(sources)
ax2.legend(loc='upper left')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

for bar in rects2:
    height = bar.get_height()
    ax2.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', color='#D55E00')

# --- 图 (c): 问题类型分布 ---
ax3 = plt.subplot(gs[1, 0])
q_labels = [item[0] for item in top_q_types]
q_vals = [item[1] for item in top_q_types]
y_pos = np.arange(len(q_labels))
ax3.barh(y_pos, q_vals, align='center', color='#4C72B0', alpha=0.85)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(q_labels)
ax3.invert_yaxis()
ax3.set_xlabel('Number of Samples')
ax3.set_title('(c) Distribution of Top Question Types', fontweight='bold')
ax3.grid(axis='x', linestyle='--', alpha=0.7)

# --- 图 (d): 医学特征关键词分布 ---
ax4 = plt.subplot(gs[1, 1])
kw_labels = [item[0] for item in top_keywords]
kw_vals = [item[1] for item in top_keywords]
y_pos_kw = np.arange(len(kw_labels))
ax4.barh(y_pos_kw, kw_vals, align='center', color='#DD8452', alpha=0.85)
ax4.set_yticks(y_pos_kw)
ax4.set_yticklabels(kw_labels)
ax4.invert_yaxis()  
ax4.set_xlabel('Frequency in <think> Chains')
ax4.set_title('(d) Top Medical Findings Keywords in CoT', fontweight='bold')
ax4.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('v1_mix_dataset_stats_auto.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('v1_mix_dataset_stats_auto.png', dpi=300, bbox_inches='tight')
print("✅ 生成完毕！图表已保存为 v1_mix_dataset_stats_auto.pdf 和 .png")