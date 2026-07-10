import json
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
import squarify  # 用于生成块分布图 (Treemap)
from wordcloud import WordCloud  # 用于生成词云

# ==========================================
# 1. 自动解析与统计模块 (Data Extraction)
# ==========================================
file_path = "m3d_ctrate_lidc_mix_v1-2.jsonl"
file_path = "/volume/med-train/users/shuyan/flemingvlcode/flemingvl8b_3d/mixed_datasets_rebalanced/m3d_ctrate_lidc_mix_v1/m3d_ctrate_lidc_mix_v1.jsonl"

source_counts = Counter()
q_len = defaultdict(list)
cot_len = defaultdict(list)
ans_len = defaultdict(list)
q_types = Counter()

# 用一个长字符串来收集所有的医学词汇，用于生成词云
all_medical_words = ""

# 核心医学词汇过滤库
med_words = ['nodule', 'lesion', 'density', 'spiculation', 'liver', 'kidney', 
             'pleural', 'effusion', 'artery', 'aorta', 'calcification', 'margin', 
             'mass', 'cyst', 'hemorrhage', 'opacity', 'ground-glass', 'consolidation']
med_words=[]
def map_source(src):
    if not src: return 'Unknown'
    src_lower = src.lower()
    if 'm3d' in src_lower: return 'M3D'
    if 'ctrate' in src_lower: return 'CT-RATE'
    if 'lidc' in src_lower: return 'LIDC'
    return 'Other'

print("正在解析数据集...")
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        
        src = map_source(data.get('_dataset_source', ''))
        source_counts[src] += 1
        
        qt = data.get('question_type', 'Unknown')
        if qt != 'Unknown':
            qt = qt.split('(')[0].strip()
            # 缩写以适应块状图
            qt = qt.replace('Single-Point', 'S.P.')
            q_types[qt] += 1
        
        conv = data.get('conversations', [])
        q_text, gpt_text = "", ""
        for turn in conv:
            if turn['from'] == 'human': q_text = turn['value']
            if turn['from'] == 'gpt': gpt_text = turn['value']
        
        think_match = re.search(r'<think>(.*?)</think>', gpt_text, re.DOTALL)
        ans_match = re.search(r'<answer>(.*?)</answer>', gpt_text, re.DOTALL)
        cot_text = think_match.group(1) if think_match else gpt_text
        ans_text = ans_match.group(1) if ans_match else ""
        
        q_len[src].append(len(q_text.split()))
        cot_len[src].append(len(cot_text.split()))
        ans_len[src].append(len(ans_text.split()))
        
        # 提取用于词云的词汇
        words = re.findall(r'\b[a-z]{4,}\b', cot_text.lower())
        for w in words:
            if w in med_words:
                all_medical_words += f"{w} "

sources = ['CT-RATE', 'M3D', 'LIDC']
samples = [source_counts[s] for s in sources]
total_samples = sum(samples)

avg_q = [np.mean(q_len[s]) for s in sources]
avg_cot = [np.mean(cot_len[s]) for s in sources]
avg_ans = [np.mean(ans_len[s]) for s in sources]

# ==========================================
# 2. 论文级可视化生成 (Visualization)
# ==========================================
print("正在生成块分布图与词云组合图...")

fig = plt.figure(figsize=(16, 12))
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1.2])

colors_sources = ['#4C72B0', '#55A868', '#DD8452']
colors_tokens = ['#8DA0CB', '#FC8D62', '#66C2A5']

# --- 图 (a): 环形图 & 统计表格 (保持不变) ---
ax1 = plt.subplot(gs[0, 0])
ax1.axis('off')
ax1.set_title('(a) Curated Dataset Composition & Statistics', fontweight='bold', pad=20)

ax_pie = fig.add_axes([0.15, 0.65, 0.22, 0.25])
labels_with_counts = [f"{s}\n{c} ({c/total_samples*100:.1f}%)" for s, c in zip(sources, samples)]
wedges, texts = ax_pie.pie(samples, labels=labels_with_counts, colors=colors_sources, startangle=140, 
                           wedgeprops=dict(width=0.4, edgecolor='w'))
ax_pie.text(0, 0, f"Total\n{total_samples}", ha='center', va='center', fontweight='bold', fontsize=14)

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
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#40466e')
    elif i == len(table_data) - 1:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#f0f0f0')

# --- 图 (b): Token 长度对比柱状图 (保持不变) ---
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

# ==========================================
# ★ 图 (c): 块分布图 (Treemap) 替代原条形图
# ==========================================
ax3 = plt.subplot(gs[1, 0])
# 提取 Top 6 的题型
top_q_types = q_types.most_common(6)
q_labels = [f"{item[0]}\n({item[1]})" for item in top_q_types]
q_sizes = [item[1] for item in top_q_types]
# 定义一套柔和好看的调色盘
treemap_colors = ['#8DA0CB', '#66C2A5', '#FC8D62', '#E78AC3', '#A6D854', '#FFD92F']

# 画 Treemap
squarify.plot(sizes=q_sizes, label=q_labels, color=treemap_colors, alpha=0.85, 
              ax=ax3, text_kwargs={'fontsize': 11, 'weight': 'bold', 'color': 'white', 'wrap': True})
ax3.axis('off') # Treemap 不需要坐标轴
ax3.set_title('(c) Distribution of Question Types (Treemap)', fontweight='bold', pad=15)

# ==========================================
# ★ 图 (d): 医学特征词云 (WordCloud) 替代原条形图
# ==========================================
ax4 = plt.subplot(gs[1, 1])

# 配置生成高颜值词云
wordcloud = WordCloud(width=800, height=500, 
                      background_color='white', 
                      colormap='copper', # 铜色系，类似你参考图右下角的颜色
                      max_words=80, 
                      contour_width=0, 
                      contour_color='steelblue',
                      prefer_horizontal=0.9).generate(all_medical_words)

ax4.imshow(wordcloud, interpolation='bilinear')
ax4.axis('off') # 词云不需要坐标轴
ax4.set_title('(d) Medical Findings Keywords (Word Cloud)', fontweight='bold', pad=15)

# ==========================================
plt.tight_layout()
plt.savefig('v1_mix_dataset_advanced_viz.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('v1_mix_dataset_advanced_viz.png', dpi=300, bbox_inches='tight')
print("✅ 生成完毕！高级组合图已保存为 v1_mix_dataset_advanced_viz.pdf/.png")