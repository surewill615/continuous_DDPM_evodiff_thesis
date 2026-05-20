import matplotlib.pyplot as plt
import os

# ═══════════════════════════════════════════════════════════════
# 1. 基础配置
# ═══════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "text.usetex": False,
})

# ═══════════════════════════════════════════════════════════════
# 2. 数据定义
# ═══════════════════════════════════════════════════════════════
col_labels = [
    "Experiment\nVariant",
    r"$\zeta$",
    r"$\eta$",
    "FID (NFE=10)" + "\n" + r"$\downarrow$",
    "Improvement\n(vs Baseline)",
]

rows = [
    ["Baseline\n(DPM-Solver++)",  r"$\mathbf{\times}$", r"$\mathbf{\times}$", "8.71", r"$\backslash$"],
    ["Variant A\n($\\zeta$-only)",  r"$\checkmark$",      r"$\mathbf{\times}$", "8.24", "+5.4%"],
    ["Variant B\n($\\eta$-only)",   r"$\mathbf{\times}$", r"$\checkmark$",      "7.36", "+15.5%"],
    ["Full EVODiff\n(Ours)",        r"$\checkmark$",      r"$\checkmark$",      r"$\mathbf{6.92}$", r"$\mathbf{+20.6\%}$"],
]

# ═══════════════════════════════════════════════════════════════
# 3. 绘图
# ═══════════════════════════════════════════════════════════════
# 【修改1】稍微增加画布高度
fig, ax = plt.subplots(figsize=(9.0, 3.2))  # 高度从2.6增加到3.2
ax.axis("off")

n_rows = len(rows)
n_cols = len(col_labels)

header_color = "#2c3e50"
header_text_color = "white"
row_colors = ["#f8f9fa", "#ecf0f1", "#f8f9fa", "#d5f5e3"]

# 【修改2】手动指定每列的宽度比例，给长单词列(第2、3列)分配更多宽度
col_widths = [0.18, 0.24, 0.24, 0.16, 0.18]

table = ax.table(
    cellText=rows,
    colLabels=col_labels,
    colWidths=col_widths, 
    cellLoc="center",
    loc="center",
)

# 样式设置
table.auto_set_font_size(False)
table.set_fontsize(9)

# 【修改3】分别控制纵向缩放和行高
# 使用scale(1.0, 1.4)减少整体纵向拉伸，但通过自定义行高来保证表头空间
table.scale(1.0, 1.4)

# 【修改4】手动设置表头行（第0行）的高度，确保文字不被截断
for key, cell in table.get_celld().items():
    row_idx, col_idx = key
    
    # 设置表头行高度更大一些
    if row_idx == 0:
        cell.set_height(0.12)  # 默认约0.08，增加到0.12
    # 数据行保持适中高度
    else:
        cell.set_height(0.09)
    
    cell.set_edgecolor("#bdc3c7")
    cell.set_linewidth(0.6)

    # 表头行样式设置
    if row_idx == 0:
        cell.set_facecolor(header_color)
        cell.set_text_props(color=header_text_color, fontweight="bold", fontsize=9)
        cell.set_linewidth(1.0)
        cell.set_edgecolor("#1a252f")
    else:
        # 数据行颜色
        cell.set_facecolor(row_colors[row_idx - 1])
        cell.set_text_props(color="#2c3e50", fontsize=8.5)

        # 最后一行的特定列加粗 (FID 和 Improvement)
        if row_idx == n_rows and col_idx in [3, 4]:
            cell.set_text_props(fontweight="bold", fontsize=9)

# 【修改5】可选：手动调整垂直对齐方式，让文字在单元格中更居中
for key, cell in table.get_celld().items():
    cell.set_text_props(va='center')  # 垂直居中

# ═══════════════════════════════════════════════════════════════
# 4. 保存
# ═══════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.5)

output_path = "ablation_fid_table.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()