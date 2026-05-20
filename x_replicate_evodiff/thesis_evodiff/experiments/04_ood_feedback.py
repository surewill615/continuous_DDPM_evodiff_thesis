"""
experiments/04_ood_feedback.py
---------------------------------------
学术级 OOD 反馈放大散点图。
展示 EVODiff 如何被方差机制约束在流形支撑区附近，
而传统求解器在 OOD 区间经历灾难性误差增长。
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# seaborn 主题必须在字体设置之前调用，否则会被覆盖
sns.set_theme(style="ticks", context="paper", font_scale=1.2)

# 中文字体设置在 seaborn 之后
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DengXian', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def generate_perfect_ood_scatter():
    np.random.seed(42)

    # ==========================================
    # 1. 构造理想化的概念数据 (符合第 3 章 OOD 发散理论)
    # ==========================================
    n_points = 350

    # EVODiff 数据点 (绿点)：被方差机制约束在流形附近，距离极小，误差极小
    evo_dM = np.random.gamma(shape=1.5, scale=0.3, size=n_points)
    evo_err = 0.2 + 0.5 * evo_dM + np.random.normal(0, 0.15, n_points)
    evo_err = np.clip(evo_err, 0.1, None)

    # 传统求解器数据点 (红点)：部分在安全区，部分因切线飞逸进入中远距离，误差呈指数爆炸
    baseline_dM = np.random.gamma(shape=2.5, scale=1.2, size=n_points)
    # 核心物理公式：误差随距离呈现二次方/指数级放大 (模拟 OOD 盲区失效)
    baseline_err = (
        0.2
        + 0.5 * baseline_dM
        + 0.8 * (baseline_dM ** 2.2)
        + np.random.normal(0, 0.8, n_points)
    )
    baseline_err = np.clip(baseline_err, 0.1, None)

    # ==========================================
    # 2. 绘图渲染
    # ==========================================
    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FFFFFF')

    # 背景阴影分区
    ax.axvspan(0, 1.5, color='#2ECC71', alpha=0.08, zorder=0)
    ax.axvspan(3.0, max(baseline_dM) + 0.5, color='#E74C3C', alpha=0.08, zorder=0)

    # 散点
    ax.scatter(
        evo_dM, evo_err,
        color='#27AE60', alpha=0.7, s=40,
        edgecolors='white', linewidth=0.5,
        label='EVODiff ',
    )
    ax.scatter(
        baseline_dM, baseline_err,
        color='#C0392B', alpha=0.5, s=40,
        edgecolors='white', linewidth=0.5,
        label='基线',
    )

    # OOD 放大趋势线 (仅对基线方法进行非线性拟合)
    z = np.polyfit(baseline_dM, baseline_err, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(0, max(baseline_dM), 100)
    ax.plot(
        x_trend, p(x_trend),
        color='#2C3E50', linewidth=3, linestyle='--', alpha=0.8,
        label='OOD 反馈放大趋势线',
    )

    # 文本标注
    ax.text(
        0.75, 25,
        '流形安全区',
        color='#27AE60', ha='center', fontweight='bold', fontsize=12,
    )
    ax.text(
        4.5, 10,
        'OOD发散区',
        color='#C0392B', ha='center', fontweight='bold', fontsize=12,
    )

    # ==========================================
    # 3. 坐标轴与图例格式化
    # ==========================================
    ax.set_xlabel(
        '推断状态偏离流形的欧氏距离 $d_{\\mathcal{M}}(x)$',
        fontsize=14, fontweight='bold', labelpad=10,
    )
    ax.set_ylabel(
        '相对得分逼近误差 $E_{\\mathrm{score}}$',
        fontsize=14, fontweight='bold', labelpad=10,
    )

    ax.set_xlim(0, max(baseline_dM) + 0.2)
    ax.set_ylim(0, 35)

    ax.grid(True, linestyle=':', alpha=0.6, color='#BDC3C7')
    ax.legend(
        loc='upper left', fontsize=11, framealpha=0.95,
        edgecolor='#BDC3C7',
    )

    plt.tight_layout()
    plt.savefig("Fig_4_2_5_OOD_Concept.png", dpi=300, bbox_inches='tight')
    print("图片已保存：Fig_4_2_5_OOD_Concept.png")


if __name__ == "__main__":
    generate_perfect_ood_scatter()