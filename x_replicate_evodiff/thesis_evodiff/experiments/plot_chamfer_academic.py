"""
experiments/plot_chamfer_academic.py
---------------------------------------
学术级 Chamfer 距离 vs NFE 曲线图
使用真实实验数据 (analytical score 轨道)
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_chamfer_curve():
    # ── 1. NFE 测试节点 ─────────────────────────────────────
    nfe_list = np.array([3, 5, 8, 10, 15, 20])

    # ── 2. 真实实验数据 (analytical score 轨道) ─────────────
    cd_ddim  = np.array([0.017066, 0.014834, 0.012302, 0.010743, 0.009662, 0.009436])
    cd_dpm   = np.array([0.018666, 0.015859, 0.011322, 0.009849, 0.009033, 0.008649])
    cd_unipc = np.array([0.118661, 0.012314, 0.009957, 0.009568, 0.008732, 0.008460])
    cd_evo   = np.array([0.029624, 0.019773, 0.014297, 0.014356, 0.009622, 0.009569])

    # ── 3. 画图 ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('white')

    styles = [
        {"name": "DDIM (一阶基线)",    "data": cd_ddim,  "color": "#7F8C8D", "marker": "v", "ls": "--", "lw": 2},
        {"name": "DPM-Solver-2",       "data": cd_dpm,   "color": "#D9534F", "marker": "s", "ls": "-",  "lw": 2.5},
        {"name": "UniPC",              "data": cd_unipc, "color": "#337AB7", "marker": "^", "ls": "-",  "lw": 2.5},
        {"name": "EVODiff (本文方法)", "data": cd_evo,   "color": "#5CB85C", "marker": "*", "ls": "-",  "lw": 3.5, "ms": 12},
    ]

    for s in styles:
        ax.plot(nfe_list, s["data"], label=s["name"], color=s["color"],
                marker=s["marker"], linestyle=s["ls"], linewidth=s["lw"],
                markersize=s.get("ms", 8), alpha=0.9)

    # ── 4. 图表格式化 ───────────────────────────────────────
    ax.set_yscale('log')

    ax.set_xlabel('函数评估次数 (NFE)', fontsize=14, fontweight='bold')
    ax.set_ylabel('流形偏离误差 (Chamfer Distance) - Log Scale', fontsize=14, fontweight='bold')
    ax.set_title('不同求解器在 Swiss Roll 流形上的 Chamfer 距离衰减曲线', fontsize=16, fontweight='bold', pad=15)

    ax.set_xticks(nfe_list)
    ax.set_xticklabels(nfe_list, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.grid(True, which="both", ls="--", alpha=0.3, color='#95a5a6')
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9, edgecolor='#BDC3C7')

    # ── 5. 高亮低 NFE 区间 ─────────────────────────────────
    ax.axvspan(3, 8, color='#F1C40F', alpha=0.1, zorder=0)
    ax.text(5.5, max(cd_unipc) * 0.8, '大步长 (Low-NFE) 高风险崩溃区',
            fontsize=12, color='#D35400', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("Fig_4_2_4_Chamfer_Curve.png", dpi=300)
    print("曲线图已生成：Fig_4_2_4_Chamfer_Curve.png")


if __name__ == "__main__":
    plot_chamfer_curve()