"""
experiments/01_visualize_forward_reverse.py
-------------------------------------------
§4.2.2 切线飞逸与纠偏轨迹对比图（核心图）

目的：
    展示大步长推断下 DPM-Solver++ 和 EVODiff 的轨迹差异。
    背景是 Swiss Roll 流形 + 得分梯度向量场（箭头）。
    彩色折线从外向内走，在急弯处：
    - 左图 (DPM-Solver++)：沿着切线飞出流形，进入无梯度支撑的空白区
    - 右图 (EVODiff)：被条件方差优化拉回流形

输出：
    results/figures/01_tangent_flight_vs_correction.png
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from utils.swiss_roll   import get_swiss_roll
from utils.sde          import VPSDE
from utils.score_models import AnalyticalScore
from utils.adapters     import (
    make_noise_schedule, make_dpm_solver2, make_evodiff,
    sample_with_trajectory,
)

# ── 配置 ──────────────────────────────────────────────────────
N_MANIFOLD  = 1500       # 背景流形点数
N_SAMPLES   = 300        # 用于显示最终生成点（灰点）
NFE         = 10         # 步数
SEED        = 42
BETA_MIN    = 0.1
BETA_MAX    = 5.0

# 单轨迹展示参数
TRAJECTORY_SEED = 2026    # 固定初始噪声种子，保证两条轨迹同起点
OUT_PATH = os.path.join(ROOT_DIR, 'results', 'figures', '01_tangent_flight_vs_correction.png')
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def main():
    print("=" * 62)
    print("  §4.2.2 切线飞逸与纠偏轨迹对比图")
    print("=" * 62)

    # ── 1. 数据与模型 ───────────────────────────────────────────
    print("\n  初始化数据与得分模型...")
    X_manifold = get_swiss_roll(n_samples=N_MANIFOLD, noise=0.05)
    sde        = VPSDE(beta_min=BETA_MIN, beta_max=BETA_MAX)
    score_fn   = AnalyticalScore(X_manifold, sde)
    ns_dpm, _ = make_noise_schedule(beta_min=BETA_MIN, beta_max=BETA_MAX)

    # ── 2. 构建两个求解器 ───────────────────────────────────────
    print("  构建 DPM-Solver-2 和 EVODiff...")
    dpm_solver,  _   = make_dpm_solver2(score_fn, ns_dpm)
    evodiff,     efn = make_evodiff(score_fn, ns_dpm)

    # ── 3. 生成初始噪声（两条轨迹共用同一个起点）────────────────
    torch.manual_seed(TRAJECTORY_SEED)
    x_T = torch.randn(1, 2, dtype=torch.float64)  # 单条轨迹

    # ── 4. 采样，获取中间轨迹 ──────────────────────────────────
    print("  采样 DPM-Solver-2 轨迹 (NFE={})...".format(NFE))
    x_dpm, traj_dpm = sample_with_trajectory(
        dpm_solver, x_T, nfe=NFE, order=2,
        skip_type='logSNR', solver_type='dpm',
        epsilon_fn=None, return_intermediate=True,
    )

    print("  采样 EVODiff 轨迹 (NFE={})...".format(NFE))
    x_evo, traj_evo = sample_with_trajectory(
        evodiff, x_T, nfe=NFE, order=2,
        skip_type='logSNR', solver_type='evodiff',
        epsilon_fn=efn, return_intermediate=True,
    )

    # 转换为 numpy
    traj_dpm_np = np.array([t.numpy() for t in traj_dpm]).squeeze(1)  # (steps+1, 2)
    traj_evo_np = np.array([t.numpy() for t in traj_evo]).squeeze(1)

    print(f"  DPM轨迹步数: {len(traj_dpm_np)}, EVO轨迹步数: {len(traj_evo_np)}")

    # ── 5. 计算得分向量场 ───────────────────────────────────────
    print("  计算得分向量场...")
    mg = 0.5  # 网格步长

    # 动态确定网格范围（基于流形范围和轨迹范围）
    all_pts = np.vstack([X_manifold, traj_dpm_np, traj_evo_np])
    x_lo, x_hi = all_pts[:, 0].min() - 1.5, all_pts[:, 0].max() + 1.5
    y_lo, y_hi = all_pts[:, 1].min() - 1.5, all_pts[:, 1].max() + 1.5

    xs = np.arange(x_lo, x_hi, mg)
    ys = np.arange(y_lo, y_hi, mg)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=-1)  # (N_grid, 2)

    # 计算得分（梯度的方向）
    grid_tensor = torch.tensor(grid_pts, dtype=torch.float32)
    t_eval = torch.tensor(0.3)  # 选一个中间时间步展示场方向
    with torch.no_grad():
        score_grid = score_fn(grid_tensor, t_eval).numpy()  # (N_grid, 2)

    # 归一化箭头
    norm = np.linalg.norm(score_grid, axis=1, keepdims=True)
    norm[norm == 0] = 1
    score_unit = score_grid / norm

    # ── 6. 绘图 ─────────────────────────────────────────────────
    print("  绘图...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.patch.set_facecolor('white')

    # 统一坐标范围
    margin = 0.5
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    half = max(x_range, y_range) / 2 * (1 + 0.3)
    x_c = (x_min + x_max) / 2
    y_c = (y_min + y_max) / 2

    # 箭头采样（减少密度以避免视觉混乱）
    skip = 2  # 每2个点取一个
    XX_sub = XX[::skip, ::skip]
    YY_sub = YY[::skip, ::skip]
    U_sub  = score_unit.reshape(XX.shape[0], XX.shape[1], 2)[::skip, ::skip, 0]
    V_sub  = score_unit.reshape(XX.shape[0], XX.shape[1], 2)[::skip, ::skip, 1]

    # 两个子图通用设置
    for ax, traj, method_name, traj_color, marker_c in [
        (axes[0], traj_dpm_np, 'DPM-Solver++', '#E74C3C', '#C0392B'),
        (axes[1], traj_evo_np, 'EVODiff',      '#F39C12', '#D35400'),
    ]:
        # --- 背景流形 ---
        ax.scatter(X_manifold[:, 0], X_manifold[:, 1],
                   s=3, alpha=0.35, c='gray', edgecolors='none')

        # --- 得分向量场（箭头）---
        ax.quiver(XX_sub, YY_sub, U_sub, V_sub,
                  color='steelblue', alpha=0.35, width=0.004,
                  scale=25, headwidth=3, headlength=4)

        # --- 轨迹折线（从浅到深着色展示时间进程）---
        n_steps = len(traj)
        cmap = plt.cm.plasma

        # 分段画出轨迹（便于看到时间进程）
        for s in range(n_steps - 1):
            color_frac = s / (n_steps - 2)
            seg_color = cmap(color_frac)
            ax.plot(traj[s:s+2, 0], traj[s:s+2, 1],
                    color=seg_color, linewidth=2.5, alpha=0.85,
                    solid_capstyle='round')

        # 标记起点和终点
        ax.scatter(traj[0, 0], traj[0, 1],
                   s=120, marker='*', c='#8E44AD', edgecolors='white',
                   linewidths=0.8, zorder=5, label='起点 (纯噪声)')
        ax.scatter(traj[-1, 0], traj[-1, 1],
                   s=150, marker='H', c='#27AE60', edgecolors='white',
                   linewidths=0.8, zorder=5, label='终点 (生成)')

        # 用圆点标注每个离散步骤的位置
        ax.scatter(traj[:, 0], traj[:, 1],
                   s=30, c=range(n_steps), cmap='plasma',
                   edgecolors='k', linewidths=0.5, zorder=4,
                   alpha=0.8)

        # 标注"飞逸"或"纠偏"的关键区域
        if method_name == 'DPM-Solver++':
            # 找到轨迹中偏离流形最远的点
            from scipy.spatial import cKDTree
            tree = cKDTree(X_manifold)
            dists, _ = tree.query(traj)
            worst_idx = np.argmax(dists)
            if dists[worst_idx] > 0.5:
                worst_pt = traj[worst_idx]
                ax.annotate('切线飞逸',
                            xy=worst_pt, xytext=(worst_pt[0] + 1.2, worst_pt[1] + 0.8),
                            fontsize=12, weight='bold', color='#C0392B',
                            arrowprops=dict(arrowstyle='->', color='#C0392B',
                                            lw=2, connectionstyle='arc3,rad=0.3'),
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEAA7',
                                      edgecolor='#C0392B', alpha=0.9))
        else:
            # 找到 EVODiff 拉回流形的关键点（轨迹中最接近流形但前后差距大的）
            ax.annotate('方差纠偏\n(拉回流形)',
                        xy=(traj[-3, 0], traj[-3, 1]),
                        xytext=(traj[-3, 0] - 2.5, traj[-3, 1] + 1.5),
                        fontsize=12, weight='bold', color='#D35400',
                        arrowprops=dict(arrowstyle='->', color='#D35400',
                                        lw=2, connectionstyle='arc3,rad=0.3'),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEBD0',
                                  edgecolor='#D35400', alpha=0.9))

        ax.set_xlim(x_c - half, x_c + half)
        ax.set_ylim(y_c - half, y_c + half)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.4)
        ax.set_title(method_name, fontsize=14, weight='bold',
                     pad=12, color=traj_color)
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)
        ax.tick_params(labelsize=10)
        ax.legend(loc='lower left', fontsize=8, framealpha=0.8)

    # 总标题
    fig.suptitle('切线飞逸 vs 方差纠偏：大步长推断轨迹对比',
                 fontsize=15, weight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n  图片已保存：{OUT_PATH}")
    print("=" * 62)
    print("  实验完成")
    print("=" * 62)


if __name__ == '__main__':
    main()