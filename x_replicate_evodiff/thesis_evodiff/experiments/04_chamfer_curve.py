"""
experiments/04_chamfer_curve.py
---------------------------------
§4.2.4 Chamfer 距离随 NFE 变化曲线（双轨版本）

支持两种 score 模式：
  --score analytical  使用解析分数（轨道一）
  --score mlp         使用 MLP 拟合分数（轨道二）

输出：
  results/figures/04_chamfer_curve_{analytical,mlp}.png
  results/metrics/04_chamfer_curve_{analytical,mlp}.csv
"""

import os, sys, time, argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from utils.swiss_roll   import get_swiss_roll
from utils.sde          import VPSDE
from utils.score_models import AnalyticalScore
from utils.adapters     import (
    make_noise_schedule, make_ddim, make_dpm_solver2,
    make_unipc, make_evodiff, sample_with_trajectory,
)

# ── 配置（可调整） ────────────────────────────────────────────
NFE_LIST    = [3, 5, 8, 10, 15, 20]
N_SAMPLES   = 1000
N_MANIFOLD  = 2000
BETA_MIN    = 0.1
BETA_MAX    = 10.0
SEED        = 42

SOLVER_STYLE = {
    'DDIM':         {'color': '#E74C3C', 'marker': 'o', 'linestyle': '-'},
    'DPM-Solver-2': {'color': '#3498DB', 'marker': 's', 'linestyle': '-'},
    'UniPC':        {'color': '#2ECC71', 'marker': '^', 'linestyle': '-'},
    'EVODiff':      {'color': '#F39C12', 'marker': 'D', 'linestyle': '-'},
}


def compute_chamfer_kdtree(x_gen_np, X_manifold_np):
    from scipy.spatial import cKDTree
    tree = cKDTree(X_manifold_np)
    dists, _ = tree.query(x_gen_np, k=1)
    return float(dists.mean())


def run_experiment(score_mode='analytical'):
    suffix = score_mode
    print(f"\n{'='*60}")
    print(f"  轨道—{'解析' if score_mode=='analytical' else 'MLP拟合'}分数")
    print(f"{'='*60}")

    OUT_PATH = os.path.join(ROOT_DIR, 'results', 'figures', f'04_chamfer_curve_{suffix}.png')
    CSV_PATH = os.path.join(ROOT_DIR, 'results', 'metrics', f'04_chamfer_curve_{suffix}.csv')
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    # ── 1. 数据与模型 ─────────────────────────────────────────
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_manifold = get_swiss_roll(n_samples=N_MANIFOLD)
    sde = VPSDE(beta_min=BETA_MIN, beta_max=BETA_MAX)

    if score_mode == 'analytical':
        score_fn = AnalyticalScore(X_manifold, sde)
        print(f"  解析分数 (beta_max={BETA_MAX})")
    else:
        from utils.mlp_score import load_mlp_score
        _, score_fn = load_mlp_score()
        print(f"  MLP 分数加载成功 (beta_max={BETA_MAX})")

    ns_dpm, ns_uni = make_noise_schedule(beta_min=BETA_MIN, beta_max=BETA_MAX)

    # ── 2. 构建求解器 ─────────────────────────────────────────
    ddim,     _   = make_ddim(score_fn, ns_dpm)
    dpm2,     _   = make_dpm_solver2(score_fn, ns_dpm)
    unipc,    _   = make_unipc(score_fn, ns_uni)
    evodiff,  efn = make_evodiff(score_fn, ns_dpm)

    solvers = [
        ('DDIM',         ddim,    'dpm',    1, None),
        ('DPM-Solver-2', dpm2,    'dpm',    2, None),
        ('UniPC',        unipc,   'unipc',  2, None),
        ('EVODiff',      evodiff, 'evodiff', 2, efn),
    ]

    # ── 3. 核心采样循环 ───────────────────────────────────────
    results = np.full((len(solvers), len(NFE_LIST)), np.nan)

    for s_idx, (name, solver, s_type, order, mfn) in enumerate(solvers):
        print(f"\n  [{name}]")
        for n_idx, nfe in enumerate(NFE_LIST):
            torch.manual_seed(SEED)
            x_T = torch.randn(N_SAMPLES, 2, dtype=torch.float64)

            x_gen, _ = sample_with_trajectory(
                solver, x_T, nfe=nfe, order=order,
                skip_type='logSNR', solver_type=s_type,
                epsilon_fn=mfn, return_intermediate=False,
            )

            cd = compute_chamfer_kdtree(x_gen.numpy(), X_manifold)
            results[s_idx, n_idx] = cd
            print(f"    NFE={nfe:2d}  |  Chamfer={cd:.6f}")

    # ── 4. 输出 ──────────────────────────────────────────────
    header = "solver," + ",".join(str(n) for n in NFE_LIST)
    lines = [header]
    for s_idx, (name, *_) in enumerate(solvers):
        vals = ",".join(f"{results[s_idx, n_idx]:.6f}" for n_idx in range(len(NFE_LIST)))
        lines.append(f"{name},{vals}")
    with open(CSV_PATH, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  CSV 已保存：{CSV_PATH}")

    print(f"\n  {'求解器':15s}", end="")
    for nfe in NFE_LIST:
        print(f"  NFE={nfe:2d}", end="")
    print()
    for s_idx, (name, *_) in enumerate(solvers):
        print(f"  {name:15s}", end="")
        for n_idx in range(len(NFE_LIST)):
            print(f"  {results[s_idx, n_idx]:.6f}", end="")
        print()

    # ── 5. 画图 ──────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.patch.set_facecolor('white')

    for s_idx, (name, *_) in enumerate(solvers):
        style = SOLVER_STYLE[name]
        cd_vals = results[s_idx, :]
        ax.plot(NFE_LIST, cd_vals, color=style['color'],
                marker=style['marker'], linestyle=style['linestyle'],
                linewidth=2, markersize=8, markeredgecolor='black',
                markeredgewidth=0.5, label=name)
        for n_idx, nfe in enumerate(NFE_LIST):
            ax.annotate(f"{cd_vals[n_idx]:.4f}", xy=(nfe, cd_vals[n_idx]),
                        xytext=(0, -14), textcoords='offset points',
                        fontsize=7, ha='center', va='top', color=style['color'])

    ax.set_yscale('log')
    ax.set_xlabel('NFE (Number of Function Evaluations)', fontsize=12)
    ax.set_ylabel('Chamfer Distance (log scale)', fontsize=12)
    title_label = 'Analytical score' if score_mode == 'analytical' else 'MLP score'
    ax.set_title(f'Chamfer Distance vs. NFE - Swiss Roll ({title_label})',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim(min(NFE_LIST) - 0.5, max(NFE_LIST) + 0.5)
    ax.set_xticks(NFE_LIST)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图片已保存：{OUT_PATH}")

    return results, solvers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score', choices=['analytical', 'mlp', 'both'],
                        default='both',
                        help='使用解析分数、MLP 分数，或两者都跑')
    args = parser.parse_args()

    modes = ['analytical', 'mlp'] if args.score == 'both' else [args.score]
    for mode in modes:
        run_experiment(score_mode=mode)

    print("\n" + "=" * 55)
    print(f"  实验完成！轨迹：{args.score}")
    print("=" * 55)


if __name__ == '__main__':
    main()