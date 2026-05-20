"""
实验 4.3：FID 计算脚本
================================================================
从 04_3_ablation.py 生成的 PNG 图像目录计算 FID。

使用方法:
    python experiments/04_3_compute_fid.py

环境要求:
    pip install clean-fid torch torchvision
"""
import sys; sys.path.insert(0, '.')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # FID 计算用 GPU 加速

import torch
import numpy as np
from cleanfid import fid

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

VARIANTS = {
    "baseline":  "Baseline (zeta=1, eta=0)",
    "zeta_only": "Variant A (zeta only)",
    "eta_only":  "Variant B (eta only)",
    "full":      "Full EVODiff (zeta+eta)",
}

BASE_DIR = "thesis_evodiff/results/metrics/ablation_4_3"


def compute_fid_for_variant(mode, label):
    """计算单个变体的 FID"""
    img_dir = os.path.join(BASE_DIR, mode)
    if not os.path.isdir(img_dir):
        print(f"  [SKIP] {label}: directory not found ({img_dir})")
        return None
    n_files = len([f for f in os.listdir(img_dir) if f.endswith('.png')])
    if n_files == 0:
        print(f"  [SKIP] {label}: directory empty (0 png files)")
        return None
    print(f"  Computing FID for {label} ({n_files} images)...")
    score = fid.compute_fid(img_dir, dataset_name="cifar10", dataset_res=32,
                            mode="clean", device=torch.device("cuda"))
    print(f"  {label}: FID = {score:.4f}")
    return score


def main():
    print("=" * 60)
    print("Experiment 4.3: FID Computation")
    print("=" * 60)

    results = {}
    for mode, label in VARIANTS.items():
        score = compute_fid_for_variant(mode, label)
        results[label] = score

    print("\n" + "=" * 60)
    print("SUMMARY: FID Scores (lower is better)")
    print("=" * 60)
    for label, score in results.items():
        if score is not None:
            print(f"  {label:35s}  FID = {score:.4f}")
        else:
            print(f"  {label:35s}  FID = N/A (images not found)")

    # 保存结果
    os.makedirs("results/metrics", exist_ok=True)
    np.savez(
        "results/metrics/ablation_4_3_fid.npz",
        variants=list(results.keys()),
        fid_scores=list(results.values()),
    )
    print("\nResults saved to results/metrics/ablation_4_3_fid.npz")
    return results


if __name__ == "__main__":
    main()