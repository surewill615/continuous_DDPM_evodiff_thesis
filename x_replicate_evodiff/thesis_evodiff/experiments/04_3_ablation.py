"""
实验 4.3：采样脚本 (CIFAR-10, EDM预训练模型)
================================================================
固定 NFE=10，4个变体各生成 10000 张图并保存到磁盘。

消融变体：
  - Baseline:  zeta=1, eta=0   (退化为无方差优化的基础求解器)
  - Variant A: zeta优化, eta=0  (仅优化大方向)
  - Variant B: zeta=1, eta优化  (仅平滑局部梯度)
  - Full:      zeta+eta都优化   (完全体 EVODiff)

FID 计算由独立的 04_3_compute_fid.py 脚本完成，避免 CUDA 上下文冲突。
"""
import sys; sys.path.insert(0, '.')
import os
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import pickle
from dnnlib.util import open_url

from solvers.evodiff_edm import EVODiff_edm


# ═══════════════════════════════════════════════════════════════
# 1. EDM VE 噪声调度 (t = sigma 直接映射)
# ═══════════════════════════════════════════════════════════════

class EDMVESchedule:
    """
    EDM 的 Variance-Exploding 噪声调度。
    
    VE-SDE: dx = sqrt(d[sigma^2]/dt) dw
    边缘分布: p(x_t | x_0) = N(x_0, sigma_t^2 I)
    
    特性:
      - alpha_t = 1 (无收缩，纯加噪)
      - sigma_t = t  (t 直接等于 sigma)
      - kappa_t = sigma_t / alpha_t = t
      - lambda_t = log(alpha_t / sigma_t) = -log(t)
    """
    def __init__(self, sigma_max=80.0, sigma_min=0.002):
        self.total_N = 1000
        self.T = sigma_max
        self.sigma_min = sigma_min
    
    def marginal_alpha(self, t):
        return torch.ones_like(t)
    
    def marginal_std(self, t):
        return t
    
    def marginal_kappa(self, t):
        return t
    
    def marginal_lambda(self, t):
        return -torch.log(t + 1e-8)
    
    def inverse_lambda(self, lam):
        return torch.exp(-lam)


# ═══════════════════════════════════════════════════════════════
# 2. EDM 模型包装器
# ═══════════════════════════════════════════════════════════════

class EDMModelWrapper:
    """
    将 EDM 原始模型 (x0 预测) 包装为噪声预测接口。
    
    EDM 模型输出 D(x, sigma) = 预测的去噪图像 x0。
    噪声预测: epsilon = (x - D(x, sigma)) / sigma
    """
    def __init__(self, ema_model):
        self.ema = ema_model
    
    def __call__(self, x, t):
        with torch.no_grad():
            d = self.ema(x, t)
        noise = (x - d) / t.view(-1, 1, 1, 1)
        return noise


# ═══════════════════════════════════════════════════════════════
# 3. 采样工具函数
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_images(solver, noise_schedule, model_wrapper,
                  batch_size=64, total_images=10000,
                  steps=10, device='cuda'):
    """
    生成 total_images 张 CIFAR-10 图像 (32x32x3)
    """
    all_images = []
    num_batches = (total_images + batch_size - 1) // batch_size
    
    sigma_max = noise_schedule.T
    sigma_min = noise_schedule.sigma_min
    
    pbar = tqdm(range(num_batches), desc=f"Generating {total_images} images")
    for _ in pbar:
        actual_batch = min(batch_size, total_images - len(all_images) * batch_size)
        if actual_batch <= 0:
            break
        
        x_T = torch.randn(actual_batch, 3, 32, 32, device=device) * sigma_max
        
        x_gen = solver.sample(
            model_fn=model_wrapper,
            x=x_T,
            steps=steps,
            t_start=sigma_max,
            t_end=sigma_min,
            order=2,
            skip_type="edm",
            method="multistep",
            lower_order_final=True,
            denoise_to_zero=False,
            solver_type="evodiff",
        )
        
        x_gen = torch.clamp(x_gen, -1.0, 1.0)
        x_gen = (x_gen + 1.0) / 2.0
        x_gen = (x_gen * 255).byte()
        
        all_images.append(x_gen.cpu())
    
    images = torch.cat(all_images, dim=0)
    return images[:total_images]


def save_images_for_fid(images, output_dir):
    """将 (N, 3, 32, 32) uint8 tensor 保存为 PNG 文件"""
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(tqdm(images, desc=f"Saving to {output_dir}")):
        path = os.path.join(output_dir, f"{i:05d}.png")
        torchvision.utils.save_image(img.float() / 255.0, path)
    print(f"Saved {len(images)} images to {output_dir}")


# ═══════════════════════════════════════════════════════════════
# 4. 实验主函数
# ═══════════════════════════════════════════════════════════════

def run_ablation_experiment():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    print("\n[1/4] Loading EDM checkpoint...")
    with open_url('checkpoints/edm-cifar10-32x32-uncond-ve.pkl') as f:
        checkpoint = pickle.load(f)
    ema = checkpoint['ema']
    ema.eval()
    ema.to(device)
    print(f"  Model loaded: img_resolution={ema.img_resolution}, "
          f"img_channels={ema.img_channels}")
    
    noise_schedule = EDMVESchedule(sigma_max=80.0, sigma_min=0.002)
    model_wrapper = EDMModelWrapper(ema)
    
    variants = [
        ("baseline",  "Baseline (zeta=1, eta=0)"),
        ("zeta_only", "Variant A (zeta only)"),
        ("eta_only",  "Variant B (eta only)"),
        ("full",      "Full EVODiff (zeta+eta)"),
    ]
    
    NFE = 10
    BATCH_SIZE = 64
    TOTAL_IMAGES = 10000
    
    results = {}
    
    for idx, (ablation_mode, display_name) in enumerate(variants):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/4] Running: {display_name}")
        print(f"{'='*60}")
        
        solver = EVODiff_edm(
            noise_schedule=noise_schedule,
            algorithm_type="data_prediction",
            correcting_x0_fn=None,
            ablation=ablation_mode,
        )
        
        output_dir = f"results/metrics/ablation_4_3/{ablation_mode}"
        images = sample_images(
            solver=solver,
            noise_schedule=noise_schedule,
            model_wrapper=model_wrapper,
            batch_size=BATCH_SIZE,
            total_images=TOTAL_IMAGES,
            steps=NFE,
            device=device,
        )
        
        save_images_for_fid(images, output_dir)
        results[ablation_mode] = {
            "name": display_name,
            "output_dir": output_dir,
        }
        print(f"  Done: {TOTAL_IMAGES} images generated.")
    
    print(f"\n{'='*60}")
    print("SAMPLING COMPLETE")
    print(f"{'='*60}")
    for mode, info in results.items():
        print(f"  {info['name']:30s} -> {info['output_dir']}")
    print(f"\nNext: run `04_3_compute_fid.py` to compute FID scores.")
    return results


if __name__ == "__main__":
    run_ablation_experiment()