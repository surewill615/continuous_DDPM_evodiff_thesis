"""
utils/mlp_score.py
------------------
MLP Score Model — denoising score matching (DSM) for noise prediction.

Network: x (2D) + t (1D) concat -> 3D -> SiLU(128) -> SiLU(128) -> 2D noise pred
Training: online batch generation from Swiss Roll, VP-SDE forward diffusion
Loss: MSE(eps_theta(x_t, t), eps)  (denoising score matching)
Inference: score(x, t) = -eps_theta(x, t) / sigma_t

修复记录：
  1. 训练循环：x_t, _ = sde.forward_sample(x0, t) 改为 x_t, noise = ...
     确保 x_t 和 noise 来自同一次采样（原版两者不一致导致 loss 无法收敛）
  2. MLPScore.__call__：marginal_prob 返回的 std 形状 (B,1)，
     去掉多余的 unsqueeze(-1) 避免维度重复
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Path setup (ensure import from project root)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from utils.swiss_roll import get_swiss_roll
from utils.sde        import VPSDE


class MLPNet(nn.Module):
    def __init__(self, data_dim=2, time_dim=16, hidden_dim=256, out_dim=2):
        super().__init__()
        self.time_dim = time_dim
        # 时间编码的频率
        half = time_dim // 2
        freqs = torch.exp(
            torch.arange(half, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        self.register_buffer('freqs', freqs)   # (half,)

        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, t):
        # t: (B,) → 正弦时间编码 (B, time_dim)
        if t.dim() == 1:
            t_emb = t.unsqueeze(-1) * self.freqs.unsqueeze(0)  # (B, half)
            t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)  # (B, time_dim)
        inp = torch.cat([x, t_emb], dim=-1)   # (B, data_dim + time_dim)
        return self.net(inp)


class MLPScore:
    """
    推理接口：把训练好的 MLPNet 包装成 score_fn(x, t) → score。

    接口和 AnalyticalScore 完全一致，可以直接替换传入 adapters.py。
    """
    def __init__(self, model: MLPNet, sde: VPSDE):
        self.model = model
        self.sde   = sde
        self.model.eval()

    @torch.no_grad()
    def __call__(self, x, t):
        """
        x: (B, 2) float32
        t: float / 0-dim tensor / (B,) tensor
        返回: (B, 2) score
        """
        # 统一 t 为 (B,) float32
        if isinstance(t, (float, int)):
            t = torch.full((x.shape[0],), float(t), dtype=torch.float32)
        elif isinstance(t, torch.Tensor):
            if t.dim() == 0:
                t = t.expand(x.shape[0]).float()
            else:
                t = t.float()

        x = x.float()

        eps_pred = self.model(x, t)          # (B, 2) noise prediction

        _, std = self.sde.marginal_prob(x, t)
        # 取第一列作为标量 sigma_t，shape → (B, 1)
        sigma_t = std
        score = -eps_pred / (sigma_t + 1e-8)  # (B, 2)
        return score


def train_mlp_score(
    n_steps      = 10_000,
    batch_size   = 256,
    lr           = 1e-3,
    data_n       = 2000,
    seed         = 42,
    ckpt_path    = None,
    print_every  = 1000,
):
    """
    训练 MLP score 网络。

    核心修复：使用 sde.forward_sample 返回的 noise（而非另外采样的 noise）
    作为训练目标，确保 x_t 和 noise 来自同一次前向过程。
    """
    if ckpt_path is None:
        ckpt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'checkpoints', 'mlp_score.pth'
        )
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    sde     = VPSDE(beta_min=0.1, beta_max=5.0)
    X_ref   = get_swiss_roll(n_samples=data_n, noise=0.05)   # (N, 2)
    X_ref_t = torch.tensor(X_ref, dtype=torch.float32)

    model = MLPNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    print("=" * 55)
    print("  MLP Score Training")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Steps:  {n_steps},  Batch: {batch_size},  lr={lr}")
    print("=" * 55)

    for step in range(1, n_steps + 1):
        # ── 在线生成训练数据 ──────────────────────────────────
        idx = torch.randint(0, data_n, (batch_size,))
        x0  = X_ref_t[idx]                          # (B, 2)
        t   = torch.rand(batch_size) * 0.99 + 0.01  # (B,) ∈ [0.01, 1.0]

        # 关键修复：x_t 和 noise 来自同一次 forward_sample
        # sde.forward_sample 返回 (x_t, noise)
        x_t, noise = sde.forward_sample(x0, t)      # 各 (B, 2)

        # ── 前向 + 损失 ───────────────────────────────────────
        eps_pred = model(x_t, t)                     # (B, 2)
        loss     = nn.functional.mse_loss(eps_pred, noise)

        # ── 反向传播 ──────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if print_every > 0 and (step % print_every == 0 or step == 1):
            avg = float(np.mean(losses[-print_every:]))
            print(f"  Step {step:5d}/{n_steps}  |  Loss={loss.item():.6f}"
                  f"  |  Avg={avg:.6f}")

    # ── 保存 ──────────────────────────────────────────────────
    torch.save({'model_state_dict': model.state_dict()}, ckpt_path)
    print(f"\n  Model saved → {ckpt_path}")
    print(f"  Final Loss:  {losses[-1]:.6f}")
    print(f"  Last 500 avg: {float(np.mean(losses[-500:])):.6f}")
    print("=" * 55)

    return model, losses


def load_mlp_score(ckpt_path=None):
    """
    加载训练好的 MLP score 模型。

    返回:
        (model, wrapper): MLPNet 实例 和 MLPScore 实例
    """
    if ckpt_path is None:
        ckpt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'checkpoints', 'mlp_score.pth'
        )

    sde   = VPSDE(beta_min=0.1, beta_max=10.0)
    model = MLPNet(data_dim=2, time_dim=32, hidden_dim=512, out_dim=2)
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    wrapper = MLPScore(model, sde)
    return model, wrapper


if __name__ == '__main__':
    model, losses = train_mlp_score()

    # ── 验证推理输出 ──────────────────────────────────────────
    print("\n  Verifying score output ...")
    sde   = VPSDE(beta_min=0.1, beta_max=10.0)
    score = MLPScore(model, sde)

    x_test = torch.randn(10, 2)
    for t_val in [0.1, 0.5, 0.9]:
        s    = score(x_test, t_val)
        norm = s.norm(dim=-1).mean().item()
        print(f"    t={t_val:.1f}  |  avg |score| = {norm:.4f}")
    print("  [Done]")