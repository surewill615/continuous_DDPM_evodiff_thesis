"""
solvers/evodiff_vp.py
---------------------
EVODiff 的 VP-SDE 适配版本。

相比原版 evodiff_edm.py 的改动（仅三处）：
  1. 移除 `from .utils import expand_dims`，改为本地定义
  2. compute_dot_product: 'bchw,bchw->bc' → 'bd,bd->b'
     (适配二维螺旋数据，形状 (B, D) 而非图像 (B,C,H,W))
  3. projection_coefficient 返回形状 (B,1) 而非 (B,C,1,1)
     (保证与 (B,D) 张量的广播正确)

算法逻辑（ζ/η 计算、B_theta、ESD 机制）完全未改动，
与论文 §3.3 的推演严格对应。

噪声调度要求：传入的 noise_schedule 对象必须有 marginal_kappa 方法。
使用 adapters.py 中的 NoiseScheduleVPWithKappa 来满足此要求。
"""

import torch
import torch.nn.functional as F
import math
import numpy as np


# ── 本地定义 expand_dims（原版从 .utils 导入）─────────────────
def expand_dims(v, dims):
    """
    将张量 v 扩展到 dims 维。
    例：v.shape=(B,)，dims=2 → v.shape=(B,1)
    """
    return v[(...,) + (None,) * (dims - 1)]


class EVODiff_VP:
    """
    EVODiff: Entropy-aware Variance Optimized Diffusion inference
    VP-SDE 适配版本（对应论文 §3.3）

    与原版 EVODiff_edm 的唯一数学差异：
      - 使用 VP-SDE 的 kappa_t = sigma_t / alpha_t
        而非 VE-SDE 的 kappa_t = t
      - compute_dot_product 适配二维数据
    算法核心（ζ、η 优化）完全相同。
    """

    def __init__(
        self,
        noise_schedule,
        algorithm_type="data_prediction",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.0,
        dynamic_thresholding_ratio=0.995,
    ):
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["data_prediction"]
        self.algorithm_type = algorithm_type
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

        # 记录每步的 zeta、eta、条件方差，供消融实验使用（§4.3）
        self.zeta_history  = []
        self.eta_history   = []
        self.cond_var_history = []

    def dynamic_thresholding_fn(self, x0, t):
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(
            torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)),
            dims
        )
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """返回噪声预测（epsilon）"""
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        返回数据预测（x0 预测）。
        对应论文中的 x_theta(x_t, t)。
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t = self.noise_schedule.marginal_alpha(t)
        sigma_t = self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0)
        return x0

    def model_fn(self, x, t):
        """统一模型调用接口，返回 x0 预测"""
        return self.data_prediction_fn(x, t)

    # ── 改动 1：compute_dot_product 适配二维数据 ─────────────────
    def compute_dot_product(self, a, b):
        """
        计算批内每个样本的点积。

        原版（图像 4D）：einsum('bchw,bchw->bc') → (B, C, 1, 1)
        本版（2D 数据）：einsum('bd,bd->b')     → (B, 1)

        返回 (B, 1) 以便与 (B, D) 张量正确广播。
        """
        dot = torch.einsum('bd,bd->b', a, b)   # (B,)
        return dot.unsqueeze(-1)               # (B, 1)

    def projection_coefficient(self, tensor_a, tensor_b, tensor_b_dot_product=None):
        """
        计算 tensor_a 在 tensor_b 方向上的投影系数：
            coeff = (A · B) / (B · B)

        返回 (B, 1)，与 (B, D) 张量广播时自动扩展。
        """
        cross_corr = self.compute_dot_product(tensor_a, tensor_b)  # (B, 1)
        if tensor_b_dot_product is None:
            var_b = self.compute_dot_product(tensor_b, tensor_b)   # (B, 1)
        else:
            var_b = tensor_b_dot_product                           # (B, 1)
        return torch.clamp(cross_corr / (var_b + 1e-8), min=-2, max=2)  # (B, 1)

    def denoise_to_zero_fn(self, x, s):
        return self.data_prediction_fn(x, s)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(
                lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1
            ).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == "time_quadratic":
            t_order = 2
            t = torch.linspace(
                t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1
            ).pow(t_order).to(device)
            return t
        else:
            raise ValueError(
                f"Unsupported skip_type {skip_type}, "
                "need 'logSNR' or 'time_uniform' or 'time_quadratic'"
            )

    def sample(
        self,
        model_fn,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=2,
        skip_type="time_uniform",
        method="multistep",
        lower_order_final=True,
        denoise_to_zero=False,
        return_intermediate=False,
    ):
        """
        EVODiff 采样主函数。

        参数：
            model_fn:   noise prediction 函数，接口 model_fn(x, t) → epsilon
            x:          初始噪声，shape (B, D)，float64
            steps:      采样步数（NFE）
            t_start:    起始时刻，默认 noise_schedule.T
            t_end:      终止时刻，默认 1/total_N
            order:      求解器阶数（1 或 2），推荐 2
            skip_type:  时间步分布，'time_uniform' 或 'logSNR'
            return_intermediate: 是否返回中间状态列表
        """
        # 绑定 model_fn，内部自动将 t 扩展为 (B,) 向量
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))

        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start

        assert t_0 > 0 and t_T > 0, (
            "Time range must be > 0. Use t_end=1e-3 instead of 0."
        )

        device = x.device
        intermediates = []

        # 清空历史记录（每次 sample 都重新记录）
        self.zeta_history.clear()
        self.eta_history.clear()
        self.cond_var_history.clear()

        # 滑动窗口缓存：避免重复计算点积
        m0_dot_m0 = None
        m1_dot_m1 = None
        m0_dot_m1 = None

        with torch.no_grad():
            if method == "multistep":
                assert steps >= order

                # ── 获取所有时间步的调度参数 ──────────────────────
                timesteps  = self.get_time_steps(
                    skip_type=skip_type, t_T=t_T, t_0=t_0,
                    N=steps, device=device
                )
                assert timesteps.shape[0] - 1 == steps

                ns          = self.noise_schedule
                all_kappas  = ns.marginal_kappa(timesteps)  # kappa_t = sigma_t/alpha_t
                all_sigmas  = ns.marginal_std(timesteps)

                # ── 初始化 ────────────────────────────────────────
                step = 0
                t   = timesteps[step]
                t_prev_list     = [t]
                model_prev_list = [self.model_fn(x, t)]

                if return_intermediate:
                    intermediates.append(x)

                # ── 前 order 步用低阶方法（DDIM/Euler）热启动 ─────
                for step in range(1, order):
                    t, s       = timesteps[step], t_prev_list[-1]
                    model_s    = model_prev_list[-1]
                    kappa_s    = all_kappas[step - 1]
                    kappa_t    = all_kappas[step]
                    sigma_s    = all_sigmas[step - 1]
                    sigma_t    = all_sigmas[step]
                    sigm_ratio = sigma_t / sigma_s

                    # 一阶更新（DDIM 等价）
                    h = 1 / kappa_t - 1 / kappa_s
                    x = sigm_ratio * x + sigma_t * h * model_s

                    model_prev_list.append(self.model_fn(x, t))
                    t_prev_list.append(t)

                # ── 主循环：二阶 EVODiff 更新 ─────────────────────
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    step_order = min(order, steps + 1 - step) if lower_order_final else order

                    current_idx  = step
                    kappa_t      = all_kappas[current_idx]
                    sigma_t      = all_sigmas[current_idx]
                    prev_idx_0   = step - 1
                    prev_idx_1   = step - 2

                    # ── 一阶步（lower order final 时使用）──────────
                    if step_order == 1:
                        s          = t_prev_list[-1]
                        model_s    = model_prev_list[-1]
                        kappa_s    = all_kappas[prev_idx_0]
                        sigma_s    = all_sigmas[prev_idx_0]
                        sigma_ratio = sigma_t / sigma_s

                        h = 1 / kappa_t - 1 / kappa_s
                        x_t = sigma_ratio * x + sigma_t * h * model_s

                        sigma_final = 1 - torch.pow(sigma_t * sigma_s, 0.5)
                        if sigma_ratio < 0.5:
                            r_dfinal = 1 - 0.5 * torch.pow(sigma_ratio, 0.5)
                        else:
                            r_dfinal = 1 - 0.5 * torch.pow(sigma_ratio - 0.5, 0.5)

                        D_finalstep = model_s - r_dfinal * model_prev_list[-2]
                        x = sigma_final * x_t + 0.5 * sigma_t * h * D_finalstep

                        # 记录（一阶步无 zeta/eta）
                        self.zeta_history.append(None)
                        self.eta_history.append(None)
                        self.cond_var_history.append(
                            float(sigma_t.item() * sigma_s.item())
                        )

                    # ── 二阶步（EVODiff 核心）──────────────────────
                    elif step_order == 2:
                        x_pre = x

                        model_prev_1 = model_prev_list[-2]
                        model_prev_0 = model_prev_list[-1]
                        kappa_prev_1 = all_kappas[prev_idx_1]
                        kappa_prev_0 = all_kappas[prev_idx_0]
                        sigma_prev_1 = all_sigmas[prev_idx_1]
                        sigma_prev_0 = all_sigmas[prev_idx_0]
                        sigma_rat0   = sigma_t / sigma_prev_0
                        sigma_ra01   = sigma_prev_0 / sigma_prev_1

                        h = 1 / kappa_t - 1 / kappa_prev_0

                        # 一阶欧拉预测（DDIM 基础步）
                        x_euler = sigma_rat0 * x_pre + sigma_t * h * model_prev_0

                        # ── §3.3.1 计算 ζ（zeta）─────────────────
                        # r_logh：对应论文中的基础比率 r_i
                        r_logh = (
                            torch.log(kappa_prev_1 / kappa_prev_0)
                            / torch.log(kappa_prev_0 / kappa_t)
                        )

                        # 初始化或更新滑动窗口点积缓存
                        if m0_dot_m0 is None:
                            m0_dot_m0 = self.compute_dot_product(model_prev_0, model_prev_0)
                            m1_dot_m1 = self.compute_dot_product(model_prev_1, model_prev_1)
                            m0_dot_m1 = self.compute_dot_product(model_prev_0, model_prev_1)

                        t_normalized = (t + t_0) / (t_T + t_0)
                        weight_t     = 0.5 * (1 - t_normalized ** 2)
                        balance_baser = torch.sqrt(sigma_rat0 / sigma_ra01)

                        # 计算方向平衡参数 r1_balance
                        r_01_pc    = torch.clamp(m0_dot_m1 / (m1_dot_m1 + 1e-8), min=-2, max=2)
                        r1_balance = (1 - weight_t) * balance_baser + weight_t * r_01_pc

                        # 方差平衡差分项 D1_0（对应 §3.3.1 的 B_theta 分量）
                        D1_0 = model_prev_0 - r1_balance * model_prev_1

                        # 精化控制因子 r_i（对应论文中 Sigmoid 映射后的 ζ*）
                        temperature = 0.25
                        ri_scale    = torch.sigmoid(temperature * r1_balance.abs())
                        r_i         = ri_scale * r_logh
                        r_i         = torch.clamp(r_i, min=0.25 * r_logh, max=1.5 * r_logh)

                        # 带方差控制的二阶探测步
                        x = x_euler + 0.5 * sigma_t * h / r_i * D1_0

                        # 更新模型列表
                        for i in range(order - 1):
                            t_prev_list[i]     = t_prev_list[i + 1]
                            model_prev_list[i] = model_prev_list[i + 1]
                        t_prev_list[-1]     = t
                        model_t             = self.model_fn(x, t)
                        model_prev_list[-1] = model_t

                        # 计算新点积
                        mt_dot_mt = self.compute_dot_product(model_t, model_t)
                        mt_dot_m0 = self.compute_dot_product(model_t, model_prev_0)

                        # ── §3.3.2 计算 η（eta）──────────────────
                        r_t0_pc    = torch.clamp(mt_dot_m0 / (m0_dot_m0 + 1e-8), min=-2, max=2)
                        r2_balance = (1 - weight_t) * balance_baser + weight_t * r_t0_pc
                        D2_0       = model_t - r2_balance * model_prev_0

                        B_pre_i_i  = 1 / r_logh / h * D1_0
                        B_next_i_i = r_logh / h * D2_0

                        # eta_star：通过投影系数求解（对应 §3.3.2 的闭式解）
                        eta_star     = 0.5 * self.projection_coefficient(
                            B_next_i_i + B_pre_i_i,
                            B_next_i_i - B_pre_i_i
                        )
                        eta_star_abs = torch.abs(eta_star)
                        eta          = 0.5 * torch.sigmoid(eta_star_abs)   # Sigmoid 映射到 (0, 0.5)
                        eta_1, eta_2 = -eta, 1 - eta

                        # B_theta：双参数加权梯度项（对应 §3.3.2 的嵌套形式）
                        B_theta = eta_1 / r_logh * D1_0 + r_logh * eta_2 * D2_0

                        # ── §3.3.1 求解 ζ（zeta）的闭式解 ────────
                        P_1           = x - x_euler - 0.5 * sigma_t * h / r_i * B_theta
                        D1_0_dot_D1_0 = self.compute_dot_product(D1_0, D1_0)
                        zeta_star     = self.projection_coefficient(
                            P_1, D1_0, D1_0_dot_D1_0
                        ) / (h * sigma_t)
                        zeta_star_abs   = torch.abs(zeta_star)
                        shift_mu        = 0.5
                        zeta_star_shift = zeta_star_abs - shift_mu

                        if steps > 20:
                            # 高 NFE 时引入一致性约束
                            D10          = (model_prev_0 - model_prev_1) / h
                            D10_dot_D10  = self.compute_dot_product(D10, D10)
                            m0_D10_pc    = self.projection_coefficient(
                                model_prev_0, D10, D10_dot_D10
                            )
                            consistency  = torch.sigmoid(m0_D10_pc * (2 - sigma_rat0 ** 2))
                            zeta         = torch.sigmoid(-zeta_star_shift * consistency)
                        else:
                            zeta = torch.sigmoid(zeta_star_shift)

                        # ── 最终状态更新 ──────────────────────────
                        x = x_euler + 0.5 * sigma_t * h / zeta * B_theta

                        # 记录 zeta、eta、条件方差（供消融实验 §4.3 使用）
                        self.zeta_history.append(
                            float(zeta.mean().item())
                        )
                        self.eta_history.append(
                            float(eta.mean().item())
                        )
                        # 条件方差代理：弥散体积 ≈ ||B_theta||^2 * (sigma_t * h / zeta)^2
                        cond_var_proxy = float(
                            (B_theta.norm(dim=-1).mean() * (sigma_t * h / zeta).mean()).item()
                        )
                        self.cond_var_history.append(cond_var_proxy)

                        # 更新滑动窗口点积缓存
                        m1_dot_m1 = m0_dot_m0
                        m0_dot_m0 = mt_dot_mt
                        m0_dot_m1 = mt_dot_m0

                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)

                    if return_intermediate:
                        intermediates.append(x.clone())

            else:
                raise ValueError(f"Unsupported method: {method}")

            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if return_intermediate:
                    intermediates.append(x.clone())

        if return_intermediate:
            return x, intermediates
        return x
