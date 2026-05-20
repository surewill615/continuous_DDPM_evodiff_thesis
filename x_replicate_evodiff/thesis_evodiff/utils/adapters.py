import sys
import os
import torch

# ── 把 solvers/ 加入搜索路径 ─────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
for _p in [_ROOT_DIR, os.path.join(_ROOT_DIR, 'solvers')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dpm_solver_pytorch import (
    NoiseScheduleVP  as _NSVPDpm,
    DPM_Solver,
    model_wrapper    as _dpm_model_wrapper,
)
from uni_pc import (
    NoiseScheduleVP  as _NSVPUni,
    UniPC,
    model_wrapper    as _uni_model_wrapper,
)


# ════════════════════════════════════════════════════════════════
# 1. 噪声调度（两个库各自实例化，参数完全一致）
# ════════════════════════════════════════════════════════════════

def make_noise_schedule(beta_min: float = 0.1,
                        beta_max: float = 5.0):
    """
    返回 (ns_dpm, ns_uni) 两个调度对象。
    两者数学上完全相同，分别供 DPM_Solver 和 UniPC 使用。

    参数与 VPSDE 的对应关系：
        beta_min  ←→  VPSDE.beta_min
        beta_max  ←→  VPSDE.beta_max
    """
    ns_dpm = _NSVPDpm(
        schedule='linear',
        continuous_beta_0=beta_min,
        continuous_beta_1=beta_max,
    )
    ns_uni = _NSVPUni(
        schedule='linear',
        continuous_beta_0=beta_min,
        continuous_beta_1=beta_max,
    )
    return ns_dpm, ns_uni


# ════════════════════════════════════════════════════════════════
# 2. score_fn 包装器
#    两个库的 model_wrapper 都支持 model_type="score"
#    内部换算：epsilon = -sigma_t * score（无需手动处理）
#
#    唯一需要处理的问题：
#    两个库内部都会把 t 扩展成 (B,) 向量再传入，
#    但 AnalyticalScore 期望标量 t（批内所有样本 t 相同）。
#    解决方案：用 _ScoreWrapper 做一次薄薄的维度处理。
# ════════════════════════════════════════════════════════════════

class _ScoreWrapper:
    """
    把 score_fn(x, t_scalar) 包装成 score_fn(x, t_vector) 的形式。

    两个库内部传入的 t 是 (B,) 的向量，批内所有元素相同。
    我们只取 t[0] 作为标量传给原始 score_fn，保持接口一致。

    同时处理精度：两个库内部用 float64，而 AnalyticalScore 用 float32。
    我们在进入 score_fn 之前把 x 转成 float32，输出再转回 float64。
    """

    def __init__(self, score_fn):
        self.score_fn = score_fn

    def __call__(self, x, t_vector):
        """
        x:        (B, D) float64 tensor（库内部传入）
        t_vector: (B,)   float64 tensor（批内元素相同）
        返回:      (B, D) float64 score tensor
        """
        # 取标量 t（批内 t 全部相同，取第一个即可）
        t_scalar = t_vector[0].float()          # float32 标量

        # 转 float32 送入 score_fn
        x_f32 = x.float()                       # (B, D) float32
        score_f32 = self.score_fn(x_f32, t_scalar)  # (B, D) float32

        # 转回 float64 返回给库
        return score_f32.double()               # (B, D) float64


# ════════════════════════════════════════════════════════════════
# 3. 三个求解器的工厂函数
# ════════════════════════════════════════════════════════════════

def make_ddim(score_fn, ns_dpm):
    """
    DDIM 求解器（等价于 DPM-Solver order=1，确定性）。

    使用方法：
        ddim, model_fn = make_ddim(score_fn, ns_dpm)
        x_gen, traj = sample_with_trajectory(ddim, model_fn, x_T, nfe=20)

    参数：
        score_fn: AnalyticalScore 或 MLP，接口为 score_fn(x, t_scalar)
        ns_dpm:   make_noise_schedule() 返回的第一个调度对象

    返回：
        ddim:     DPM_Solver 实例（order=1 时等价于 DDIM）
        model_fn: 包装后的 model_fn，可直接传给 ddim.sample()
    """
    wrapped = _ScoreWrapper(score_fn)
    model_fn = _dpm_model_wrapper(
        wrapped,
        ns_dpm,
        model_type="score",       # 库内部自动做 epsilon = -sigma_t * score
        guidance_type="uncond",
    )
    ddim = DPM_Solver(
        model_fn,
        ns_dpm,
        algorithm_type="dpmsolver++",
    )
    return ddim, model_fn


def make_dpm_solver2(score_fn, ns_dpm):
    """
    DPM-Solver-2 求解器（二阶，每步 2 次网络调用）。

    使用方法：
        solver, model_fn = make_dpm_solver2(score_fn, ns_dpm)
        x_gen, traj = sample_with_trajectory(solver, model_fn, x_T, nfe=10)

    参数：
        score_fn: AnalyticalScore 或 MLP，接口为 score_fn(x, t_scalar)
        ns_dpm:   make_noise_schedule() 返回的第一个调度对象

    返回：
        solver:   DPM_Solver 实例（order=2）
        model_fn: 包装后的 model_fn
    """
    wrapped = _ScoreWrapper(score_fn)
    model_fn = _dpm_model_wrapper(
        wrapped,
        ns_dpm,
        model_type="score",
        guidance_type="uncond",
    )
    solver = DPM_Solver(
        model_fn,
        ns_dpm,
        algorithm_type="dpmsolver++",
    )
    return solver, model_fn


def make_unipc(score_fn, ns_uni):
    """
    UniPC 求解器。
    该版本 model_wrapper 不支持 model_type="score"，
    因此手动把 score 转换成 noise prediction 再传入。
    换算公式：epsilon = -sigma_t * score
    """
    wrapped = _ScoreWrapper(score_fn)

    # 手动构造 noise_pred_fn，绕过 model_wrapper 的类型限制
    def noise_pred_fn(x, t_vector):
        """
        x:        (B, D) float64
        t_vector: (B,)   float64，批内元素相同
        返回:      (B, D) float64 noise prediction
        """
        score = wrapped(x, t_vector)           # (B, D) float64
        sigma_t = ns_uni.marginal_std(t_vector)  # (B,)  float64
        # sigma_t 是 (B,)，score 是 (B, D)，需要 unsqueeze 广播
        epsilon = -sigma_t.unsqueeze(-1) * score  # (B, D) float64
        return epsilon

    solver = UniPC(
        noise_pred_fn,
        ns_uni,
        algorithm_type="noise_prediction",
        variant='bh1',)
    return solver, noise_pred_fn


# ════════════════════════════════════════════════════════════════
# 4. 统一采样接口（带轨迹记录）
# ════════════════════════════════════════════════════════════════

def sample_with_trajectory(solver, x_T, nfe,
                           t_start=1.0, t_end=1e-3,
                           order=2, skip_type='time_uniform',
                           solver_type='dpm',
                           epsilon_fn=None,
                           return_intermediate=False): 
    """
    统一采样接口，返回最终样本和完整轨迹。

    参数：
        solver:      make_ddim / make_dpm_solver2 / make_unipc 返回的 solver
        x_T:         初始噪声，shape (N, D)，建议用 float64
        nfe:         网络前向调用次数（= 采样步数对于一阶，≈ 采样步数/2 对于二阶）
        t_start:     起始时刻，默认 1.0
        t_end:       终止时刻，默认 1e-3（不能为 0，会除零）
        order:       求解器阶数，DDIM=1，DPM-Solver-2=2，UniPC=2
        skip_type:   时间步分布，'time_uniform' 或 'logSNR'
        solver_type: 'dpm'（DPM-Solver / DDIM）或 'unipc'

    返回：
        x_gen:  最终生成样本，shape (N, D) float32
        traj:   轨迹列表，每个元素是该时间步的 (N, D) float32 tensor
                traj[0] = x_T，traj[-1] = x_gen
    """
    # 确保输入是 float64（库内部要求）
    x_T_64 = x_T.double()

    traj_64 = [x_T_64.cpu()]

    with torch.no_grad():
        if solver_type == 'unipc':
            # UniPC 的 sample 接口
            x_gen_64 = solver.sample(
                x_T_64,
                steps=nfe,
                t_start=t_start,
                t_end=t_end,
                order=order,
                skip_type=skip_type,
                lower_order_final=True,
            )
        elif solver_type == 'evodiff':
            result = solver.sample(
                epsilon_fn,
                x_T_64,
               steps=nfe,
               t_start=t_start,
               t_end=t_end,
               order=order,
               skip_type=skip_type,
               lower_order_final=True,
               return_intermediate=return_intermediate,  # 改这里，不再硬编码
             )
            if return_intermediate:
                x_gen_64, intermediates_64 = result
                traj_64 = [x_T_64.cpu()] + [s.cpu() for s in intermediates_64]
            else:
                x_gen_64 = result
    
        else:
            # DPM-Solver / DDIM 的 sample 接口
            x_gen_64 = solver.sample(
                x_T_64,
                steps=nfe,
                t_start=t_start,
                t_end=t_end,
                order=order,
                skip_type=skip_type,
                lower_order_final=True,
                return_intermediate=return_intermediate,
            )
            if return_intermediate:
                x_gen_64, intermediates_64 = x_gen_64
                traj_64 = [x_T_64.cpu()] + [s.cpu() for s in intermediates_64]

    if not return_intermediate:
        traj_64.append(x_gen_64.cpu())

    # 转回 float32 方便后续处理
    x_gen   = x_gen_64.float()
    traj    = [s.float() for s in traj_64]

    return x_gen, traj



# ════════════════════════════════════════════════════════════════
# 5. 快速验证（直接运行此文件时执行）
# ════════════════════════════════════════════════════════════════
class NoiseScheduleVPWithKappa:
    """
    在 NoiseScheduleVP 基础上添加 marginal_kappa 方法的包装类。

    kappa_t = sigma_t / alpha_t = sqrt(1-alpha_bar_t) / sqrt(alpha_bar_t)

    EVODiff 使用 kappa 而非 lambda 来计算步长，这是与 DPM-Solver 的
    主要调度差异。
    """
    def __init__(self, ns_dpm):
        """
        ns_dpm: make_noise_schedule() 返回的第一个调度对象
                (dpm_solver_pytorch.NoiseScheduleVP 实例)
        """
        self._ns = ns_dpm
        # EVODiff 内部用到这两个属性
        self.total_N = 1000
        self.T       = 1.0

    # ── 直接代理给底层 NoiseScheduleVP 的方法 ────────────────────
    def marginal_log_mean_coeff(self, t):
        return self._ns.marginal_log_mean_coeff(t)

    def marginal_alpha(self, t):
        return self._ns.marginal_alpha(t)

    def marginal_std(self, t):
        return self._ns.marginal_std(t)

    def marginal_lambda(self, t):
        return self._ns.marginal_lambda(t)

    def inverse_lambda(self, lamb):
        return self._ns.inverse_lambda(lamb)

    # ── 新增：EVODiff 需要的 marginal_kappa ───────────────────────
    def marginal_kappa(self, t):
        """
        kappa_t = sigma_t / alpha_t

        对于 VP-SDE：
            alpha_t = sqrt(alpha_bar_t)
            sigma_t = sqrt(1 - alpha_bar_t)
        所以：
            kappa_t = sqrt(1 - alpha_bar_t) / sqrt(alpha_bar_t)

        当 t → 0 时 kappa_t → 0（数据端，噪声极小）
        当 t → 1 时 kappa_t → 大数（噪声端）

        加 1e-8 防止 t=0 时除零。
        """
        alpha_t = self._ns.marginal_alpha(t)
        sigma_t = self._ns.marginal_std(t)
        return sigma_t / (alpha_t + 1e-8)


# ════════════════════════════════════════════════════════════════
# 7. EVODiff 工厂函数
# ════════════════════════════════════════════════════════════════

def make_evodiff(score_fn, ns_dpm):
    """
    EVODiff 求解器工厂函数。

    使用方法：
        solver = make_evodiff(score_fn, ns_dpm)
        x_gen, traj = sample_with_trajectory(
            solver, x_T, nfe=10, order=2, solver_type='evodiff'
        )

    参数：
        score_fn: AnalyticalScore 或 MLP，接口为 score_fn(x, t_scalar)
        ns_dpm:   make_noise_schedule() 返回的第一个调度对象

    返回：
        solver: EVODiff_VP 实例
    """
    from evodiff_vp import EVODiff_VP

    # 1. 构造带 kappa 的调度对象
    ns_with_kappa = NoiseScheduleVPWithKappa(ns_dpm)

    # 2. 包装 score_fn → epsilon_fn（EVODiff 内部期望 noise prediction）
    wrapped = _ScoreWrapper(score_fn)

    def epsilon_fn(x, t_vector):
        """
        将 score_fn 输出转换为 noise prediction。
        epsilon = -score * sigma_t
        """
        score   = wrapped(x, t_vector)                     # (B, D) float64
        sigma_t = ns_with_kappa.marginal_std(t_vector)     # (B,)   float64
        epsilon = -sigma_t.unsqueeze(-1) * score           # (B, D) float64
        return epsilon

    # 3. 实例化 EVODiff_VP
    solver = EVODiff_VP(
        noise_schedule=ns_with_kappa,
        algorithm_type="data_prediction",
        correcting_x0_fn=None,
    )

    return solver, epsilon_fn


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from utils.swiss_roll import get_swiss_roll
    from utils.sde        import VPSDE
    from utils.score_models import AnalyticalScore
    import numpy as np

    print("=" * 50)
    print("adapters.py 快速验证")
    print("=" * 50)

    # 数据与得分函数
    X     = torch.tensor(get_swiss_roll(n_samples=1000), dtype=torch.float32)
    sde   = VPSDE(beta_max=5.0)
    score = AnalyticalScore(X.numpy(), sde)

    # 调度
    ns_dpm, ns_uni = make_noise_schedule(beta_min=0.1, beta_max=5.0)
    print(f"调度验证: alpha_bar(0.5) = {ns_dpm.marginal_alpha(torch.tensor(0.5)).item():.4f}")

    # 初始噪声
    torch.manual_seed(42)
    N   = 10
    x_T = torch.randn(N, 2, dtype=torch.float64)

    # 验证三个求解器
    for name, solver_obj, s_type, ord_ in [
        ('DDIM',         make_ddim(score, ns_dpm),         'dpm',   1),
        ('DPM-Solver-2', make_dpm_solver2(score, ns_dpm),  'dpm',   2),
        ('UniPC',        make_unipc(score, ns_uni),         'unipc', 2),
    ]:
        solver, _ = solver_obj
        x_gen, traj = sample_with_trajectory(
            solver, x_T, nfe=20, order=ord_, solver_type=s_type
        )

        # 计算到流形的平均最近邻距离（Chamfer 近似）
        X_ref = X.numpy()
        x_np  = x_gen.numpy()
        dists = []
        for pt in x_np:
            d = np.linalg.norm(X_ref - pt, axis=1).min()
            dists.append(d)
        print(f"{name:15s}: shape={tuple(x_gen.shape)}, "
              f"avg_chamfer={np.mean(dists):.4f}, "
              f"traj_len={len(traj)}")
            # 验证 EVODiff
    solver_evo, epsilon_fn = make_evodiff(score, ns_dpm)
    x_gen, traj = sample_with_trajectory(
        solver_evo, x_T, nfe=20, order=2,
        solver_type='evodiff', epsilon_fn=epsilon_fn
)
    dists = [np.linalg.norm(X.numpy() - pt, axis=1).min() for pt in x_gen.numpy()]
    print(f"{'EVODiff':15s}: shape={tuple(x_gen.shape)}, avg_chamfer={np.mean(dists):.4f}")

    print("\n[OK] 三个求解器全部正常运行")
