import sys; sys.path.insert(0, '.')
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from utils.swiss_roll import get_swiss_roll
from utils.sde import VPSDE

# 数据与模型初始化
X = torch.tensor(get_swiss_roll(n_samples=1000, noise=0.05), dtype=torch.float32)
sde = VPSDE(beta_max=5.0)

# 绘图参数
time_steps = [0.0, 0.05, 0.15, 0.3, 1.0]
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
fig.patch.set_facecolor('white')

# 用角度给原始螺旋上色（保留到加噪后，方便追踪结构）
theta = np.linspace(0, 4 * np.pi, 1000)
colors = plt.cm.viridis(theta / (4 * np.pi))  # 螺旋从内到外颜色渐变

for i, t in enumerate(time_steps):
    ax = axes[i]
    
    # 前向采样
    if t == 0:
        Xt = X
    else:
        Xt, _ = sde.forward_sample(X, torch.tensor(t))
    
    Xnp = Xt.numpy()
    
    # 计算正方形边界
    x_min, x_max = Xnp[:, 0].min(), Xnp[:, 0].max()
    y_min, y_max = Xnp[:, 1].min(), Xnp[:, 1].max()
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    half_size = max(x_max - x_min, y_max - y_min) / 2 * 1.15
    
    # 动态渲染参数
    point_size = max(3, 12 * (1 - 0.6 * t))
    alpha_val = max(0.3, 0.85 - 0.5 * t)
    
    # 绘制散点
    ax.scatter(Xnp[:, 0], Xnp[:, 1],
               s=point_size,
               alpha=alpha_val,
               c=colors,
               edgecolors='none')
    
    # 坐标轴设置
    ax.set_xlim(x_center - half_size, x_center + half_size)
    ax.set_ylim(y_center - half_size, y_center + half_size)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linewidth=0.5, linestyle='--')
    ax.tick_params(labelsize=9)
    
    # 标题：显示信噪比
    ab = sde.alpha_bar(torch.tensor(t)).item()
    snr = torch.sqrt(torch.tensor(ab)).item() / torch.sqrt(torch.tensor(1 - ab + 1e-8)).item()
    ax.set_title(f't   = {t:6.2f}\nSNR = {snr:6.2f}', fontsize=12, weight='bold', pad=8)
    
    if i == 0:
        ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_xlabel('$x_1$', fontsize=12)

# 添加颜色条说明螺旋位置
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap='viridis',
                           norm=mcolors.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('螺旋位置\n(内 → 外)', fontsize=10)
cbar.set_ticks([0, 0.5, 1.0])
cbar.set_ticklabels(['0', '0.5', '1.0'])

# 总标题
fig.suptitle('二维阿基米德螺旋流形上的前向扩散过程\n'
             r'(VP-SDE, $\beta_{\min}=0.1$, $\beta_{\max}=5.0$)',
             fontsize=13, y=1.02)

plt.tight_layout(rect=[0, 0, 0.91, 1])
plt.savefig('results/figures/00_forward_test.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()