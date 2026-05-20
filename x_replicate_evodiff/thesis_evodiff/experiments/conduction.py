import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def generate_realistic_illustration_notitle():
    np.random.seed(42)
    
    # ==========================================
    # 1. 构造具有真实“厚度”与噪声分布的数据流形
    # ==========================================
    n_samples = 3000
    theta = np.linspace(1.5 * np.pi, 4.5 * np.pi, n_samples)
    r = theta
    x_base = r * np.cos(theta)
    y_base = r * np.sin(theta)
    
    dx = np.cos(theta + np.pi/2)
    dy = np.sin(theta + np.pi/2)
    thickness_noise = np.random.normal(0, 0.5, n_samples)
    bg_x = x_base + dx * thickness_noise + np.random.normal(0, 0.2, n_samples)
    bg_y = y_base + dy * thickness_noise + np.random.normal(0, 0.2, n_samples)

    # ==========================================
    # 2. 模拟真实神经网络的 OOD “垃圾梯度”场
    # ==========================================
    x_grid = np.linspace(-16, 16, 32)
    y_grid = np.linspace(-16, 16, 32)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    R = np.hypot(X, Y)
    
    U_ideal = -Y - X * 0.3
    V_ideal = X - Y * 0.3
    
    ood_factor = np.clip((R - 6) / 8, 0, 1) 
    chaos_U = np.random.randn(*X.shape) * 20
    chaos_V = np.random.randn(*X.shape) * 20
    
    U = U_ideal * (1 - ood_factor) + chaos_U * ood_factor
    V = V_ideal * (1 - ood_factor) + chaos_V * ood_factor
    
    magnitude = np.hypot(U, V)
    U_norm = U / (magnitude + 1e-8)
    V_norm = V / (magnitude + 1e-8)

    # ==========================================
    # 3. 模拟轨迹
    # ==========================================
    # 轨迹 A (DPM-Solver++)
    traj_dpm = np.array([
        [-12.0, -12.0],  # NFE=0 (起点)
        [-8.5, -9.5],    # NFE=1
        [-4.5, -6.5],    # NFE=2 
        [-0.5, -2.5],    # NFE=3 
        [4.5, 0.5],      # NFE=4 
        [8.5, 3.5],      # NFE=5 
        [11.5, 8.0]      # NFE=6 
    ])

    # 轨迹 B (EVODiff)
    traj_evo = np.array([
        [-12.0, -12.0],  # NFE=0 
        [-8.8, -9.2],    # NFE=1
        [-4.2, -6.8],    # NFE=2 
        [-0.8, -3.0],    # NFE=3 
        [1.2, 1.5],      # NFE=4 
        [-1.5, 4.5],     # NFE=5 
        [-4.0, 1.5]      # NFE=6 
    ])

    # ==========================================
    # 4. 采用学术级配色与渲染
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('#FAFAFA') 
    
    dpm_color = '#D9534F' 
    evo_color = '#5CB85C' 
    
    solvers = [("DPM-Solver++ (基线方法)", traj_dpm, dpm_color, "切线飞逸区\n(梯度失效)"), 
               ("EVODiff (本文方法)", traj_evo, evo_color, "向心拉回\n(流形支撑)")]

    for i, (name, traj, color, text_anno) in enumerate(solvers):
        ax = axes[i]
        
        ax.scatter(bg_x, bg_y, s=6, alpha=0.25, c='#7F8C8D', edgecolors='none', zorder=1)
        ax.quiver(X, Y, U_norm, V_norm, ood_factor, cmap='coolwarm', 
                  alpha=0.5, scale=45, width=0.003, zorder=2)
        
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2.5, zorder=3, alpha=0.9)
        ax.scatter(traj[:, 0], traj[:, 1], color=color, s=50, edgecolors='white', linewidths=1.5, zorder=4)
        
        ax.scatter(traj[0, 0], traj[0, 1], color='#2C3E50', s=100, marker='s', 
                   facecolors='none', linewidths=2, zorder=5, label='起点 (纯噪声)')
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=180, marker='*', 
                   edgecolors='white', linewidths=1.5, zorder=5, label='终点 (生成状态)')
        
        circle = plt.Circle((traj[-1, 0], traj[-1, 1]), 2.8, color=color, fill=False, 
                            linestyle=':', linewidth=2.5, zorder=6, alpha=0.8)
        ax.add_patch(circle)
        
        # 【核心修改点】: 左图偏移量改为 (-90, -70)，即向左向下移动；右图保持在右下 (-20, -35)
        xytext_offset = (-110, -70) if i == 0 else (20, -35)
        
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.5, alpha=0.85)
        ax.annotate(text_anno, xy=(traj[-1, 0], traj[-1, 1]), 
                    xytext=xytext_offset, textcoords='offset points', 
                    color=color, fontweight='bold', fontsize=11,
                    bbox=bbox_props,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2, shrinkA=5, shrinkB=35))

        ax.set_xlim(-16, 16)
        ax.set_ylim(-16, 16)
        ax.set_aspect('equal')
        
        ax.set_title(name, fontsize=15, fontweight='bold', pad=12, color='#34495E')
        ax.grid(True, alpha=0.3, linestyle='-.', color='#BDC3C7')
        
        custom_lines = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#7F8C8D', markersize=8, alpha=0.5),
            Line2D([0], [0], color=color, lw=2.5),
            Line2D([0], [0], marker='s', color='w', markeredgecolor='#2C3E50', markersize=8, lw=2),
            Line2D([0], [0], marker='*', color='w', markerfacecolor=color, markersize=14)
        ]
        ax.legend(custom_lines, ['训练数据流形 (点云)', '离散推断轨迹', '起点 $x_T$', '终点 $x_0$'], 
                  loc='upper right', fontsize=10, framealpha=0.9, edgecolor='#BDC3C7')

    plt.tight_layout()
    plt.savefig("Fig_4_2_Realistic_Illustration_NoTitle.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    generate_realistic_illustration_notitle()