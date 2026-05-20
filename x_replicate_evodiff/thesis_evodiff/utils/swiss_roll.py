import numpy as np

def get_swiss_roll(n_samples=1000, noise=0.05, seed=42):
    """
    
    参数:
        n_samples: 采样点数量
        noise:     初始结构噪声（模拟真实数据的测量误差）
        seed:      随机种子，保证可复现性
    
    返回:
        X: (n_samples, 2) float32 数组，范围约 [-1, 1]
    """
    rng = np.random.RandomState(seed)
    
    # 生成螺旋角度参数：绕两圈（0 到 4π）
    theta = np.linspace(0, 4 * np.pi, n_samples)
    
    # 半径随角度线性增长：r = theta / (4π)，范围 [0, 1]
    r = theta / (4 * np.pi)
    
    # 转为笛卡尔坐标
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # 加少量初始噪声（模拟真实数据）
    x += rng.randn(n_samples) * noise
    y += rng.randn(n_samples) * noise
    
    # 拼合并标准化到 [-1, 1]
    X = np.stack([x, y], axis=1).astype(np.float32)
    X = X / np.abs(X).max()
    
    return X


def visualize_manifold(X, title="Archimedean Spiral"):
    """快速可视化工具函数（调试用）"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], s=5, alpha=0.6, c='steelblue')
    plt.title(title)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    X = get_swiss_roll(n_samples=1000)
    print(f"形状: {X.shape}")
    print(f"x 范围: [{X[:,0].min():.3f}, {X[:,0].max():.3f}]")
    print(f"y 范围: [{X[:,1].min():.3f}, {X[:,1].max():.3f}]")
    visualize_manifold(X)