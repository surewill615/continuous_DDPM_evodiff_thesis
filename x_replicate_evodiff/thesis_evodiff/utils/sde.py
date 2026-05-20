import torch

class VPSDE:
    """Variance-Preserving SDE,与 §2.2.1 推导一致"""
    def __init__(self, beta_min=0.1, beta_max=5.0, T=1.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
    
    def beta(self, t):
        """确保 t 是浮点数"""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def alpha_bar(self, t):
        """\bar{\alpha}_t = exp(-\int_0^t \beta(s) ds)"""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        else:
            t = t.float()
        
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
        alpha_bar = torch.exp(-integral)
        
        # 调试信息:打印出来看是否正常
        # print(f"t={t.item():.2f}, alpha_bar={alpha_bar.item():.4f}")
        
        return alpha_bar
    
    def marginal_prob(self, x0, t):
        """前向边缘分布的均值与标准差
        x0: (N, D)
        t: 标量 或 (N,) 向量
        """
        ab = self.alpha_bar(t)                  # 标量 或 (N,)
        sqrt_ab = torch.sqrt(ab)                # 标量 或 (N,)
        sqrt_1m_ab = torch.sqrt(1.0 - ab)       # 标量 或 (N,)
        
        # 处理维度：当 t 是 (N,) 向量时，unsqueeze 成 (N, 1) 以便与 (N, D) 广播
        if sqrt_ab.dim() > 0:
            sqrt_ab   = sqrt_ab.unsqueeze(-1)   # (N,) → (N, 1)
            sqrt_1m_ab = sqrt_1m_ab.unsqueeze(-1)  # (N,) → (N, 1)
        
        mean = sqrt_ab * x0                     # (N, 1) * (N, D) → (N, D)
        std  = sqrt_1m_ab                       # 标量 或 (N, 1)
        
        return mean, std
    
    def forward_sample(self, x0, t):
        """从 q(x_t | x_0) 采样"""
        mean, std = self.marginal_prob(x0, t)
        noise = torch.randn_like(x0)
        return mean + std * noise, noise