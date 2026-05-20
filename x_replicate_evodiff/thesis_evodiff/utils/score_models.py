import torch

class AnalyticalScore:
    """基于高斯混合的解析 score 函数。
    p_t(x) = (1/N) \sum_i N(x; sqrt(ab_t) x0_i, (1-ab_t) I)
    grad log p_t(x) 可解析计算。
    """
    def __init__(self, data, sde):
        # data: (N, 2) 原始 Swiss Roll
        self.data = torch.tensor(data, dtype=torch.float32)
        self.sde = sde
    
    def __call__(self, x, t):
        """x: (B, 2), t: 标量"""
        if isinstance(t, torch.Tensor):
            ab = self.sde.alpha_bar(t.detach().clone().float())
        else:
            ab = self.sde.alpha_bar(torch.tensor(t, dtype=torch.float32))
        mean_data = torch.sqrt(ab) * self.data  # (N, 2)
        var = 1.0 - ab
        
        # 计算每个 x 对每个 mean_data 的 log 概率
        # diff: (B, N, 2)
        diff = x.unsqueeze(1) - mean_data.unsqueeze(0)
        log_probs = -0.5 * (diff ** 2).sum(-1) / var  # (B, N)
        
        # softmax 加权
        weights = torch.softmax(log_probs, dim=1)  # (B, N)
        
        # score = -E[diff/var | weighted]
        score = -(weights.unsqueeze(-1) * diff).sum(1) / var  # (B, 2)
        return score