import torch

class BaseSolver:
    """求解器基类"""
    def __init__(self, sde, score_fn, nfe=100):
        self.sde = sde
        self.score_fn = score_fn
        self.nfe = nfe
    
    def step(self, x, t, dt):
        """单步更新，由子类实现"""
        raise NotImplementedError
    
    def solve(self, x_init, T=1.0):
        """从初始状态x_init求解到时间T"""
        x = x_init
        dt = T / self.nfe
        trajectory = [x.clone()]
        times = [0.0]
        
        for i in range(self.nfe):
            t = i * dt
            x = self.step(x, t, dt)
            trajectory.append(x.clone())
            times.append(t + dt)
        
        return torch.stack(trajectory), torch.tensor(times)