import torch

def chamfer_distance(x, y):
    """计算两个点集之间的Chamfer距离"""
    # x: (N, D), y: (M, D)
    # 简化实现，实际可能需要更高效的实现
    dist = torch.cdist(x, y)  # (N, M)
    min_dist_x = dist.min(dim=1)[0].mean()
    min_dist_y = dist.min(dim=0)[0].mean()
    return min_dist_x + min_dist_y