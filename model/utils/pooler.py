import torch
import torch.nn.functional as F

def custom_pool_2d(x : torch.Tensor, N, M):
    # global pooling
    x = F.avg_pool2d(x, x.size()[2:])
    x = x.view(N, M, -1, 1, 1).mean(dim=1)
    
    return x