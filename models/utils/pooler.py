import torch
import torch.nn.functional as F


def custom_pool_2d(x: torch.Tensor, N: int, M: int):
    '''
    Args:
    ----
    x : torch.Tensor
        input tensor with shape (N, M, C, H, W)
    N : int
        number of samples
    M : int
        number of nodes
    '''
    # global pooling
    x = F.avg_pool2d(x, x.size()[2:])
    x = x.view(N, M, -1, 1, 1).mean(dim=1)

    return x
