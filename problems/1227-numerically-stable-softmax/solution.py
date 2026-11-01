import torch

def softmax(t, dim):
    """Numerically stable softmax along dim.

    Args:
        t (torch.Tensor): input tensor
        dim (int): dimension along which to apply softmax

    Returns:
        torch.Tensor: tensor of same shape as t; slices along dim sum to 1
    """
    # TODO: subtract max along dim, exp, then normalize
    max_val = torch.max(t)
    t = t - max_val
	
    t = torch.exp(t) 
    return t / torch.sum(t, dim, keepdim=True)
    pass
