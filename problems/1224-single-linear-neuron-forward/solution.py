import torch
import torch.nn as nn


def single_neuron_forward(x):
    """Forward pass of one fixed linear neuron.

    Args:
        x: torch.Tensor of shape (1, 3).

    Returns:
        Python float, the neuron output.
    """
    model = nn.Linear(in_features=3, out_features=1)
    weight = torch.tensor([0.5, -0.2, 0.3])
    bias = torch.tensor(0.1)
    
    with torch.no_grad():
        model.weight.copy_(weight)
        model.bias.copy_(bias)
        
    y = model(x)
    return y.item()
    pass
