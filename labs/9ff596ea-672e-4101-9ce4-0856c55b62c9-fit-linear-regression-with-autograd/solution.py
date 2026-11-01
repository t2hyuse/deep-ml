# --- cell 1 ---
import torch
import numpy as np


def fit_linear_regression(X, y, lr=0.1, steps=500):
    """Fit y ~= X @ w + b with full-batch GD using only autograd.

    Args:
        X: Float tensor (N, D)
        y: Float tensor (N,) or (N, 1)
        lr: learning rate
        steps: number of gradient descent iterations

    Returns:
        w: Float tensor (D,) learned weights (no grad)
        b: Float tensor scalar learned bias (no grad)
    """
    # TODO: ensure y is shape (N,)
    # TODO: initialize w (D,) and b with requires_grad=True
    # TODO: for each step:
    #   - predict, compute MSE loss
    #   - loss.backward()
    #   - manual GD update under torch.no_grad()
    #   - zero gradients
    # TODO: return detached w, b
    
    y = y.reshape(-1) 
    
    N, D = X.shape
    
    w = torch.zeros(D, requires_grad = True)
    b = torch.zeros((), requires_grad = True)
    
    for i in range (steps):
        diff = X @ w + b - y
        loss = (diff * diff).mean()
        loss.backward()
        
        with torch.no_grad():           
            w -= lr * w.grad
            b -= lr * b.grad

        w.grad.zero_()                    
        b.grad.zero_()

    return w.detach(), b.detach()
    return w, b