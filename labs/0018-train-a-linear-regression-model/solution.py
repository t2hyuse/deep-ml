# --- cell 1 ---
import numpy as np

def train(X, y, W, b):
    epochs = 100
    lr = 0.1
    n = X.shape[0]
    W = W.copy()
    
    for _ in range(epochs):
        error = (X @ W + b) - y          # prediction - target
        
        grad_W = X.T @ error / n
        grad_b = error.mean()
        
        W = W - lr * grad_W
        b = b - lr * grad_b
    
    return W, b
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    pass