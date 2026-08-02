import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    # Write code here
    a = np.array(a)
    b = np.array(b)
    d = l2_norm(a-b)
    y = np.array(y)
    loss = (y*d**2) + ((1-y)*np.maximum(0.0,margin-d)**2)
    if reduction == "mean":
        return loss.mean()
    return loss.sum()
    

def l2_norm(x):
    return np.sqrt(np.sum(x**2,axis=-1))