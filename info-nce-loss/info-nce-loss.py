import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    # Write code here
    Z1 = np.array(Z1)
    Z2 = np.array(Z2)
    s = (Z1 @ Z2.T)/ temperature
    return -np.mean(np.log(softmax(s).diagonal()))

def softmax(X):
    shifted_x = np.exp(X-np.max(X))
    return shifted_x/np.sum(shifted_x,axis=-1)