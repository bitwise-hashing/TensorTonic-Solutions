import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    # Write code here
    z1=np.array(Z1)
    z2=np.array(Z2)
    s = z1@z2.T/temperature
    return -np.mean(np.log(softmax(s).diagonal()))

def softmax(x):
    exp_x_stable = np.exp(x-np.max(x))
    return exp_x_stable/np.sum(exp_x_stable,axis=-1)