import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    anchor = np.array(anchor)
    positive = np.array(positive)
    negative = np.array(negative)
    return np.mean(np.maximum(0, dist(anchor,positive)-dist(anchor,negative)+margin))

def dist(x,y):
    return np.sum((x-y)**2,axis=-1)