import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return -np.mean(np.log(np.take_along_axis(y_pred, np.expand_dims(y_true, axis=-1),axis=1)))
    