import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    neg_x = -np.array(x)
    return 1./(1. + np.exp(neg_x))

    