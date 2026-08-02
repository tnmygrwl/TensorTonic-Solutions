import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """

    # process both -x_i and +x_i seperately for numerical stability

    values = np.asarray(x, dtype=np.float64)

    result = np.empty_like(values)

    pos_mask = values >=0
    neg_mask = ~pos_mask

    result[pos_mask] =  1./(1. + np.exp(-values[pos_mask]))
    result[neg_mask] = np.exp(values[neg_mask]) /( 1.+np.exp(values[neg_mask]))

    return result
    