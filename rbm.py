import numpy as np
from utils import sample, sigmoid


def gibbs_sample(visible, w, bh, bv, num_steps=50):
    ## performace gibbs sampling for visible
    def gibbs_step(visible):
        h_probs = sigmoid(bh + np.matmul(visible, w))
        h_samples = sample(h_probs)
        v_probs = bv + np.matmul(h_samples, w.T)
        v_samples = sample(v_probs, 'gaussian')
        return h_probs, h_samples, v_probs, v_samples
    
    inputs = visible
    h_probs0 = None
    h_probs1 = None
    for k in range(num_steps):
        if k == 0:
            h_probs0, _, _, v_sample = gibbs_step(inputs)
        else:
            _, _, _, v_sample = gibbs_step(inputs)
        inputs = v_sample
    h_probs1 = sigmoid(bh + np.matmul(v_sample, w))
    return v_sample, h_probs0, h_probs1


def gradients(v, v_sample, bh_t, w):
    sigmoid1 = sigmoid(np.matmul(v_sample, w) - bh_t)
    sigmoid2 = sigmoid(np.matmul(v, w) - bh_t)
    dw = np.matmul(v_sample.T, sigmoid1) - np.matmul(v.T, sigmoid2)
    dbh = sigmoid1 - sigmoid2
    # dw = sigmoid(np.matmul(v_sample, w) - bh_t)
    # dw = np.matmul(v_sample.T, hprobs1) - np.matmul(v.T, hprobs0)
    # dbh = hprobs1 - hprobs0
    dbv = v_sample - v
    return dw, dbh, dbv