__author__ = 'fucus'
import numpy as np

def cross_category_loss_grad(pred, labels):
    if len(pred.shape) <= 1:
        n = 1
    else:
        n = pred.shape[0]
    cost = -1.0 * np.sum(np.log(pred) * labels) / n  #scalar
    grad = -1.0 * labels / pred / n # (n, Dy)
    return cost, grad
