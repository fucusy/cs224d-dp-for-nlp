__author__ = 'fucus'
import numpy as np
from q1_softmax import softmax

def cross_category_loss_grad(pred, labels):
    if len(pred.shape) <= 1:
        n = 1
    else:
        n = pred.shape[0]
    cost = -1.0 * np.sum(np.log(pred) * labels) / n  #scalar
    grad = -1.0 * labels / pred / n # (n, Dy)
    return cost, grad


def score_to_loss_grad(score, labels):
    if len(score.shape) <= 1:
        n = 1
    else:
        n = score.shape[0]
    result = softmax(score)
    cost = -1.0 * np.sum(np.log(result) * labels) / n  #scalar

    # my gradient, but it's wrong
    # softmax gradient
    softmax_grad = (1 - result) * result
    d_result = (labels / result) * (-1.0 * n)
    grad = softmax_grad * d_result
    grad = labels * (result - 1) / n



    # the correct gradient
    grad = (result - labels) / n

    return cost, grad