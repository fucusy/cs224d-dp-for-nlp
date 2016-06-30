__author__ = 'fucus'
import numpy as np
from q1_softmax import softmax_loss_grad

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
    softmax_result, softmax_grad = softmax_loss_grad(score)
    cost, cross_grad = cross_category_loss_grad(softmax_result, labels)

    # the correct gradient
    grad = (softmax_result - labels) / n


    # my gradient
    grad = cross_grad * softmax_grad

    return cost, grad