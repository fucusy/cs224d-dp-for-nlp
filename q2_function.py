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


    if len(softmax_grad.shape) < 3:
        # my gradient
        grad = softmax_grad.dot(cross_grad)
    else:
        grad = np.zeros(labels.shape)
        for i in range(softmax_grad.shape[0]):
            grad[i, :] = softmax_grad[i].dot(cross_grad[i])

    return cost, grad