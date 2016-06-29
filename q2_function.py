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
    grad = labels * (result - 1) / n

    dscores = result
    classfy_label = np.zeros(n)
    for i in range(n):
        for j in range(len(labels[i])):
            if labels[i][j] == 1:
                classfy_label[i] = j
                break
    for i in range(n):
        dscores[i][classfy_label[i]] -= 1

    dscores /= n


    return cost, dscores