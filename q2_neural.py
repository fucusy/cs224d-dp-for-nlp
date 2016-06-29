import numpy as np
import random

from q1_softmax import softmax_loss_grad
from q2_sigmoid import sigmoid_loss_grad
from q2_gradcheck import gradcheck_naive
from q2_function import score_to_loss_grad

def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """
    ### Unpack network parameters (do not modify)

    n = len(data)
    classfy_label = np.zeros(n)
    for i in range(n):
        for j in range(len(labels[i])):
            if labels[i][j] == 1:
                classfy_label[i] = j
                break

    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    hidden_layer = np.maximum(0, np.dot(data, W1) + b1) # ReLU activation
    scores = np.dot(hidden_layer, W2) + b2

    probs, layer2_grad = softmax_loss_grad(scores) # (n, Dy)

    cost, dscores = score_to_loss_grad(scores, labels)

    gradW2 = np.dot(hidden_layer.T, dscores)
    gradb2 = np.sum(dscores, axis=0)
    # next backprop into hidden layer
    dhidden = np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0
    # finally into W,b

    gradW1 = np.dot(data.T, dhidden)
    gradb1 = np.sum(dhidden, axis=0)

    # layer1 = np.dot(data, W1) + b1   # (n, H)
    #
    # layer1_sigmoid, layer1_grad = sigmoid_loss_grad(layer1) # (n, H)
    #
    # scores = np.dot(layer1_sigmoid, W2) + b2 # (n, Dy)
    #
    # layer2_softmax, layer2_grad = softmax_loss_grad(scores) # (n, Dy)
    #
    # # do gradient
    # cost, d_layer2_softmax = cross_category_loss_grad(layer2_softmax, labels)
    #
    # d_scores = layer2_grad * d_layer2_softmax # (n, Dy)
    #
    # gradW2 = layer1_sigmoid.T.dot(d_scores) #(H, Dy)
    #
    # gradb2 = d_scores.sum(axis=0) # (1, Dy)
    #
    # d_layer1_sigmoid = np.dot(d_scores, W2.T) # (n, H)
    #
    # d_layer1 = layer1_grad * d_layer1_sigmoid # (n, H)
    #
    # gradW1 = np.dot(data.T, d_layer1)  #(Dx, H)
    #
    # gradb1 = d_layer1.sum(axis=0) # (1, H)

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
