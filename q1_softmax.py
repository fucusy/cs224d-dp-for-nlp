import numpy as np
import random

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    You should also make sure that your code works for one
    dimensional inputs (treat the vector as a row), you might find
    it helpful for your later problems.

    You must implement the optimization in problem 1(a) of the
    written assignment!
    """
    shape = x.shape

    if len(shape) == 0:
        return np.array(1)
    elif len(shape) == 1:
        min = np.min(x)
        new_x = np.zeros(x.shape)
        for i in range(len(x)):
            new_x[i] = x[i] - min
        exp_x = []
        for num in new_x:
            exp_x.append(np.exp(num))
        exp_x = np.array(exp_x)
        result = []
        sum_exp_x = np.sum(exp_x)
        for num in exp_x:
            result.append(num * 1.0 / sum_exp_x)
        result = np.array(result)
        return result
    else:
        result = []
        for num in x:
            result.append(softmax(num))
        result = np.array(result)
        return result

def softmax_loss_grad(x):
    result = softmax(x)
    grad = result * (1 - result)
    return result, grad



def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6
    test2 = softmax(np.array([[1001,1002],[3,4]]))

    print test2
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6


    print "You should verify these results!\n"

def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    ### END YOUR CODE

if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
