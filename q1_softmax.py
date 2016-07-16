import numpy as np
import random
from q2_gradcheck_fuc import gradcheck_naive

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
    return softmax_loss_grad(x)[0]

def softmax_loss_grad(x):
    shape = x.shape
    if len(shape) == 0:
        return np.array(1), np.array(0)
    elif len(shape) == 1:
        n = shape[0]
        new_x = x - np.min(x)
        exp_x = np.exp(new_x)
        result = exp_x / np.sum(exp_x)
        grad = np.zeros((n, n))
        i_equal_j = result * (1 - result)
        for i in range(n):
            for j in range(n):
                if j == i:
                    grad[i][j] = i_equal_j[i]
                else:
                    grad[i][j] = -1.0 * result[i] * result[j]
        return result, grad
    else:
        result = []
        grad = []
        for num in x:
            tmp_result, tmp_grad = softmax_loss_grad(num)
            result.append(tmp_result)
            grad.append(tmp_grad)
        result = np.array(result)
        grad = np.array(grad)
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


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."

    print "checking softmax_loss_grad"
    gradcheck_naive(softmax_loss_grad, np.array(123.456))      # scalar test
    gradcheck_naive(softmax_loss_grad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(softmax_loss_grad, np.random.randn(4,5))   # 2-D test


if __name__ == "__main__":
    test_softmax_basic()
    your_sanity_checks()