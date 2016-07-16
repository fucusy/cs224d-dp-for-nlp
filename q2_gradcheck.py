import numpy as np
from q2_gradcheck_fuc import gradcheck_naive


from q1_softmax import softmax_loss_grad, softmax
from q2_sigmoid import sigmoid_loss_grad
from q2_function import cross_category_loss_grad, score_to_loss_grad

def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running sanity checks..."
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print ""

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

    print "checking sigmoid_loss_grad"
    gradcheck_naive(sigmoid_loss_grad, np.array(123.456))      # scalar test
    gradcheck_naive(sigmoid_loss_grad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(sigmoid_loss_grad, np.random.randn(4,5))   # 2-D test

    print "checking cross_category_loss_grad"
    gradcheck_naive(lambda x: cross_category_loss_grad(x, np.array(134.1)), np.array(123.456))      # scalar test

    l1 = softmax(np.random.randn(3,))
    l2 = softmax(np.random.randn(4, 5))
    gradcheck_naive(lambda x: cross_category_loss_grad(x, l1), softmax(np.random.randn(3,)))    # 1-D test
    gradcheck_naive(lambda x: cross_category_loss_grad(x, l2), softmax(np.random.randn(4, 5)))    # 2-D test


    print "checking score_to_loss_grad"

    l1 = softmax(np.random.randn(3,))
    l2 = softmax(np.random.randn(4, 5))
    gradcheck_naive(lambda x: score_to_loss_grad(x, l1), np.random.randn(3,))    # 1-D test
    gradcheck_naive(lambda x: score_to_loss_grad(x, l2), np.random.randn(4, 5))    # 2-D test

if __name__ == "__main__":
    l1 = softmax(np.random.randn(3,))
    l2 = softmax(np.random.randn(4, 5))
    gradcheck_naive(lambda x: score_to_loss_grad(x, l1), np.random.randn(3,))    # 1-D test
    gradcheck_naive(lambda x: score_to_loss_grad(x, l2), np.random.randn(4, 5))    # 2-D test
    # sanity_check()
    # your_sanity_checks()
