import numpy as np
import random
from q1_softmax import softmax_loss_grad, softmax
from q2_sigmoid import sigmoid_loss_grad
from q2_function import cross_category_loss_grad, score_to_loss_grad


# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 

    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        ### try modifying x[ix] with h defined above to compute numerical gradients
        ### make sure you call random.setstate(rndstate) before calling f(x) each time, this will make it 
        ### possible to test cost functions with built in randomness later


        if len(it.multi_index) == 0:
            left_x = np.array(x)
            left_x -= h
            right_x = np.array(x)
            right_x += h
        else:
            if len(it.multi_index) == 1:
                ix = ix[0]
            left_x = np.array(x)
            left_x[ix] -= h
            right_x = np.array(x)
            right_x[ix] += h


        diff = f(right_x)[0] - f(left_x)[0]
        numgrad = diff / (2.0 * h)

        if hasattr(numgrad, "shape") and len(numgrad.shape) > 0:
            numgrad = numgrad[ix]

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return
        it.iternext() # Step to next dimension

    print "Gradient check passed!"

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
    sanity_check()
    your_sanity_checks()
