__author__ = 'fucus'
import numpy as np
import random


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

        r_cost, r_grad = f(right_x)
        l_cost, l_grad = f(left_x)

        diff = r_cost - l_cost
        numgrad = diff / (2.0 * h)

        # if hasattr(numgrad, "shape") and len(numgrad.shape) > 0:
        #     numgrad = numgrad[ix]

        # Compare gradients
        reldiff = abs(np.sum(numgrad - grad[ix])) / max(1, abs(np.sum(numgrad)), abs(np.sum(grad[ix])))
        if reldiff > 1e-4:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %s \t Numerical gradient: %s" % (grad[ix], numgrad)
            return
        it.iternext() # Step to next dimension

    print "Gradient check passed!"
