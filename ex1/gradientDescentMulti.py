from computeCostMulti import computeCostMulti
import numpy as np


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    J_history = []
    m = y.size  # number of training examples

    for i in range(num_iters):
        #   ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        #work through hypothesis and subtract y
        h_gd = np.dot(X,theta)
        e_gd = (h_gd-y)
        #finish with dot product of xT and error
        grad = np.dot(np.transpose(X),e_gd)
        #solve for theta
        theta = theta-(alpha*grad/m)
        #print(computeCost(X,y,theta))

        # ============================================================

        # Save the cost J in every iteration
        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history