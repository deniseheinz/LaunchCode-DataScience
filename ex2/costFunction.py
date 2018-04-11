from numpy import log
import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X,y):
    """ computes the cost of using theta as the
    parameter for logistic regression and the
    gradient of the cost w.r.t. to the parameters."""

# Initialize some useful values
    m = y.size # number of training examples
    J = 0

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
    
    z=sigmoid(np.dot(X,theta))

    J = -(1.0/m)*(np.sum(y*log(z)+((1-y)*log(1-z))))
    
    return J
