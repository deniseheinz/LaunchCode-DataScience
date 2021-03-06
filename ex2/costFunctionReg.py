#from costFunction import costFunction
import numpy as np
from sigmoid import sigmoid
from numpy import log
def costFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    # Initialize some useful values
    m = len(y)   # number of training examples
    J = 0
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta

    z=sigmoid(np.dot(X,theta))
    J = -(1.0/m)*(np.sum(y.values.flatten()*log(z)+((1-y.values.flatten())*log(1-z)))) + (Lambda/(2.0*m)*np.sum(theta[1:]**2))# =============================================================

    return J
