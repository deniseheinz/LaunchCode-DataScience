from numpy import asfortranarray, squeeze, asarray

#from gradientFunction import gradientFunction
from sigmoid import sigmoid

def gradientFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    m = len(y)   # number of training examples
    grad = 0
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the gradient of a particular choice of theta.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta

    z=sigmoid(X.dot(theta))
    grad = (1.0/m)*X.T.dot(z-y.values.flatten())+(1.0/m)*Lambda*theta

# =============================================================
    return grad