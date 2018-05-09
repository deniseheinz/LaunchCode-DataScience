import sys
sys.path.append('../')
from ex2.sigmoid import sigmoid
import numpy as np

def sigmoidGradient(z):
    """computes the gradient of the sigmoid function
    evaluated at z. This should work regardless if z is a matrix or a
    vector. In particular, if z is a vector or matrix, you should return
    the gradient for each element."""

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the gradient of the sigmoid function evaluated at
#               each value of z (z can be a matrix, vector or scalar).

    m=len(y)
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)
    #h=sigmoid(np.dot(X,theta.T))
    h=sigmoid(np.dot(X,theta.T))
    #grad=(1.0/m)*X.T.dot(h-y)
    grad=(1.0/m)*np.dot(X.T,(h-y))
    #grad=((1.0/m)*np.dot(X.T,(h-y))).T
    gradreg=grad.T+np.multiply((Lambda/m),theta)
    gradreg[0,0]=np.sum(np.multiply(X[:,0],(h-y)))/m
    return gradreg
 
# =============================================================

    return g
