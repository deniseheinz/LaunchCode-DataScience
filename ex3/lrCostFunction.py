#import sys
#sys.path.append("../ex2/")
import numpy as np


#from ex2.costFunctionReg import costFunctionReg
def sigmoid(z):
    """computes the sigmoid of z."""
    return 1.0/(1+np.exp(-z))
    

    
#============================================================


def lrCostFunction(theta, X, y, Lambda):
    """computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#
# Hint: The computation of the cost function and gradients can be
#       efficiently vectorized. For example, consider the computation
#
#           sigmoid(X * theta)
#
#       Each row of the resulting matrix will contain the value of the
#       prediction for that example. You can make use of this to vectorize
#       the cost function and gradient computations. 
#
    m = len(y)   # number of training examples
    J = 0
    X=np.matrix(X)
    y=np.matrix(y)
    theta=np.matrix(theta)
    Lambda=np.matrix(Lambda)
    #initial_theta = np.zeros(X.shape[1])
# from ex2.costfunctionreg    
    h=sigmoid(X.dot(theta.T))
    one=np.dot(y.T,np.log(h))
    two=np.dot((1-y).T,np.log(1-h))
    reg=(Lambda/(2.0*m))*np.sum(np.power(theta[1:],2))
    J=(-1.0/m)*(one+two)+reg
    return J
    
    #grad = (1.0/m)*X.T.dot(z-y.values.flatten())+(1.0/m)*Lambda*theta
    
    #grad = 1./m*(X.T*J-y)
    # =============================================================

   
    
def gradient(theta, X, y, Lambda):
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
 