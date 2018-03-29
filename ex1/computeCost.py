import numpy as np

def computeCost(X, y, theta):
    """
       computes the cost of using theta as the parameter for linear 
       regression to fit the data points in X and y
    """
    #m is number of examples
    m = y.size
    
    J=0
   
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.
   #define the hypothesis
    h = np.dot(X,theta)
    #subtract Y
    e = (h-y)
    #square, sum all and divide by 2m
    J = np.sum(e**2)/(2*m)
    

# =========================================================================

    return(J)


