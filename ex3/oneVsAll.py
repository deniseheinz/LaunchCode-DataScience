import numpy as np
from scipy.optimize import minimize

from lrCostFunction import lrCostFunction
from lrCostFunction import gradient

#=========================================================================

#=======================================================================

def oneVsAll(X, y, num_labels, Lambda):
    """trains multiple logistic regression classifiers and returns all
        the classifiers in a matrix all_theta, where the i-th row of all_theta
        corresponds to the classifier for label i
    """

# Some useful variables
    m, n = X.shape

# You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))

# Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))

# ====================== YOUR CODE HERE ======================
# Instructions: You should complete the following code to train num_labels
#               logistic regression classifiers with regularization
#               parameter lambda. 
#
# Hint: theta(:) will return a column vector.
#
# Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
#       whether the ground truth is true/false for this class.
#
# Note: For this assignment, we recommend using fmincg to optimize the cost
#       function. It is okay to use a for-loop (for c = 1:num_labels) to
#       loop over the different classes.

    # Set Initial theta
    #initial_theta = np.zeros((n + 1, 1))
    
 
    for c in range(1,num_labels+1):
        #create true/false for values equal to c
        initial_theta = np.zeros((n + 1, 1))
        y_c=np.array(y==c)
#find minimal value, taken from exercise 2
        result = minimize(fun=lrCostFunction, x0=initial_theta, args=(X, y_c, Lambda), method='L-BFGS-B',
               jac=gradient)
               #options={'gtol': 1e-4, 'disp': False, 'maxiter': 1000})        
        all_theta[c-1,:] = result.x
        #cost = result.fun
    # This function will return theta and the cost(?)



# =========================================================================

    return all_theta

