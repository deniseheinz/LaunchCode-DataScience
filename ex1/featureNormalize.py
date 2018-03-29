import numpy as np


def featureNormalize(X):
    """
       returns a normalized version of X where
       the mean value of each feature is 0 and the standard deviation
       is 1. This is often a good preprocessing step to do when
       working with learning algorithms.
    """
    #X_norm, mu, sigma = X,0,0
    X_norm, mu, sigma = X,[],[]
    
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: First, for each feature dimension, compute the mean
    #               of the feature and subtract it from the dataset,
    #               storing the mean value in mu. Next, compute the
    #               standard deviation of each feature and divide
    #               each feature by it's standard deviation, storing
    #               the standard deviation in sigma.
    #
    #               Note that X is a matrix where each column is a
    #               feature and each row is an example. You need
    #               to perform the normalization separately for
    #               each feature.
    #
    
    # Hint: You might find the 'mean' and 'std' functions useful.
    features=range(0,np.size(X,1))
    for i in features:
        single_mean = np.mean(X[:,i]) 
        mu.append(single_mean)
        single_std = np.std(X[:,i])
        sigma.append(single_std)
    mu_matrix=np.array([mu])
    mu_matrix.transpose()
    m=np.size(X,0)
    mu_math=np.repeat(mu_matrix,m,axis=0)
    sigma_matrix=np.array([sigma])
    X_norm=np.divide((X - mu_math),sigma_matrix)
        
    
    
    
# ============================================================

    return X_norm, mu, sigma
