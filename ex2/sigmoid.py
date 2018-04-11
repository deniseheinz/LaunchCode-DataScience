from numpy import e

def sigmoid(z):
    """computes the sigmoid of z."""
    g=0
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the sigmoid of each value of z (z can be a matrix,
#               vector or scalar).
    g=1.0/(1+e**-z)

# =============================================================

    return g