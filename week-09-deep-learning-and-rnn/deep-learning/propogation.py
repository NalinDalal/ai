# for the simple 1D linear function $J(\theta) = \theta x$.
def forward_propagation(x, theta):
    """
    Implement the linear forward propagation (compute J) for J(theta) = theta * x
    """
    J = np.dot(theta, x)
    return J

def backward_propagation(x, theta):
    """
    Computes the derivative of J with respect to theta
    """
    dtheta = x
    return dtheta

# Test the functions
x, theta = 2, 4
J = forward_propagation(x, theta)
print("J =", J)
dtheta = backward_propagation(x, theta)
print("dtheta =", dtheta)