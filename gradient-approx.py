def gradient_check(x, theta, epsilon=1e-7):
    """
    Numerically approximates the gradient and compares to the analytical gradient.
    """
    # Compute gradapprox using finite differences
    thetaplus = theta + epsilon
    thetaminus = theta - epsilon
    J_plus = forward_propagation(x, thetaplus)
    J_minus = forward_propagation(x, thetaminus)
    gradapprox = (J_plus - J_minus) / (2 * epsilon)
    
    # Analytical gradient
    grad = backward_propagation(x, theta)
    
    # Compute relative difference
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator
    
    if difference < 1e-7:
        print("The gradient is correct!")
    else:
        print("The gradient is wrong!")
    return difference

# Test gradient checking
x, theta = 2, 4
difference = gradient_check(x, theta)
print("difference =", difference)