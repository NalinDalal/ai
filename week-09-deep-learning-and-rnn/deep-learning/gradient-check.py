"""
1. Import Required Libraries and Utility Functions
2. 1D Gradient Checking: Forward and Backward Propagation
3. 1D Gradient Checking: Numerical Gradient Approximation
4. 1D Gradient Checking: Comparison and Validation
5. N-Dimensional Gradient Checking: Forward Propagation
6. N-Dimensional Gradient Checking: Backward Propagation
7. N-Dimensional Gradient Checking: Numerical Gradient Approximation
8. N-Dimensional Gradient Checking: Comparison and Validation
"""

#library imports
import numpy as np

#utility function for 1D case
# Activation functions

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

# Utility functions for vectorization (for N-dimensional case)
def dictionary_to_vector(parameters):
    """
    Roll all parameters dictionary values into a single vector.
    Returns: vector, keys
    """
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        new_vector = np.reshape(parameters[key], (-1, 1))
        keys = keys + [key]*new_vector.shape[0]
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1
    return theta, keys

def vector_to_dictionary(theta):
    """
    Unrolls a vector into the parameters dictionary.
    """
    parameters = {}
    parameters["W1"] = theta[0:20].reshape((5, 4))
    parameters["b1"] = theta[20:25].reshape((5, 1))
    parameters["W2"] = theta[25:40].reshape((3, 5))
    parameters["b2"] = theta[40:43].reshape((3, 1))
    parameters["W3"] = theta[43:46].reshape((1, 3))
    parameters["b3"] = theta[46:47].reshape((1, 1))
    return parameters

def gradients_to_vector(gradients):
    """
    Roll all gradients dictionary values into a single vector.
    """
    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        new_vector = np.reshape(gradients[key], (-1, 1))
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1
    return theta

# Test case generator for N-dimensional gradient check
def gradient_check_n_test_case():
    np.random.seed(1)
    X = np.random.randn(4,3)
    Y = (np.random.randn(1,3) > 0)
    W1 = np.random.randn(5,4)
    b1 = np.random.randn(5,1)
    W2 = np.random.randn(3,5)
    b2 = np.random.randn(3,1)
    W3 = np.random.randn(1,3)
    b3 = np.random.randn(1,1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    return X, Y, parameters


def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7, print_msg=False):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
    """
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    for i in range(num_parameters):
        thetaplus = np.copy(parameters_values)
        thetaplus[i][0] += epsilon
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaplus))
        thetaminus = np.copy(parameters_values)
        thetaminus[i][0] -= epsilon
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus))
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if print_msg:
        if difference > 2e-7:
            print(f"There is a mistake in the backward propagation! difference = {difference}")
        else:
            print(f"Your backward propagation works perfectly fine! difference = {difference}")
    return difference


# Run the N-dimensional gradient check
def run_nd_gradient_check():
    X, Y, parameters = gradient_check_n_test_case()
    cost, cache = forward_propagation_n(X, Y, parameters)
    gradients = backward_propagation_n(X, Y, cache)
    difference = gradient_check_n(parameters, gradients, X, Y, print_msg=True)
    print("N-dimensional gradient check difference:", difference)

run_nd_gradient_check()