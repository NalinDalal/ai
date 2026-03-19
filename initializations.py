"""
initializations.py - Weight initialization methods for neural networks
"""
import numpy as np
import math

def initialize_parameters(layers_dims, method="he"):
    """
    Initialize parameters for a neural network.
    Args:
        layers_dims: List of layer sizes.
        method: "zeros", "random", or "he".
    Returns:
        parameters: dict of initialized weights and biases.
    """
    parameters = {}
    L = len(layers_dims)
    np.random.seed(3)
    for l in range(1, L):
        if method == "zeros":
            parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        elif method == "random":
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 10
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        elif method == "he":
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * math.sqrt(2. / layers_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        else:
            raise ValueError("Unknown initialization method: {}".format(method))
    return parameters
