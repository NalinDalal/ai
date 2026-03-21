"""
model.py - Neural network model class with different optimization methods
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    sigmoid,
    relu,
    compute_loss,
    forward_propagation,
    backward_propagation,
    update_parameters,
    predict,
)
from optimization import (
    update_parameters_with_gd,
    random_mini_batches,
    initialize_velocity,
    update_parameters_with_momentum,
    initialize_adam,
    update_parameters_with_adam,
)
from initializations import initialize_parameters


class NeuralNetwork:
    def __init__(self, layers_dims, parameters, lambd=0.0, keep_prob=1.0):
        self.layers_dims = layers_dims
        self.parameters = parameters
        self.lambd = lambd
        self.keep_prob = keep_prob

    def train(self, X, Y, num_iterations=15000, learning_rate=0.01, print_cost=True):
        costs = []
        for i in range(num_iterations):
            a3, cache = forward_propagation(
                X, self.parameters, keep_prob=self.keep_prob
            )
            cost = compute_loss(a3, Y, parameters=self.parameters, lambd=self.lambd)
            grads = backward_propagation(
                X, Y, cache, parameters=self.parameters, lambd=self.lambd
            )
            self.parameters = update_parameters(self.parameters, grads, learning_rate)
            if print_cost and i % 1000 == 0:
                print(f"Cost after iteration {i}: {cost}")
                costs.append(cost)
        return costs

    def evaluate(self, X, Y):
        predictions = predict(X, Y, self.parameters, keep_prob=1.0)
        accuracy = np.mean(predictions == Y)
        return accuracy


def model(
    X,
    Y,
    layers_dims,
    optimizer="gd",
    learning_rate=0.0007,
    mini_batch_size=64,
    beta=0.9,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    num_epochs=10000,
    print_cost=True,
    lambd=0.0,
    keep_prob=1.0,
):
    """
    3-layer neural network model which can be run in different optimizer modes.

    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    optimizer -- the optimizer: "gd", "momentum", "adam"
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs
    lambd -- L2 regularization parameter
    keep_prob -- dropout probability

    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    L = len(layers_dims)
    costs = []
    t = 0
    seed = 10
    m = X.shape[1]

    print(f"\nThe number of training examples is: {m}")
    print(f"The mini-batch size: {mini_batch_size}")

    parameters = initialize_parameters(layers_dims)

    v, s = None, None
    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    for i in range(num_epochs):
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch

            a3, caches = forward_propagation(
                minibatch_X, parameters, keep_prob=keep_prob
            )
            cost_total += compute_loss(
                a3, minibatch_Y, parameters=parameters, lambd=lambd
            )
            grads = backward_propagation(
                minibatch_X, minibatch_Y, caches, parameters=parameters, lambd=lambd
            )

            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(
                    parameters, grads, v, beta, learning_rate
                )
            elif optimizer == "adam":
                t = t + 1
                parameters, v, s = update_parameters_with_adam(
                    parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon
                )

        cost_avg = cost_total / m

        if print_cost and i % 1000 == 0:
            print(f"Cost after epoch {i}: {cost_avg}")
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)

    if print_cost:
        plt.plot(costs)
        plt.ylabel("cost")
        plt.xlabel("epochs (per 100)")
        plt.title(f"Learning rate = {learning_rate}, Optimizer = {optimizer}")
        plt.show()

    return parameters
