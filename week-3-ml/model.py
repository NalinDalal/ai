"""
model.py - Neural network model class
"""
import numpy as np
from utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation, update_parameters, predict

class NeuralNetwork:
    def __init__(self, layers_dims, parameters, lambd=0.0, keep_prob=1.0):
        self.layers_dims = layers_dims
        self.parameters = parameters
        self.lambd = lambd
        self.keep_prob = keep_prob

    def train(self, X, Y, num_iterations=15000, learning_rate=0.01, print_cost=True):
        costs = []
        for i in range(num_iterations):
            a3, cache = forward_propagation(X, self.parameters, keep_prob=self.keep_prob)
            cost = compute_loss(a3, Y, parameters=self.parameters, lambd=self.lambd)
            grads = backward_propagation(X, Y, cache, parameters=self.parameters, lambd=self.lambd)
            self.parameters = update_parameters(self.parameters, grads, learning_rate)
            if print_cost and i % 1000 == 0:
                print(f"Cost after iteration {i}: {cost}")
                costs.append(cost)
        return costs

    def evaluate(self, X, Y):
        predictions = predict(X, Y, self.parameters, keep_prob=1.0)  # No dropout at test time
        accuracy = np.mean(predictions == Y)
        return accuracy
