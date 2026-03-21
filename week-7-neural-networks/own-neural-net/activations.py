"""
activations.py
Defines activation functions: ReLU, Softmax, Sigmoid, Linear.
"""
import numpy as np

class Activation_ReLU:
    """ReLU activation."""
    def forward(self, inputs, training=False):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
    def predictions(self, outputs):
        return outputs

class Activation_Softmax:
    """Softmax activation."""
    def forward(self, inputs, training=False):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

class Activation_Sigmoid:
    """Sigmoid activation."""
    def forward(self, inputs, training=False):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
    def predictions(self, outputs):
        return (outputs > 0.5) * 1

class Activation_Linear:
    """Linear activation (for regression)."""
    def forward(self, inputs, training=False):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
    def predictions(self, outputs):
        return outputs
