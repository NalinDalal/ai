"""
activations.py
Activation functions for RNN.
"""

import numpy as np


class Activation:
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, dvalues):
        raise NotImplementedError


class ActivationTanh(Activation):
    def forward(self, inputs):
        self.output = np.tanh(inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output**2)
        return self.dinputs


class ActivationSigmoid(Activation):
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-np.clip(inputs, -500, 500)))
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues * self.output * (1 - self.output)
        return self.dinputs


class ActivationReLU(Activation):
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.output <= 0] = 0
        return self.dinputs


class ActivationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        self.output = probabilities
        return self.output

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for i, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )
            self.dinputs[i] = np.dot(jacobian_matrix, single_dvalues)
        return self.dinputs

    def predictions(self, outputs):
        return np.argmax(outputs, axis=-1)


class ActivationLinear:
    def forward(self, inputs):
        self.output = inputs
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        return self.dinputs
