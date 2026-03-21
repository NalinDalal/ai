"""
losses.py
Loss functions for RNN.
"""

import numpy as np


class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        raise NotImplementedError


class LossCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[np.arange(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)

    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        labels = len(y_pred[0])

        if len(y_true.shape) == 1:
            y_true_onehot = np.eye(labels)[y_true]
        else:
            y_true_onehot = y_true

        self.dinputs = -y_true_onehot / y_pred
        self.dinputs = self.dinputs / samples
        return self.dinputs


class LossSparseCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = y_pred.shape[1]
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        y_true = y_true.flatten()

        correct_confidences = y_pred_clipped[y_true, np.arange(samples)]
        negative_log_likelihoods = -np.log(correct_confidences)

        return np.mean(negative_log_likelihoods)

    def backward(self, y_pred, y_true):
        samples = y_pred.shape[1]
        y_true = y_true.flatten()

        y_true_onehot = np.zeros_like(y_pred)
        y_true_onehot[y_true, np.arange(samples)] = 1

        self.dinputs = -y_true_onehot / y_pred
        self.dinputs = self.dinputs / samples
        return self.dinputs


class LossMSE(Loss):
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        self.dinputs = 2 * (y_pred - y_true) / samples
        return self.dinputs
