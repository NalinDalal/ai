"""
losses.py
Defines loss functions: Categorical, Binary, MSE, MAE.
"""
import numpy as np

class Loss:
    """Base loss class."""
    def regularization_loss(self, trainable_layers):
        reg_loss = 0
        for layer in trainable_layers:
            if hasattr(layer, 'weights'):
                if layer.weight_regularizer_l1 > 0:
                    reg_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
                if layer.weight_regularizer_l2 > 0:
                    reg_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
                if layer.bias_regularizer_l1 > 0:
                    reg_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
                if layer.bias_regularizer_l2 > 0:
                    reg_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        return reg_loss

    def remember_trainable_layers(self, trainable_layers):
        """Store reference to trainable layers for regularization loss calculation."""
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, include_regularization=False):
        """Calculate data loss and optionally regularization loss."""
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        if not include_regularization:
            return data_loss
        reg_loss = self.regularization_loss(self.trainable_layers)
        return data_loss, reg_loss

class Loss_CategoricalCrossentropy(Loss):
    """Categorical cross-entropy loss."""
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Loss_BinaryCrossentropy(Loss):
    """Binary cross-entropy loss."""
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples

class Loss_MeanSquaredError(Loss):
    """Mean squared error loss (L2)."""
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)
        return sample_losses
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

class Loss_MeanAbsoluteError(Loss):
    """Mean absolute error loss (L1)."""
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples
