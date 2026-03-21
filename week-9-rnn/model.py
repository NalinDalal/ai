"""
model.py
RNN Model class with gradient clipping and perplexity.
"""

import numpy as np
from rnn import RNNLayer


class RNNModel:
    """RNN model for sequence modeling tasks."""

    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.grad_clip_val = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, loss, optimizer, grad_clip_val=None):
        self.loss = loss
        self.optimizer = optimizer
        self.grad_clip_val = grad_clip_val

    def forward(self, X, h0=None):
        current_input = X

        for layer in self.layers:
            if isinstance(layer, RNNLayer):
                output = layer.forward_sequence(current_input, h0)
                current_input = output

        return current_input

    def backward(self, doutput):
        for layer in reversed(self.layers):
            if isinstance(layer, RNNLayer):
                layer.backward_sequence(doutput)

    def train_step(self, X, y):
        output = self.forward(X)

        loss = self.loss.forward(output, y)

        doutput = self.loss.backward(output, y)

        self.backward(doutput)

        if self.grad_clip_val is not None:
            for layer in self.layers:
                if isinstance(layer, RNNLayer):
                    params_dict = {
                        "W_xh": (layer.cell.W_xh, layer.cell.dW_xh),
                        "W_hh": (layer.cell.W_hh, layer.cell.dW_hh),
                        "b_h": (layer.cell.b_h, layer.cell.db_h),
                        "W_hy": (layer.cell.W_hy, layer.cell.dW_hy),
                        "b_y": (layer.cell.b_y, layer.cell.db_y),
                    }
                    self.optimizer.clip_gradients(self.grad_clip_val, params_dict)

        for layer in self.layers:
            if isinstance(layer, RNNLayer):
                layer.update_weights(self.optimizer)

        return loss

    def predict(self, X, h0=None):
        output = self.forward(X, h0)
        predictions = np.argmax(output, axis=0)
        return predictions

    def perplexity(self, loss):
        return np.exp(loss)

    def fit(self, X, y, epochs=100, print_every=10, validation_data=None):
        for epoch in range(1, epochs + 1):
            total_loss = 0
            n_samples = 0

            for seq, target in zip(X, y):
                X_onehot = seq.reshape(-1, seq.shape[1], 1)
                loss = self.train_step(X_onehot, np.array([target]))
                total_loss += loss
                n_samples += 1

            avg_loss = total_loss / n_samples

            if epoch % print_every == 0:
                ppl = self.perplexity(avg_loss)
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Perplexity = {ppl:.4f}")

                if validation_data is not None:
                    X_val, y_val = validation_data
                    val_loss = 0
                    val_samples = 0
                    for seq, target in zip(X_val, y_val):
                        X_onehot = seq.reshape(-1, seq.shape[1], 1)
                        output = self.forward(X_onehot)
                        val_loss += self.loss.forward(output, np.array([target]))
                        val_samples += 1
                    val_avg_loss = val_loss / val_samples
                    val_ppl = self.perplexity(val_avg_loss)
                    print(
                        f"  Validation: Loss = {val_avg_loss:.4f}, Perplexity = {val_ppl:.4f}"
                    )
