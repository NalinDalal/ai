"""
rnn_cell.py - Simple RNN cell for learning
"""

import numpy as np


class RNNCell:
    def __init__(self, n_input, n_hidden):
        self.n_input = n_input
        self.n_hidden = n_hidden

        scale = 0.1
        self.W_xh = np.random.randn(n_hidden, n_input) * scale
        self.W_hh = np.random.randn(n_hidden, n_hidden) * scale
        self.b_h = np.zeros((n_hidden, 1))

        self.W_hy = np.random.randn(n_input, n_hidden) * scale
        self.b_y = np.zeros((n_input, 1))

    def forward(self, x, h_prev):
        self.x = x
        self.h_prev = h_prev
        self.h_raw = np.dot(self.W_xh, x) + np.dot(self.W_hh, h_prev) + self.b_h
        self.h = np.tanh(self.h_raw)
        self.y = np.dot(self.W_hy, self.h) + self.b_y
        return self.h, self.y

    def backward(self, dh_next, dy):
        batch = self.x.shape[1]

        dW_hy = np.dot(dy, self.h.T) / batch
        db_y = np.sum(dy, axis=1, keepdims=True) / batch
        dh = np.dot(self.W_hy.T, dy) + dh_next

        dh_raw = dh * (1 - self.h**2)

        dW_xh = np.dot(dh_raw, self.x.T) / batch
        dW_hh = np.dot(dh_raw, self.h_prev.T) / batch
        db_h = np.sum(dh_raw, axis=1, keepdims=True) / batch

        self.dW_xh = dW_xh
        self.dW_hh = dW_hh
        self.db_h = db_h
        self.dW_hy = dW_hy
        self.db_y = db_y

        dx = np.dot(self.W_xh.T, dh_raw)
        dh_prev = np.dot(self.W_hh.T, dh_raw)

        return dx, dh_prev

    def update(self, lr=0.01):
        self.W_xh -= lr * self.dW_xh
        self.W_hh -= lr * self.dW_hh
        self.b_h -= lr * self.db_h
        self.W_hy -= lr * self.dW_hy
        self.b_y -= lr * self.db_y
