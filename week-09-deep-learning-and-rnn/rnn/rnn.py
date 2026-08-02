"""
rnn.py - RNN Layer with proper BPTT
"""

import numpy as np
from rnn_cell import RNNCell


class RNNLayer:
    def __init__(self, n_input, n_hidden):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.cell = RNNCell(n_input, n_hidden)

    def forward_sequence(self, X, h0=None):
        n_in, seq_len, batch = X.shape

        if h0 is None:
            h0 = np.zeros((self.n_hidden, batch))

        self.X = X
        self.seq_len = seq_len
        self.batch = batch
        self.h0 = h0

        self.H = np.zeros((self.n_hidden, seq_len, batch))
        self.Y = np.zeros((self.n_input, seq_len, batch))
        h = h0

        for t in range(seq_len):
            self.cell.x = X[:, t, :]
            self.cell.h_prev = h
            h, y = self.cell.forward(X[:, t, :], h)
            self.H[:, t, :] = h
            self.Y[:, t, :] = y

        self.final_h = h

        return self.Y[:, -1, :]

    def backward_sequence(self, dY):
        dh_next = np.zeros((self.n_hidden, self.batch))

        self.cell.dW_xh = np.zeros_like(self.cell.W_xh)
        self.cell.dW_hh = np.zeros_like(self.cell.W_hh)
        self.cell.db_h = np.zeros_like(self.cell.b_h)
        self.cell.dW_hy = np.zeros_like(self.cell.W_hy)
        self.cell.db_y = np.zeros_like(self.cell.b_y)

        for t in reversed(range(self.seq_len)):
            dy = dY if t == self.seq_len - 1 else np.zeros((self.n_input, self.batch))

            dh = np.dot(self.cell.W_hy.T, dy) + dh_next

            self.cell.h = self.H[:, t, :]
            self.cell.x = self.cell.x if t == self.seq_len - 1 else self.X[:, t, :]
            self.cell.h_prev = self.H[:, t - 1, :] if t > 0 else self.h0
            self.cell.h_raw = (
                np.dot(self.cell.W_xh, self.cell.x)
                + np.dot(self.cell.W_hh, self.cell.h_prev)
                + self.cell.b_h
            )

            dh_raw = dh * (1 - self.cell.h**2)

            dW_xh = np.dot(dh_raw, self.cell.x.T) / self.batch
            dW_hh = np.dot(dh_raw, self.cell.h_prev.T) / self.batch
            db_h = np.sum(dh_raw, axis=1, keepdims=True) / self.batch
            dW_hy = np.dot(dy, self.cell.h.T) / self.batch
            db_y = np.sum(dy, axis=1, keepdims=True) / self.batch

            self.cell.dW_xh += dW_xh
            self.cell.dW_hh += dW_hh
            self.cell.db_h += db_h
            self.cell.dW_hy += dW_hy
            self.cell.db_y += db_y

            dh_next = np.dot(self.cell.W_hh.T, dh_raw)

    def update(self, lr=0.01):
        self.cell.W_xh -= lr * self.cell.dW_xh
        self.cell.W_hh -= lr * self.cell.dW_hh
        self.cell.b_h -= lr * self.cell.db_h
        self.cell.W_hy -= lr * self.cell.dW_hy
        self.cell.b_y -= lr * self.cell.db_y
