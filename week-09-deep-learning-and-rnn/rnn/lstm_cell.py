"""
lstm_cell.py - LSTM Cell with proper BPTT
"""

import numpy as np


class LSTMCell:
    def __init__(self, n_input, n_hidden):
        self.n_input = n_input
        self.n_hidden = n_hidden

        scale = 0.1

        self.W_xi = np.random.randn(n_hidden, n_input) * scale
        self.W_hi = np.random.randn(n_hidden, n_hidden) * scale
        self.b_i = np.zeros((n_hidden, 1))

        self.W_xf = np.random.randn(n_hidden, n_input) * scale
        self.W_hf = np.random.randn(n_hidden, n_hidden) * scale
        self.b_f = np.zeros((n_hidden, 1))

        self.W_xo = np.random.randn(n_hidden, n_input) * scale
        self.W_ho = np.random.randn(n_hidden, n_hidden) * scale
        self.b_o = np.zeros((n_hidden, 1))

        self.W_xc = np.random.randn(n_hidden, n_input) * scale
        self.W_hc = np.random.randn(n_hidden, n_hidden) * scale
        self.b_c = np.zeros((n_hidden, 1))

        self.W_hy = np.random.randn(n_input, n_hidden) * scale
        self.b_y = np.zeros((n_input, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x, h_prev, C_prev):
        self.x = x
        self.h_prev = h_prev
        self.C_prev = C_prev

        self.i = self.sigmoid(
            np.dot(self.W_xi, x) + np.dot(self.W_hi, h_prev) + self.b_i
        )
        self.f = self.sigmoid(
            np.dot(self.W_xf, x) + np.dot(self.W_hf, h_prev) + self.b_f
        )
        self.o = self.sigmoid(
            np.dot(self.W_xo, x) + np.dot(self.W_ho, h_prev) + self.b_o
        )
        self.C_tilde = np.tanh(
            np.dot(self.W_xc, x) + np.dot(self.W_hc, h_prev) + self.b_c
        )

        self.C = self.f * C_prev + self.i * self.C_tilde
        self.h = self.o * np.tanh(self.C)

        self.y = np.dot(self.W_hy, self.h) + self.b_y

        return self.h, self.C, self.y

    def backward(self, dh_next, dC_next, dy):
        batch = self.x.shape[1]

        dW_hy = np.dot(dy, self.h.T) / batch
        db_y = np.sum(dy, axis=1, keepdims=True) / batch
        dh = np.dot(self.W_hy.T, dy) + dh_next

        do = dh * np.tanh(self.C)
        do_raw = do * self.o * (1 - self.o)

        dC = dh * self.o * (1 - np.tanh(self.C) ** 2)
        dC = dC + dC_next

        di = dC * self.C_tilde
        di_raw = di * self.i * (1 - self.i)

        df = dC * self.C_prev
        df_raw = df * self.f * (1 - self.f)

        dC_tilde = dC * self.i
        dC_tilde_raw = dC_tilde * (1 - self.C_tilde**2)

        self.dW_xi = np.dot(di_raw, self.x.T) / batch
        self.dW_hi = np.dot(di_raw, self.h_prev.T) / batch
        self.db_i = np.sum(di_raw, axis=1, keepdims=True) / batch

        self.dW_xf = np.dot(df_raw, self.x.T) / batch
        self.dW_hf = np.dot(df_raw, self.h_prev.T) / batch
        self.db_f = np.sum(df_raw, axis=1, keepdims=True) / batch

        self.dW_xo = np.dot(do_raw, self.x.T) / batch
        self.dW_ho = np.dot(do_raw, self.h_prev.T) / batch
        self.db_o = np.sum(do_raw, axis=1, keepdims=True) / batch

        self.dW_xc = np.dot(dC_tilde_raw, self.x.T) / batch
        self.dW_hc = np.dot(dC_tilde_raw, self.h_prev.T) / batch
        self.db_c = np.sum(dC_tilde_raw, axis=1, keepdims=True) / batch

        self.dW_hy = dW_hy
        self.db_y = db_y

        dx_i = np.dot(self.W_xi.T, di_raw)
        dx_f = np.dot(self.W_xf.T, df_raw)
        dx_o = np.dot(self.W_xo.T, do_raw)
        dx_c = np.dot(self.W_xc.T, dC_tilde_raw)
        dx = dx_i + dx_f + dx_o + dx_c

        dh_prev = (
            np.dot(self.W_hi.T, di_raw)
            + np.dot(self.W_hf.T, df_raw)
            + np.dot(self.W_ho.T, do_raw)
            + np.dot(self.W_hc.T, dC_tilde_raw)
        )

        dC_prev = dC * self.f

        return dx, dh_prev, dC_prev

    def update(self, lr=0.01):
        self.W_xi -= lr * self.dW_xi
        self.W_hi -= lr * self.dW_hi
        self.b_i -= lr * self.db_i

        self.W_xf -= lr * self.dW_xf
        self.W_hf -= lr * self.dW_hf
        self.b_f -= lr * self.db_f

        self.W_xo -= lr * self.dW_xo
        self.W_ho -= lr * self.dW_ho
        self.b_o -= lr * self.db_o

        self.W_xc -= lr * self.dW_xc
        self.W_hc -= lr * self.dW_hc
        self.b_c -= lr * self.db_c

        self.W_hy -= lr * self.dW_hy
        self.b_y -= lr * self.db_y
