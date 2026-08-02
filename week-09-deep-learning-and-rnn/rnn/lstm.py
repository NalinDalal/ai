"""
lstm.py - LSTM Layer with proper BPTT
"""

import numpy as np


class LSTMLayer:
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

    def forward_sequence(self, X, h0=None, C0=None):
        n_in, seq_len, batch = X.shape

        if h0 is None:
            h0 = np.zeros((self.n_hidden, batch))
        if C0 is None:
            C0 = np.zeros((self.n_hidden, batch))

        self.X = X
        self.seq_len = seq_len
        self.batch = batch
        self.h0 = h0
        self.C0 = C0

        self.xs = []
        self.h_prevs = []
        self.C_prevs = []
        self.is_ = []
        self.fs = []
        self.os = []
        self.C_tildes = []
        self.Cs = []
        self.hs = []

        h = h0
        C = C0

        for t in range(seq_len):
            x = X[:, t, :]
            self.xs.append(x)
            self.h_prevs.append(h)
            self.C_prevs.append(C)

            i = self.sigmoid(np.dot(self.W_xi, x) + np.dot(self.W_hi, h) + self.b_i)
            f = self.sigmoid(np.dot(self.W_xf, x) + np.dot(self.W_hf, h) + self.b_f)
            o = self.sigmoid(np.dot(self.W_xo, x) + np.dot(self.W_ho, h) + self.b_o)
            C_tilde = np.tanh(np.dot(self.W_xc, x) + np.dot(self.W_hc, h) + self.b_c)

            C = f * C + i * C_tilde
            h = o * np.tanh(C)

            self.is_.append(i)
            self.fs.append(f)
            self.os.append(o)
            self.C_tildes.append(C_tilde)
            self.Cs.append(C)
            self.hs.append(h)

        self.final_h = h
        self.final_C = C

        return np.dot(self.W_hy, h) + self.b_y

    def backward_sequence(self, dY):
        batch = self.batch

        dh_next = np.zeros((self.n_hidden, batch))
        dC_next = np.zeros((self.n_hidden, batch))

        self.dW_xi = np.zeros_like(self.W_xi)
        self.dW_hi = np.zeros_like(self.W_hi)
        self.db_i = np.zeros_like(self.b_i)

        self.dW_xf = np.zeros_like(self.W_xf)
        self.dW_hf = np.zeros_like(self.W_hf)
        self.db_f = np.zeros_like(self.b_f)

        self.dW_xo = np.zeros_like(self.W_xo)
        self.dW_ho = np.zeros_like(self.W_ho)
        self.db_o = np.zeros_like(self.b_o)

        self.dW_xc = np.zeros_like(self.W_xc)
        self.dW_hc = np.zeros_like(self.W_hc)
        self.db_c = np.zeros_like(self.b_c)

        self.dW_hy = np.zeros_like(self.W_hy)
        self.db_y = np.zeros_like(self.b_y)

        for t in reversed(range(self.seq_len)):
            dy = dY if t == self.seq_len - 1 else np.zeros((self.n_input, batch))

            dW_hy = np.dot(dy, self.hs[t].T) / batch
            db_y = np.sum(dy, axis=1, keepdims=True) / batch
            dh = np.dot(self.W_hy.T, dy) + dh_next

            do = dh * np.tanh(self.Cs[t])
            do_raw = do * self.os[t] * (1 - self.os[t])

            dC = dh * self.os[t] * (1 - np.tanh(self.Cs[t]) ** 2)
            dC = dC + dC_next

            di = dC * self.C_tildes[t]
            di_raw = di * self.is_[t] * (1 - self.is_[t])

            df = dC * self.C_prevs[t]
            df_raw = df * self.fs[t] * (1 - self.fs[t])

            dC_tilde = dC * self.is_[t]
            dC_tilde_raw = dC_tilde * (1 - self.C_tildes[t] ** 2)

            self.dW_xi += np.dot(di_raw, self.xs[t].T) / batch
            self.dW_hi += np.dot(di_raw, self.h_prevs[t].T) / batch
            self.db_i += np.sum(di_raw, axis=1, keepdims=True) / batch

            self.dW_xf += np.dot(df_raw, self.xs[t].T) / batch
            self.dW_hf += np.dot(df_raw, self.h_prevs[t].T) / batch
            self.db_f += np.sum(df_raw, axis=1, keepdims=True) / batch

            self.dW_xo += np.dot(do_raw, self.xs[t].T) / batch
            self.dW_ho += np.dot(do_raw, self.h_prevs[t].T) / batch
            self.db_o += np.sum(do_raw, axis=1, keepdims=True) / batch

            self.dW_xc += np.dot(dC_tilde_raw, self.xs[t].T) / batch
            self.dW_hc += np.dot(dC_tilde_raw, self.h_prevs[t].T) / batch
            self.db_c += np.sum(dC_tilde_raw, axis=1, keepdims=True) / batch

            self.dW_hy += dW_hy
            self.db_y += db_y

            dh_next = (
                np.dot(self.W_hi.T, di_raw)
                + np.dot(self.W_hf.T, df_raw)
                + np.dot(self.W_ho.T, do_raw)
                + np.dot(self.W_hc.T, dC_tilde_raw)
            )

            dC_next = dC * self.fs[t]

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
