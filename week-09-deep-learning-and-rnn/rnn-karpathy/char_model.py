"""
char_model.py - Multi-layer CharLSTM from Scratch (NumPy)
Best of both: d2l.ai implementation style + Karpathy's character-level approach
"""

import numpy as np
from lstm import LSTMCell


class CharLSTM:
    def __init__(self, vocab_size, hidden_size=256, num_layers=2, dropout=0.0):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = []
        for i in range(num_layers):
            input_size = vocab_size if i == 0 else hidden_size
            self.layers.append(LSTMCell(input_size, hidden_size))

        self.W_hy = np.random.randn(self.vocab_size, self.hidden_size) * np.sqrt(
            2.0 / self.hidden_size
        )
        self.b_y = np.zeros((self.vocab_size, 1))

        self.xs_per_layer = None
        self.hs = None
        self.Cs = None
        self.final_h = None
        self.final_C = None

    def forward(self, X):
        seq_len = X.shape[0]

        self.xs_per_layer = [[] for _ in range(self.num_layers)]
        self.hs = [np.zeros((self.hidden_size, 1)) for _ in range(seq_len + 1)]
        self.Cs = [np.zeros((self.hidden_size, 1)) for _ in range(seq_len + 1)]

        h = self.hs[0]
        C = self.Cs[0]

        for t in range(seq_len):
            x = X[t].reshape(-1, 1)
            self.xs_per_layer[0].append(x)
            for li, cell in enumerate(self.layers):
                h, C, _ = cell.forward(x, h, C)
                x = h
                if li < self.num_layers - 1:
                    self.xs_per_layer[li + 1].append(x)
            self.hs[t + 1] = h
            self.Cs[t + 1] = C

        self.final_h = self.hs[seq_len]
        self.final_C = self.Cs[seq_len]
        logits = np.dot(self.W_hy, self.final_h) + self.b_y
        return logits

    def backward(self, dy):
        seq_len = len(self.xs_per_layer[0])

        self.dW_hy = np.zeros_like(self.W_hy)
        self.db_y = np.zeros_like(self.b_y)
        for cell in self.layers:
            cell.dW_xi = np.zeros_like(cell.W_xi)
            cell.dW_hi = np.zeros_like(cell.W_hi)
            cell.db_i = np.zeros_like(cell.b_i)
            cell.dW_xf = np.zeros_like(cell.W_xf)
            cell.dW_hf = np.zeros_like(cell.W_hf)
            cell.db_f = np.zeros_like(cell.b_f)
            cell.dW_xo = np.zeros_like(cell.W_xo)
            cell.dW_ho = np.zeros_like(cell.W_ho)
            cell.db_o = np.zeros_like(cell.b_o)
            cell.dW_xc = np.zeros_like(cell.W_xc)
            cell.dW_hc = np.zeros_like(cell.W_hc)
            cell.db_c = np.zeros_like(cell.b_c)

        dh_next = np.zeros((self.hidden_size, 1))
        dC_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(seq_len)):
            self.dW_hy += np.dot(dy, self.final_h.T)
            self.db_y += dy
            dh = np.dot(self.W_hy.T, dy) + dh_next
            dC = dC_next

            for li in reversed(range(self.num_layers)):
                cell = self.layers[li]
                x = self.xs_per_layer[li][t]
                h_prev = self.hs[t]
                C_prev = self.Cs[t]

                cell.x = x
                cell.h_prev = h_prev
                cell.C_prev = C_prev

                do = dh * np.tanh(cell.C)
                do_raw = do * cell.o * (1 - cell.o)
                dC = dC + dh * cell.o * (1 - np.tanh(cell.C) ** 2)
                di = dC * cell.C_tilde
                di_raw = di * cell.i * (1 - cell.i)
                df = dC * cell.C_prev
                df_raw = df * cell.f * (1 - cell.f)
                dC_tilde = dC * cell.i
                dC_tilde_raw = dC_tilde * (1 - cell.C_tilde**2)

                cell.dW_xi += np.dot(di_raw, x.T)
                cell.dW_hi += np.dot(di_raw, h_prev.T)
                cell.db_i += di_raw
                cell.dW_xf += np.dot(df_raw, x.T)
                cell.dW_hf += np.dot(df_raw, h_prev.T)
                cell.db_f += df_raw
                cell.dW_xo += np.dot(do_raw, x.T)
                cell.dW_ho += np.dot(do_raw, h_prev.T)
                cell.db_o += do_raw
                cell.dW_xc += np.dot(dC_tilde_raw, x.T)
                cell.dW_hc += np.dot(dC_tilde_raw, h_prev.T)
                cell.db_c += dC_tilde_raw

                dh_next = (
                    np.dot(cell.W_hi.T, di_raw)
                    + np.dot(cell.W_hf.T, df_raw)
                    + np.dot(cell.W_ho.T, do_raw)
                    + np.dot(cell.W_hc.T, dC_tilde_raw)
                )
                dC_next = dC * cell.f

    def update(self, lr=0.01):
        self.W_hy -= lr * self.dW_hy
        self.b_y -= lr * self.db_y
        for cell in self.layers:
            cell.W_xi -= lr * cell.dW_xi
            cell.W_hi -= lr * cell.dW_hi
            cell.b_i -= lr * cell.db_i
            cell.W_xf -= lr * cell.dW_xf
            cell.W_hf -= lr * cell.dW_hf
            cell.b_f -= lr * cell.db_f
            cell.W_xo -= lr * cell.dW_xo
            cell.W_ho -= lr * cell.dW_ho
            cell.b_o -= lr * cell.db_o
            cell.W_xc -= lr * cell.dW_xc
            cell.W_hc -= lr * cell.dW_hc
            cell.b_c -= lr * cell.db_c

    def sample(self, h, C, seed_idx, temperature=1.0):
        x = np.zeros((self.vocab_size, 1))
        x[seed_idx, 0] = 1
        for li, cell in enumerate(self.layers):
            h, C, _ = cell.forward(x, h, C)
            x = h
        logits = np.dot(self.W_hy, h) + self.b_y
        probs = np.exp(logits / temperature - np.max(logits / temperature, axis=0))
        probs = probs / np.sum(probs, axis=0, keepdims=True)
        idx = int(np.argmax(np.random.rand() < np.cumsum(probs.flatten())))
        return idx, h, C

    def generate(
        self, seed_text, char2idx, idx2char, seq_len, num_chars=100, temperature=1.0
    ):
        h = np.zeros((self.hidden_size, 1))
        C = np.zeros((self.hidden_size, 1))
        for char in seed_text[-seq_len:]:
            idx = char2idx.get(char, 0)
            x = np.zeros((self.vocab_size, 1))
            x[idx, 0] = 1
            for cell in self.layers:
                h, C, _ = cell.forward(x, h, C)
                x = h
        result = list(seed_text)
        last_idx = char2idx.get(seed_text[-1], 0)
        for _ in range(num_chars):
            idx, h, C = self.sample(h, C, last_idx, temperature)
            result.append(idx2char[idx])
            last_idx = idx
        return "".join(result)
