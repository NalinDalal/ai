"""
data_utils.py - Data utilities for character-level language modeling
"""

import numpy as np


class TextData:
    def __init__(self, text):
        self.text = text
        self.chars = sorted(list(set(text)))
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.vocab_size = len(self.chars)

    def encode(self, char):
        return self.char2idx.get(char, 0)

    def decode(self, idx):
        return self.idx2char.get(idx, "")

    def get_batches(self, seq_len, batch_size):
        sequences = []
        targets = []
        for i in range(0, len(self.text) - seq_len, seq_len // 2):
            seq = self.text[i : i + seq_len]
            target = self.text[i + 1 : i + seq_len + 1]
            sequences.append([self.char2idx[c] for c in seq])
            targets.append([self.char2idx[c] for c in target])

        for _ in range(len(sequences) % batch_size):
            sequences.append(sequences[0])
            targets.append(targets[0])

        sequences = np.array(sequences).reshape(-1, batch_size, seq_len)
        targets = np.array(targets).reshape(-1, batch_size, seq_len)
        return sequences, targets

    def one_hot(self, indices, seq_len, batch_size):
        one_hot = np.zeros((batch_size, seq_len, self.vocab_size))
        for b in range(batch_size):
            for t in range(seq_len):
                one_hot[b, t, indices[b, t]] = 1
        return one_hot.transpose(2, 1, 0)

    @staticmethod
    def load_from_file(filepath):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return TextData(text)
