"""
main_lstm.py - LSTM Character-Level Language Model
"""

import numpy as np
from lstm import LSTMLayer


def clip_gradients(grads, clip_val=1.0):
    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
    clip_coef = clip_val / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads.values():
            g *= clip_coef


def main():
    print("=" * 60)
    print("LSTM from Scratch - Character-Level Language Model")
    print("=" * 60)

    text = "hello world hello world" * 5
    chars = sorted(list(set(text)))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    vocab_size = len(chars)

    print(f'\nTraining text: "{text[:50]}..."')
    print(f"Vocab ({vocab_size}): {''.join(chars)}")

    seq_len = 5
    X_seqs = []
    y_chars = []

    for i in range(len(text) - seq_len):
        X_seqs.append([char2idx[c] for c in text[i : i + seq_len]])
        y_chars.append(char2idx[text[i + seq_len]])

    X_onehot = np.zeros((vocab_size, seq_len, len(X_seqs)))
    for i, seq in enumerate(X_seqs):
        for t, idx in enumerate(seq):
            X_onehot[idx, t, i] = 1

    print(f"Sequences: {len(X_seqs)}, Seq length: {seq_len}")

    n_hidden = 32
    lr = 0.1
    epochs = 300
    grad_clip = 5.0

    lstm = LSTMLayer(vocab_size, n_hidden)

    print(f"\nModel: vocab={vocab_size}, hidden={n_hidden}")
    print(f"Training: lr={lr}, grad_clip={grad_clip}, epochs={epochs}")

    for epoch in range(1, epochs + 1):
        total_loss = 0

        indices = np.random.permutation(len(X_seqs))
        for i in indices:
            X = X_onehot[:, :, i : i + 1]
            y = y_chars[i]

            output = lstm.forward_sequence(X)

            probs = np.exp(output - np.max(output))
            probs = probs / np.sum(probs)

            loss = -np.log(probs[y, 0] + 1e-10)
            total_loss += loss

            doutput = probs.copy()
            doutput[y, 0] -= 1

            lstm.backward_sequence(doutput)

            grads = {
                "dW_xi": lstm.dW_xi,
                "dW_hi": lstm.dW_hi,
                "dW_xf": lstm.dW_xf,
                "dW_hf": lstm.dW_hf,
                "dW_xo": lstm.dW_xo,
                "dW_ho": lstm.dW_ho,
                "dW_xc": lstm.dW_xc,
                "dW_hc": lstm.dW_hc,
                "dW_hy": lstm.dW_hy,
            }
            clip_gradients(grads, grad_clip)

            lstm.update(lr)

        avg_loss = total_loss / len(X_seqs)
        perplexity = np.exp(avg_loss)

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, Perplexity={perplexity:.2f}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    print("\nPredictions:")
    correct = 0
    for i, seq in enumerate(X_seqs):
        seq_str = "".join([idx2char[idx] for idx in seq])
        target = idx2char[y_chars[i]]

        X = X_onehot[:, :, i : i + 1]
        output = lstm.forward_sequence(X)
        probs = np.exp(output - np.max(output))
        probs = probs / np.sum(probs)
        pred = idx2char[np.argmax(probs)]

        match = "✓" if pred == target else "✗"
        if pred == target:
            correct += 1
        if i < 10:
            print(f"  '{seq_str}' -> '{pred}' (target: '{target}') {match}")

    print(f"\nAccuracy: {correct}/{len(X_seqs)} = {100 * correct / len(X_seqs):.1f}%")

    print("\nText generation:")
    for prefix in ["hello", "world", "llo w"]:
        result = list(prefix)

        for _ in range(20):
            seq_idx = [char2idx[c] for c in result[-seq_len:]]
            X = np.zeros((vocab_size, seq_len, 1))
            for t, idx in enumerate(seq_idx):
                X[idx, t, 0] = 1

            output = lstm.forward_sequence(X)
            probs = np.exp(output - np.max(output))
            probs = probs / np.sum(probs)
            next_idx = np.argmax(probs)
            result.append(idx2char[next_idx])

        print(f"  '{prefix}' -> '{''.join(result)}'")


if __name__ == "__main__":
    main()
