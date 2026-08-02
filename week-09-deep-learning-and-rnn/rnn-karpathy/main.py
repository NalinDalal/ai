"""
main.py - Character-Level LSTM Training (Karpathy Style)
Best of both: clean NumPy implementation + real text generation
"""

import numpy as np
from char_model import CharLSTM
from data_utils import TextData


def clip_gradients(model, clip_val=5.0):
    total_norm = np.sqrt(
        sum(
            np.sum(param**2)
            for param in [model.W_hy, model.b_y]
            + [
                p
                for cell in model.layers
                for p in [
                    cell.W_xi,
                    cell.W_hi,
                    cell.b_i,
                    cell.W_xf,
                    cell.W_hf,
                    cell.b_f,
                    cell.W_xo,
                    cell.W_ho,
                    cell.b_o,
                    cell.W_xc,
                    cell.W_hc,
                    cell.b_c,
                ]
            ]
        )
    )
    clip_coef = clip_val / (total_norm + 1e-6)
    if clip_coef < 1:
        for cell in model.layers:
            cell.dW_xi *= clip_coef
            cell.dW_hi *= clip_coef
            cell.db_i *= clip_coef
            cell.dW_xf *= clip_coef
            cell.dW_hf *= clip_coef
            cell.db_f *= clip_coef
            cell.dW_xo *= clip_coef
            cell.dW_ho *= clip_coef
            cell.db_o *= clip_coef
            cell.dW_xc *= clip_coef
            cell.dW_hc *= clip_coef
            cell.db_c *= clip_coef
        model.dW_hy *= clip_coef
        model.db_y *= clip_coef


def softmax_cross_entropy(logits, target_idx):
    probs = np.exp(logits - np.max(logits, axis=0))
    probs = probs / np.sum(probs, axis=0, keepdims=True)
    nll = -np.log(probs[target_idx, 0] + 1e-10)
    return nll, probs


def main():
    print("=" * 70)
    print("Character-Level LSTM - The Unreasonable Effectiveness of RNNs")
    print("=" * 70)

    text = """
A tale of two cities

It was the best of times, it was the worst of times, it was the age of wisdom, 
it was the age of foolishness, it was the epoch of belief, it was the epoch of 
incredulity, it was the season of Light, it was the season of Darkness, it was 
the spring of hope, it was the winter of despair, we had everything before us, 
we had nothing before us, we were all going direct to Heaven, we were all going 
direct the other way. The period was so far like the present period, that some 
of its noisiest authorities insisted on its being received, for good or for evil, 
in the superlative degree of comparison only. There were a king with a large 
jaw and a queen with a plain face, on the throne of England. There were a king 
with a large jaw and a queen with a fair face, on the throne of France. In both 
countries it was clearer than crystal to the lords of the State preserves of 
laces and ice, that the all-important question was about to be settled by the 
strong hand of war. This speech was the signal for the Tuileries and the Champs 
Elysees and the whole city to pour into the streets.
""".strip()

    data = TextData(text)
    print(f"\nText length: {len(text)} characters")
    print(f"Vocabulary: {data.vocab_size} unique characters")
    print(f"Characters: {''.join(data.chars)}")

    seq_len = 25
    hidden_size = 128
    num_layers = 2
    epochs = 100
    lr = 0.1
    grad_clip = 5.0

    model = CharLSTM(data.vocab_size, hidden_size, num_layers)
    print(
        f"\nModel: vocab={data.vocab_size}, hidden={hidden_size}, layers={num_layers}"
    )
    print(f"Training: seq_len={seq_len}, lr={lr}, epochs={epochs}")

    n_seqs = len(text) - seq_len
    print(f"Total sequences per epoch: {n_seqs}")

    for epoch in range(1, epochs + 1):
        total_loss = 0

        for i in range(n_seqs):
            seq = text[i : i + seq_len]
            target = text[i + seq_len]

            X = np.zeros((seq_len, data.vocab_size))
            for t, char in enumerate(seq):
                X[t, data.char2idx[char]] = 1
            target_idx = data.char2idx[target]

            logits = model.forward(X)
            loss, probs = softmax_cross_entropy(logits, target_idx)
            total_loss += loss

            dy = probs.copy()
            dy[target_idx, 0] -= 1

            model.backward(dy)
            clip_gradients(model, grad_clip)
            model.update(lr)

        avg_loss = total_loss / n_seqs
        perplexity = np.exp(avg_loss)

        if epoch % 25 == 0:
            print(
                f"\nEpoch {epoch:4d}: Loss={avg_loss:.4f}, Perplexity={perplexity:.2f}"
            )
            for temp in [0.5, 1.0]:
                sample = model.generate(
                    "It was", data.char2idx, data.idx2char, seq_len, 100, temp
                )
                print(f"  [T={temp}] {sample[:80]}...")

    print("\n" + "=" * 70)
    print("Final samples at different temperatures:")
    print("=" * 70)
    for temp in [0.3, 0.5, 0.8, 1.0, 1.5]:
        sample = model.generate(
            "It was", data.char2idx, data.idx2char, seq_len, 200, temp
        )
        print(f"\nTemperature={temp}:\n{sample[:150]}...")

    print("\n" + "=" * 70)
    print(
        "Training complete! The model has learned to generate text character by character."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
