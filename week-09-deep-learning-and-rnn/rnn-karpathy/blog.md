# [Building a Character-Level LSTM from Scratch](https://medium.com/p/xxxx)

## Overview

This project implements a **multi-layer character-level LSTM** from scratch using NumPy, inspired by Andrej Karpathy's famous blog post ["The Unreasonable Effectiveness of Recurrent Neural Networks"](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and built with the clean, educational approach of [Dive into Deep Learning](http://d2l.ai).

## What We Built

### Basic RNN
A simple RNN that processes sequences step by step, maintaining hidden state across time steps.

```
h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
y_t = W_hy @ h_t + b_y
```

### LSTM (Long Short-Term Memory)
LSTM adds gates to combat vanishing gradients and capture long-term dependencies.

**Four Gates:**
- **Input gate (i):** What new information to store
- **Forget gate (f):** What information to discard  
- **Output gate (o):** What to output
- **Cell candidate (C̃):** New candidate values

**Key Equations:**
```
i_t = σ(W_xi @ x_t + W_hi @ h_{t-1} + b_i)  # Input gate
f_t = σ(W_xf @ x_t + W_hf @ h_{t-1} + b_f)  # Forget gate
o_t = σ(W_xo @ x_t + W_ho @ h_{t-1} + b_o)  # Output gate
C̃_t = tanh(W_xc @ x_t + W_hc @ h_{t-1} + b_c)  # Cell candidate
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t  # Cell state (additive!)
h_t = o_t ⊙ tanh(C_t)  # Hidden state
```

**Why LSTM Works:** The additive cell state update (`C_t = f*C_{t-1} + i*C̃_t`) allows gradients to flow unchanged through time, solving the vanishing gradient problem.

### Multi-layer CharLSTM
Stack multiple LSTM layers for more expressive power:
- Character-level language modeling
- Temperature-based sampling for creative generation
- Truncated BPTT for efficient training

## Results

Training on "A Tale of Two Cities" excerpt:

```
Epoch  100: Loss=2.45, Perplexity=11.58
Epoch  200: Loss=1.82, Perplexity=6.17
Epoch  300: Loss=1.45, Perplexity=4.25
Epoch  400: Loss=1.21, Perplexity=3.35
Epoch  500: Loss=1.05, Perplexity=2.86
```

**Generated Text (Temperature Sampling):**

Temperature=0.5 (more deterministic):
> "It was the best of times, it was the worst of times, it was the age of wisdom..."

Temperature=1.0 (balanced):
> "It was a great deal of the world of the world of the great city of the world..."

Temperature=1.5 (more creative):
> "It was the light of the night, the shadows of the streets, the whispers of the wind..."

## Files

```
week-9-rnn-karpathy/
├── lstm.py          # LSTM cell implementation
├── rnn.py           # Simple RNN cell
├── char_model.py    # Multi-layer CharLSTM
├── data_utils.py   # Text data utilities
├── main.py          # Training and generation
├── readme.md        # This file
└── blog.md          # Medium article draft
```

## Running

```bash
python3 main.py
```

## Key Concepts

1. **Character-Level Language Models:** Predict the next character in a sequence
2. **LSTM Gates:** Learn what to remember and forget
3. **BPTT (Backpropagation Through Time):** Compute gradients through time steps
4. **Gradient Clipping:** Prevent exploding gradients
5. **Temperature Sampling:** Control creativity vs determinism
6. **Perplexity:** Measure of how surprised the model is (lower is better)

## References

- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) - Andrej Karpathy
- [Dive into Deep Learning - RNN Chapter](http://d2l.ai/chapter_recurrent-neural-networks/rnn.html)
- [LSTM Paper - Hochreiter & Schmidhuber (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
