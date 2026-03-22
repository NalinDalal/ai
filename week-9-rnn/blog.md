# [Building RNN and LSTM from Scratch](https://medium.com/p/d8e9759e47a8?postPublishedType=initial)

## Overview

This project implements Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks from scratch using NumPy, following the d2l.ai approach.

## What We Built

### Basic RNN
A simple RNN that processes sequences step by step, maintaining hidden state across time steps.

**Forward Pass:**
```
h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
y_t = W_hy @ h_t + b_y
```

**Backpropagation Through Time (BPTT):**
Gradients flow backward through time, computing partial derivatives at each step.

### LSTM
LSTM adds gates to combat vanishing gradients and capture long-term dependencies.

**Three Gates:**
- **Input gate (i):** What new information to store
- **Forget gate (f):** What information to discard  
- **Output gate (o):** What to output

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

## Results

Both models trained on "hello world" text:

| Model | Accuracy | Perplexity |
|-------|----------|------------|
| RNN   | 96.4%    | 1.07       |
| LSTM  | 96.4%    | 1.07       |

**Generated Text:**
- Input: "hello" → Output: "hello world hello world h"

## Files

```
week-9-rnn/
├── readme.md       # Technical documentation
├── rnn_cell.py    # RNN cell implementation
├── rnn.py         # RNN layer
├── lstm.py        # LSTM layer with gates
├── lstm_cell.py   # LSTM cell helper
├── losses.py      # Cross-entropy loss
├── optimizers.py   # SGD, Adam optimizers
├── main.py        # RNN demo
├── main_lstm.py   # LSTM demo
└── blog.md        # This file
```

## Key Concepts Learned

1. **Sequence Processing:** RNNs naturally handle sequential data
2. **Hidden State:** Memory that captures previous time step information
3. **BPTT:** Backpropagating through time to compute gradients
4. **Gradient Clipping:** Preventing exploding gradients
5. **LSTM Gates:** Learning what to remember and forget
6. **Perplexity:** Measuring language model quality

## Running the Code

```bash
# RNN Demo
python3 main.py

# LSTM Demo  
python3 main_lstm.py
```

## References

- [Dive into Deep Learning - RNN Chapter](http://d2l.ai/chapter_recurrent-neural-networks/rnn.html)
- [LSTM Paper - Hochreiter & Schmidhuber (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
