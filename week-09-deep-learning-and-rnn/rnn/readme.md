# Recurrent Neural Networks from Scratch

- [Awesome Deep Learning](https://github.com/ChristosChristofidis/awesome-deep-learning) — reference list, not a curriculum

## RNN (Basic)

### Forward Pass
$$h^{(t)} = \tanh(W_{xh} x^{(t)} + W_{hh} h^{(t-1)} + b_h)$$
$$y^{(t)} = W_{hy} h^{(t)} + b_y$$

### Backpropagation Through Time (BPTT)
Gradients flow backward through time with potential vanishing/exploding problems.

---

## LSTM (Long Short-Term Memory)

LSTM adds **gates** to combat vanishing gradients and capture long-term dependencies.

### Gates
- **Input gate (i)**: what new info to store
- **Forget gate (f)**: what info to discard
- **Output gate (o)**: what to output

### Cell State
The cell state acts as memory, passing through time with additive updates (prevents vanishing).

### Equations

**Gates:**
$$i^{(t)} = \sigma(W_{xi} x^{(t)} + W_{hi} h^{(t-1)} + b_i)$$
$$f^{(t)} = \sigma(W_{xf} x^{(t)} + W_{hf} h^{(t-1)} + b_f)$$
$$o^{(t)} = \sigma(W_{xo} x^{(t)} + W_{ho} h^{(t-1)} + b_o)$$

**Cell candidate:**
$$\tilde{C}^{(t)} = \tanh(W_{xc} x^{(t)} + W_{hc} h^{(t-1)} + b_c)$$

**Cell state update:**
$$C^{(t)} = f^{(t)} \odot C^{(t-1)} + i^{(t)} \odot \tilde{C}^{(t)}$$

**Hidden state:**
$$h^{(t)} = o^{(t)} \odot \tanh(C^{(t)})$$

### Why LSTM Works
- Cell state updates via **addition** (not multiplication)
- Gates control information flow
- Gradient flows through cell state unchanged (constant error carousel)

---

## Gradient Clipping

Prevents exploding gradients:
```python
total_norm = sqrt(sum(grad**2 for grad in gradients))
clip_coef = clip_val / (total_norm + eps)
if clip_coef < 1:
    for grad in gradients:
        grad *= clip_coef
```

## Perplexity
$$\text{Perplexity} = \exp(-\frac{1}{N} \sum_i \log p(x_i))$$

Lower = better model.

---

## Files

| File | Description |
|------|-------------|
| `rnn_cell.py` | Basic RNN cell |
| `rnn.py` | RNN layer |
| `lstm.py` | LSTM layer with gates |
| `lstm_cell.py` | LSTM cell (helper) |
| `losses.py` | Cross-entropy loss |
| `main.py` | RNN demo |
| `main_lstm.py` | LSTM demo |

## Running

```bash
# RNN
python3 main.py

# LSTM
python3 main_lstm.py
```

---

## Comparison: RNN vs LSTM

| Aspect | RNN | LSTM |
|--------|-----|------|
| Gates | None | Input, Forget, Output |
| Memory | Hidden state only | Cell state + Hidden state |
| Gradient flow | Via hidden state | Via cell state (additive) |
| Long dependencies | Suffers vanishing | Handles well |
| Parameters | 3 weight matrices | 8 weight matrices |

## Key Insight

LSTM's cell state update: `C_t = f * C_{t-1} + i * C_tilde`

This additive path allows gradients to flow unchanged through time, solving the vanishing gradient problem!
