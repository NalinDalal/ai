# Recurrent Neural Networks from Scratch (Karpathy Edition)

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
- **Input gate (i):** what new info to store
- **Forget gate (f):** what info to discard
- **Output gate (o):** what to output

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

## Character-Level LSTM (Karpathy Style)

### Multi-layer Stacked LSTM
Stack multiple LSTM layers for more expressive power.

### Temperature Sampling
Control the "creativity" of text generation:
- **Low temperature (0.3-0.5):** More deterministic, conservative
- **Medium temperature (0.8-1.0):** Balanced
- **High temperature (1.2-1.5):** More creative, riskier

### Truncated BPTT
Split long sequences into chunks for efficient training.

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

Lower = better model (measures how surprised the model is).

---

## Files

| File | Description |
|------|-------------|
| `rnn.py` | Basic RNN cell |
| `lstm.py` | LSTM cell (helper) |
| `char_model.py` | Multi-layer CharLSTM model |
| `data_utils.py` | Text data utilities |
| `main.py` | Training & generation demo |

## Running

```bash
python3 main.py
```

---

## Comparison: RNN vs LSTM vs CharLSTM

| Aspect | RNN | LSTM | CharLSTM |
|--------|-----|------|----------|
| Gates | None | Input, Forget, Output | Input, Forget, Output |
| Memory | Hidden state only | Cell state + Hidden | Cell + Multi-layer |
| Gradient flow | Via hidden state | Via cell state (additive) | Via cell (unchanged) |
| Long dependencies | Suffers vanishing | Handles well | Handles well |
| Parameters | ~4 | ~8 per gate | ~8 per gate × layers |
| Use case | Simple sequences | Long sequences | Character-level text |

## Key Insight

LSTM's cell state update: `C_t = f * C_{t-1} + i * C_tilde`

This additive path allows gradients to flow unchanged through time, solving the vanishing gradient problem!

## References

- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) - **Andrej Karpathy**
- [Dive into Deep Learning](http://d2l.ai/chapter_recurrent-neural-networks/rnn.html)
- [LSTM Paper - Hochreiter & Schmidhuber (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
