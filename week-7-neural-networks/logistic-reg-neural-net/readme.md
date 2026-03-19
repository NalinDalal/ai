# Logistic Regression with a Neural Network Mindset

> **Source:** [DLS C1 W2 — Logistic Regression as a Neural Network](https://nbviewer.org/github/amanchadha/coursera-deep-learning-specialization/blob/master/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%202/Logistic%20Regression%20as%20a%20Neural%20Network/Logistic.ipynb)
> **Context:** Week 7, Day 4 — first programming assignment in DLS C1
> **See also:** [Perceptron notes](../perceptron/readme.md) | [Week 7 schedule](../readme.md)

---

## 1. Why Logistic Regression = Simplest Neural Network

Logistic regression is a **single-neuron neural network** — it takes an input vector, applies a linear transformation followed by a non-linear activation (sigmoid), and outputs a probability. The [perceptron](../perceptron/readme.md) used a step function; this replaces it with a smooth, differentiable sigmoid so we can use **gradient descent** instead of the perceptron learning rule.

```
Input (x)  →  [Linear: z = wᵀx + b]  →  [Sigmoid: a = σ(z)]  →  ŷ ∈ (0,1)
```

---

## 2. Problem Statement

- **Dataset:** "data.h5" containing cat/non-cat RGB images
  - Training: `m_train = 209` images, shape `(209, 64, 64, 3)`
  - Test: `m_test = 50` images, shape `(50, 64, 64, 3)`
- **Goal:** Binary classification — is it a cat (`y=1`) or not (`y=0`)?
- **Feature vector:** Each image flattened to shape `(12288, 1)` where `12288 = 64 × 64 × 3`

---

## 3. Preprocessing

| Step | What | Why |
|------|------|-----|
| Flatten | `(m, 64, 64, 3)` → `(12288, m)` | Each column = one example as a vector |
| Normalize | Divide by 255 | Pixel values → `[0, 1]`; helps gradient descent converge |

```python
X_flatten = X_orig.reshape(X_orig.shape[0], -1).T   # shape (n, m)
X = X_flatten / 255.0
```

---

## 4. Mathematical Formulation

### 4.1 Forward Propagation (single example)

$$z^{(i)} = w^T x^{(i)} + b$$

$$\hat{y}^{(i)} = a^{(i)} = \sigma(z^{(i)}) = \frac{1}{1 + e^{-z^{(i)}}}$$

### 4.2 Loss Function (single example)

$$\mathcal{L}(a^{(i)}, y^{(i)}) = -\left[ y^{(i)} \log(a^{(i)}) + (1 - y^{(i)}) \log(1 - a^{(i)}) \right]$$

### 4.3 Cost Function (all examples)

$$J = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(a^{(i)}, y^{(i)})$$

### 4.4 Backward Propagation (gradients)

$$\frac{\partial J}{\partial w} = \frac{1}{m} X (A - Y)^T$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (a^{(i)} - y^{(i)})$$

### 4.5 Parameter Update

$$w := w - \alpha \frac{\partial J}{\partial w}$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

where $\alpha$ is the learning rate.

---

## 5. Algorithm Architecture

```
1. Initialize parameters      →  w = zeros, b = 0
2. For each iteration:
   a. Forward propagation     →  A = σ(wᵀX + b)
   b. Compute cost            →  J = -(1/m) Σ [y·log(a) + (1-y)·log(1-a)]
   c. Backward propagation    →  dw = (1/m) X(A-Y)ᵀ,  db = (1/m) Σ(a-y)
   d. Update parameters       →  w -= α·dw,  b -= α·db
3. Use learned (w, b) to predict: ŷ = 1 if σ(wᵀx + b) > 0.5 else 0
```

### Building Blocks (functions)

| Function | Input | Output |
|----------|-------|--------|
| `sigmoid(z)` | scalar or array `z` | `1 / (1 + exp(-z))` |
| `initialize_with_zeros(dim)` | number of features | `w` (dim, 1), `b = 0` |
| `propagate(w, b, X, Y)` | params + data | `grads`, `cost` |
| `optimize(w, b, X, Y, ...)` | params + data + hyperparams | learned `params`, `grads`, `costs` |
| `predict(w, b, X)` | learned params + data | `Y_prediction` (0/1) |
| `model(...)` | train + test data | complete results dict |

---

## 6. Key Results (from notebook)

| Metric | Value |
|--------|-------|
| Training accuracy | ~99.04% |
| Test accuracy | ~70.0% |
| Learning rate used | 0.005 |
| Iterations | 2000 |

**Observation:** The model overfits — training accuracy >> test accuracy. Later courses (DLS C2) address this with regularization.

### Learning Rate Comparison

| α | Train Acc | Test Acc | Notes |
|---|-----------|----------|-------|
| 0.01 | 99.52% | 68.0% | Good — may oscillate |
| 0.001 | 88.99% | 64.0% | Conservative |
| 0.0001 | 68.42% | 36.0% | Under-trained |

---

## 7. Key Takeaways

1. **Preprocessing matters** — flatten + normalize images before feeding to the model
2. **Vectorization** — avoid explicit loops; use `np.dot()` for matrix operations
3. **Logistic regression is a 1-layer NN** — single sigmoid neuron, trained by gradient descent
4. **Learning rate is a hyperparameter** — too large → oscillation; too small → slow convergence
5. **Overfitting** — when train acc >> test acc, need regularization (covered in DLS C2)

---

## 8. Connection to Previous Work

| Concept | Perceptron (Rosenblatt, 1958) | Logistic Regression |
|---------|-------------------------------|---------------------|
| Activation | Step function (0 or 1) | Sigmoid (smooth 0→1) |
| Learning rule | Perceptron rule (discrete) | Gradient descent (continuous) |
| Loss | Not well-defined | Cross-entropy |
| Differentiable? | No | Yes — enables backprop |
| Can learn XOR? | No | No (still linear) |

> The perceptron learning rule from [Paper 2](../perceptron/readme.md) adjusts weights by `±lr × input` on error. Logistic regression replaces this with gradient-based optimization on a smooth loss surface — the foundation for training deeper networks.

---

## 9. Files

| File | Description |
|------|-------------|
| `readme.md` | This document — notes & math |
| `logistic_regression.py` | Standalone implementation with synthetic data |

---

## Bibliography

- [DLS C1 W2 Notebook](https://nbviewer.org/github/amanchadha/coursera-deep-learning-specialization/blob/master/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%202/Logistic%20Regression%20as%20a%20Neural%20Network/Logistic.ipynb)
- [Implementing a Neural Network from Scratch](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/)
- [Why normalize by subtracting mean?](https://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c)