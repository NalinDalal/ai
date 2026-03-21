## Neural Network Initialization: Why and How

- **Initialization** is critical for training deep neural networks efficiently.
- Poor initialization can cause slow learning, vanishing/exploding gradients, or failure to break symmetry (all neurons learn the same thing).
- Three main initialization strategies:
  1. **Zeros**: All weights set to zero. Fails to break symmetry; network doesn’t learn.
  2. **Random (large)**: Weights set to large random values. Breaks symmetry, but can cause unstable gradients and slow learning.
  3. **He Initialization**: Weights set to random values scaled by $\sqrt{2/\text{fan-in}}$ (fan-in = number of input units). Works best for ReLU activations.

### Key Takeaways

- Always initialize weights randomly (not zeros).
- Use He initialization for ReLU networks.
- Modularize code for easy swapping/testing of initialization and other components.

---

### Modular init.py Design

- **init.py**: Main entry point, exposes modular functions for initialization, model building, training, and evaluation.
- **initializations.py**: Contains all initialization methods (zeros, random, He, Xavier, etc.).
- **model.py**: Contains the neural network model logic (forward, backward, update, etc.).
- **utils.py**: Helper functions (activation, loss, plotting, etc.).
- **datasets.py**: Data loading and preprocessing.

---

**What you should remember:**
- Different initializations lead to different results
- Random initialization is used to break symmetry and make sure different hidden units can learn different things
- Don't intialize to values that are too large
- He initialization works well for networks with ReLU activations.

## Regularization: Theory & Implementation

### Theory
Regularization helps prevent overfitting in neural networks by adding a penalty to the loss function, discouraging overly complex models.

**Common Types:**
- **L2 Regularization (Ridge):** Adds $\lambda \sum W^2$ to the loss, penalizing large weights. Makes the model simpler and more generalizable.
- **L1 Regularization (Lasso):** Adds $\lambda \sum |W|$ to the loss, encouraging sparsity (some weights become zero).
- **Dropout:** Randomly sets a fraction of activations to zero during training, forcing the network to not rely on any single neuron.

**Effect:**
Regularization reduces variance (overfitting) at the cost of a slight increase in bias, leading to better generalization on unseen data.

---

### Implementation Plan
1. **L2 Regularization:**  
  - Add an L2 penalty term to your loss function:  
    $J = \text{original loss} + \frac{\lambda}{2m} \sum_l \|W^{[l]}\|^2$
  - Update gradients to include the L2 term.

2. **Dropout:**  
  - During forward propagation, randomly set some activations to zero with probability $p$.
  - Scale activations during training to keep expected values consistent.

Regularization reduces overfitting and improves model generalization.

---

## Optimization Methods: Gradient Descent, Momentum, Adam

Until now, we've used basic gradient descent to update parameters. Advanced optimization methods can speed up learning and achieve better results.

### 1. Mini-Batch Gradient Descent

Instead of using all m examples (batch GD) or just 1 example (SGD), mini-batch GD uses intermediate-sized batches.

**Steps:**
1. **Shuffle**: Randomly shuffle training data (X, Y) keeping alignment
2. **Partition**: Split into mini-batches of size `mini_batch_size`

```python
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    # Shuffle
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    
    # Partition into complete mini-batches
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size:(k+1)*mini_batch_size]
    
    # Handle last mini-batch (may be smaller)
```

### 2. Gradient Descent Update

$$W^{[l]} = W^{[l]} - \alpha \cdot dW^{[l]}$$
$$b^{[l]} = b^{[l]} - \alpha \cdot db^{[l]}$$

### 3. Momentum

Momentum uses exponentially weighted averages of past gradients to smooth out updates.

**Update Rule:**
$$v_{dW^{[l]}} = \beta v_{dW^{[l]}} + (1 - \beta) dW^{[l]}$$
$$W^{[l]} = W^{[l]} - \alpha v_{dW^{[l]}}$$

- $\beta$ (momentum): typically 0.9
- Helps reduce oscillations, faster convergence

### 4. Adam (Adaptive Moment Estimation)

Combines Momentum and RMSProp. Maintains two moving averages:
- $v$: exponentially weighted average of gradients (first moment)
- $s$: exponentially weighted average of squared gradients (second moment)

**Update Rule:**
$$v_{dW^{[l]}} = \beta_1 v_{dW^{[l]}} + (1 - \beta_1) dW^{[l]}$$
$$v^{corrected}_{dW^{[l]}} = \frac{v_{dW^{[l]}}}{1 - \beta_1^t}$$
$$s_{dW^{[l]}} = \beta_2 s_{dW^{[l]}} + (1 - \beta_2) (dW^{[l]})^2$$
$$s^{corrected}_{dW^{[l]}} = \frac{s_{dW^{[l]}}}{1 - \beta_2^t}$$
$$W^{[l]} = W^{[l]} - \alpha \frac{v^{corrected}_{dW^{[l]}}}{\sqrt{s^{corrected}_{dW^{[l]}}} + \varepsilon}$$

- $\beta_1$: typically 0.9
- $\beta_2$: typically 0.999
- $\varepsilon$: 1e-8 (prevents division by zero)

### Summary

| Method | Pros | Cons |
|--------|------|------|
| GD | Simple | Slow, can get stuck |
| Mini-batch GD | Faster, better convergence | Requires tuning batch size |
| Momentum | Reduces oscillations | Needs $\beta$ tuning |
| Adam | Fast, adaptive learning rates | More hyperparameters |

**Key Takeaways:**
- Mini-batch size powers of 2 (16, 32, 64, 128) work well
- Adam usually converges fastest with minimal tuning
- Momentum helps in curved valleys
- Bias correction needed for Adam in early iterations

---

## TensorFlow Framework Transition

Transition from NumPy implementation to TensorFlow for faster development and automatic differentiation.

### Key Differences

| NumPy | TensorFlow |
|-------|------------|
| Manual forward/backward | Automatic with `GradientTape` |
| NumPy arrays | `tf.Tensor` |
| Manual parameter updates | `optimizer.apply_gradients()` |
| No computation graph | `@tf.function` for optimization |

### Core Functions

```python
# tf.Variable - mutable parameters (like our NumPy arrays)
W1 = tf.Variable(initializer(shape=(25, 12288)), name="W1")

# @tf.function - compiles to optimized graph
@tf.function
def forward_propagation(X, parameters):
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.keras.activations.relu(Z1)
    ...

# GradientTape - automatic differentiation
with tf.GradientTape() as tape:
    Z3 = forward_propagation(minibatch_X, parameters)
    minibatch_cost = compute_cost(Z3, minibatch_Y)

grads = tape.gradient(minibatch_cost, trainable_variables)
optimizer.apply_gradients(zip(grads, trainable_variables))

# tf.data.Dataset - efficient data pipeline
dataset = tf.data.Dataset.from_tensor_slices((X, Y))
dataset = dataset.batch(32).prefetch(8)
```

### Files
- **tensorflow_model.py**: Full TensorFlow implementation

### Running
```bash
source venv/bin/activate
python tensorflow_model.py
```

### Advantages of TensorFlow
- Automatic differentiation (no manual backprop)
- GPU acceleration
- Optimized computation graphs
- Built-in optimizers (`tf.keras.optimizers.Adam`, etc.)
- Efficient data pipelines with `tf.data`