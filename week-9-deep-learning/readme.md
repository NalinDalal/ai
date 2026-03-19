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