## [Rumelhart, Hinton & Williams (1986) — "Learning Representations by Back-propagating Errors"](https://github.com/georgezoto/Convolutional-Neural-Networks/blob/master/Papers/1986%20Backpro%20Learning%20representations%20by%20back-propagating%20errors%20-%20Rumelhart,%20Hinton,%20Williams.pdf)

> **Authors:** David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams
> **Published in:** Nature, Vol 323, pp 533-536, 9 October 1986
> **Significance:** Demonstrated that backpropagation (gradient descent via the chain rule) can learn useful internal representations in multi-layer networks — directly solving the "credit assignment problem" that had stalled neural network research for nearly two decades after Minsky & Papert's critique of the perceptron.

---

### 1. Historical Context

- **Minsky & Papert (1969)** proved that single-layer perceptrons cannot learn XOR or any non-linearly-separable function. This effectively killed neural network funding for ~15 years ("AI Winter").
- The idea of multi-layer networks existed, but there was no efficient algorithm for training the hidden layers — the **credit assignment problem**: how do you know which hidden unit is responsible for the error at the output?
- **Werbos (1974)** described backpropagation in his PhD thesis, and **Parker (1985)** independently rediscovered it, but neither gained traction.
- Rumelhart, Hinton & Williams (1986) provided the clearest formulation, proved it worked on non-trivial tasks, and published in Nature — giving the method the visibility it needed.
- This paper revived neural network research and directly led to the modern deep learning era.

---

### 2. The Problem

**What problem is it solving?**
How to train a neural network with one or more hidden layers so that the hidden units learn useful internal representations of the input — without being explicitly told what those representations should be.

**What is the intended goal?**
Provide a general-purpose learning procedure for multi-layer networks that adjusts all weights (input→hidden and hidden→output) to minimize the error between the network's output and the desired output.

**What makes it hard?**
- Single-layer perceptrons have a clear error signal at the output — you can directly assign blame to each weight.
- In a multi-layer network, hidden units don't have a "desired output" — you only know the final error. You need a way to propagate that error backward through the layers to assign credit/blame to every weight in the network.
- This requires computing the **partial derivative of the error with respect to every weight**, including weights in layers far from the output.

---

### 3. Key Idea — The Generalized Delta Rule

The paper proposes a two-phase procedure:

1. **Forward pass:** Input propagates through the network, layer by layer, to produce an output.
2. **Backward pass:** The error (difference between actual and desired output) propagates backward through the network using the **chain rule of calculus**, computing how much each weight contributed to the error.

The weight update rule (gradient descent):

$$\Delta w_{ij} = -\eta \frac{\partial E}{\partial w_{ij}}$$

Where:
- $\Delta w_{ij}$ = change in weight from unit $j$ to unit $i$
- $\eta$ = learning rate (small positive constant)
- $E$ = total error (loss)
- $\frac{\partial E}{\partial w_{ij}}$ = how much $w_{ij}$ contributed to the error

The insight: **the chain rule lets you decompose** $\frac{\partial E}{\partial w_{ij}}$ **into a product of local derivatives**, computed layer by layer from output back to input.

---

### 4. Mathematical Derivation

#### 4a. Network Structure

- Units indexed by $i, j, k, \ldots$
- $y_i$ = output of unit $i$
- $x_i = \sum_j w_{ij} y_j$ = total input to unit $i$ (weighted sum of incoming activations)
- $y_i = f(x_i)$ where $f$ is a differentiable, non-linear activation function

#### 4b. Activation Function

The paper uses the **logistic sigmoid**:

$$f(x) = \frac{1}{1 + e^{-x}}$$

Key property — the derivative is expressible in terms of the output itself:

$$f'(x) = f(x)(1 - f(x)) = y(1 - y)$$

This makes computation efficient: you already have $y$ from the forward pass.

#### 4c. Error Measure

For a set of input-output training pairs, the total error is:

$$E = \frac{1}{2} \sum_c \sum_j (t_{j,c} - y_{j,c})^2$$

Where:
- $c$ indexes over training cases
- $j$ indexes over output units
- $t_{j,c}$ = target (desired) output for unit $j$ on case $c$
- $y_{j,c}$ = actual output

For a single training case:

$$E = \frac{1}{2} \sum_j (t_j - y_j)^2$$

#### 4d. The Chain Rule Decomposition

To compute $\frac{\partial E}{\partial w_{ij}}$, apply the chain rule:

$$\frac{\partial E}{\partial w_{ij}} = \frac{\partial E}{\partial x_i} \cdot \frac{\partial x_i}{\partial w_{ij}}$$

Since $x_i = \sum_j w_{ij} y_j$:

$$\frac{\partial x_i}{\partial w_{ij}} = y_j$$

Define the **error signal** (delta) for unit $i$:

$$\delta_i = -\frac{\partial E}{\partial x_i}$$

Then:

$$\frac{\partial E}{\partial w_{ij}} = -\delta_i \cdot y_j$$

And the weight update becomes:

$$\Delta w_{ij} = \eta \cdot \delta_i \cdot y_j$$

#### 4e. Computing Delta — Output Units

For an output unit $j$:

$$\delta_j = -\frac{\partial E}{\partial x_j} = -\frac{\partial E}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_j}$$

$$\frac{\partial E}{\partial y_j} = -(t_j - y_j)$$

$$\frac{\partial y_j}{\partial x_j} = f'(x_j) = y_j(1 - y_j)$$

Therefore:

$$\boxed{\delta_j = (t_j - y_j) \cdot y_j(1 - y_j)} \quad \text{(output unit)}$$

#### 4f. Computing Delta — Hidden Units

For a hidden unit $i$ that feeds into units $k$ in the next layer:

$$\delta_i = -\frac{\partial E}{\partial x_i} = -\frac{\partial E}{\partial y_i} \cdot f'(x_i)$$

The key chain rule step — error propagates backward:

$$\frac{\partial E}{\partial y_i} = -\sum_k \delta_k \cdot w_{ki}$$

(sum over all units $k$ that unit $i$ feeds into)

Therefore:

$$\boxed{\delta_i = f'(x_i) \cdot \sum_k \delta_k \cdot w_{ki}} \quad \text{(hidden unit)}$$

This is the **backpropagation equation** — the delta at a hidden unit is computed from the deltas at the next layer, weighted by the connecting weights, and scaled by the local derivative.

#### 4g. Full Algorithm Summary

```
1. FORWARD PASS:
   For each layer l = 1, 2, ..., L:
       x_i = Σ_j w_ij · y_j       (weighted sum)
       y_i = f(x_i)                (activation)

2. COMPUTE OUTPUT ERROR:
   E = ½ Σ_j (t_j - y_j)²

3. BACKWARD PASS:
   For output units:
       δ_j = (t_j - y_j) · f'(x_j)

   For hidden units (from last hidden layer back to first):
       δ_i = f'(x_i) · Σ_k δ_k · w_ki

4. WEIGHT UPDATE:
   For all weights:
       w_ij ← w_ij + η · δ_i · y_j
```

---

### 5. Why Differentiable Activations Matter

The M-P neuron (1943) and Rosenblatt's perceptron (1958) used **step functions** (0 or 1 output). Step functions have zero derivative almost everywhere — you cannot compute $\frac{\partial E}{\partial w}$ through them.

Backpropagation requires a **smooth, differentiable** activation function so that:
- The chain rule produces non-zero gradients
- Small weight changes produce proportional output changes

The sigmoid $\sigma(x) = \frac{1}{1+e^{-x}}$ was the original choice. Modern networks use:
- **tanh:** $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ — zero-centered, derivative $= 1 - \tanh^2(x)$
- **ReLU:** $\max(0, x)$ — avoids vanishing gradient for positive inputs (not in this paper, introduced later)

---

### 6. Results from the Paper

The paper demonstrated backpropagation on several tasks:

#### 6a. XOR Problem
- 2 inputs, 2 hidden units, 1 output
- The network learned the correct XOR mapping — something a perceptron provably cannot do
- The hidden units learned to create a **linearly separable representation** of the XOR inputs

#### 6b. Symmetry Detection
- Input: binary vector of varying length
- Task: detect whether the input is symmetric (palindrome)
- Hidden units learned to encode "position pairs" — comparing first with last, second with second-to-last, etc.

#### 6c. Family Tree Relationships
- Input: (person, relationship) → output: person
- 24 people, 12 relationships
- Hidden units learned to encode features like nationality, generation, and branch of family tree — **without being told these features exist**

> This was the headline result: the network **discovered meaningful internal representations** purely from input-output pairs.

---

### 7. Strengths

1. **Solves the credit assignment problem** — gives a principled way to train hidden units
2. **General-purpose** — works on any differentiable network architecture, any number of layers
3. **Learns internal representations** — hidden units discover features that are useful for the task, not hand-engineered
4. **Mathematically elegant** — the entire algorithm follows from the chain rule and gradient descent
5. **Computationally tractable** — cost is $O(\text{weights})$ per training example (linear in network size)

### 8. Limitations & What Breaks

1. **Vanishing gradients** — in deep networks, gradients shrink exponentially as they propagate backward through many sigmoid layers ($f'(x) \leq 0.25$ for sigmoid). Deep networks couldn't train → addressed later by ReLU, batch normalization, residual connections.
2. **Local minima** — gradient descent can get stuck in local minima (though in practice, saddle points are more common than true local minima in high dimensions).
3. **Slow convergence** — vanilla gradient descent with a fixed learning rate is slow. Later addressed by momentum, RMSProp, Adam.
4. **No guarantee of finding the global optimum** — the loss surface is non-convex.
5. **Requires labeled data** — purely supervised; no unsupervised or reinforcement learning.
6. **Sensitive to hyperparameters** — learning rate, network architecture, weight initialization all matter significantly.
7. **Biological plausibility is debatable** — real neurons probably don't propagate precise error gradients backward through synapses (though this is still debated).

---

### 9. Connection to Previous Papers

```
McCulloch & Pitts (1943)     → neurons compute, but NO learning
        ↓
Rosenblatt (1958)            → perceptron LEARNS, but only 1 layer → can't do XOR
        ↓
Minsky & Papert (1969)       → proved the limitation, killed funding
        ↓
Rumelhart et al. (1986)      → backprop lets you train HIDDEN LAYERS
                                → solves XOR, learns internal representations
                                → revives neural networks
        ↓
Modern deep learning          → same algorithm, better activations, 
                                 better optimization, more data, GPUs
```

---

### 10. Key Takeaway

Backpropagation is the **chain rule applied systematically to compute gradients in a multi-layer network**, combined with gradient descent to update weights. It solved the fundamental problem that had blocked neural network progress for two decades: how to train hidden layers. The hidden units learn to create internal representations that are useful for the task — representations the programmer never specified. This single idea, published in a 4-page Nature paper, is the foundation of all modern deep learning.

---

### 11. Momentum (Extension from the Paper)

The paper also introduced **momentum** to accelerate convergence:

$$\Delta w_{ij}(t) = \eta \cdot \delta_i \cdot y_j + \alpha \cdot \Delta w_{ij}(t-1)$$

Where $\alpha \in [0, 1)$ is the momentum coefficient. This smooths out oscillations and speeds up training in flat regions of the loss surface.

---

### 12. [Implementation](./backprop_nn.py)

Single-hidden-layer neural network in NumPy (tanh hidden, sigmoid output) with gradient checking.
See `backprop_nn.py` for the full implementation.