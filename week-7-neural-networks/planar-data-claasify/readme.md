## Planar Data Classification — One Hidden Layer Neural Network

> **Source:** Deep Learning Specialization, Course 1, Week 3 (Andrew Ng, Coursera)
> **Significance:** First hands-on implementation of a shallow neural network (one hidden layer) for binary classification. Bridges theory (backpropagation) and practice (NumPy code).

---

### 1. Context & Motivation
- Classic toy problem: classify points in a 2D plane ("planar data") into two classes.
- Linear classifiers (e.g., perceptron, logistic regression) fail for non-linear boundaries.
- Introduces the power of neural networks to learn complex decision boundaries.

---

### 2. Model Architecture
- **Input:** $X \in \mathbb{R}^{2 \times m}$ (2 features, $m$ examples)
- **Hidden layer:** $n_h$ units, activation $\tanh$
- **Output layer:** 1 unit, activation $\sigma$ (sigmoid)
- **Parameters:** $W_1, b_1, W_2, b_2$

---

### 3. Forward & Backward Pass
- **Forward:** $Z_1 = W_1 X + b_1$, $A_1 = \tanh(Z_1)$, $Z_2 = W_2 A_1 + b_2$, $A_2 = \sigma(Z_2)$
- **Loss:** Cross-entropy for binary classification
- **Backward:** Compute gradients $dW_1, db_1, dW_2, db_2$ via chain rule
- **Update:** Gradient descent

---

### 4. Implementation Notes
- All steps coded from scratch in NumPy
- Visualize decision boundary after training
- Compare with logistic regression baseline
- Key learning: how hidden layers enable non-linear separation

---

### 5. Reference Assignment
- [Planar Data Classification Notebook (external)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%203/Planar%20Data%20Classification%20v1.ipynb)
- Follow notebook for step-by-step implementation

---

### 6. Further Reading
- [Backpropagation Paper](../backpropogation/readme.md)
- [Perceptron](../perceptron/readme.md)
- [McCulloch-Pitts Neuron](../neuron-as-logic-gate/readme.md)
