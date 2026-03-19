"""
Backpropagation from Scratch — Rumelhart, Hinton & Williams (1986)

Single-hidden-layer neural network in NumPy:
  - Hidden layer: tanh activation
  - Output layer: sigmoid activation
  - Loss: binary cross-entropy
  - Gradient checking to verify derivatives

Demonstrates: XOR, and a simple 2D classification task.
"""

import numpy as np

# ─────────────────────────── Activation Functions ───────────────────────────


def sigmoid(z):
    """Logistic sigmoid — maps R → (0, 1)."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a):
    """Derivative of sigmoid, expressed in terms of its own output a = σ(z)."""
    return a * (1.0 - a)


def tanh(z):
    """Hyperbolic tangent — maps R → (-1, 1). Zero-centered."""
    return np.tanh(z)


def tanh_derivative(a):
    """Derivative of tanh, expressed in terms of its own output a = tanh(z)."""
    return 1.0 - a ** 2


# ─────────────────────────── Parameter Initialization ──────────────────────


def initialize_parameters(n_input, n_hidden, n_output, seed=42):
    """
    Initialize weights with small random values and biases with zeros.

    Parameters
    ----------
    n_input  : int — number of input features
    n_hidden : int — number of hidden units
    n_output : int — number of output units

    Returns
    -------
    params : dict with W1, b1, W2, b2
    """
    np.random.seed(seed)
    params = {
        "W1": np.random.randn(n_hidden, n_input) * 0.5,  # (h, n)
        "b1": np.zeros((n_hidden, 1)),  # (h, 1)
        "W2": np.random.randn(n_output, n_hidden) * 0.5,  # (o, h)
        "b2": np.zeros((n_output, 1)),  # (o, 1)
    }
    return params


# ─────────────────────────── Forward Pass ──────────────────────────────────


def forward(X, params):
    """
    Forward propagation through a 2-layer network.

    X      : (n_input, m) — m training examples
    params : dict with W1, b1, W2, b2

    Returns
    -------
    A2    : (n_output, m) — predictions
    cache : dict with Z1, A1, Z2, A2 for the backward pass
    """
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    # Hidden layer — tanh
    Z1 = W1 @ X + b1  # (h, m)
    A1 = tanh(Z1)  # (h, m)

    # Output layer — sigmoid
    Z2 = W2 @ A1 + b2  # (o, m)
    A2 = sigmoid(Z2)  # (o, m)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


# ─────────────────────────── Loss Function ─────────────────────────────────


def compute_loss(A2, Y):
    """
    Binary cross-entropy loss.

    A2 : (1, m) — predictions
    Y  : (1, m) — true labels

    Returns
    -------
    loss : scalar
    """
    m = Y.shape[1]
    eps = 1e-8  # numerical stability
    loss = -(1.0 / m) * np.sum(Y * np.log(A2 + eps) + (1 - Y) * np.log(1 - A2 + eps))
    return float(np.squeeze(loss))


# ─────────────────────────── Backward Pass ─────────────────────────────────


def backward(X, Y, params, cache):
    """
    Backpropagation — compute gradients of loss w.r.t. all parameters.

    X      : (n_input, m)
    Y      : (1, m)
    params : dict
    cache  : dict from forward pass

    Returns
    -------
    grads : dict with dW1, db1, dW2, db2
    """
    m = X.shape[1]
    W2 = params["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    # ── Output layer gradients ──
    # dL/dZ2 = A2 - Y  (for sigmoid + cross-entropy, the math simplifies)
    dZ2 = A2 - Y  # (1, m)
    dW2 = (1.0 / m) * (dZ2 @ A1.T)  # (o, h)
    db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)  # (o, 1)

    # ── Hidden layer gradients (backprop through tanh) ──
    # δ_hidden = (W2^T · δ_output) * tanh'(Z1)
    dA1 = W2.T @ dZ2  # (h, m)
    dZ1 = dA1 * tanh_derivative(A1)  # (h, m)
    dW1 = (1.0 / m) * (dZ1 @ X.T)  # (h, n)
    db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)  # (h, 1)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads


# ─────────────────────────── Gradient Checking ─────────────────────────────


def gradient_check(X, Y, params, epsilon=1e-7):
    """
    Numerical gradient checking — verify analytical gradients are correct.

    Computes numerical gradients via finite differences:
        dθ ≈ [L(θ+ε) - L(θ-ε)] / 2ε

    Compares with analytical gradients from backward().

    Returns
    -------
    diff : float — relative difference (should be < 1e-5)
    """
    # Get analytical gradients
    A2, cache = forward(X, params)
    grads = backward(X, Y, params, cache)

    # Flatten parameters and gradients into vectors
    param_keys = ["W1", "b1", "W2", "b2"]
    grad_keys = ["dW1", "db1", "dW2", "db2"]

    theta = np.concatenate([params[k].ravel() for k in param_keys])
    d_theta = np.concatenate([grads[k].ravel() for k in grad_keys])

    num_params = theta.shape[0]
    d_theta_numerical = np.zeros(num_params)

    # Compute numerical gradient for each parameter
    for i in range(num_params):
        theta_plus = theta.copy()
        theta_plus[i] += epsilon

        theta_minus = theta.copy()
        theta_minus[i] -= epsilon

        # Rebuild params from flattened vector
        params_plus = _unflatten(theta_plus, params, param_keys)
        params_minus = _unflatten(theta_minus, params, param_keys)

        A2_plus, _ = forward(X, params_plus)
        A2_minus, _ = forward(X, params_minus)

        loss_plus = compute_loss(A2_plus, Y)
        loss_minus = compute_loss(A2_minus, Y)

        d_theta_numerical[i] = (loss_plus - loss_minus) / (2.0 * epsilon)

    # Relative difference
    numerator = np.linalg.norm(d_theta - d_theta_numerical)
    denominator = np.linalg.norm(d_theta) + np.linalg.norm(d_theta_numerical)
    diff = numerator / (denominator + 1e-8)

    return diff, d_theta, d_theta_numerical


def _unflatten(flat_vector, params_template, keys):
    """Rebuild params dict from a flat vector, using template for shapes."""
    rebuilt = {}
    idx = 0
    for k in keys:
        shape = params_template[k].shape
        size = params_template[k].size
        rebuilt[k] = flat_vector[idx : idx + size].reshape(shape)
        idx += size
    return rebuilt


# ─────────────────────────── Training Loop ─────────────────────────────────


def train(X, Y, n_hidden=4, learning_rate=0.5, epochs=10000, print_every=1000, seed=42):
    """
    Train a single-hidden-layer NN on data (X, Y).

    Parameters
    ----------
    X             : (n_input, m)
    Y             : (1, m)
    n_hidden      : int — number of hidden units
    learning_rate : float — η
    epochs        : int
    print_every   : int — print loss every N epochs

    Returns
    -------
    params : trained parameters
    losses : list of loss values
    """
    n_input = X.shape[0]
    n_output = Y.shape[0]
    params = initialize_parameters(n_input, n_hidden, n_output, seed=seed)
    losses = []

    for epoch in range(1, epochs + 1):
        # Forward
        A2, cache = forward(X, params)

        # Loss
        loss = compute_loss(A2, Y)
        losses.append(loss)

        # Backward
        grads = backward(X, Y, params, cache)

        # Update weights (gradient descent)
        params["W1"] -= learning_rate * grads["dW1"]
        params["b1"] -= learning_rate * grads["db1"]
        params["W2"] -= learning_rate * grads["dW2"]
        params["b2"] -= learning_rate * grads["db2"]

        if epoch % print_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:>5d}  |  Loss: {loss:.6f}")

    return params, losses


def predict(X, params):
    """Return binary predictions (threshold = 0.5)."""
    A2, _ = forward(X, params)
    return (A2 > 0.5).astype(int)


# ─────────────────────────── Demos ─────────────────────────────────────────


def demo_xor():
    """
    XOR — the problem that killed the perceptron.
    Backpropagation solves it with a single hidden layer.
    """
    print("=" * 60)
    print("DEMO 1: XOR Problem")
    print("=" * 60)

    # XOR dataset: 4 examples, 2 features each
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])  # (2, 4)
    Y = np.array([[0, 1, 1, 0]])  # (1, 4)

    print(f"\nInputs:\n{X.T}")
    print(f"Targets: {Y.ravel()}")

    # Train
    print("\nTraining...")
    params, losses = train(X, Y, n_hidden=4, learning_rate=1.0, epochs=10000, print_every=2000)

    # Predict
    preds = predict(X, params)
    A2, _ = forward(X, params)
    print(f"\nPredictions (raw):  {np.round(A2.ravel(), 4)}")
    print(f"Predictions (bin):  {preds.ravel()}")
    print(f"Targets:            {Y.ravel()}")
    print(f"Correct: {np.sum(preds == Y)}/{Y.size}")

    # Gradient check
    print("\nGradient check...")
    diff, _, _ = gradient_check(X, Y, params)
    print(f"  Relative difference: {diff:.2e}")
    if diff < 1e-5:
        print("  ✓ Gradients are correct!")
    else:
        print("  ✗ Gradient mismatch — check implementation.")


def demo_circle():
    """
    2D classification: points inside vs outside a circle.
    Non-linearly separable — requires hidden layer.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Circle Classification (non-linear boundary)")
    print("=" * 60)

    np.random.seed(1)
    m = 200
    X = np.random.randn(2, m) * 2  # (2, 200)
    radius = np.sum(X ** 2, axis=0)  # distance from origin squared
    Y = (radius < 2.0).astype(float).reshape(1, m)  # inside circle → 1

    print(f"\n{m} points, {int(Y.sum())} inside circle, {m - int(Y.sum())} outside")

    # Train
    print("\nTraining...")
    params, losses = train(
        X, Y, n_hidden=8, learning_rate=1.0, epochs=15000, print_every=3000, seed=3
    )

    # Accuracy
    preds = predict(X, params)
    accuracy = np.mean(preds == Y) * 100
    print(f"\nTraining accuracy: {accuracy:.1f}%")

    # Gradient check on a small subset
    print("\nGradient check (on 4-point subset)...")
    diff, _, _ = gradient_check(X[:, :4], Y[:, :4], params)
    print(f"  Relative difference: {diff:.2e}")
    if diff < 1e-5:
        print("  ✓ Gradients are correct!")
    else:
        print("  ✗ Gradient mismatch — check implementation.")


# ─────────────────────────── Main ──────────────────────────────────────────

if __name__ == "__main__":
    demo_xor()
    demo_circle()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        """
Key insights from Rumelhart, Hinton & Williams (1986):

1. The chain rule lets you compute ∂E/∂w for EVERY weight in the
   network — including hidden layers — in a single backward pass.

2. Hidden units learn useful internal representations WITHOUT being
   told what to represent. The error signal at the output is enough.

3. XOR is trivially solved by a 2-input, 2-hidden, 1-output network
   — something a perceptron provably cannot do.

4. Gradient checking confirms that our analytical derivatives match
   numerical finite-difference approximations (relative diff < 1e-5).

This is the foundation of all modern deep learning.
"""
    )
