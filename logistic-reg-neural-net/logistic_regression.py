"""
Logistic Regression with a Neural Network Mindset
===================================================
Based on DLS C1 W2 assignment.

Implements binary classification using logistic regression as a single-neuron
neural network. Uses synthetic data (colored blobs) instead of the original
cat dataset (which requires h5py and the proprietary .h5 file).

Architecture:
    Input (n features) → Linear (z = wᵀx + b) → Sigmoid (a = σ(z)) → ŷ ∈ {0, 1}

Math:
    Forward:   A = σ(wᵀX + b)
    Cost:      J = -(1/m) Σ [y·log(a) + (1-y)·log(1-a)]
    Backward:  dw = (1/m) X(A-Y)ᵀ,  db = (1/m) Σ(a-y)
    Update:    w -= α·dw,  b -= α·db

See also: ../perceptron/perceptron_and_or.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Helper Functions
# ---------------------------------------------------------------------------

def sigmoid(z):
    """
    Compute the sigmoid activation function.

    σ(z) = 1 / (1 + e^{-z})

    Parameters
    ----------
    z : float or np.ndarray
        Linear output — scalar or array of any shape.

    Returns
    -------
    s : same shape as z
        Sigmoid of z, values in (0, 1).
    """
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    """
    Create a zero-vector for weights and initialize bias to 0.

    Parameters
    ----------
    dim : int
        Number of input features (e.g. num_px * num_px * 3 for images).

    Returns
    -------
    w : np.ndarray, shape (dim, 1)
        Weight vector initialized to zeros.
    b : float
        Bias scalar initialized to 0.
    """
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b


# ---------------------------------------------------------------------------
# 2. Forward & Backward Propagation
# ---------------------------------------------------------------------------

def propagate(w, b, X, Y):
    """
    Compute cost and gradients for one forward + backward pass.

    Forward propagation:
        A = σ(wᵀX + b)
        J = -(1/m) Σ [y·log(a) + (1-y)·log(1-a)]

    Backward propagation:
        dw = (1/m) X(A - Y)ᵀ
        db = (1/m) Σ(a - y)

    Parameters
    ----------
    w : np.ndarray, shape (n, 1)
        Weight vector.
    b : float
        Bias scalar.
    X : np.ndarray, shape (n, m)
        Data matrix — each column is one example.
    Y : np.ndarray, shape (1, m)
        True labels (0 or 1).

    Returns
    -------
    grads : dict
        {"dw": dw, "db": db} — gradients for w and b.
    cost : float
        Negative log-likelihood cost.
    """
    m = X.shape[1]

    # --- Forward ---
    A = sigmoid(np.dot(w.T, X) + b)                          # (1, m)
    cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # --- Backward ---
    dw = (1 / m) * np.dot(X, (A - Y).T)                      # (n, 1)
    db = (1 / m) * np.sum(A - Y)                              # scalar

    # Sanity checks
    assert dw.shape == w.shape
    cost = np.squeeze(cost)

    grads = {"dw": dw, "db": db}
    return grads, cost


# ---------------------------------------------------------------------------
# 3. Optimization (Gradient Descent)
# ---------------------------------------------------------------------------

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    Learn w and b by running gradient descent for num_iterations steps.

    Update rule per iteration:
        w := w - α · dw
        b := b - α · db

    Parameters
    ----------
    w : np.ndarray, shape (n, 1)
        Initial weight vector.
    b : float
        Initial bias.
    X : np.ndarray, shape (n, m)
        Training data.
    Y : np.ndarray, shape (1, m)
        Training labels.
    num_iterations : int
        Number of gradient descent iterations.
    learning_rate : float
        Step size α.
    print_cost : bool
        If True, print cost every 100 iterations.

    Returns
    -------
    params : dict
        {"w": w, "b": b} — learned parameters.
    grads : dict
        {"dw": dw, "db": db} — final gradients.
    costs : list[float]
        Cost recorded every 100 iterations (for plotting).
    """
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        # Gradient descent update
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"  Cost after iteration {i:4d}: {cost:.6f}")

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs


# ---------------------------------------------------------------------------
# 4. Prediction
# ---------------------------------------------------------------------------

def predict(w, b, X):
    """
    Predict binary labels using learned logistic regression parameters.

    Steps:
        1. Compute A = σ(wᵀX + b)
        2. Threshold: ŷ = 1 if a ≥ 0.5, else 0

    Parameters
    ----------
    w : np.ndarray, shape (n, 1)
        Learned weights.
    b : float
        Learned bias.
    X : np.ndarray, shape (n, m)
        Data to predict on.

    Returns
    -------
    Y_prediction : np.ndarray, shape (1, m)
        Predicted labels (0 or 1).
    """
    m = X.shape[1]
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)       # probabilities, shape (1, m)
    Y_prediction = (A >= 0.5) * 1.0        # vectorized thresholding

    assert Y_prediction.shape == (1, m)
    return Y_prediction


# ---------------------------------------------------------------------------
# 5. Full Model
# ---------------------------------------------------------------------------

def model(X_train, Y_train, X_test, Y_test,
          num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Build the complete logistic regression model.

    Pipeline:
        initialize → optimize (gradient descent) → predict

    Parameters
    ----------
    X_train : np.ndarray, shape (n, m_train)
    Y_train : np.ndarray, shape (1, m_train)
    X_test  : np.ndarray, shape (n, m_test)
    Y_test  : np.ndarray, shape (1, m_test)
    num_iterations : int
    learning_rate : float
    print_cost : bool

    Returns
    -------
    d : dict
        Contains costs, predictions, learned params, and hyperparams.
    """
    # 1. Initialize
    w, b = initialize_with_zeros(X_train.shape[0])

    # 2. Gradient descent
    params, grads, costs = optimize(
        w, b, X_train, Y_train, num_iterations, learning_rate, print_cost
    )
    w = params["w"]
    b = params["b"]

    # 3. Predict
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # 4. Print accuracy
    train_acc = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    test_acc = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
    print(f"  Train accuracy: {train_acc:.2f}%")
    print(f"  Test accuracy:  {test_acc:.2f}%")

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
    }
    return d


# ---------------------------------------------------------------------------
# 6. Synthetic Dataset (replaces the original cat/non-cat h5 data)
# ---------------------------------------------------------------------------

def make_blobs(n_samples=400, n_features=4, seed=42):
    """
    Generate two Gaussian blobs for binary classification.

    Class 0: centered at -1 (in each feature)
    Class 1: centered at +1

    Parameters
    ----------
    n_samples : int
        Total samples (split 50/50 between classes).
    n_features : int
        Dimensionality of each sample.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (n_features, n_samples)
        Feature matrix (each column = one example).
    Y : np.ndarray, shape (1, n_samples)
        Labels (0 or 1).
    """
    rng = np.random.RandomState(seed)
    half = n_samples // 2

    X0 = rng.randn(half, n_features) - 1      # class 0 cluster
    X1 = rng.randn(half, n_features) + 1      # class 1 cluster

    X = np.vstack([X0, X1]).T                  # shape (n_features, n_samples)
    Y = np.hstack([np.zeros(half), np.ones(half)]).reshape(1, -1)

    # Shuffle
    perm = rng.permutation(n_samples)
    X = X[:, perm]
    Y = Y[:, perm]

    return X, Y


# ---------------------------------------------------------------------------
# 7. Unit Tests for Each Building Block
# ---------------------------------------------------------------------------

def run_tests():
    """Quick sanity checks matching the expected outputs from the notebook."""
    print("=" * 60)
    print("UNIT TESTS")
    print("=" * 60)

    # Test sigmoid
    result = sigmoid(np.array([0, 2]))
    print(f"\nsigmoid([0, 2]) = {result}")
    assert np.allclose(result, [0.5, 0.88079708]), "sigmoid test failed"
    print("  ✓ sigmoid OK")

    # Test initialize_with_zeros
    w, b = initialize_with_zeros(2)
    assert w.shape == (2, 1) and np.all(w == 0) and b == 0
    print("  ✓ initialize_with_zeros OK")

    # Test propagate
    w = np.array([[1.0], [2.0]])
    b = 2.0
    X = np.array([[1.0, 2.0, -1.0], [3.0, 4.0, -3.2]])
    Y = np.array([[1, 0, 1]])
    grads, cost = propagate(w, b, X, Y)
    print(f"\n  dw = {grads['dw'].flatten()}")
    print(f"  db = {grads['db']:.15f}")
    print(f"  cost = {cost:.15f}")
    assert np.allclose(grads["dw"], [[0.99845601], [2.39507239]])
    assert np.isclose(grads["db"], 0.00145557813678, atol=1e-10)
    assert np.isclose(cost, 5.801545319394553)
    print("  ✓ propagate OK")

    # Test optimize
    params, grads, costs = optimize(
        w, b, X, Y, num_iterations=100, learning_rate=0.009
    )
    print(f"\n  w = {params['w'].flatten()}")
    print(f"  b = {params['b']:.10f}")
    assert np.allclose(params["w"], [[0.19033591], [0.12259159]], atol=1e-5)
    print("  ✓ optimize OK")

    # Test predict
    w = np.array([[0.1124579], [0.23106775]])
    b = -0.3
    X = np.array([[1.0, -1.1, -3.2], [1.2, 2.0, 0.1]])
    preds = predict(w, b, X)
    print(f"\n  predictions = {preds}")
    assert np.array_equal(preds, [[1.0, 1.0, 0.0]])
    print("  ✓ predict OK")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 8. Main — Train on Synthetic Data + Compare Learning Rates
# ---------------------------------------------------------------------------

def main():
    run_tests()

    # --- Generate synthetic data ---
    X, Y = make_blobs(n_samples=400, n_features=4, seed=42)

    # 80/20 train-test split
    split = int(0.8 * X.shape[1])
    X_train, X_test = X[:, :split], X[:, split:]
    Y_train, Y_test = Y[:, :split], Y[:, split:]

    print(f"\n{'=' * 60}")
    print("DATASET")
    print(f"{'=' * 60}")
    print(f"  X_train shape: {X_train.shape}   (n_features, m_train)")
    print(f"  Y_train shape: {Y_train.shape}")
    print(f"  X_test shape:  {X_test.shape}    (n_features, m_test)")
    print(f"  Y_test shape:  {Y_test.shape}")

    # --- Train the model ---
    print(f"\n{'=' * 60}")
    print("TRAINING (lr=0.5, 2000 iterations)")
    print(f"{'=' * 60}")
    d = model(
        X_train, Y_train, X_test, Y_test,
        num_iterations=2000, learning_rate=0.5, print_cost=True,
    )

    # --- Compare different learning rates (like Section 6 in the notebook) ---
    print(f"\n{'=' * 60}")
    print("LEARNING RATE COMPARISON")
    print(f"{'=' * 60}")
    learning_rates = [1.0, 0.1, 0.01, 0.001]
    for lr in learning_rates:
        print(f"\n  Learning rate: {lr}")
        model(
            X_train, Y_train, X_test, Y_test,
            num_iterations=1500, learning_rate=lr, print_cost=False,
        )

    # --- Print cost trajectory ---
    print(f"\n{'=' * 60}")
    print("COST TRAJECTORY (first 10 checkpoints)")
    print(f"{'=' * 60}")
    for i, c in enumerate(d["costs"][:10]):
        print(f"  iteration {i * 100:4d} → cost = {c:.6f}")

    print(f"\nDone. Learned weight vector shape: {d['w'].shape}")


if __name__ == "__main__":
    main()
