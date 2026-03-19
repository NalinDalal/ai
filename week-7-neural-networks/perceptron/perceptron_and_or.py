"""
Perceptron Implementation — Learning AND & OR Gates
Based on Rosenblatt's 1958 perceptron model.

The perceptron learns by adjusting weights through reinforcement:
  - If correct: no change
  - If wrong: adjust weights by ±learning_rate × input
"""

import numpy as np


class Perceptron:
    """A single-layer perceptron with adjustable weights and bias."""

    def __init__(self, n_inputs, learning_rate=0.1, threshold=0.5):
        self.weights = np.random.uniform(-0.5, 0.5, n_inputs)
        self.bias = np.random.uniform(-0.5, 0.5)
        self.lr = learning_rate
        self.threshold = threshold

    def activate(self, x):
        """All-or-nothing activation (as described by Rosenblatt)."""
        return 1 if x >= self.threshold else 0

    def predict(self, inputs):
        """Compute weighted sum and apply activation."""
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.activate(weighted_sum)

    def train(self, training_data, labels, epochs=100):
        """
        Train using the perceptron learning rule.
        Returns list of errors per epoch for visualization.
        """
        errors_per_epoch = []
        for epoch in range(epochs):
            total_error = 0
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                if error != 0:
                    self.weights += self.lr * error * np.array(inputs)
                    self.bias += self.lr * error
                    total_error += abs(error)
            errors_per_epoch.append(total_error)
            if total_error == 0:
                print(f"  Converged at epoch {epoch + 1}")
                break
        return errors_per_epoch


def test_gate(name, inputs, labels):
    """Train and evaluate a perceptron on a logic gate."""
    print(f"\n{'='*40}")
    print(f"  Learning {name} Gate")
    print(f"{'='*40}")

    p = Perceptron(n_inputs=2, learning_rate=0.1, threshold=0.5)
    errors = p.train(inputs, labels, epochs=100)

    print(f"  Final weights: {p.weights}")
    print(f"  Final bias:    {p.bias:.4f}")
    print(f"  Truth table:")
    all_correct = True
    for x, y in inputs:
        pred = p.predict([x, y])
        expected = labels[inputs.tolist().index([x, y])]
        status = "✓" if pred == expected else "✗"
        if pred != expected:
            all_correct = False
        print(f"    {int(x)} {name} {int(y)} = {pred}  {status}")

    print(f"  Result: {'PASSED' if all_correct else 'FAILED'}")
    return errors


def main():
    # AND gate truth table
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    and_labels = np.array([0, 0, 0, 1])
    or_labels = np.array([0, 1, 1, 1])
    nand_labels = np.array([1, 1, 1, 0])

    print("Rosenblatt Perceptron — Logic Gate Learning")
    print("=" * 40)

    and_errors = test_gate("AND", inputs, and_labels)
    or_errors = test_gate("OR", inputs, or_labels)
    nand_errors = test_gate("NAND", inputs, nand_labels)

    # XOR — should FAIL (not linearly separable)
    xor_labels = np.array([0, 1, 1, 0])
    print(f"\n{'='*40}")
    print("  Attempting XOR Gate (expected to fail)")
    print(f"{'='*40}")
    p_xor = Perceptron(n_inputs=2, learning_rate=0.1, threshold=0.5)
    xor_errors = p_xor.train(inputs, xor_labels, epochs=100)
    print(f"  Final errors after 100 epochs: {xor_errors[-1]}")
    print("  XOR is NOT linearly separable — a single perceptron cannot learn it.")
    print("  This limitation was later highlighted by Minsky & Papert (1969).")

    # Plot learning curves if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, errs, name in zip(
            axes,
            [and_errors, or_errors, nand_errors, xor_errors],
            ["AND", "OR", "NAND", "XOR"],
        ):
            ax.plot(errs, "b-", linewidth=1.5)
            ax.set_title(f"{name} Gate")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Total Errors")
            ax.set_ylim(-0.1, max(errs) + 1)
            ax.grid(True, alpha=0.3)

        plt.suptitle("Perceptron Learning Curves (Rosenblatt, 1958)", fontsize=14)
        plt.tight_layout()
        plt.savefig("perceptron_learning_curves.png", dpi=150)
        plt.show()
        print("\n  Plot saved to perceptron_learning_curves.png")
    except ImportError:
        print("\n  matplotlib not available — skipping plot.")


if __name__ == "__main__":
    main()
