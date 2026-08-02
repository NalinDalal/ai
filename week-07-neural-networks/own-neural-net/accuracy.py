"""
accuracy.py
Defines accuracy calculation classes for classification and regression.
"""
import numpy as np

class Accuracy:
    """Base accuracy class."""
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        return accuracy

class Accuracy_Categorical(Accuracy):
    """Accuracy for classification models."""
    def init(self, y):
        pass
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

class Accuracy_Regression(Accuracy):
    """Accuracy for regression models."""
    def __init__(self):
        self.precision = None
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision
