"""
data.py
Data loading and generation utilities.
"""
import numpy as np

def spiral_data(samples, classes):
    """Generate a spiral dataset for classification."""
    X = np.zeros((samples, 2))
    y = np.zeros(samples, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples // classes * class_number, samples // classes * (class_number + 1))
        r = np.linspace(0.0, 1, samples // classes)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, samples // classes) + np.random.randn(samples // classes) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = class_number
    return X, y
