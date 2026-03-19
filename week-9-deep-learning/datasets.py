"""
datasets.py - Data loading and preprocessing
"""
import numpy as np

def load_dataset():
    # Placeholder: implement your dataset loading logic here
    # Return train_X, train_Y, test_X, test_Y
    # Example dummy data:
    train_X = np.random.randn(2, 300)
    train_Y = (np.random.rand(1, 300) > 0.5).astype(int)
    test_X = np.random.randn(2, 100)
    test_Y = (np.random.rand(1, 100) > 0.5).astype(int)
    return train_X, train_Y, test_X, test_Y
