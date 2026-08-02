"""
NumPy Notes
-----------

numpy is basically a library written to handle all ofnumerical computings in python
not everything needs to be written from scratch, you can rely on external code also

NumPy is a library for numerical computing in Python.

Key Concepts:
- Arrays: Homogeneous, fast, multi-dimensional containers for numbers.
- np.array(): Create an array from a Python list.
- np.zeros(), np.ones(): Create arrays filled with zeros or ones.
- np.arange(start, stop, step): Create arrays with regularly spaced values.
- Array operations: Addition, multiplication, slicing, broadcasting.
- np.mean(), np.std(), np.dot(): Compute mean, standard deviation, dot product.

Function/Method Explanations:
- np.array([..]): Converts a list to a NumPy array.
- np.ones(n): Creates a 1D array of length n filled with 1s.
- np.zeros(n): Creates a 1D array of length n filled with 0s.
- np.arange(start, stop, step): Creates an array from start to stop (exclusive) with given step.
- a + b: Element-wise addition of arrays a and b.
- a * 2: Multiplies every element of a by 2 (broadcasting).
- np.mean(a): Computes the mean (average) of array a.
- np.dot(a, b): Computes the dot product of two arrays.
"""
import numpy as np

def numpy_examples():
    """ """
    a = np.array([1, 2, 3])  # Create a NumPy array from a list
    b = np.ones(3)           # Array of ones, length 3
    c = np.zeros(3)          # Array of zeros, length 3
    d = np.arange(0, 10, 2)  # Array: [0, 2, 4, 6, 8]
    print("Array a:", a)
    print("Ones:", b)
    print("Zeros:", c)
    print("Arange:", d)
    print("a + b:", a + b)         # Element-wise addition
    print("a * 2:", a * 2)         # Element-wise multiplication (broadcasting)
    print("Mean of a:", np.mean(a)) # Mean of array
    print("Dot product a·b:", np.dot(a, b)) # Dot product

if __name__ == "__main__":
    numpy_examples()
