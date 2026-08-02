v2 = np.array([4, 5, 6])

# Linear Algebra: Vectors, Dot Product, Matrix Multiplication, Eigenvalues (conceptual)
#
# - np.array([...]): Create a NumPy array (vector or matrix)
# - np.dot(a, b): Dot product of two vectors
# - np.matmul(A, B): Matrix multiplication
# - np.linalg.eig(M): Eigenvalues and eigenvectors of a matrix

import numpy as np

# Vectors
v1 = np.array([1, 2, 3])  # Create a vector
v2 = np.array([4, 5, 6])  # Create another vector
print('Vector v1:', v1)
print('Vector v2:', v2)

# Dot Product
# The dot product is the sum of products of corresponding elements
print('Dot product:', np.dot(v1, v2))

# Matrix Multiplication
A = np.array([[1, 2], [3, 4]])  # 2x2 matrix
B = np.array([[2, 0], [1, 2]])  # 2x2 matrix
print('Matrix A:\n', A)
print('Matrix B:\n', B)
print('A x B =\n', np.matmul(A, B))  # Matrix multiplication

# Eigenvalues/Eigenvectors (conceptual)
# For a square matrix M, if Mx = λx, then λ is an eigenvalue and x is an eigenvector.
M = np.array([[2, 0], [0, 3]])  # Diagonal matrix
eigenvalues, eigenvectors = np.linalg.eig(M)  # Compute eigenvalues and eigenvectors
print('Eigenvalues:', eigenvalues)
print('Eigenvectors:\n', eigenvectors)

"""
Linear Algebra Notes
-------------------
- Vectors: Ordered lists of numbers, e.g., [1, 2, 3].
- Matrix: 2D array of numbers, e.g., [[1, 2], [3, 4]].
- Matrix Addition: Add corresponding elements.
- Matrix Multiplication: Dot product of rows and columns.
- Dot Product: Sum of products of corresponding vector elements.
- Eigenvalues/Eigenvectors: For matrix M, if Mx = λx, λ is eigenvalue, x is eigenvector.
"""
import numpy as np

def matrix_addition(A, B):
    """Add two matrices.

    :param A: param B:
    :param B: 

    """
    return np.add(A, B)

def matrix_multiplication(A, B):
    """Multiply two matrices.

    :param A: param B:
    :param B: 

    """
    return np.matmul(A, B)

def dot_product(v1, v2):
    """Dot product of two vectors.

    :param v1: param v2:
    :param v2: 

    """
    return np.dot(v1, v2)

def eigen(M):
    """

    :param M: 

    """
    return np.linalg.eig(M)

# Example usage
if __name__ == "__main__":
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[2, 0], [1, 2]])
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    print("A+B=\n", matrix_addition(A, B))
    print("A x B=\n", matrix_multiplication(A, B))
    print("Dot product:", dot_product(v1, v2))
    vals, vecs = eigen(A)
    print("Eigenvalues:", vals)
    print("Eigenvectors:\n", vecs)
