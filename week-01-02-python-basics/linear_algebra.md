# Linear Algebra Notes

## Vectors
A vector is an ordered list of numbers. Example:

```
v = [1, 2, 3]
```

## Matrix Operations
A matrix is a 2D array of numbers. Example:

```
A = [[1, 2], [3, 4]]
```

### Addition
Add corresponding elements:
```
A + B = [[a11+b11, a12+b12], [a21+b21, a22+b22]]
```

### Multiplication
Matrix multiplication is not element-wise. For A (m x n) and B (n x p):
```
C = AB, where cij = sum(Aik * Bkj)
```

### Dot Product
For vectors a and b:
```
a · b = a1*b1 + a2*b2 + ... + an*bn
```

### Eigenvalues/Eigenvectors (Conceptual)
For a square matrix M, if Mx = λx, then λ is an eigenvalue and x is an eigenvector.

---

## Example (Python)
```python
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 2]])
print('A+B =\n', A+B)
print('A x B =\n', np.matmul(A, B))
```
