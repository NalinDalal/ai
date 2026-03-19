# Matrix Operations from Scratch

# Function to create a matrix
def create_matrix(rows, cols, fill=0):
    """

    :param rows: param cols:
    :param fill: Default value = 0)
    :param cols: 

    """
    return [[fill for _ in range(cols)] for _ in range(rows)]

# Function to print a matrix nicely
def print_matrix(matrix):
    """

    :param matrix: 

    """
    for row in matrix:
        print(row)
    print()

# Matrix addition
def add_matrices(A, B):
    """

    :param A: param B:
    :param B: 

    """
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrices must have the same dimensions for addition.")
    
    result = create_matrix(len(A), len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] + B[i][j]
    return result

# Matrix subtraction
def subtract_matrices(A, B):
    """

    :param A: param B:
    :param B: 

    """
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrices must have the same dimensions for subtraction.")
    
    result = create_matrix(len(A), len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] - B[i][j]
    return result

# Scalar multiplication
def scalar_multiply(matrix, scalar):
    """

    :param matrix: param scalar:
    :param scalar: 

    """
    result = create_matrix(len(matrix), len(matrix[0]))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            result[i][j] = matrix[i][j] * scalar
    return result

# Matrix multiplication
def multiply_matrices(A, B):
    """

    :param A: param B:
    :param B: 

    """
    if len(A[0]) != len(B):
        raise ValueError("Number of columns in A must equal number of rows in B for multiplication.")
    
    result = create_matrix(len(A), len(B[0]))
    for i in range(len(A)):
        for j in range(len(B[0])):
            sum = 0
            for k in range(len(A[0])):
                sum += A[i][k] * B[k][j]
            result[i][j] = sum
    return result

# Transpose of a matrix
def transpose(matrix):
    """

    :param matrix: 

    """
    rows, cols = len(matrix), len(matrix[0])
    result = create_matrix(cols, rows)
    for i in range(rows):
        for j in range(cols):
            result[j][i] = matrix[i][j]
    return result


# Example Usage
A = [[1, 2, 3],
     [4, 5, 6]]

B = [[7, 8, 9],
     [10, 11, 12]]

print("Matrix A:")
print_matrix(A)

print("Matrix B:")
print_matrix(B)

print("A + B:")
print_matrix(add_matrices(A, B))

print("A - B:")
print_matrix(subtract_matrices(A, B))

print("2 * A:")
print_matrix(scalar_multiply(A, 2))

C = [[1, 2],
     [3, 4],
     [5, 6]]

print("A * C:")
print_matrix(multiply_matrices(A, C))

print("Transpose of A:")
print_matrix(transpose(A))

