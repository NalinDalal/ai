#!/usr/bin/env python3
"""
CLI Matrix Calculator
Supports addition, subtraction, and multiplication of matrices
"""

def print_matrix(matrix, label="Matrix"):
    """Pretty print a matrix

    :param matrix: param label:  (Default value = "Matrix")
    :param label: Default value = "Matrix")

    """
    print(f"\n{label}:")
    for row in matrix:
        print("  ", " ".join(f"{val:8.2f}" for val in row))
    print()

def get_matrix_input(matrix_name):
    """Get matrix dimensions and values from user

    :param matrix_name: 

    """
    print(f"\nEnter {matrix_name}:")
    
    while True:
        try:
            rows = int(input(f"  Number of rows: "))
            cols = int(input(f"  Number of columns: "))
            if rows <= 0 or cols <= 0:
                print("  Dimensions must be positive integers.")
                continue
            break
        except ValueError:
            print("  Please enter valid integers.")
    
    matrix = []
    print(f"  Enter {rows} rows of {cols} values each (space-separated):")
    
    for i in range(rows):
        while True:
            try:
                row_input = input(f"    Row {i+1}: ")
                row = [float(x) for x in row_input.split()]
                if len(row) != cols:
                    print(f"    Error: Expected {cols} values, got {len(row)}")
                    continue
                matrix.append(row)
                break
            except ValueError:
                print("    Error: Please enter valid numbers")
    
    return matrix

def add_matrices(m1, m2):
    """Add two matrices

    :param m1: param m2:
    :param m2: 

    """
    rows = len(m1)
    cols = len(m1[0])
    
    if rows != len(m2) or cols != len(m2[0]):
        raise ValueError("Matrices must have the same dimensions for addition")
    
    result = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(m1[i][j] + m2[i][j])
        result.append(row)
    
    return result

def subtract_matrices(m1, m2):
    """Subtract matrix m2 from m1

    :param m1: param m2:
    :param m2: 

    """
    rows = len(m1)
    cols = len(m1[0])
    
    if rows != len(m2) or cols != len(m2[0]):
        raise ValueError("Matrices must have the same dimensions for subtraction")
    
    result = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(m1[i][j] - m2[i][j])
        result.append(row)
    
    return result

def multiply_matrices(m1, m2):
    """Multiply two matrices

    :param m1: param m2:
    :param m2: 

    """
    rows1 = len(m1)
    cols1 = len(m1[0])
    rows2 = len(m2)
    cols2 = len(m2[0])
    
    if cols1 != rows2:
        raise ValueError(f"Cannot multiply: columns of first matrix ({cols1}) must equal rows of second matrix ({rows2})")
    
    result = []
    for i in range(rows1):
        row = []
        for j in range(cols2):
            val = sum(m1[i][k] * m2[k][j] for k in range(cols1))
            row.append(val)
        result.append(row)
    
    return result

def display_menu():
    """Display operation menu"""
    print("\n" + "="*50)
    print("         MATRIX CALCULATOR")
    print("="*50)
    print("\n  1. Add matrices")
    print("  2. Subtract matrices")
    print("  3. Multiply matrices")
    print("  4. Exit")
    print()

def main():
    """Main program loop"""
    while True:
        display_menu()
        
        choice = input("Select operation (1-4): ").strip()
        
        if choice == '4':
            print("\nThank you for using Matrix Calculator!")
            break
        
        if choice not in ['1', '2', '3']:
            print("\nInvalid choice. Please select 1-4.")
            continue
        
        try:
            matrix1 = get_matrix_input("Matrix 1")
            print_matrix(matrix1, "Matrix 1")
            
            matrix2 = get_matrix_input("Matrix 2")
            print_matrix(matrix2, "Matrix 2")
            
            if choice == '1':
                result = add_matrices(matrix1, matrix2)
                print_matrix(result, "Result (Matrix 1 + Matrix 2)")
            
            elif choice == '2':
                result = subtract_matrices(matrix1, matrix2)
                print_matrix(result, "Result (Matrix 1 - Matrix 2)")
            
            elif choice == '3':
                result = multiply_matrices(matrix1, matrix2)
                print_matrix(result, "Result (Matrix 1 × Matrix 2)")
        
        except ValueError as e:
            print(f"\nError: {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
