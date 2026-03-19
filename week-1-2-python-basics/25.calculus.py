# Calculus: Derivatives and Gradients using sympy
import sympy as sp

# Derivative
x = sp.symbols('x')
f = x**2 + 3*x + 2
f_prime = sp.diff(f, x)
print('Function:', f)
print('Derivative:', f_prime)

# Gradient (for multivariable functions)
y = sp.symbols('y')
g = x**2 + y**2
gradient = [sp.diff(g, var) for var in (x, y)]
print('Function:', g)
print('Gradient:', gradient)
