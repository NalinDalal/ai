# Calculus Notes

## Differentiation (Derivatives)
The derivative measures how a function changes as its input changes. Notation: f'(x) or df/dx.

Example: If f(x) = x², then f'(x) = 2x.

## Integration
Integration is the reverse of differentiation. It finds the area under a curve. Notation: ∫f(x)dx.

Example: ∫x dx = (1/2)x² + C

## Gradients
For multivariable functions, the gradient is a vector of partial derivatives.

---

## Example (Python, using sympy)
```python
import sympy as sp
x = sp.symbols('x')
f = x**2 + 3*x + 2
f_prime = sp.diff(f, x)
print('Derivative:', f_prime)
# Integration
f_int = sp.integrate(f, x)
print('Integral:', f_int)
```
