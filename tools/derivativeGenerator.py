# This is a help funciton to pre-calculate the derivative of a function.

import sympy as sym

n = sym.Symbol('n', real=True)
dfs = sym.diff(1 / (1 + sym.exp(-n)), n)
print("The derivative of transfer function is ")
print(dfs)
