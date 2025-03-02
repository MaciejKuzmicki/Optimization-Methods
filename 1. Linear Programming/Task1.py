import numpy as np
import cvxpy as co
from cvxopt.modeling import variable, op
# Zadanie 1: Mieszanka paszowa
x1 = variable()
x2 = variable()
x3 = variable()

c1 = (0.8*x1 + 0.3*x2 + 0.1*x3 >= 0.3)
c2 = (0.01*x1 + 0.4*x2 + 0.7*x3 >= 0.7)
c3 = (0.15*x1 + 0.1*x2 + 0.2*x3 >= 0.1)
c4 = (x1 >= 0)
c5 = (x2 >= 0)
c6 = (x3 >= 0)

p1 = op(300*x1 + 500*x2 + 800*x3, [c1,c2,c3,c4,c5,c6])
p1.solve()
print("Zadanie 1 - optymalne wartości:", round(x1.value[0], 2), round(x2.value[0], 2), round(x3.value[0], 2))