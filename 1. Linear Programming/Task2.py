import numpy as np
import cvxpy as co
from cvxopt.modeling import variable, op
# Zadanie 2: Optymalne śniadanie
x1 = variable()
x2 = variable()
x3 = variable()

c1 = (70*x1 + 121*x2 + 65*x3 >= 2000)
c12 = (70*x1 + 121*x2 + 65*x3 <= 2250)

c2 = (107*x1 + 500*x2 >= 5000)
c22 = (107*x1 + 500*x2 <= 10000)

c3 = (45*x1 + 40*x2 + 60*x3 >= 0)
c32 = (45*x1 + 40*x2 + 60*x3 <= 1000)

c4 = (0 <= x1 <= 10)
c5 = (0 <= x2 <= 10)
c6 = (0 <= x3 <= 10)

p1 = op(0.15*x1 + 0.25*x2 + 0.05*x3, [c1,c12,c2, c22, c3,c32,c4,c5,c6])
p1.solve()
print("Zadanie 2 - optymalne wartości:", round(x1.value[0], 2), round(x2.value[0], 2), round(x3.value[0], 2))