import numpy as np
import cvxpy as co
from cvxopt.modeling import variable, op
from scipy.optimize import linprog
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
print("Zadanie 1 - CVXOPT - optymalne wartości:", round(x1.value[0], 2), round(x2.value[0], 2), round(x3.value[0], 2))

x = co.Variable(3, nonneg=True)
cons = [0.8*x[0] + 0.3*x[1] + 0.1*x[2] >= 0.3,
        0.01*x[0] + 0.4*x[1] + 0.7*x[2] >= 0.7,
        0.15*x[0] + 0.1*x[1] + 0.2*x[2] >= 0.1]
obj = co.Minimize(300*x[0] + 500*x[1] + 800*x[2])
prob = co.Problem(obj, cons)
prob.solve()
print("Zadanie 1 - CVXPY - optymalne wartości:", np.round(x.value, 2))

c = [300, 500, 800]
A = [[-0.8, -0.3, -0.1],
     [-0.01, -0.4, -0.7],
     [-0.15, -0.1, -0.2]]
b = [-0.3, -0.7, -0.1]
res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)]*3)
print("Zadanie 1 - Linprog - optymalne wartości:", np.round(res.x, 2))