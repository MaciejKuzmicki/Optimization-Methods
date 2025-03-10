import numpy as np
import cvxpy as co
from cvxopt.modeling import variable, op
from scipy.optimize import linprog
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
print("Zadanie 2 - CVXOPT - optymalne wartości:", round(x1.value[0], 2), round(x2.value[0], 2), round(x3.value[0], 2))

x = co.Variable(3)
cons = [70*x[0] + 121*x[1] + 65*x[2] >= 2000,
        70*x[0] + 121*x[1] + 65*x[2] <= 2250,
        107*x[0] + 500*x[1] >= 5000,
        107*x[0] + 500*x[1] <= 10000,
        45*x[0] + 40*x[1] + 60*x[2] >= 0,
        45*x[0] + 40*x[1] + 60*x[2] <= 1000,
        x[0] >= 0, x[0] <= 10,
        x[1] >= 0, x[1] <= 10,
        x[2] >= 0, x[2] <= 10]
obj = co.Minimize(0.15*x[0] + 0.25*x[1] + 0.05*x[2])
prob = co.Problem(obj, cons)
prob.solve()
print("Zadanie 2 - CVXPY - optymalne wartości:", np.round(x.value, 2))

c = [0.15, 0.25, 0.05]
A = [[-70, -121, -65],
     [70, 121, 65],
     [-107, -500, 0],
     [107, 500, 0],
     [-45, -40, -60],
     [45, 40, 60]]
b = [-2000, 2250, -5000, 10000, 0, 1000]
res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, 10)]*3)
print("Zadanie 2 - Linprog - optymalne wartości:", np.round(res.x, 2))