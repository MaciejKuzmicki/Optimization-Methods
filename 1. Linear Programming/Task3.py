from cvxopt.modeling import variable, op
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog

# Zadanie 3: Leki
x_LekI = variable()
x_LekII = variable()
x_SurI = variable()
x_SurII = variable()

c1 = (0.01 * x_SurI + 0.02 * x_SurII - 0.50 * x_LekI - 0.60 * x_LekII == 0)
c2 = (x_SurI + x_SurII <= 1000)
c3 = (90.00 * x_LekI + 100.00 * x_LekII <= 2000)
c4 = (40.00 * x_LekI + 50.00 * x_LekII <= 800)
c5 = (100.00 * x_SurI + 199.90 * x_SurII + 700.00 * x_LekI + 800.00 * x_LekII <= 100000)
c6 = (x_LekI >= 0)
c7 = (x_LekII >= 0)
c8 = (x_SurI >= 0)
c9 = (x_SurII >= 0)

p1 = op(100.00 * x_SurI + 199.90 * x_SurII + 700.00 * x_LekI + 800.00 * x_LekII - (6500.00 * x_LekI + 7100.00 * x_LekII), [c1, c2, c3, c4, c5, c6, c7, c8, c9])
p1.solve()

print("Zadanie 3 - CVXOPT - optymalne wartości:", round(x_LekI.value[0], 3), round(x_LekII.value[0], 3), round(x_SurI.value[0], 3), round(x_SurII.value[0], 3))

x = cp.Variable(4)
cons = [0.01*x[2] + 0.02*x[3] - 0.50*x[0] - 0.60*x[1] == 0,
        x[2] + x[3] <= 1000,
        90*x[0] + 100*x[1] <= 2000,
        40*x[0] + 50*x[1] <= 800,
        100*x[2] + 199.90*x[3] + 700*x[0] + 800*x[1] <= 100000,
        x[0] >= 0,
        x[1] >= 0,
        x[2] >= 0,
        x[3] >= 0]
obj = cp.Minimize(100*x[2] + 199.90*x[3] + 700*x[0] + 800*x[1] - 6500*x[0] - 7100*x[1])
prob = cp.Problem(obj, cons)
prob.solve()
print("Zadanie 3 - CVXPY - optymalne wartości:", np.round(x.value, 3))

c = [-5800, -6300, 100, 199.90]
A_eq = [[-0.50, -0.60, 0.01, 0.02]]
b_eq = [0]
A_ub = [[0, 0, 1, 1],
        [90, 100, 0, 0],
        [40, 50, 0, 0],
        [700, 800, 100, 199.90]]
b_ub = [1000, 2000, 800, 100000]
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)]*4)
print("Zadanie 3 - linprog - optymalne wartości:", np.round(res.x, 3))