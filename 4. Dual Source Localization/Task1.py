import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

sensors = np.array([[1.8, 2.5],
                    [2.0, 1.7],
                    [1.5, 1.5],
                    [1.5, 2.0],
                    [2.5, 1.5]])
d = np.array([2.00, 1.24, 0.59, 1.31, 1.44])
m = sensors.shape[0]
n = sensors.shape[1]

A = np.zeros((m, n + 1))
b = np.zeros(m)
for k in range(m):
    A[k, :n] = -2 * sensors[k]
    A[k, n] = 1
    b[k] = d[k]**2 - np.dot(sensors[k], sensors[k])

Q = np.diag([1, 1, 0])
c = np.array([0, 0, -0.5])

AtA = A.T @ A
Atb = A.T @ b
norm_b_sq = np.sum(b**2)

mu = cp.Variable()
t_var = cp.Variable()

M_top = AtA + mu * Q
M_right = Atb - mu * c
M = cp.bmat([[M_top, M_right.reshape((3, 1))],
             [M_right.reshape((1, 3)), cp.reshape(t_var, (1, 1))]])

constraints = [M >> 0]
obj = cp.Minimize(t_var - norm_b_sq)
prob = cp.Problem(obj, constraints)
prob.solve(solver=cp.SCS)

mu_opt = mu.value
t_opt = t_var.value

M_mat = AtA + mu_opt * Q
rhs = Atb - mu_opt * c
z = np.linalg.solve(M_mat, rhs)
x_est = z[:n]
t_est = z[n]

print("Szacowane położenie źródła x:", x_est)

def f0(x):
    s = 0
    for k in range(m):
        s += (np.linalg.norm(x - sensors[k])**2 - d[k]**2)**2
    return s

x1_vals = np.linspace(0, 3, 300)
x2_vals = np.linspace(0, 3, 300)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
F = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        F[i, j] = f0(np.array([X1[i, j], X2[i, j]]))

plt.figure(figsize=(8, 6))
contour = plt.contour(X1, X2, F, levels=50, cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)
plt.plot(sensors[:, 0], sensors[:, 1], 'ro', markersize=8, label='Sensory')
plt.plot(x_est[0], x_est[1], 'bx', markersize=12, label='Szacowane położenie źródła')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Poziomice funkcji f₀(x) z sensorami i szacowanym źródłem')
plt.legend()
plt.show()
