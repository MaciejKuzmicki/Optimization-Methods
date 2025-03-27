import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

sensors = np.array([
    [1.8, 2.5],
    [2.0, 1.7],
    [1.5, 1.5],
    [1.5, 2.0],
    [2.5, 1.5]
])
d = np.array([2.00, 1.24, 0.59, 1.31, 1.44])

def f0(x):
    s = 0.0
    for i in range(len(d)):
        diff = np.linalg.norm(x - sensors[i])**2 - d[i]**2
        s += diff**2
    return s

A = np.hstack([-2 * sensors, np.ones((len(d), 1))])
b = np.array([d[i]**2 - np.linalg.norm(sensors[i])**2 for i in range(len(d))])
Q = np.diag([1, 1, 0])
c = np.array([0, 0, -0.5])
mu = cp.Variable()
t = cp.Variable()
M_top_left = A.T @ A + mu * Q
M_top_right = (A.T @ b - mu * c)[:, None]
M = cp.bmat([[M_top_left, M_top_right],
             [M_top_right.T, cp.reshape(t, (1,1))]])
prob = cp.Problem(cp.Minimize(t), [M >> 0])
prob.solve(solver=cp.SCS)
M_matrix = A.T @ A + mu.value * Q
rhs = A.T @ b - mu.value * c
z = np.linalg.solve(M_matrix, rhs)
x_opt = z[:2]
print("Optymalne położenie źródła:", x_opt)
print("Wartość funkcji celu:", f0(x_opt))

x1_vals = np.linspace(0, 3, 400)
x2_vals = np.linspace(0, 3, 400)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = np.zeros_like(X1)

for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        point = np.array([X1[i, j], X2[i, j]])
        Z[i, j] = f0(point)

plt.figure(figsize=(8, 6))
contour = plt.contour(X1, X2, Z, levels=50, cmap='viridis')
plt.colorbar(contour, label='Wartość f₀(x)')

plt.scatter(sensors[:, 0], sensors[:, 1], color='red', marker='o', s=80, label='Sensory')
plt.scatter(x_opt[0], x_opt[1], color='blue', marker='x', s=100, label='Oszacowane źródło')

plt.title('Poziomice funkcji celu f₀(x) i lokalizacja źródła')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.legend()
plt.show()
