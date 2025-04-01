import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def f(x):
    return np.array([
        x[0] + np.exp(-x[1]),
        x[0]**2 + 2*x[1] + 1
    ])

def df(x):
    return np.array([
        [1,             -np.exp(-x[1])],
        [2*x[0],        2]
    ])

def g(x):
    return x[0] + x[0]**3 + x[1] + x[1]**2

def dg(x):
    return np.array([
        1 + 3*x[0]**2,
        1 + 2*x[1]
    ])

def L_A(x, z, mu):
    return np.concatenate([
        f(x),
        [np.sqrt(mu) * g(x) + z / (2 * np.sqrt(mu))]
    ])

def obj_fun(x, z, mu):
    return L_A(x, z, mu)

k_max = 10
x = np.array([0.5, -0.5]) 
z = 0                      
mu = 1 

res_x = np.zeros((k_max, 2))
optimality_residuals = []
feasibility_residuals = []
mus = []

for k in range(k_max):
    res_x[k, :] = x.copy()

    result = least_squares(obj_fun, x, args=(z, mu), method='lm', verbose=0)
    old_x = x.copy()
    x = result.x

    z = z + 2 * mu * g(x)

    f_val = f(x)
    df_val = df(x)
    dg_val = dg(x).reshape(-1, 1)

    feasibility_res = np.linalg.norm(g(x))
    optimality_res = np.linalg.norm(2 * df_val.T @ f_val + dg_val * z)

    feasibility_residuals.append(feasibility_res)
    optimality_residuals.append(optimality_res)
    mus.append(mu)

    if feasibility_res >= 0.25 * np.linalg.norm(g(old_x)):
        mu *= 2

X, Y = np.meshgrid(np.arange(-3, 3.01, 0.01),
                   np.arange(-3, 3.01, 0.01))
Z = np.zeros_like(X)
g_Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        Z[i, j] = np.linalg.norm(f(point))**2
        g_Z[i, j] = g(point)

plt.figure(figsize=(6,5))
plt.grid(True)
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.title('Augmented Lagrangian Iterations')
plt.xlabel('x1')
plt.ylabel('x2')

for i in range(k_max):
    plt.plot(res_x[i, 0], res_x[i, 1], 'ro')

plt.contour(X, Y, g_Z, levels=[0], colors='r')

for level in range(2, 18, 2):
    plt.contour(X, Y, Z, levels=[level], colors='b')

plt.show()

log10_FR = np.log(np.array(feasibility_residuals))
log10_OR = np.log(np.array(optimality_residuals))
k_vals = np.arange(1, k_max + 1)

print("\nTabela rezyduów:")
print("k\tlog10(FR)\tlog10(OR)")
for k, fr, or_val in zip(k_vals, log10_FR, log10_OR):
    print(f"{k}\t{fr:10.4f}\t{or_val:10.4f}")

print("\nWspółrzędne iteracji:")
for i, coords in enumerate(res_x):
    print(f"Iteracja {i}: x = [{coords[0]:.4f}, {coords[1]:.4f}]")
