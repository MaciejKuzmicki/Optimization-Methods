import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat('twoInertialData.mat')
t = data['t'].squeeze()
y = data['y'].squeeze()

def h_double(x, t):
    k, T1, T2 = x
    return k * (1 - (T1 * np.exp(-t / T1) - T2 * np.exp(-t / T2)) / (T1 - T2))

def residuals(x):
    return h_double(x, t) - y

def jacobian(x):
    k, T1, T2 = x
    N = len(t)
    J = np.zeros((N, 3))
    Q = T1 - T2
    exp_T1 = np.exp(-t / T1)
    exp_T2 = np.exp(-t / T2)
    F = (T1 * exp_T1 - T2 * exp_T2) / Q
    J[:, 0] = 1 - F
    u = T1 * exp_T1 - T2 * exp_T2
    u_T1 = exp_T1 * (1 + t / T1)
    dF_dT1 = (u_T1 * Q - u) / (Q**2)
    J[:, 1] = -k * dF_dT1
    u_T2 = -exp_T2 * (1 + t / T2)
    dF_dT2 = (u_T2 * Q + u) / (Q**2)
    J[:, 2] = -k * dF_dT2
    return J

k_max = 35
x0 = np.array([1.0, 1.0, 2.0])
X = np.zeros((3, k_max+1))
X[:, 0] = x0
L = 1.0
lambda_history = [L]
obj_history = [np.linalg.norm(residuals(x0))**2]
x_current = x0.copy()

for i in range(k_max):
    J_current = jacobian(x_current)
    f_current = residuals(x_current)
    A = J_current.T @ J_current + L * np.eye(3)
    b = - J_current.T @ f_current
    delta = np.linalg.solve(A, b)
    x_new = x_current + delta
    if np.linalg.norm(residuals(x_new)) < np.linalg.norm(f_current):
        x_current = x_new
        L = 0.8 * L
    else:
        L = 2.0 * L
    X[:, i+1] = x_current
    lambda_history.append(L)
    obj_history.append(np.linalg.norm(residuals(x_current))**2)

x_opt = x_current
print("Oszacowane parametry:")
print("k  =", x_opt[0])
print("T1 =", x_opt[1])
print("T2 =", x_opt[2])

t_plot = np.linspace(t[0], t[-1], 1000)
plt.figure()
plt.plot(t, y, 'rs', label='Dane')
plt.plot(t_plot, h_double(x0, t_plot), 'b-', linewidth=2, label='Model (start)')
plt.plot(t_plot, h_double(x_opt, t_plot), 'k-', linewidth=2, label='Model (dopasowany)')
plt.xlabel('t')
plt.ylabel('h(t)')
plt.title('Dopasowanie modelu układu podwójnie inercyjnego')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(X[0, :], 'k-', linewidth=2)
plt.ylabel('k')
plt.grid(True)
plt.subplot(3,1,2)
plt.plot(X[1, :], 'k-', linewidth=2)
plt.ylabel('T1')
plt.grid(True)
plt.subplot(3,1,3)
plt.plot(X[2, :], 'k-', linewidth=2)
plt.ylabel('T2')
plt.xlabel('Iteracja')
plt.grid(True)
plt.suptitle('Ewolucja parametrów')

plt.figure()
plt.plot(lambda_history, 'ko', linestyle='None', markersize=6)
plt.xlabel('Iteracja')
plt.ylabel('lambda')
plt.title('Ewolucja parametru zaufania')
plt.grid(True)

plt.figure()
plt.plot(obj_history, 'ko', linestyle='None', markersize=6)
plt.xlabel('Iteracja')
plt.ylabel('Wartość funkcji celu')
plt.title('Ewolucja funkcji celu')
plt.grid(True)

plt.show()
