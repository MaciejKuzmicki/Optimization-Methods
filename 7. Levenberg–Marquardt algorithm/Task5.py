import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat('reductionData.mat')
t = data['t'].squeeze()
y = data['y'].squeeze()

t_scaled = t * 1e3
y_scaled = y * 1e3

def h_osc(x, t):
    k_, gamma_, beta_ = x
    return k_ * (
        1.0
        - np.exp(-gamma_ * t) * (
            np.cos(beta_ * t)
            + (gamma_ / beta_) * np.sin(beta_ * t)
        )
    )

def residuals(x):
    return h_osc(x, t_scaled) - y_scaled

def jacobian(x):
    k_, gamma_, beta_ = x
    N = len(t_scaled)
    J = np.zeros((N, 3))
    exp_term = np.exp(-gamma_ * t_scaled)
    cos_term = np.cos(beta_ * t_scaled)
    sin_term = np.sin(beta_ * t_scaled)
    
    J[:, 0] = 1.0 - exp_term * (
        cos_term + (gamma_ / beta_) * sin_term
    )

    J[:, 1] = k_ * exp_term * (
        t_scaled * cos_term
        + ((t_scaled * gamma_) - 1.0) / beta_ * sin_term
    )

    J[:, 2] = k_ * exp_term * (
        (t_scaled + (gamma_ / (beta_**2))) * sin_term
        - (gamma_ / beta_) * t_scaled * cos_term
    )
    return J

k_max = 30
n_params = 3
x0 = np.array([1, 1, 1])
X = np.zeros((n_params, k_max + 1))
X[:, 0] = x0
L = 1.0
lambda_history = [L]
obj_history = [np.linalg.norm(residuals(x0))**2]
x_current = x0.copy()

for k in range(k_max):
    J_current = jacobian(x_current)
    f_current = residuals(x_current)
    A_matrix = J_current.T @ J_current + L * np.eye(n_params)
    b = -J_current.T @ f_current
    delta = np.linalg.solve(A_matrix, b)
    x_new = x_current + delta

    if np.linalg.norm(residuals(x_new)) < np.linalg.norm(f_current):
        x_current = x_new
        L = 0.8 * L
    else:
        L = 2.0 * L
    
    X[:, k + 1] = x_current
    lambda_history.append(L)
    obj_history.append(np.linalg.norm(residuals(x_current))**2)

x_opt = x_current

x_opt_true = np.array([
    x_opt[0] / 1e3, 
    x_opt[1] * 1e3,
    x_opt[2] * 1e3 
])

print("Oszacowane parametry (k, gamma, beta):")
print("k     =", x_opt_true[0])
print("gamma =", x_opt_true[1])
print("beta  =", x_opt_true[2])

T_est = 1.0 / np.sqrt(x_opt_true[1]**2 + x_opt_true[2]**2)
xi_est = x_opt_true[1] * T_est

print("\nZ tego wynika:")
print("T   =", T_est)
print("xi  =", xi_est)

t_plot = np.linspace(t_scaled[0], t_scaled[-1], 1000)

plt.figure()
plt.plot(t_scaled, y_scaled, 'rs', label='Odpowiedź 4-rzędowa (dane)')
plt.plot(t_plot, h_osc(x0, t_plot), 'b-', linewidth=2, label='Model 2-rz. (start)')
plt.plot(t_plot, h_osc(x_opt, t_plot), 'k-', linewidth=2, label='Model 2-rz. (dopasowany)')
plt.xlabel('t (ms)')
plt.ylabel('h(t) [scaled]')
plt.title('Redukcja rzędu: dopasowanie członu oscylacyjnego 2-rz.')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(X[0, :], 'k-', linewidth=2)
plt.ylabel('k')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(X[1, :], 'k-', linewidth=2)
plt.ylabel('gamma')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(X[2, :], 'k-', linewidth=2)
plt.ylabel('beta')
plt.xlabel('Iteracja')
plt.grid(True)
plt.suptitle('Ewolucja parametrów')

plt.figure()
plt.plot(lambda_history, 'ko', linestyle='None', markersize=6)
plt.xlabel('Iteracja')
plt.ylabel('lambda')
plt.title('Ewolucja lambda')
plt.grid(True)

plt.figure()
plt.plot(obj_history, 'ko', linestyle='None', markersize=6)
plt.xlabel('Iteracja')
plt.ylabel('Funkcja celu')
plt.title('Ewolucja funkcji celu')
plt.grid(True)

plt.show()
