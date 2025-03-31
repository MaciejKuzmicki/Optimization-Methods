import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat('inertialData.mat')
t = data['t'].squeeze() 
y = data['y'].squeeze()

def model(x, t):
    k_, T_ = x
    return k_ * (1.0 - np.exp(-t / T_))

def residuals(x):
    return model(x, t) - y

def jacobian(x):
    k_, T_ = x
    J = np.zeros((len(t), 2))
    exp_term = np.exp(-t / T_)
    
    J[:, 0] = 1.0 - exp_term
    J[:, 1] = -k_ * exp_term * (t / (T_**2))
    
    return J

k_max = 35
n_params = 2

x0 = np.array([1.0, 1.0])

X = np.zeros((n_params, k_max + 1))
X[:, 0] = x0

L = 1.0

lambda_history = [L]
obj_history = [np.linalg.norm(residuals(x0))**2]

x_current = x0.copy()

for k in range(k_max):
    J_current = jacobian(x_current)
    f_current = residuals(x_current)
    
    # Rozwiązujemy równanie
    A_matrix = J_current.T @ J_current + L * np.eye(n_params)
    b = -J_current.T @ f_current
    delta = np.linalg.solve(A_matrix, b)
    
    x_new = x_current + delta
    
    # Jeśli norma reszt się zmniejsza, przyjmujemy x_new, inaczej powiększamy L
    if np.linalg.norm(residuals(x_new)) < np.linalg.norm(f_current):
        x_current = x_new
        L = 0.8 * L
    else:
        L = 2.0 * L
    
    X[:, k+1] = x_current
    lambda_history.append(L)
    obj_history.append(np.linalg.norm(residuals(x_current))**2)

x_opt = x_current
print("Optymalne parametry:")
print("k =", x_opt[1])
print("T =", x_opt[0])

t_plot = np.linspace(t[0], t[-1], 400)

plt.figure()
plt.plot(t, y, 'rs', label='Pomiary')
plt.plot(t_plot, model(x0, t_plot), 'b-', linewidth=2, label='Przybliżenie początkowe')
plt.plot(t_plot, model(x_opt, t_plot), 'k-', linewidth=2, label='Dopasowany model')
plt.xlabel('t [s]')
plt.ylabel('h(t)')
plt.title('Dopasowanie: $h(t) = k(1 - e^{-t/T})$')
plt.legend()
plt.grid(True)

plt.figure(figsize=(8,6))

plt.subplot(2,1,1)
plt.plot(X[0, :], 'k-', linewidth=2)
plt.ylabel('$k$')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(X[1, :], 'k-', linewidth=2)
plt.ylabel('$T$')
plt.xlabel('Numer iteracji')
plt.grid(True)

plt.suptitle('Ewolucja parametrów w kolejnych iteracjach')

plt.figure()
plt.plot(lambda_history, 'ko', linestyle='None', markersize=6)
plt.xlabel('Numer iteracji')
plt.ylabel('$\\lambda$')
plt.title('Ewolucja parametru zaufania $\\lambda(k)$')
plt.grid(True)

plt.figure()
plt.plot(obj_history, 'ko', linestyle='None', markersize=6)
plt.xlabel('Numer iteracji')
plt.ylabel('Wartość funkcji celu')
plt.title('Ewolucja wartości funkcji celu')
plt.grid(True)

plt.show()
