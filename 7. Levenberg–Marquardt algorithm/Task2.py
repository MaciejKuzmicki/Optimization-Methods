import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat('LM04Data.mat')
t = data['t'].squeeze()  
y = data['y'].squeeze()

# Definicja modelu

def model(x, t):
    A, a, omega, phi = x
    return A * np.exp(-a * t) * np.sin(omega * t + phi)

def residuals(x):
    return model(x, t) - y

def jacobian(x):
    A, a, omega, phi = x
    exp_term = np.exp(-a * t)
    sin_term = np.sin(omega * t + phi)
    cos_term = np.cos(omega * t + phi)
    
    J = np.zeros((len(t), 4))
    J[:, 0] = exp_term * sin_term                      # pochodna względem A
    J[:, 1] = -A * t * exp_term * sin_term             # pochodna względem a
    J[:, 2] = A * t * exp_term * cos_term              # pochodna względem omega
    J[:, 3] = A * exp_term * cos_term                  # pochodna względem phi
    
    return J

# Ustawienia algorytmu LM

k_max = 35
n_params = 4
# Przybliżenie początkowe
x0 = np.array([1.2, 1.4, 56.5, 1.57])
X = np.zeros((n_params, k_max+1))
X[:, 0] = x0

L = 1.0  # początkowy parametr zaufania

lambda_history = [L]
obj_history = [np.linalg.norm(residuals(x0))**2]

x_current = x0.copy()

for k in range(k_max):
    J_current = jacobian(x_current)
    f_current = residuals(x_current)
    
    # Rozwiązujemy równanie
    A_matrix = J_current.T @ J_current + L * np.eye(n_params)
    b = - J_current.T @ f_current
    delta = np.linalg.solve(A_matrix, b)
    
    x_new = x_current + delta
    
    # Jeśli norma reszt maleje, akceptujemy nowe przybliżenie i zmniejszamy L,
    # w przeciwnym wypadku zwiększamy L i pozostawiamy x_current
    if np.linalg.norm(residuals(x_new)) < np.linalg.norm(f_current):
        x_current = x_new
        L = 0.8 * L
    else:
        L = 2 * L
    
    X[:, k+1] = x_current
    lambda_history.append(L)
    obj_history.append(np.linalg.norm(residuals(x_current))**2)

x_opt = x_current
print("Optymalne parametry:")
print("A =", x_opt[0])
print("a =", x_opt[1])
print("omega =", x_opt[2])
print("phi =", x_opt[3])

t_plot = np.linspace(t[0], t[-1], 1000)
plt.figure()
plt.plot(t, y, 'rs', label='Pomiary')
plt.plot(t_plot, model(x0, t_plot), 'b-', linewidth=2, label='Przybliżenie początkowe')
plt.plot(t_plot, model(x_opt, t_plot), 'k-', linewidth=2, label='Dopasowany model')
plt.xlabel('t [s]')
plt.ylabel('y')
plt.title('Dopasowanie modelu: $y = Ae^{-at}\\sin(\\omega t + \\varphi)$')
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(lambda_history, 'ko', linestyle='None', markersize=6)
plt.xlabel('Numer iteracji')
plt.ylabel('$\\lambda$')
plt.title('Ewolucja parametru zaufania $\\lambda(k)$')
plt.grid(True)
plt.xlim(0, 25)
plt.ylim(0, 2)

plt.figure()
plt.plot(obj_history, 'ko', linestyle='None', markersize=6)
plt.xlabel('Numer iteracji')
plt.ylabel('Wartość funkcji celu')
plt.title('Ewolucja wartości funkcji celu')
plt.grid(True)
plt.ylim(1, 50)
plt.xlim(0, 25)

plt.show()
