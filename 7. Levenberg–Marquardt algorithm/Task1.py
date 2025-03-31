import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

#Zadanie polega na dopasowaniu modelu sinusoidalnego do danych pomiarowych 
# przy użyciu algorytmu Levenberga–Marquardta, który iteracyjnie modyfikuje 
# parametry modelu w celu minimalizacji sumy kwadratów różnic między danymi a modelem.

data = loadmat('LM01Data.mat')
t = data['t'].squeeze()
y = data['y'].squeeze()

# Definicja modelu
def model(x, t):
    A, omega, phi = x
    return A * np.sin(omega * t + phi)

# Wektor reszt
def residuals(x):
    return model(x, t) - y

# Macierz Jacobiego
def jacobian(x):
    A, omega, phi = x
    J = np.zeros((len(t), 3))
    J[:, 0] = np.sin(omega * t + phi)
    J[:, 1] = A * t * np.cos(omega * t + phi)
    J[:, 2] = A * np.cos(omega * t + phi)
    return J

# Ustawienia algorytmu LM

k_max = 35
n_params = 3
x0 = np.array([1.0, 100*np.pi, 0.0])
X = np.zeros((n_params, k_max+1))
X[:, 0] = x0

# Początkowy parametr zaufania
L = 1.0

# Listy do zapisu historii parametru lambda oraz funkcji celu
lambda_history = [L]
obj_history = [np.linalg.norm(residuals(x0))**2]

# Iteracyjna procedura Levenberga–Marquardta

x_current = x0.copy()

for k in range(k_max):
    # Obliczamy macierz Jacobiego oraz wektor reszt
    J_current = jacobian(x_current)
    f_current = residuals(x_current)
    
    # Rozwiązujemy równanie z treści zadania
    A_matrix = J_current.T @ J_current + L * np.eye(n_params)
    b = - J_current.T @ f_current
    delta = np.linalg.solve(A_matrix, b)
    
    x_new = x_current + delta
    
    # Jeśli funkcja celu maleje to zapisujemy nowe przybliżenie
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
print("omega =", x_opt[1])
print("phi =", x_opt[2])

t_plot = np.linspace(t[0], t[-1], 1000)
plt.figure()
plt.plot(t, y, 'rs', label='Pomiary')
plt.plot(t_plot, model(x0, t_plot), 'b-', linewidth=2, label='Przybliżenie początkowe')
plt.plot(t_plot, model(x_opt, t_plot), 'k-', linewidth=2, label='Dopasowany model')
plt.xlabel('t [s]')
plt.ylabel('y')
plt.legend()
plt.title('Dopasowanie modelu: $y = A\\sin(\\omega t + \\varphi)$')
plt.grid(True)

plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(X[0,:], 'k-', linewidth=2)
plt.ylabel('A')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(X[1,:], 'k-', linewidth=2)
plt.ylabel('$\\omega$')
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(X[2,:], 'k-', linewidth=2)
plt.ylabel('$\\varphi$')
plt.xlabel('Numer iteracji')
plt.grid(True)

plt.suptitle('Ewolucja parametrów w kolejnych iteracjach')

plt.figure()
plt.plot(lambda_history, 'ko', markersize=6)
plt.xlabel('Numer iteracji')
plt.ylabel('$\\lambda$')
plt.title('Ewolucja parametru zaufania $\\lambda(k)$')
plt.grid(True)

plt.figure()
plt.plot(obj_history, 'ko', markersize=6)
plt.xlabel('Numer iteracji')
plt.ylabel('Wartość funkcji celu')
plt.title('Ewolucja wartości funkcji celu')
plt.grid(True)

plt.show()
