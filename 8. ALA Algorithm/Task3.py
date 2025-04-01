import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def f(x, A, b):
    return A.dot(x) - b

def g(x):
    return x**2 - 1

def objective(x, A, b):
    return np.linalg.norm(A.dot(x) - b)**2

def augmentedLagrangianBooleanLS(A, b, max_iter=100):
    m, n = A.shape
    mu = 1.0
    z = np.zeros(n)
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    objective_values = []

    for k in range(max_iter):
        x_old = x.copy()
        fun = lambda x: np.concatenate((
            f(x, A, b),
            np.sqrt(mu) * g(x) + (z / (2 * np.sqrt(mu)))
        ))
        sol = least_squares(fun, x, method='lm')
        x = sol.x

        feasible_x = np.sign(x)

        z = z + 2 * mu * g(feasible_x)

        objective_values.append(objective(feasible_x, A, b))

        if np.linalg.norm(g(feasible_x)) >= 0.25 * np.linalg.norm(g(np.sign(x_old))):
            mu *= 2

    return feasible_x, objective_values

n = 10
m = 10
np.random.seed(0) 
A = np.random.randn(m, n)
b = np.random.randn(m)

feasible_x, objective_values = augmentedLagrangianBooleanLS(A, b)

# Wypisanie wyników
print("Rozwiązanie dopuszczalne (feasible x):")
print(feasible_x)
print("\nWartość funkcji celu dla rozwiązania:")
print(objective_values[-1])

plt.figure()
plt.plot(objective_values, marker='o')
plt.xlabel('Iteracja')
plt.ylabel('Wartość funkcji celu')
plt.title('Objective Value vs. Iteration for n=m=10')
plt.show()
