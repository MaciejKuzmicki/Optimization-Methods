import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

data = loadmat('isoPerimData.mat')
C = float(data['C'][0][0])
F = data['F'].flatten().astype(int)
L = float(data['L'][0][0])
N = int(data['N'][0][0])
a = float(data['a'][0][0])
y_fixed_all = data['y_fixed'].flatten()
y_fixed = y_fixed_all[F-1]

def solve_problem(objective, constraints):
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return problem.value, y.value

# 1. Maksymalizacja pola pod krzywą
y = cp.Variable(N + 1)
objective = cp.Maximize(cp.sum(y[1:]))
constraints = [
    cp.norm(cp.hstack([a/N, y[2:] - y[1:-1]]), 2) <= L,
    y[F - 1] == y_fixed,
    y[0] == 0,
    y[N] == 0,
]
for i in range(N - 1):
    constraints.append(y[i + 2] - 2 * y[i + 1] + y[i] <= C * (a/N)**2)
    constraints.append(y[i + 2] - 2 * y[i + 1] + y[i] >= -C * (a/N)**2)

max_area, y_max = solve_problem(objective, constraints)
max_area *= a / N

# 2. Minimalizacja pola pod krzywą
objective = cp.Minimize(cp.sum(y[1:]))
min_area, y_min = solve_problem(objective, constraints)
min_area *= a / N

# 3. Minimalizacja pola przy nieujemnych zmiennych
constraints.append(y >= 0)
min_nonneg_area, y_min_nonneg = solve_problem(objective, constraints)
min_nonneg_area *= a / N

# 4. Maksymalizacja pola bez ograniczenia na krzywiznę
constraints = [
    cp.sum(cp.norm(cp.hstack([(a/N)*np.ones((N,1)), cp.reshape(y[1:] - y[:-1], (N,1))]), axis=1)) <= L,
    y[F - 1] == y_fixed,
    y[0] == 0,
    y[N] == 0,
]
objective = cp.Maximize(cp.sum(y[1:]))
max_no_curvature_area, y_max_no_curvature = solve_problem(objective, constraints)
max_no_curvature_area *= a / N

print(f"1. Maksymalizacja pola: A = {max_area:.4f}")
print(f"2. Minimalizacja pola: A = {min_area:.4f}")
print(f"3. Minimalizacja pola (nieujemne zmienne): A = {min_nonneg_area:.4f}")
print(f"4. Maksymalizacja pola (bez krzywizny): A = {max_no_curvature_area:.4f}")

x_values = np.linspace(0, a, N + 1)
x_fixed = x_values[F - 1]

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x_values, y_max, label='Maksymalizacja pola')
plt.plot(x_fixed, y_fixed, 'ro', label='Punkty stałe')
plt.title('Maksymalizacja pola')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x_values, y_min, label='Minimalizacja pola')
plt.plot(x_fixed, y_fixed, 'ro', label='Punkty stałe')
plt.title('Minimalizacja pola')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x_values, y_min_nonneg, label='Min. pola (nieujemne zmienne)')
plt.plot(x_fixed, y_fixed, 'ro', label='Punkty stałe')
plt.title('Minimalizacja pola (nieujemne zmienne)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x_values, y_max_no_curvature, label='Maks. pola (bez krzywizny)')
plt.plot(x_fixed, y_fixed, 'ro', label='Punkty stałe')
plt.title('Maksymalizacja pola (bez krzywizny)')
plt.legend()

plt.tight_layout()
plt.show()