import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.io import loadmat
from scipy.optimize import linprog

data = loadmat('Data01.mat')
t = data['t'].squeeze()
y = data['y'].squeeze()
n = len(y)

D = np.zeros((n-1, n))
for i in range(n-1):
    D[i, i] = -1
    D[i, i+1] = 1

q_values = [0.5, 1, 2]
tau_values = [0.1, 1, 10]

v_CVX_q = []
for q in q_values:
    v = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(y - v))
    constraints = [cp.norm1(D @ v) <= q]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    v_CVX_q.append(v.value)

v_CVX_tau = []
for tau in tau_values:
    v = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(y - v) + tau * cp.norm1(D @ v))
    prob = cp.Problem(objective)
    prob.solve(solver=cp.ECOS)
    v_CVX_tau.append(v.value)

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(t, y, 'k--', linewidth=1.5, label='y')
for i, q in enumerate(q_values):
    plt.plot(t, v_CVX_q[i], linewidth=1.5, label=f'q = {q:.2f}')
plt.title('CVXPY: Rekonstrukcja z ograniczeniem ||D*v||₁ ≤ q')
plt.xlabel('t')
plt.ylabel('Sygnał')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, y, 'k--', linewidth=1.5, label='y')
for i, tau in enumerate(tau_values):
    plt.plot(t, v_CVX_tau[i], linewidth=1.5, label=f'tau = {tau:.2f}')
plt.title('CVXPY: Rekonstrukcja z LASSO (||y-v||₂² + tau||D*v||₁)')
plt.xlabel('t')
plt.ylabel('Sygnał')
plt.legend()

plt.tight_layout()
plt.show()

q_lp = 1

N = n
M = n - 1
N_total = 2 * N + M

f = np.concatenate([np.zeros(N), np.ones(N), np.zeros(M)])

A1 = np.hstack([np.eye(N), -np.eye(N), np.zeros((N, M))])
b1 = y
A2 = np.hstack([-np.eye(N), -np.eye(N), np.zeros((N, M))])
b2 = -y
A3 = np.hstack([D, np.zeros((M, N)), -np.eye(M)])
b3 = np.zeros(M)
A4 = np.hstack([-D, np.zeros((M, N)), -np.eye(M)])
b4 = np.zeros(M)
A5 = np.hstack([np.zeros((1, 2 * N)), np.ones((1, M))])
b5 = np.array([q_lp])

A_lp = np.vstack([A1, A2, A3, A4, A5])
b_lp = np.concatenate([b1, b2, b3, b4, b5])

lb = -np.inf * np.ones(N_total)
lb[N:2*N] = 0
lb[2*N:] = 0
bounds = [(lb[i], None) for i in range(N_total)]

res_lp = linprog(f, A_ub=A_lp, b_ub=b_lp, bounds=bounds, method='highs')
v_lp = res_lp.x[:N]

tau_lp = 0.1
f2 = np.concatenate([np.zeros(N), np.ones(N), tau_lp * np.ones(M)])
A_lp2 = np.vstack([A1, A2, A3, A4])
b_lp2 = np.concatenate([b1, b2, b3, b4])
res_lp2 = linprog(f2, A_ub=A_lp2, b_ub=b_lp2, bounds=bounds, method='highs')
v_lp2 = res_lp2.x[:N]

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(t, y, 'k--', linewidth=1.5, label='y')
plt.plot(t, v_lp, 'r', linewidth=1.5, label=f'v (q = {q_lp:.2f})')
plt.title('LP: Rekonstrukcja (||y-v||₁, ||D*v||₁ ≤ q) - linprog')
plt.xlabel('t')
plt.ylabel('Sygnał')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, y, 'k--', linewidth=1.5, label='y')
plt.plot(t, v_lp2, 'b', linewidth=1.5, label=f'v (tau = {tau_lp:.2f})')
plt.title('LP: Rekonstrukcja (||y-v||₁ + tau||D*v||₁) - linprog')
plt.xlabel('t')
plt.ylabel('Sygnał')
plt.legend()

plt.tight_layout()
plt.show()
