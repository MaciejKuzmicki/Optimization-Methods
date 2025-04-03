import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

def sekwencyjna_metoda_bariery_lp_og(c, A, b, Aeq, beq, x0, t_init, gamma, epsilon):
    x = x0
    t = t_init
    k = 0
    x_history = []
    while True:
        def psi_t(x):
            ineq_barrier = - (1/t) * np.sum(np.log(b - A @ x)) if A is not None else 0
            return c.T @ x + ineq_barrier
        def grad_psi_t(x):
            ineq_grad = (1/t) * A.T @ (1 / (b - A @ x)) if A is not None else 0
            return c + ineq_grad
        def hess_psi_t(x):
            ineq_hess = (1/t) * A.T @ np.diag((1 / (b - A @ x))**2) @ A if A is not None else 0
            return ineq_hess
        x_k_history = [x]
        max_iter = 100
        alpha = 0.01
        beta = 0.5
        for _ in range(max_iter):
            grad = grad_psi_t(x)
            hess = hess_psi_t(x)
            try:
                if Aeq is not None:
                    hess_eq = np.vstack([np.hstack([hess, Aeq.T]), np.hstack([Aeq, np.zeros((Aeq.shape[0], Aeq.shape[0]))])])
                    grad_eq = np.hstack([grad, np.zeros(Aeq.shape[0])])
                    hess_eq_reg = hess_eq + epsilon * np.eye(hess_eq.shape[0])
                    delta_x_eq = -np.linalg.solve(hess_eq_reg, grad_eq)
                    delta_x = delta_x_eq[:x.shape[0]]
                else:
                    hess_reg = hess + epsilon * np.eye(hess.shape[0])
                    delta_x = -np.linalg.solve(hess_reg, grad)
            except np.linalg.LinAlgError:
                print("Macierz osobliwa")
                break
            t_line = 1
            while (A is not None and np.any(b - A @ (x + t_line * delta_x) <= 0)) or \
                   psi_t(x + t_line * delta_x) > psi_t(x) + alpha * t_line * grad.T @ delta_x:
                t_line *= beta
                if t_line < 1e-10:
                    break
            x = x + t_line * delta_x
            x_k_history.append(x)
            if np.linalg.norm(grad) < 1e-6:
                break
        x_k_star = x
        x_history.append(x_k_star)
        x = x_k_star
        k += 1
        if (A is not None and A.shape[0] / t <= epsilon) or (A is None and 1 / t <= epsilon):
            return x_k_star, x_history
        t = gamma * t

c = np.array([-0.5, 0.5])
A = np.array([[0.4873, -0.8732],
              [0.6072, 0.7946],
              [0.9880, -0.1546],
              [-0.2142, -0.9768],
              [0.9124, 0.4093]])
b = np.array([1, 1, 1, 1, 1])
Aeq = np.array([[1, 1]])
beq = np.array([0.5])

x0 = np.array([0.25, 0.25])
t_init = 1
gamma = 2.5
epsilon = 1e-6

# Rozwiązanie zadania LP
x_opt, x_history = sekwencyjna_metoda_bariery_lp_og(c, A, b, Aeq, beq, x0, t_init, gamma, epsilon)
print("Rozwiązanie optymalne (metoda SBM - ogólna postać):", x_opt)

# Porównanie z funkcją linprog
res = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq)
print("Rozwiązanie optymalne (linprog):", res.x)

# Wykres obszaru dopuszczalnego, punktów centralnych i poziomic funkcji bariery
V = np.array([[0.1562, 0.9127, 1.0338, 0.8086, -1.3895, -0.8782],
              [-1.0580, -0.6358, 0.1386, 0.6406, 2.3203, -0.8311]])

plt.figure(figsize=(8, 6))

# Rysowanie wielokomórki
plt.fill(V[0, :], V[1, :], color='lightgray')

# Rysowanie ograniczeń nierównościowych
for i in range(A.shape[0]):
    x_plot = np.linspace(-2, 2, 100)
    y_plot = (b[i] - A[i, 0] * x_plot) / A[i, 1]
    plt.plot(x_plot, y_plot, '--', label=f'Ograniczenie {i+1}')

# Rysowanie ograniczenia równościowego
x_eq_plot = np.linspace(-2, 2, 100)
y_eq_plot = beq[0] - x_eq_plot
plt.plot(x_eq_plot, y_eq_plot, '-', label='Ograniczenie równościowe')

# Rysowanie punktów centralnych
x_history = np.array(x_history)
plt.plot(x_history[:, 0], x_history[:, 1], 'r-', marker='o', markersize=3, label='Ścieżka centralna')

# Rysowanie poziomic funkcji celu
x_range = np.linspace(-2, 2, 100)
y_range = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = c[0] * X + c[1] * Y
plt.contour(X, Y, levels=20, colors='red', alpha=0.5)

# Rysowanie rozwiązania
plt.plot(x_opt[0], x_opt[1], 'r*', markersize=10, label='Rozwiązanie (SBM)')
plt.plot(res.x[0], res.x[1], 'g*', markersize=10, label='Rozwiązanie (linprog)')

plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.title('Sekwencyjna Metoda Bariery dla LP (Ogólna Postać)')
plt.legend()
plt.grid(True)
plt.show()