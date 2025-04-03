import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

def sekwencyjna_metoda_bariery_lp(c, A, b, x0, t_init, gamma, epsilon):
    x = x0
    t = t_init
    k = 0
    x_history = []
    while True:
        def psi_t(x):
            return c.T @ x - (1/t) * np.sum(np.log(b - A @ x))
        def grad_psi_t(x):
            return c + (1/t) * A.T @ (1 / (b - A @ x))
        def hess_psi_t(x):
            return (1/t) * A.T @ np.diag((1 / (b - A @ x))**2) @ A
        x_k_history = [x]
        max_iter = 100
        alpha = 0.01
        beta = 0.5
        for _ in range(max_iter):
            grad = grad_psi_t(x)
            hess = hess_psi_t(x)
            try:
                delta_x = -np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                print("Macierz osobliwa")
                break
            t_line = 1
            while np.any(b - A @ (x + t_line * delta_x) <= 0) or \
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
        if A.shape[0] / t <= epsilon:
            return x_k_star, x_history
        t = gamma * t

def sekwencyjna_metoda_bariery_lp_faza_I(c, A, b, t_init, gamma, epsilon):
    n = A.shape[1]
    A_faza_I = np.hstack((A, -np.ones((A.shape[0], 1))))
    c_faza_I = np.hstack((np.zeros(n), 1))
    x0_tilde = np.zeros(n + 1)
    s0 = 1 + np.max(A @ np.zeros(n) - b) 
    x0_tilde[-1] = s0
    x_opt_faza_I, _ = sekwencyjna_metoda_bariery_lp(c_faza_I, A_faza_I, b, x0_tilde, t_init, gamma, epsilon)
    if x_opt_faza_I is not None:
        if x_opt_faza_I[-1] < -epsilon:
            x0 = x_opt_faza_I[:-1]
        elif x_opt_faza_I[-1] <= 0:
            print("Zadanie wyjściowe nie ma ściśle dopuszczalnego rozwiązania.")
            return None
        else:
            print("Zadanie wyjściowe nie ma rozwiązania (infeasible).")
            return None
        x_opt, _ = sekwencyjna_metoda_bariery_lp(c, A, b, x0, t_init, gamma, epsilon)
        return x_opt
    else:
        return None

c = np.array([-0.5, 0.5])
A = np.array([[0.4873, -0.8732],
              [0.6072, 0.7946],
              [0.9880, -0.1546],
              [-0.2142, -0.9768],
              [-0.9871, -0.1601],
              [0.9124, 0.4093]])
b = np.array([1, 1, 1, 1, 1, 1])
t_init = 1
gamma = 2.5
epsilon = 1e-6

x_opt_faza_I = sekwencyjna_metoda_bariery_lp_faza_I(c, A, b, t_init, gamma, epsilon)

if x_opt_faza_I is not None:
    print("Rozwiązanie optymalne (metoda SBM z fazą I):", x_opt_faza_I)
    from scipy.optimize import linprog
    res = linprog(c, A_ub=A, b_ub=b)
    print("Rozwiązanie optymalne (linprog):", res.x)
    V = np.array([[0.1562, 0.9127, 1.0338, 0.8086, -1.3895, -0.8782],
                  [-1.0580, -0.6358, 0.1386, 0.6406, 2.3203, -0.8311]])
    plt.figure(figsize=(8, 6))
    plt.fill(V[0, :], V[1, :], color='lightgray')
    for i in range(A.shape[0]):
        x_plot = np.linspace(-2, 2, 100)
        y_plot = (b[i] - A[i, 0] * x_plot) / A[i, 1]
        plt.plot(x_plot, y_plot, '--', label=f'Ograniczenie {i+1}')
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = c[0] * X + c[1] * Y
    plt.contour(X, Y, levels=20, colors='red', alpha=0.5)
    plt.plot(x_opt_faza_I[0], x_opt_faza_I[1], 'r*', markersize=10, label='Rozwiązanie (SBM z fazą I)')
    plt.plot(res.x[0], res.x[1], 'g*', markersize=10, label='Rozwiązanie (linprog)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.title('Sekwencyjna Metoda Bariery dla LP (Faza I)')
    plt.legend()
    plt.grid(True)
    plt.show()