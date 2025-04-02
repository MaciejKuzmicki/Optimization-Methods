import numpy as np

# Funkcja cel
def f(x):
    term1 = np.exp(x[0] + 3*x[1] - 0.1)
    term2 = np.exp(-x[0] - 0.1)
    P = (1/8) * np.array([[7, np.sqrt(3)], [np.sqrt(3), 5]])
    xc = np.array([1, 1])
    diff = x - xc
    term3 = diff.T @ P @ diff
    return term1 + term2 + term3

# Gradient funkcji celu
def grad_f(x):
    term1 = np.exp(x[0] + 3*x[1] - 0.1)
    term2 = np.exp(-x[0] - 0.1)
    grad1 = np.array([term1 - term2, 3*term1])
    P = (1/8) * np.array([[7, np.sqrt(3)], [np.sqrt(3), 5]])
    xc = np.array([1, 1])
    grad2 = 2 * P.dot(x - xc)
    return grad1 + grad2

# Aktualizacja SR1:
def update_SR1(H, dx, dg):
    y = dx - H.dot(dg)
    denom = np.dot(y, dg)
    if np.abs(denom) < 1e-8:
        return H
    return H + np.outer(y, y) / denom

# Aktualizacja DFP:
def update_DFP(H, dx, dg):
    denom1 = np.dot(dg, dx)
    denom2 = np.dot(dg, H.dot(dg))
    if np.abs(denom1) < 1e-8 or np.abs(denom2) < 1e-8:
        return H
    term1 = np.outer(dx, dx) / denom1
    term2 = np.outer(H.dot(dg), H.dot(dg)) / denom2
    return H + term1 - term2

# Aktualizacja BFGS:
def update_BFGS(H, dx, dg):
    dgTdx = np.dot(dg, dx)
    if np.abs(dgTdx) < 1e-8:
        return H
    term1 = np.outer(dx, dx) / dgTdx
    term2 = np.outer(H.dot(dg), H.dot(dg)) / np.dot(dg, H.dot(dg))
    return H - term2 + term1

# Implementacja metody quasi-newtonowskiej z backtracking line search:
def quasi_newton_method(method, x0, tol=1e-4, alpha=0.5, beta=0.5, max_iter=1000):
    x = x0.copy()
    n = len(x)
    H = np.eye(n)
    iter_num = 0

    while iter_num < max_iter:
        g = grad_f(x)
        v = - H.dot(g)
        inner_product = -np.dot(g, v)
        if inner_product < tol:
            break

        s = 1.0
        while f(x + s * v) > f(x) + s * alpha * np.dot(g, v):
            s = beta * s

        dx = s * v
        x_new = x + dx
        g_new = grad_f(x_new)
        dg = g_new - g

        if method.lower() == 'sr1':
            H = update_SR1(H, dx, dg)
        elif method.lower() == 'dfp':
            H = update_DFP(H, dx, dg)
        elif method.lower() == 'bfgs':
            H = update_BFGS(H, dx, dg)

        x = x_new
        iter_num += 1

    return x, f(x), iter_num

def main():
    x0 = np.array([2.0, -2.0])
    methods = ['SR1', 'DFP', 'BFGS']
    results = {}

    for m in methods:
        x_opt, f_opt, iters = quasi_newton_method(m, x0)
        results[m] = (x_opt, f_opt, iters)
        print(f"Metoda {m}:")
        print(f"  x* = {x_opt}")
        print(f"  f(x*) = {f_opt:.6f}")
        print(f"  Liczba iteracji: {iters}\n")

if __name__ == '__main__':
    main()
