import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def f_func(x):
    return np.array([(x[0]-1)**2 + (x[1]-1)**2 + (x[2]-1)**2])

def df_func(x):
    return np.array([2*(x[0]-1), 2*(x[1]-1), 2*(x[2]-1)])

def g1(x):
    return x[0]**2 + 0.5*x[1]**2 + x[2]**2 - 1

def g2(x):
    return 0.8*x[0]**2 + 2.5*x[1]**2 + x[2]**2 + 2*x[0]*x[2] - x[0] - x[1] - x[2] - 1

def g_func(x):
    return np.array([g1(x), g2(x)])

def dg_func(x):
    row1 = np.array([2*x[0],         x[1],         2*x[2]])
    row2 = np.array([1.6*x[0] + 2*x[2] - 1, 5*x[1] - 1, 2*x[2] + 2*x[0] - 1])
    return np.vstack((row1, row2))

def L_A(x, z, mu):
    return np.concatenate((f_func(x), np.sqrt(mu)*g_func(x) + (1/(2*np.sqrt(mu)))*z))

def augmented_lagrangian_method():
    tol = 1e-5
    max_iter = 100
    
    x = np.array([0.0, 0.0, 0.0])
    z = np.array([0.0, 0.0])
    mu = 1.0

    feasibility_residuals = []
    optimality_residuals = []
    mu_history = []

    for k in range(max_iter):
        x_old = x.copy()
        sol = least_squares(lambda x: L_A(x, z, mu), x, method='lm')
        x = sol.x

        z = z + 2*mu*g_func(x)

        fr = np.linalg.norm(g_func(x))

        term1 = 2 * f_func(x)[0] * df_func(x)
        term2 = np.dot(dg_func(x).T, z)
        or_val = np.linalg.norm(term1 + term2)

        feasibility_residuals.append(fr)
        optimality_residuals.append(or_val)
        mu_history.append(mu)

        print(f"Augmented Iter {k:3d}: x = {x}, ||g(x)|| = {fr:.3e}, OR = {or_val:.3e}, mu = {mu:.3e}")

        if fr < tol and or_val < tol:
            break

        if np.linalg.norm(g_func(x)) >= 0.25 * np.linalg.norm(g_func(x_old)):
            mu *= 2

    print("\nRozwiązanie końcowe (Augmented Lagrangian):")
    print(x)
    
    iters = np.arange(len(feasibility_residuals))
    plt.figure(figsize=(10, 6))
    plt.subplot(3,1,1)
    plt.semilogy(iters, feasibility_residuals, marker='o')
    plt.title('Augmented Lagrangian Method')
    plt.ylabel('||g(x)|| (FR)')
    plt.subplot(3,1,2)
    plt.semilogy(iters, optimality_residuals, marker='o')
    plt.ylabel('Optimality Residual (OR)')
    plt.subplot(3,1,3)
    plt.semilogy(iters, mu_history, marker='o')
    plt.xlabel('Iteracja')
    plt.ylabel('mu')
    plt.tight_layout()
    plt.show()

def Penalty(x, mu):
    return np.concatenate((f_func(x), np.sqrt(mu)*g_func(x)))

def penalty_algorithm():
    tol = 1e-5
    max_iter = 100
    
    x = np.array([0.0, 0.0, 0.0])
    mu = 1.0

    feasibility_residuals = []
    optimality_residuals = []
    mu_history = []

    for k in range(max_iter):
        x_old = x.copy()
        sol = least_squares(lambda x: Penalty(x, mu), x, method='lm')
        x = sol.x

        term1 = 2 * f_func(x)[0] * df_func(x)
        term2 = np.sum(dg_func(x), axis=0)  
        or_val = np.linalg.norm(term1 + term2)

        fr = np.linalg.norm(g_func(x))
        feasibility_residuals.append(fr)
        optimality_residuals.append(or_val)
        mu_history.append(mu)

        print(f"Penalty Alg Iter {k:3d}: x = {x}, ||g(x)|| = {fr:.3e}, OR = {or_val:.3e}, mu = {mu:.3e}")

        if fr < tol and or_val < tol:
            break

        mu *= 2

    print("\nRozwiązanie końcowe (Penalty Algorithm):")
    print(x)

    iters = np.arange(len(feasibility_residuals))
    plt.figure(figsize=(10, 6))
    plt.subplot(3,1,1)
    plt.semilogy(iters, feasibility_residuals, marker='o')
    plt.title('Penalty Algorithm')
    plt.ylabel('||g(x)|| (FR)')
    plt.subplot(3,1,2)
    plt.semilogy(iters, optimality_residuals, marker='o')
    plt.ylabel('Optimality Residual (OR)')
    plt.subplot(3,1,3)
    plt.semilogy(iters, mu_history, marker='o')
    plt.xlabel('Iteracja')
    plt.ylabel('mu')
    plt.tight_layout()
    plt.show()

print("==== Augmented Lagrangian Method ====")
augmented_lagrangian_method()

print("\n==== Penalty Algorithm ====")
penalty_algorithm()
