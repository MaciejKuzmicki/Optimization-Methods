import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import cvxpy as cp

sqrt3 = np.sqrt(3)
P = (1/8) * np.array([[7, sqrt3],
                      [sqrt3, 5]])
xc = np.array([1, 1])

def f1(x):
    return np.exp(x[0] + 3*x[1] - 0.1) + np.exp(-x[0] - 0.1) + \
           (P[0,0]*(x[0]-xc[0])**2 + 2*P[0,1]*(x[0]-xc[0])*(x[1]-xc[1]) + P[1,1]*(x[1]-xc[1])**2)

def grad_f1(x):
    grad_exp = np.array([
        np.exp(x[0] + 3*x[1] - 0.1) - np.exp(-x[0] - 0.1),
        3*np.exp(x[0] + 3*x[1] - 0.1)
    ])
    grad_quad = 2 * (P @ (x - xc))
    return grad_exp + grad_quad

def hess_f1(x):
    H_exp = np.array([
        [np.exp(x[0] + 3*x[1] - 0.1) + np.exp(-x[0] - 0.1), 3*np.exp(x[0] + 3*x[1] - 0.1)],
        [3*np.exp(x[0] + 3*x[1] - 0.1), 9*np.exp(x[0] + 3*x[1] - 0.1)]
    ])
    H_quad = 2*P
    return H_exp + H_quad

def newton_classic(f, grad_f, hess_f, x0, tol=1e-4, max_iter=100):
    x = x0.copy()
    iterates = [x.copy()]
    f_vals = [f(x)]
    for i in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        H = hess_f(x)
        delta = np.linalg.solve(H, g)
        x = x - delta
        iterates.append(x.copy())
        f_vals.append(f(x))
    return x, np.array(iterates), np.array(f_vals)

def newton_damped(f, grad_f, hess_f, x0, tol=1e-4, max_iter=100, alpha=0.5, beta=0.5):
    x = x0.copy()
    iterates = [x.copy()]
    f_vals = [f(x)]
    for i in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        H = hess_f(x)
        delta = np.linalg.solve(H, g)
        v = -delta
        s = 1.0
        while f(x + s*v) > f(x) + alpha * s * np.dot(g, v):
            s = beta * s
        x = x + s*v
        iterates.append(x.copy())
        f_vals.append(f(x))
    return x, np.array(iterates), np.array(f_vals)

x0_task1 = np.array([2, -2])
tol = 1e-4

x_classic, iters_classic, fvals_classic = newton_classic(f1, grad_f1, hess_f1, x0_task1, tol)
x_damped, iters_damped, fvals_damped = newton_damped(f1, grad_f1, hess_f1, x0_task1, tol, alpha=0.5, beta=0.5)

x_cvx = cp.Variable(2)
objective1 = cp.Minimize(cp.exp(x_cvx[0] + 3*x_cvx[1] - 0.1) + cp.exp(-x_cvx[0] - 0.1) +
                         cp.quad_form(x_cvx - xc, P))
problem1 = cp.Problem(objective1)
result_cvx = problem1.solve()
x_cvx_val = x_cvx.value

x_fmin = fmin(f1, x0_task1, xtol=tol, disp=False)

x1_vals = np.linspace(-3, 3, 300)
x2_vals = np.linspace(-2, 2, 300)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
quad_term = P[0,0]*(X1 - xc[0])**2 + 2*P[0,1]*(X1 - xc[0])*(X2 - xc[1]) + P[1,1]*(X2 - xc[1])**2
Z1 = np.exp(X1 + 3*X2 - 0.1) + np.exp(-X1 - 0.1) + quad_term
levels = np.linspace(np.min(Z1), np.max(Z1), 500)

fig1, axs1 = plt.subplots(1, 2, figsize=(12,6))

cs1 = axs1[0].contour(X1, X2, Z1, levels=levels, cmap='viridis')
axs1[0].clabel(cs1, inline=True, fontsize=8)
axs1[0].plot(iters_classic[:,0], iters_classic[:,1], 'o', color='blue', label='Newton klasyczny')
axs1[0].plot(x_cvx_val[0], x_cvx_val[1], 's', color='green', markersize=8, label='CVXPY')
axs1[0].plot(x_fmin[0], x_fmin[1], 'd', color='red', markersize=8, label='fmin')
axs1[0].set_title('Newton klasyczny')
axs1[0].set_xlabel('x1')
axs1[0].set_ylabel('x2')
axs1[0].set_xlim([-3, 3])
axs1[0].set_ylim([-2, 2])
axs1[0].legend()

cs2 = axs1[1].contour(X1, X2, Z1, levels=levels, cmap='viridis')
axs1[1].clabel(cs2, inline=True, fontsize=8)
axs1[1].plot(iters_damped[:,0], iters_damped[:,1], 'o', color='orange', label='Newton z tłumieniem')
axs1[1].plot(x_cvx_val[0], x_cvx_val[1], 's', color='green', markersize=8, label='CVXPY')
axs1[1].plot(x_fmin[0], x_fmin[1], 'd', color='red', markersize=8, label='fmin')
axs1[1].set_title('Newton z tłumieniem')
axs1[1].set_xlabel('x1')
axs1[1].set_ylabel('x2')
axs1[1].set_xlim([-3, 3])
axs1[1].set_ylim([-2, 2])
axs1[1].legend()

plt.tight_layout()

print("Zadanie 1:")
print("Newton klasyczny, x* =", x_classic, " f(x*) =", f1(x_classic))
print("Newton z tłumieniem, x* =", x_damped, " f(x*) =", f1(x_damped))
print("CVXPY, x* =", x_cvx_val, " f(x*) =", f1(x_cvx_val))
print("fmin, x* =", x_fmin, " f(x*) =", f1(x_fmin))

def f2(x, t):
    diff = x - xc
    quad_val = diff.T @ P @ diff
    if 1 - quad_val <= 0:
        return np.inf
    return t*(np.exp(x[0] + 3*x[1] - 0.1) + np.exp(-x[0] - 0.1)) - np.log(1 - quad_val)

def grad_f2(x, t):
    diff = x - xc
    quad_val = diff.T @ P @ diff
    if 1 - quad_val <= 0:
        return np.array([np.nan, np.nan])
    grad_exp = np.array([
        np.exp(x[0] + 3*x[1] - 0.1) - np.exp(-x[0] - 0.1),
        3*np.exp(x[0] + 3*x[1] - 0.1)
    ])
    grad_barrier = (2 * (P @ diff)) / (1 - quad_val)
    return t * grad_exp + grad_barrier

def hess_f2(x, t):
    diff = x - xc
    quad_val = diff.T @ P @ diff
    if 1 - quad_val <= 0:
        return np.full((2,2), np.nan)
    H_exp = np.array([
        [np.exp(x[0] + 3*x[1] - 0.1) + np.exp(-x[0] - 0.1), 3*np.exp(x[0] + 3*x[1] - 0.1)],
        [3*np.exp(x[0] + 3*x[1] - 0.1), 9*np.exp(x[0] + 3*x[1] - 0.1)]
    ])
    term1 = 2*P/(1 - quad_val)
    term2 = 4 * (P @ diff[:, None] @ diff[None, :] @ P) / ((1 - quad_val)**2)
    return t * H_exp + term1 + term2

def newton_damped_task2(f, grad_f, hess_f, x0, t, tol=1e-4, max_iter=100, alpha=0.3, beta=0.8):
    x = x0.copy()
    iterates = [x.copy()]
    f_vals = [f(x, t)]
    for i in range(max_iter):
        g = grad_f(x, t)
        if np.linalg.norm(g) < tol:
            break
        H = hess_f(x, t)
        delta = np.linalg.solve(H, g)
        v = -delta
        s = 1.0
        while f(x + s*v, t) > f(x, t) + alpha * s * np.dot(g, v):
            s = beta * s
        x = x + s*v
        iterates.append(x.copy())
        f_vals.append(f(x, t))
    return x, np.array(iterates), np.array(f_vals)

t_values = [0.1, 1.0, 10.0]
results_task2 = {}

x1_vals = np.linspace(0, 2, 300)
x2_vals = np.linspace(0, 2, 300)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

fig3, axs3 = plt.subplots(1, 3, figsize=(18,6))
for idx, t in enumerate(t_values):
    x0_task2 = xc.copy()
    x_final, iters, fvals = newton_damped_task2(f2, grad_f2, hess_f2, x0_task2, t, tol, alpha=0.3, beta=0.8)
    results_task2[t] = (x_final, iters, fvals)
    
    Z2 = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            pt = np.array([X1[i,j], X2[i,j]])
            Z2[i,j] = f2(pt, t)
    finite_mask = np.isfinite(Z2)
    lev = np.linspace(np.min(Z2[finite_mask]), np.max(Z2[finite_mask]), 50)
    
    axs3[idx].contour(X1, X2, Z2, levels=lev, cmap='viridis')
    axs3[idx].plot(iters[:,0], iters[:,1], 'o', label=f'Newton z tłumieniem, t={t}')
    axs3[idx].set_title(f't = {t}')
    axs3[idx].set_xlabel('x1')
    axs3[idx].set_ylabel('x2')
    axs3[idx].legend()

t_fixed = 1.0
x_cvx2 = cp.Variable(2)
quad_expr = cp.quad_form(x_cvx2 - xc, P)
objective2 = cp.Minimize(
    t_fixed * (cp.exp(x_cvx2[0] + 3*x_cvx2[1] - 0.1) + cp.exp(-x_cvx2[0] - 0.1))
    - cp.log(cp.reshape(1 - quad_expr, ()))
)
constraints = [quad_expr <= 1 - 1e-6]
problem2 = cp.Problem(objective2, constraints)
result2 = problem2.solve()
x_cvx2_val = x_cvx2.value

def f2_penalty(x, t):
    diff = x - xc
    quad_val = diff.T @ P @ diff
    if quad_val >= 1:
        return 1e6 + (quad_val - 1)**2
    return f2(x, t)

x_fmin2 = fmin(lambda x: f2_penalty(x, t_fixed), xc, xtol=tol, disp=False)

print("\nZadanie 2 (t = 1.0):")
print("Newton z tłumieniem, x* =", results_task2[t_fixed][0], " f(x*) =", f2(results_task2[t_fixed][0], t_fixed))
print("CVXPY, x* =", x_cvx2_val, " f(x*) =", f2(x_cvx2_val, t_fixed))
print("fmin, x* =", x_fmin2, " f(x*) =", f2(x_fmin2, t_fixed))

fig_combined, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

ax_left.plot(range(len(fvals_classic)), fvals_classic, 'o', color='blue', label='Newton klasyczny')
ax_left.plot(range(len(fvals_damped)), fvals_damped, 'o', color='orange', label='Newton z tłumieniem')
ax_left.set_title('Wartość f(x_k) vs. iteracja (Zadanie 1)')
ax_left.set_xlabel('Iteracja')
ax_left.set_ylabel('f(x)')
ax_left.legend()

for t in t_values:
    fvals_t = results_task2[t][2]
    ax_right.plot(range(len(fvals_t)), fvals_t, 'o', label=f't = {t}')
ax_right.set_title('Wartość f(x_k) vs. iteracja (Zadanie 2)')
ax_right.set_xlabel('Iteracja')
ax_right.set_ylabel('f(x)')
ax_right.legend()

plt.tight_layout()
plt.show()
