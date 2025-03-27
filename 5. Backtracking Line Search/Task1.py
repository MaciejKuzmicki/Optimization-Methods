import numpy as np
import matplotlib.pyplot as plt

def backtracking_search(phi, phi0, phi0_prime, sinitial, alpha, beta):
    s = sinitial
    iterations = [s]
    while phi(s) > phi0 + alpha * s * phi0_prime:
        s = beta * s
        iterations.append(s)
    return s, iterations

def phi1(s):
    return 20*s**2 - 44*s + 29

phi1_0 = phi1(0)
phi1_prime0 = -44

alpha1 = 0.3
beta1 = 0.5
sinitial1 = 1.0

s_star1, iters1 = backtracking_search(phi1, phi1_0, phi1_prime0, sinitial1, alpha1, beta1)
print("Dla phi1(s)=20s^2-44s+29, s* =", s_star1)
print("Iteracje s:", iters1)

s_vals1 = np.linspace(0, 2, 400)
phi_vals1 = phi1(s_vals1)
tangent1 = phi1_0 + phi1_prime0 * s_vals1
armijo_line1 = phi1_0 + alpha1 * phi1_prime0 * s_vals1

plt.figure(figsize=(8, 5))
plt.plot(s_vals1, phi_vals1, label=r'$\varphi_1(s)=20s^2-44s+29$', linewidth=2)
plt.plot(s_vals1, tangent1, 'k--', label=r'$\ell(s)=\varphi(0)+\varphi\'(0)s$')
plt.plot(s_vals1, armijo_line1, 'r--', label=r'$\bar{\ell}(s)=\varphi(0)+\alpha\varphi\'(0)s$')
plt.axvline(s_star1, color='g', linestyle=':', label=r'$s^*$')
plt.scatter(s_star1, phi1(s_star1), color='g')
plt.title(r'Backtracking Search dla $\varphi_1(s)=20s^2-44s+29$')
plt.xlabel('s')
plt.ylabel(r'$\varphi(s)$')
plt.legend()
plt.grid(True)
plt.show()


def phi2(s):
    return 40*s**3 + 20*s**2 - 44*s + 29

phi2_0 = phi2(0)
phi2_prime0 = -44

alpha2 = 0.4
beta2 = 0.9
sinitial2 = 1.0

s_star2, iters2 = backtracking_search(phi2, phi2_0, phi2_prime0, sinitial2, alpha2, beta2)
print("Dla phi2(s)=40s^3+20s^2-44s+29, s* =", s_star2)
print("Iteracje s:", iters2)

s_vals2 = np.linspace(0, 2, 400)
phi_vals2 = phi2(s_vals2)
tangent2 = phi2_0 + phi2_prime0 * s_vals2
armijo_line2 = phi2_0 + alpha2 * phi2_prime0 * s_vals2

plt.figure(figsize=(8, 5))
plt.plot(s_vals2, phi_vals2, label=r'$\varphi_2(s)=40s^3+20s^2-44s+29$', linewidth=2)
plt.plot(s_vals2, tangent2, 'k--', label=r'$\ell(s)=\varphi(0)+\varphi\'(0)s$')
plt.plot(s_vals2, armijo_line2, 'r--', label=r'$\bar{\ell}(s)=\varphi(0)+\alpha\varphi\'(0)s$')
plt.axvline(s_star2, color='g', linestyle=':', label=r'$s^*$')
plt.scatter(s_star2, phi2(s_star2), color='g')
plt.title(r'Backtracking Search dla $\varphi_2(s)=40s^3+20s^2-44s+29$')
plt.xlabel('s')
plt.ylabel(r'$\varphi(s)$')
plt.legend()
plt.grid(True)
plt.show()
