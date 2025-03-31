import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import least_squares, curve_fit

data = loadmat('LM01Data.mat')
t = data['t'].squeeze()
y = data['y'].squeeze()

def sinusoid_model(t, A, omega, phi):
    return A * np.sin(omega * t + phi)

# Metoda least_squares (odpowiednik lsqnonlin)
def residuals(x, t, y):
    A, omega, phi = x
    return A * np.sin(omega * t + phi) - y

def jacobian(x, t, y):
    A, omega, phi = x
    J = np.empty((len(t), 3))
    J[:, 0] = np.sin(omega * t + phi)
    J[:, 1] = A * t * np.cos(omega * t + phi)
    J[:, 2] = A * np.cos(omega * t + phi)
    return J

p0 = [1.0, 100*np.pi, 0.0]
res_ls = least_squares(residuals, p0, jac=jacobian, args=(t, y))
print("Parametry (least_squares):", res_ls.x)

# Metoda curve_fit (odpowiednik lsqcurvefit)
popt, _ = curve_fit(sinusoid_model, t, y, p0=p0)
print("Parametry (curve_fit):", popt)

t_plot = np.linspace(t[0], t[-1], 1000)
plt.figure()
plt.plot(t, y, 'rs', label='Pomiary')
plt.plot(t_plot, sinusoid_model(t_plot, *p0), 'b-', linewidth=2, label='Przybliżenie początkowe')
plt.plot(t_plot, sinusoid_model(t_plot, *res_ls.x), 'k-', linewidth=2, label='Dopasowany model (least_squares)')
plt.plot(t_plot, sinusoid_model(t_plot, *popt), 'g--', linewidth=2, label='Dopasowany model (curve_fit)')
plt.xlabel('t [s]')
plt.ylabel('y')
plt.title('Dopasowanie modelu sinusoidalnego (Zadanie 1_2)')
plt.legend()
plt.grid(True)
plt.show()
