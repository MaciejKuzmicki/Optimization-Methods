import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import least_squares, curve_fit

data = loadmat('twoInertialData.mat')
t = data['t'].squeeze()
y = data['y'].squeeze()

def model(t, k, T1, T2):
    return k * (1 - (T1 * np.exp(-t/T1) - T2 * np.exp(-t/T2))/(T1 - T2))

def residuals_func(x, t, y):
    k, T1, T2 = x
    return k * (1 - (T1 * np.exp(-t/T1) - T2 * np.exp(-t/T2))/(T1 - T2)) - y

p0 = [1.0, 1.0, 2.0]

res_ls = least_squares(residuals_func, p0, args=(t,y))
print("Zadanie 4 – least_squares:", res_ls.x)

popt, _ = curve_fit(model, t, y, p0=p0)
print("Zadanie 4 – curve_fit:", popt)

t_plot = np.linspace(t[0], t[-1], 1000)
plt.figure()
plt.plot(t, y, 'rs', label='Dane')
plt.plot(t_plot, model(t_plot, *p0), 'b-', linewidth=2, label='Przybliżenie początkowe')
plt.plot(t_plot, model(t_plot, *res_ls.x), 'k-', linewidth=2, label='Least squares')
plt.plot(t_plot, model(t_plot, *popt), 'g--', linewidth=2, label='curve_fit')
plt.xlabel('t')
plt.ylabel('h(t)')
plt.title('Zadanie 4')
plt.legend()
plt.grid(True)
plt.show()
