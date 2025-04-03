import numpy as np

def znajdz_x(A, b):
    epsilon = 1e-6

    x = np.random.rand(A.shape[1])

    if np.all(np.dot(A, x) < b - epsilon):
        return x
    else:
        return None

A = np.array([[0.4873, -0.8732],
              [0.6072, 0.7946],
              [0.9880, -0.1546],
              [-0.2142, -0.9768],
              [-0.9871, -0.1601],
              [0.9124, 0.4093]])
b = np.array([1, 1, 1, 1, 1, 1])

x = znajdz_x(A, b)

if x is not None:
    print("Znaleziony wektor x:", x)
else:
    print("Nie można znaleźć wektora x spełniającego warunek.")