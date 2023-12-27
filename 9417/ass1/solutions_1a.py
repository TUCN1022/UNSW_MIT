# -*- encoding:utf-8 -*-
import numpy as np
import sympy as sp

b = np.array([3, 2, -2])
A = np.array([[1, 2, 1, -1],[-1, 1, 0, 2],[0, -1, -2, 1]])
alpha = 0.1
gamma = 0.2
x = np.array([1, 1, 1, 1])


def f(x):
    return 0.5 * np.linalg.norm(np.square(np.dot(A, x.reshape(-1, 1)) - b.reshape(-1, 1)),
                                2) + gamma / 2 * np.linalg.norm(np.square(np.square(x.reshape(-1, 1))), 2)

def grad(x):
    return A.T @ (A @ x.reshape(-1, 1) - b.reshape(-1, 1)) + gamma * x.reshape(-1, 1)

derivation_norm = 1
k = 0
res = list()

while derivation_norm >= 0.001:
    derivation = grad(x)
    derivation_norm = np.linalg.norm(derivation, 2)
    data = (k, x)
    x = x - alpha * derivation.reshape(1, -1)

    res.append(data)
    k += 1

for i in range(5):
    print('k={}, x{}={}'.format(res[i][0], res[i][0], np.around(res[i][1], 4)))

for i in range(-5, 0):
    print('k={}, x{}={}'.format(res[i][0], res[i][0], np.around(res[i][1], 4)))


