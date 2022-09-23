import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    y1 = y[0]
    y2 = y[1]

    y1p = y2
    y2p = 6*y2 - y1 + np.exp(2*x)

    return np.array([y1p, y2p])


def step(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h * k1 / 2)
    k3 = f(x + h / 2, y + h * k2 / 2)
    k4 = f(x + h, y + h * k3)

    xn = x + h
    yn = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return xn, yn


def mrk(f, x0, y0, h, N):
    x_vals = [x0]
    y_vals = [y0]

    for i in range(1, N):
        new_x, new_y = step(f, x_vals[i-1], y_vals[i-1], h)
        x_vals.append(new_x)
        y_vals.append(new_y)
    
    return x_vals, np.array(y_vals)


def exact(x):
    return -0.5 * x * np.exp(2*x)


x0 = 0
y0 = np.array([0, -1/2])
h = 0.001
N = 1500

true_x = np.linspace(x0, x0 + h*N, int(10e4))
true_y = exact(true_x)

result_x, result_y = mrk(f, x0, y0, h, N)

plt.plot(true_x, true_y, label="Exact")
plt.plot(result_x, result_y[:, 0], label="Estimated")
plt.legend()
plt.show()
