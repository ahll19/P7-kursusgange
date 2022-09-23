import numpy as np
import matplotlib.pyplot as plt


def f(t, y):
    xp = 2 - 0.9 * y[1]
    yp = -0.004 * y[0] * y[1]

    return np.array([xp, yp])


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


x0 = 0
y0 = [20, 6]
h = 0.01
N = 828

res_x, res_y = mrk(f, x0, y0, h, N)

plt.plot(res_x, res_y[:, 0], label="Team A")
plt.plot(res_x, res_y[:, 1], label="Team B")
plt.legend()
plt.show()
