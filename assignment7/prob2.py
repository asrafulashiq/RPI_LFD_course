# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

sns.set_style("whitegrid")

# %%
# calculate gradient
f_xy = lambda x, y: x**2 + 2 * y**2 + 2 * np.sin(2 * np.pi * x) * np.sin(
    2 * np.pi * y)
grad_f_x = lambda x, y: 2 * x + 4 * np.pi * np.cos(2 * np.pi * x) * np.sin(
    2 * np.pi * y)
grad_f_y = lambda x, y: 4 * y + 4 * np.pi * np.sin(2 * np.pi * x) * np.cos(
    2 * np.pi * y)

# %% [markdown]
# # (a)

# %%
# gradient descent for eta=0.01
x, y = 0.1, 0.1
max_iters = 50

list_val = [f_xy(x, y)]

eta = 0.01
for i in range(max_iters):
    x_prev, y_prev = x, y
    x = x - eta * grad_f_x(x_prev, y_prev)
    y = y - eta * grad_f_y(x_prev, y_prev)
    val = f_xy(x, y)
    list_val.append(val)

plt.plot(list_val)
plt.xlabel('iteration')
plt.ylabel('f(x,y)')

# %%
# gradient descent for eta=0.01
x, y = 0.1, 0.1
max_iters = 50

list_val = [f_xy(x, y)]

eta = 0.1
for i in range(max_iters):
    x_prev, y_prev = x, y
    x = x - eta * grad_f_x(x_prev, y_prev)
    y = y - eta * grad_f_y(x_prev, y_prev)
    val = f_xy(x, y)
    list_val.append(val)

plt.plot(list_val)
plt.xlabel('iteration')
plt.ylabel('f(x,y)')

# %% [markdown]
# # (b)

# %%
# gradient descent for eta=0.01
max_iters = 50
eta = 0.01

print(f"Learning rate eta: {eta}")
for (x, y) in [(0.1, 0.1), (1, 1), (-0.5, -0.5), (-1, -1)]:

    min_val = f_xy(x, y)
    loc_min = (x, y)

    print(f'Starting point: ({x:+.2f}, {y:+.2f}) |', end='  ')

    for i in range(max_iters):
        x_prev, y_prev = x, y
        x = x - eta * grad_f_x(x_prev, y_prev)
        y = y - eta * grad_f_y(x_prev, y_prev)
        val = f_xy(x, y)
        if val < min_val:
            min_val = val
            loc_min = (x, y)
    print(
        f"optimum loc: ({loc_min[0]:+2.4f}, {loc_min[1]:+2.4f}) |  minimum value: {min_val:+.4f}"
    )

# %%
# gradient descent for eta=0.01
max_iters = 50
eta = 0.1

print(f"Learning rate eta: {eta}")

for (x, y) in [(0.1, 0.1), (1, 1), (-0.5, -0.5), (-1, -1)]:

    min_val = f_xy(x, y)
    loc_min = (x, y)

    print(f'Starting point: ({x:+.2f}, {y:+.2f}) |', end='  ')

    for i in range(max_iters):
        x_prev, y_prev = x, y
        x = x - eta * grad_f_x(x_prev, y_prev)
        y = y - eta * grad_f_y(x_prev, y_prev)
        val = f_xy(x, y)
        if val < min_val:
            min_val = val
            loc_min = (x, y)
    print(
        f"optimum loc: ({loc_min[0]:+2.4f}, {loc_min[1]:+2.4f}) |  minimum value: {min_val:+.4f}"
    )

# %%
