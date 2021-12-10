# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('whitegrid')

# %%
x1 = np.array([1, 0])
x2 = np.array([-1, 0])

# decision line
fig, ax = plt.subplots()
ax.scatter(x1[0], x1[1], c='r', marker='o')
ax.scatter(x2[0], x2[1], c='g', marker='x')

# plot line
yy = np.linspace(-3, 3, 100)
xx = np.zeros_like(yy)
ax.plot(xx, yy, color='k', label='x-space')

transform_z = lambda x: np.array([x[0]**3 - x[1], x[0] * x[1]])

z1 = transform_z(x1)
z2 = transform_z(x2)

print(z1)
print(z2)

# decision line
xx = np.linspace(-1.5, 1.5, 100)
yy = xx**3
ax.plot(xx, yy, color='b', label='z-space')
ax.legend()

# %%
