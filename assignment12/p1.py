# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from copy import deepcopy

# %%

theta = np.tanh  # activation function
theta_prime = lambda x: 1 - np.tanh(
    x)**2  # derivative of the activation function

# %%
# forward pass
x = np.array([1, 2])
y = 1


class NNLayer:
    def __init__(self,
                 dims=[2, 2, 1],
                 out_theta=np.tanh,
                 out_theta_prime=lambda x: 1 - np.tanh(x)**2,
                 theta=np.tanh,
                 theta_prime=lambda x: 1 - np.tanh(x)**2) -> None:
        self.W = []
        for i in range(len(dims) - 1):
            self.W.append(0.25 * np.ones((dims[i] + 1, dims[i + 1])))

        self.out_theta = out_theta
        self.out_theta_prime = out_theta_prime
        self.theta = theta
        self.theta_prime = theta_prime

    def forward(self, x, W=None):
        # first forward pass
        if W is None:
            W = self.W
        num_layer = len(W)
        a = x
        self.input = x
        s_list = []
        a_list = []
        for i in range(num_layer):
            s = np.dot(np.append(a, 1), W[i])
            if i == len(W) - 1:
                a = self.out_theta(s)
            else:
                a = self.theta(s)
            s_list.append(s)
            a_list.append(a)

        self.a_list = a_list
        self.s_list = s_list
        return a

    def calc_loss(self, pred, y):
        # calculate loss
        e_in = np.mean(1 / 4. * (pred - y)**2)
        return e_in

    def calc_grad(self, x, y):
        pred = self.forward(x)

        self.d_list = []

        # compute sensitivity
        d = 1. / 4 * 2 * (pred - y) * self.out_theta_prime(self.s_list[-1])
        self.d_list.append(d)
        for l in range(len(self.W) - 2, -1, -1):
            # print(l)
            d = self.W[l + 1][:-1].dot(d) * self.theta_prime(self.s_list[l])
            self.d_list.append(d)
        self.d_list = list(reversed(self.d_list))

        # gradient descent: calculate gradient G1, G2
        self.G_list = []
        for i in range(len(self.W)):
            if i == 0:
                a = self.input
            else:
                a = self.a_list[i - 1]
            a = np.append(a, 1)
            G = a[:, None] @ self.d_list[i][:, None].T
            self.G_list.append(G)
            # G2 = a1[:, None] @ d2[:, None].T
        return self.G_list

    def calc_grad_numerical(self, x, y, eps=1e-4):
        G_list = []

        for l in range(len(self.W)):
            r, c = self.W[l].shape
            G = np.zeros((r, c))
            for i in range(r):
                for j in range(c):
                    W_copy = deepcopy(self.W)
                    W_copy[l][i, j] += eps
                    pred_plus = self.forward(x, W_copy)
                    loss_plus = self.calc_loss(pred_plus, y)
                    W_copy[l][i, j] -= 2 * eps
                    pred_minus = self.forward(x, W_copy)
                    loss_minus = self.calc_loss(pred_minus, y)
                    G[i, j] = (loss_plus - loss_minus) / (2 * eps)
            G_list.append(G)
        return G_list


# %%
# ----------------------------------- 1(a) ----------------------------------- #

nnLayer = NNLayer(dims=[2, 2, 1], theta=theta, theta_prime=theta_prime)
grads = nnLayer.calc_grad(x, y)
print(grads[0])
print(grads[1])

grads_numerical = nnLayer.calc_grad_numerical(x, y)
print(grads_numerical[0])
print(grads_numerical[1])

assert np.allclose(grads[0], grads_numerical[0], atol=1e-4)
assert np.allclose(grads[1], grads_numerical[1], atol=1e-4)

# %%
# ----------------------------------- 1(b) ----------------------------------- #
nnLayer = NNLayer(dims=[2, 2, 1],
                  out_theta=lambda x: x,
                  out_theta_prime=lambda x: np.ones_like(x))
grads = nnLayer.calc_grad(x, y)
print(grads[0])
print(grads[1])

grads_numerical = nnLayer.calc_grad_numerical(x, y)
print(grads_numerical[0])
print(grads_numerical[1])

assert np.allclose(grads[0], grads_numerical[0], atol=1e-4)
assert np.allclose(grads[1], grads_numerical[1], atol=1e-4)

# %%
