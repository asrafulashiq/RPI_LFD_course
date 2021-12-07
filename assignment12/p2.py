# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
import collections
from tqdm import tqdm

import seaborn as sns

sns.set_style('whitegrid')

# %%
# ----------------------- Data loading and preprocessig ---------------------- #

file_train = 'ZipDigits.train'
file_test = 'ZipDigits.test'


def read_data(file_name):
    digits = []
    images = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            line = [float(i) for i in line]
            digit = int(line[0])
            data = line[1:]
            images.append(data)
            digits.append(digit)
    return np.array(images), np.array(digits)


def get_features(file_name):
    images, digits = read_data(file_name)
    images = images.reshape(-1, 16, 16)
    images = (images + 1) / 2  # normalize between [0,1]

    # Extract feature
    intensities = images.reshape(-1, 16 * 16).mean(axis=-1)
    symmetry = np.power(images[..., ::-1] - images,
                        2).reshape(images.shape[0], -1).mean(axis=-1)

    ind_1 = (digits == 1)
    ind_not_1 = (digits != 1)
    labels = np.zeros(digits.shape[0])
    labels[ind_1] = 1
    labels[ind_not_1] = -1
    features = np.concatenate((intensities[:, None], symmetry[:, None]),
                              axis=1)
    return features, labels


# ------------------------------ train features ------------------------------ #
features_train, labels_train = get_features(file_train)

# ------------------------------- test features ------------------------------ #
features_test, labels_test = get_features(file_test)

# ----------------------------- combine the data ----------------------------- #
features = np.concatenate((features_train, features_test), axis=0)
labels = np.concatenate((labels_train, labels_test), axis=0)

# %%
# normalize features
import sklearn.preprocessing as preprocessing

features = preprocessing.MinMaxScaler(
    feature_range=(-1, 1)).fit_transform(features)

# split test and train
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, train_size=300, random_state=78)


# %%
class NNLayer:
    def __init__(self,
                 dims=[2, 2, 1],
                 lr=1e-4,
                 alpha=0.9,
                 beta=0.5,
                 theta=np.tanh,
                 theta_prime=lambda x: 1 - np.tanh(x)**2) -> None:
        self.W = []
        for i in range(len(dims) - 1):
            # self.W.append(np.ones((dims[i]+1, dims[i+1])))
            self.W.append(1e-4 * np.random.random((dims[i] + 1, dims[i + 1])))

        self.theta = theta
        self.theta_prime = theta_prime
        self.lr = lr
        self.lr_init = lr
        self.alpha = alpha
        self.beta = beta

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
            if i == num_layer - 1:
                a = s
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
        loss = self.calc_loss(pred, y)

        self.d_list = []

        # compute sensitivity
        # d = 1. / 4 * 2 * (pred - y) * self.theta_prime(self.s_list[-1])
        d = 1. / 4 * 2 * (pred - y) * self.s_list[-1]
        self.d_list.append(d)
        for l in range(len(self.W) - 2, -1, -1):
            # print(l)
            d = self.W[l + 1][:-1].dot(d) * self.theta_prime(self.s_list[l])
            self.d_list.append(d)
        self.d_list = list(reversed(self.d_list))

        # gradient descent: calculate gradient G1, G2
        G_list = []
        for i in range(len(self.W)):
            if i == 0:
                a = self.input
            else:
                a = self.a_list[i - 1]
            a = np.append(a, 1)
            G = a[:, None] @ self.d_list[i][:, None].T
            G_list.append(G)
        return G_list, loss

    def predict(self, x):
        pred = self.forward(x)
        pred = np.sign(pred)
        return pred

    def train_batch(self, X, Y):
        GG = None
        N = len(X)
        Losses = None
        for x, y in zip(X, Y):
            G_list, loss = self.calc_grad(x, y)
            if GG is None:
                GG = [1. / N * g for g in G_list]
                Losses = 1. / N * loss
            else:
                for i in range(len(GG)):
                    GG[i] += 1. / N * G_list[i]
                Losses += 1. / N * loss
        for i in range(len(self.W)):
            self.W[i] -= self.lr * GG[i]

    def train_single(self, x, y):
        GG, loss_w = self.calc_grad(x, y)

        # calculate E(w+)
        w_prime = []
        for i in range(len(self.W)):
            w_prime.append(self.W[i] - self.lr * GG[i])

        _pred = self.forward(x, w_prime)
        loss_w_prime = self.calc_loss(_pred, y)

        if loss_w_prime < loss_w:
            self.W = w_prime
            self.lr = self.lr * self.alpha
        else:
            self.lr = self.lr * self.beta

        return loss_w


# %%
# ------------------------------------ (a) ----------------------------------- #

nn = NNLayer(dims=[2, 2, 1], lr=1e-1, alpha=0.9, beta=0.5)
losses = []

pbar = tqdm(range(1000000))
for iteration in pbar:
    for x, y in zip(features_train, labels_train):
        loss = nn.train_single(x, y)
        losses.append(loss)

        pbar.set_postfix({'loss': loss})

plt.plot(losses)

# %%
losses

# %%
