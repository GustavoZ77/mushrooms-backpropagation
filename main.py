import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_circles

sigmoid = lambda  x: 1.0/(1.0 + np.exp(-x))
sigmoid_prime = lambda  x: sigmoid(x)*(1.0-sigmoid(x))

tanh = lambda x :np.tanh(x)
tanh_prime = lambda x: 1.0 - x**2

err_calc = lambda yp, yr: np.mean((yp - yr) ** 2)
err_calc_prime = lambda yp, yr: (yp - yr)

class NN_layer:

    def __init__(self, inputs, nnumber, act_f):
        if act_f == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif act_f == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime
        self.bias = np.random.rand(1,nnumber) *2 -1
        self.weights = np.random.rand(inputs,nnumber) *2 -1

class NN:

    df = pd.read_csv('mushroom_dataset.csv')
    df = df.drop('veil-type', axis=1)
    df2 = df
    n = 5000
    p = 22
    y=[]
    dict={}
    error=[]

    # transform to decimal value
    def __init__(self):
        self.X_all = []
        self.err = 100

    def transfrom_to_binary(self, value, column):
        dt = self.dict.get(column)
        index = dt.index(value)
        new_val = 1/index if index > 0 else 0
        return new_val

    def prepare_dataset(self):

        for data in self.df.columns.values:
            list = self.df[data].unique().tolist()
            self.dict[data] = list

        # apply tranformaion to new dataframe
        for data in self.df.columns.values:
            self.df[data] = self.df[data].apply(self.transfrom_to_binary, 1, args=[data])

    def read_dataset(self):
        self.df2 = self.df.sample(200)
        self.y = self.df2["mushroom"].to_numpy()
        del self.df2["mushroom"]

        self.X_all = self.df2.values
        self.y = self.y[:, np.newaxis]

        # self.y = self.df.iloc[3000:6000, 21].values
        # self.X_all = self.df.iloc[3000:6000, 0:22].values
        # self.y = self.y[:, np.newaxis]

        # plt.scatter(self.X_all[:48488, 1], self.X_all[:48488, 21], c='salmon', marker='o', label='Comible')
        # plt.scatter(self.X_all[8000:48489, 1],self. X_all[8000:48489, 21], c='skyblue', marker='X', label='Venenoso')
        # plt.axis("equal")
        # plt.show()

    def create_nn(self,topology, act_f):

        nn = []

        for index, layer in enumerate(topology[:-1]):
            nn.append(NN_layer(topology[index],topology[index + 1], act_f))
        return nn

    def train(self, nn, inputs, outputs, l_rate):
        out = [(None, inputs)]
        deltas = []
        for i, layer in enumerate(nn):
            sum = out[-1][1] @ nn[i].weights + nn[i].bias
            res = nn[i].activation(sum)
            out.append((sum, res))

        #backpropagation
        for index in reversed(range(0, len(nn))):

            sum = out[index+1][0]
            res = out[index+1][1]

            if index == len(nn) - 1:
                deltas.insert(0, err_calc_prime(res, outputs) * nn[i].activation_prime(sum))
            else:
                deltas.insert(0, deltas[0] @ _W.T * nn[i].activation_prime(sum))

            _W = nn[index].weights

            nn[index].bias = nn[index].bias - np.mean(deltas[0], axis=0, keepdims=True) * l_rate
            nn[index].weights = nn[index].weights - out[index][1].T @ deltas[0] * l_rate

        return out[-1][1]

    def test(self, nn):
        df = self.df.sample(100)
        del df["mushroom"]

        out = [(None,df.values)]
        for i, layer in enumerate(nn):
            sum = out[-1][1] @ nn[i].weights + nn[i].bias
            res = nn[i].activation(sum)
            out.append((sum, res))
            k = np.array(out[-1][1])
            df["result"] = k.tolist()
            pd.set_option('display.max_columns', 3)
            pd.set_option('display.max_rows', 50)
            print(df)
        return out[-1][1]


nn = NN()
nn.prepare_dataset()
topology = [21, 8, 4, 2, 1]
layer = nn.create_nn(topology, "sigmoid")

errors=[]
for i in range(1000):
    nn.read_dataset()
    res = nn.train(layer, nn.X_all, nn.y, 1)
    err = err_calc(res, nn.y)
    if i % 25 == 0:
        errors.append(err)
    if err < 0.22:
        break

print("Iterations: ", i)
print("Min err: ", err)

plt.plot(range(len(errors)), errors)
plt.show()

nn.read_dataset()
nn.test(layer)
