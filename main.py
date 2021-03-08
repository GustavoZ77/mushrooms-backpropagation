import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.ma import clip

pd.set_option('display.max_columns', 4)
pd.set_option('display.max_rows', 30)

#activation funciton

sigmoid = lambda  x: 1.0/(1.0 + np.exp(-x))
sigmoid_prime = lambda  x: sigmoid(x)*(1.0-sigmoid(x))

tanh = lambda x :np.tanh(x)
tanh_prime = lambda x: 1.0 - x**2

err_calc = lambda yp, yr: np.mean((yp - yr) ** 2)
err_calc_prime = lambda yp, yr: (yp - yr)

rectified = lambda x: np.clip(x > -1, 0.1, 1.0)
rectified_prime = lambda x: np.clip(x > -1, 0.1, 1.0)

#layer class
class NN_layer:

    def __init__(self, inputs, nnumber, act_f):
        if act_f == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif act_f == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime
        elif act_f == 'relu':
            self.activation = rectified
            self.activation_prime = rectified_prime

        self.bias = np.random.rand(1,nnumber) *2 -1
        self.weights = np.random.rand(inputs,nnumber) *2 -1

#neural network class
class NN:

    df = pd.read_csv('mushroom_dataset.csv')
    df = df.drop('veil-type', axis=1)
    df2 = df
    y=[]
    dict={}

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
        self.df2 = self.df.sample(100)
        self.y = self.df2["mushroom"].to_numpy()
        del self.df2["mushroom"]

        self.X_all = self.df2.values
        self.y = self.y[:, np.newaxis]

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
        df["probability"] = k.tolist()
        df["result"] = df["probability"].apply(lambda x: "EDIBLE" if x[0] < 0.5 else "POISONOUS")
        print(df)
        return out[-1][1]


#creating the object
nn = NN()
nn.prepare_dataset()
topology = [21, 8, 4, 2, 1]
net = nn.create_nn(topology, "relu")

errors=[]
#train-epochs
for i in range(5000):
    nn.read_dataset()
    res = nn.train(net, nn.X_all, nn.y, 0.05)
    err = err_calc(res, nn.y)
    if i % 25 == 0:
        errors.append(err)
    if err < 0.1:
        break

print("Iterations: ", i)
print("Min err: ", err)

plt.plot(range(len(errors)), errors)
plt.show()

#test
nn.test(net)

