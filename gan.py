import time
import random
import os
import numpy as np
from matplotlib import pyplot as plt

def partition_list(iter, size):
    batch = []
    for i in iter:
        batch.append(i)
        if (len(batch) >= size):
            yield batch
            batch = []
    yield batch

class Linear:
    @staticmethod
    def f(z): return z
    @staticmethod
    def grad(z): return 1

class Sigmoid:
    @staticmethod
    def f(z):
        return 1 / (1 + np.exp(-z))
    @staticmethod
    def grad(z):
        s = Sigmoid.f(z)
        return s * (1 - s)

class Softmax:
    @staticmethod
    def f(z):
        e = np.exp(z - z.max())
        return e / np.sum(e)
    @staticmethod
    def grad(z):
        s = Sigmoid.f(z)
        return s * (1 - s)

class TanH:
    @staticmethod
    def f(z):
        return np.tanh(z)
    @staticmethod
    def grad(z):
        return 1.0 - np.tanh(z)**2

class ReLU:
    @staticmethod
    def f(z): return np.maximum(z, 0)
    @staticmethod
    def grad(z): return np.greater(z, 0).astype(int)

class LeakyReLU:
    @staticmethod
    def f(z):
        return np.where(z > 0, z, z * 0.01)
    @staticmethod
    def grad(z):
        return np.where(z > 0, 1, 0.01)

class Activation:
    def __init__(self, a, z):
        self.a = a
        self.z = z

# cost function derivatives
def quadratic_cost(a, y):
    return a - y

def cross_entropy_cost(a, y):
    return (-y / a) + ((1 - y)/(1 - a))

class Layer:
    def __init__(self, layersize, inputsize, activation=Sigmoid):
        self.w = np.random.randn(layersize, inputsize) / np.sqrt(inputsize)
        self.b = np.random.randn(layersize)
        self.a = activation    

    def activate(self, z):
        return self.a.f(z)
        
    def activate_grad(self, z):
        return self.a.grad(z)
    
    def forward(self, x):
        z = np.dot(self.w, x) + self.b
        return Activation(self.activate(z), z)

    def backprop(self, x, z, da):
        pass

class NN:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        a = x
        for layer in self.layers:
            a = layer.forward(a).a
        return a

    def backprop(self, x, y):
        activations = [Activation(x, None)]
        for layer in self.layers:
            activations.append(layer.forward(activations[-1].a))

        deltas = [None] * len(self.layers)

        # output layer
        output_delta = self.cost_derivative(activations[-1].a, y, activations[-1].z)
        output_delta_w = np.outer(output_delta, activations[-2].a)
        deltas[-1] = (output_delta, output_delta_w)
        
        # hidden layers
        for i in range(-2, -(len(self.layers) + 1), -1):
            delta_b = np.dot(self.layers[i + 1].w.T, deltas[i + 1][0]) * self.layers[i].activate_grad(activations[i].z)
            delta_w = np.outer(delta_b, activations[i - 1].a)
            deltas[i] = (delta_b, delta_w)

        return deltas

    def cost_derivative(self, a, y, z):
        #quadratic cost derivative
        #return (a - y) * sigmoid_derivative(z)
        #cross entropy derivative
        return a - y

    def train(self, training_data, test_data, epochs, batch_size, learning_rate, regularisation):
        costs = []
        accuracies = []

        k = learning_rate / batch_size
        r = 1 - (learning_rate * regularisation / len(training_data))
        
        for i in range(epochs):
            print("epoch", i)
            np.random.shuffle(training_data)

            for t_batch in partition_list(training_data, batch_size):
                delta_b = [np.zeros(l.b.shape) for l in self.layers]
                delta_w = [np.zeros(l.w.shape) for l in self.layers]

                for (x, y) in t_batch:
                    deltas = self.backprop(x, y)
                    for i, (d_b, d_w) in enumerate(deltas):
                        delta_b[i] += d_b
                        delta_w[i] += d_w
                
                for (d_b, d_w, layer) in zip(delta_b, delta_w, self.layers):
                    layer.w = (r * layer.w) - (k * d_w)
                    layer.b = layer.b - (k * d_b)

            cost, accuracy = self.test(test_data)
            costs.append(cost)
            accuracies.append(accuracy)

    def test(self, testdata):
        cost = 0
        matches = 0
        n = len(testdata)

        for (x, y) in testdata:
            ay = self.forward(x)
            if np.argmax(ay) == np.argmax(y):
                matches += 1
            # quadratic cost
            cost += (0.5 * np.linalg.norm(ay - y)**2)

        cost = cost / n
        print("cost:", cost)
        print("accuracy:", matches, "/", n)
        return cost, matches / n


    def forward2(self, x):
        activations = [Activation(x, None)]
        for layer in self.layers:
            activations.append(layer.forward(activations[-1].a))
        return activations


    def train2(self, x, y, k, r, cost=cross_entropy_cost):

        activations = [Activation(x, None)]
        for layer in self.layers:
            activations.append(layer.forward(activations[-1].a))
        
        deltas = [None] * len(self.layers)

        # output layer
        output_delta = cost(activations[-1].a, y) * self.layers[-1].activate_grad(activations[-1].z) #quadratic_cost(activations[-1].a, y)# 
        output_delta_w = np.outer(output_delta, activations[-2].a)
        deltas[-1] = (output_delta, output_delta_w)
        
        # hidden layers
        for i in range(-2, -(len(self.layers) + 1), -1):
            delta_b = np.dot(self.layers[i + 1].w.T, deltas[i + 1][0]) * self.layers[i].activate_grad(activations[i].z)
            delta_w = np.outer(delta_b, activations[i - 1].a)
            deltas[i] = (delta_b, delta_w)

        dx = np.dot(self.layers[0].w.T, deltas[0][0])

        # update parameters
        for (d_b, d_w), layer in zip(deltas, self.layers):
            layer.w = (r * layer.w) - (k * d_w)
            layer.b = layer.b - (k * d_b)

        return dx


def main():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    def encode(tx, ty):
        x = tx.flatten() / 255
        y = np.zeros(10)
        y[ty] = 1
        return x, y

    #dataset = [encode(tx, ty) for tx, ty in zip(x_train, y_train)]

    sample_size = 10000
    dataset = x_train[np.random.choice(x_train.shape[0], sample_size, replace=False)]#[:,::2,::2]
    w, h = dataset.shape[1], dataset.shape[2]
    dataset = dataset.reshape(sample_size, w*h) / 255

    g_input = 40
    d_input = w*h

    k = 0.1
    r = 1 - (k * 5 / len(dataset))
    epochs = 100

    G = NN([
        Layer(140, g_input, activation=ReLU),
        Layer(80, 140, activation=ReLU),
        Layer(d_input, 80, activation=Sigmoid)
    ])

    D = NN([
        Layer(200, d_input, activation=LeakyReLU),
        Layer(90, 200, activation=LeakyReLU),
        Layer(1, 90, activation=Sigmoid)
    ])

    def save_image(name, img):
        print("saving", name)
        plt.imsave(name, np.clip(np.round(img * 255), 0, 255).reshape(w,h), cmap='gray', vmax=255)

    def test_cost():
        print("test_cost:")
        cost = 0
        for x in dataset:
            cost += np.log(D.forward(x))
        for z in np.random.rand(sample_size, g_input):
            cost += np.log(1 - D.forward(G.forward(z)))

        cost /= (sample_size * 2)
        print(cost)
    
    g_i = np.random.rand(g_input)
    
    if not os.path.exists("gen_out"):
        os.makedirs("gen_out")

    save_image("gen_out/mnist0.png", dataset[0])
    save_image("gen_out/grandom.png", G.forward(g_i))

    test_cost()
    for e in range(epochs):
        print("epoch", e)

        zs = np.random.randn(sample_size, g_input) 
        np.random.shuffle(dataset)

        for i in range(len(dataset)):
            x = dataset[i]
            g = G.forward(zs[i])

            D.train2(x, 1, k, r)
            d = D.train2(g, 0, k, r)

            G.train2(zs[i], None, k, r, cost= lambda a, y: -d)

        test_cost()
        img = G.forward(g_i)
        print(np.max(img), np.min(img))
        save_image("gen_out/g%d.png" % e, img)
    
    cost = np.zeros(1)
    for x in dataset[np.random.choice(dataset.shape[0], 100, replace=False)]:
        cost += np.square(D.forward(x) - 1)
    cost /= 100
    print("test", cost)

if __name__ == "__main__":
    main()