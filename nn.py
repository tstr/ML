# Simple feedforward nn

import json
import time
import numpy as np

def partition_list(iter, size):
    batch = []
    for i in iter:
        batch.append(i)
        if (len(batch) >= size):
            yield batch
            batch = []
    yield batch

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

class NN:
    class Activation:
        def __init__(self, a, z):
            self.a = a
            self.z = z

    class Layer:
        def __init__(self, w, b):
            self.w = w
            self.b = b
        
        @staticmethod
        def new(layersize, inputsize):
            return NN.Layer(
                np.random.randn(layersize, inputsize) / np.sqrt(inputsize), # weight matrix
                np.random.randn(layersize)                                  # bias vector
            )

        def activate(self, x):
            z = np.dot(self.w, x) + self.b
            return NN.Activation(sigmoid(z), z)

    @staticmethod
    def load(filepath):
        nn = NN()
        with open(filepath) as fp:
            nn.layers = [NN.Layer(j["weights"], j["bias"]) for j in json.load(fp)]
        return nn
    
    def __init__(self, layers=None):
        if layers:
            self.layers = [NN.Layer.new(layers[i], layers[i-1]) for i in range(1, len(layers))]
        else:
            self.layers = []

    def feedforward(self, x):
        a = x
        for layer in self.layers:
            a = layer.activate(a).a
        return a

    def backprop(self, x, y):
        activations = [NN.Activation(x, None)]
        for layer in self.layers:
            activations.append(layer.activate(activations[-1].a))

        deltas = [None] * len(self.layers)

        # output layer
        output_delta = self.cost_derivative(activations[-1].a, y, activations[-1].z)
        output_delta_w = np.outer(output_delta, activations[-2].a)
        deltas[-1] = (output_delta, output_delta_w)
        
        # hidden layers
        for i in range(-2, -(len(self.layers) + 1), -1):
            delta_b = np.dot(self.layers[i + 1].w.T, deltas[i + 1][0]) * sigmoid_derivative(activations[i].z)
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

        self.save_graph(costs, accuracies)

    def test(self, testdata):
        cost = 0
        matches = 0
        n = len(testdata)

        for (x, y) in testdata:
            ay = self.feedforward(x)
            if np.argmax(ay) == np.argmax(y):
                matches += 1
            # quadratic cost
            cost += (0.5 * np.linalg.norm(ay - y)**2)

        cost = cost / n
        print("cost:", cost)
        print("accuracy:", matches, "/", n)
        return cost, matches / n

    def debug_print(self):
        for i, l in enumerate(self.layers):
            print("layer", i)
            print("weights", l.w.shape, l.w)
            print("biases", l.b)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, NN.Layer):
                return { "weights": obj.w, "bias": obj.b }
            return json.JSONEncoder.default(self, obj)
    
    def save_model(self, name):
        with open(name, 'w') as f:
            json.dump(self.layers, f, cls=NN.NumpyEncoder)

    def save_graph(self, cost, accuracies):
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(cost)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.savefig("mse.png")
        
        plt.figure()
        plt.plot(accuracies)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.savefig("accuracy.png")


def encode_x(x):
    return x.flatten() / 255

def encode_y(y):
    a = np.zeros(10)
    a[y] = 1
    return a

def encode_data(x_inputs, y_outputs):
    return list(map(lambda p: (encode_x(p[0]), encode_y(p[1])), zip(x_inputs, y_outputs)))

def main():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    nn = NN([784, 200, 90, 10])
    
    print("---------------------------------------------")
    print("begin training")

    n = time.time()
    nn.train(encode_data(x_train, y_train), encode_data(x_test, y_test), 30, 10, 0.1, 5)
    print(time.time() - n, "s")

    print("end training")

    print("saved to nn.json")
    nn.save_model("nn.json")

if __name__ == "__main__":
    main()