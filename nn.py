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
    class Layer:
        def __init__(self, w, b):
            self.w = w
            self.b = b
        
        @staticmethod
        def new(layersize, inputsize):
            return NN.Layer(
                np.random.randn(layersize, inputsize), # weight matrix
                np.zeros(layersize)                    # bias vector
            )

        def activate(self, x):
            z = np.dot(self.w, x) + self.b
            return sigmoid(z), z

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

    def feedforward(self, inputs):
        acts = inputs
        for layer in self.layers:
            acts, _ = layer.activate(acts)
        return acts

    def backprop(self, x, y):
        a1 = x
        a2, z2 = self.layers[0].activate(a1)
        a3, z3 = self.layers[1].activate(a2)

        # output layer
        o_error = self.cost_derivative(a3, y) * sigmoid_derivative(z3)
        o_error_w = np.outer(o_error, a2)

        # hidden layer
        h_error = np.dot(self.layers[1].w.T, o_error) * sigmoid_derivative(z2)
        h_weight_error = np.outer(h_error, a1)

        return [
            (h_error, h_weight_error),
            (o_error, o_error_w),
        ]

    def cost_derivative(self, a, y):
        return a - y

    def train(self, training_data, test_data, epochs, batch_size, learning_rate):
        mses = []
        for i in range(epochs):
            print("epoch", i)
            np.random.shuffle(training_data)

            k = learning_rate / batch_size
            for t_batch in partition_list(training_data, batch_size):
                delta_b = [np.zeros(l.b.shape) for l in self.layers]
                delta_w = [np.zeros(l.w.shape) for l in self.layers]

                for (x, y) in t_batch:
                    deltas = self.backprop(x, y)
                    for i, (d_b, d_w) in enumerate(deltas):
                        delta_b[i] += d_b
                        delta_w[i] += d_w
                
                for (d_b, d_w, layer) in zip(delta_b, delta_w, self.layers):
                    layer.w = layer.w - (k * d_w)
                    layer.b = layer.b - (k * d_b)

            m = self.test(test_data)
            mses.append(m)

        self.save_graph(mses)

    def test(self, testdata):
        mse = 0
        matches = 0

        for (x, y) in testdata:
            r = self.feedforward(x)
            cost = r - y
            mse = mse + np.dot(cost, cost)
            if np.argmax(r) == np.argmax(y):
                matches = matches + 1

        mse = mse / (2 * len(testdata))
        print("MSE:", mse)
        print("matches:", matches, "/", len(testdata))
        return mse

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

    def save_graph(self, mses):
        from matplotlib import pyplot as plt
        plt.plot(mses)
        plt.xlabel("epochs")
        plt.ylabel("mean squared error")
        plt.savefig("mse.png")


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

    nn = NN([784, 32, 10])

    print("---------------------------------------------")
    print("begin training")

    n = time.time()
    nn.train(encode_data(x_train, y_train), encode_data(x_test, y_test), 30, 10, 3)
    print(time.time() - n, "s")

    print("end training")

    print("saved to nn.json")
    nn.save_model("nn.json")

if __name__ == "__main__":
    main()