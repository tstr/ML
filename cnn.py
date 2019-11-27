# Simple convolutional neural network

import numpy as np
from nn import NN
from matplotlib import pyplot as plt
import logging as log
import sys

if __name__ == "__main__":
    log.basicConfig(level=log.DEBUG, filename="cnn.log", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

#######################################################################################

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class ConvolutionLayer:
    def __init__(self, inputs: (int, int, int), kernel: int, features: int, stride=1, padding=0):
        (idepth, ih, iw) = inputs
        self.w = np.random.randn(features, idepth, kernel, kernel) / np.sqrt(kernel*kernel)
        self.b = np.zeros((features, idepth))
        self.in_shape = (idepth, ih, ih)
        ow = (((iw - kernel) + 2 * padding) // stride) + 1
        oh = (((ih - kernel) + 2 * padding) // stride) + 1
        self.out_shape = (features, oh, ow)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.features = features

    def activate(self, z):
        return z #sigmoid(z)


    def convolve(self, x: np.ndarray):
        x = x.reshape(self.in_shape)
        #x = np.pad(x, ((0,0),(0,self.padding),(0,self.padding)))
        out = np.zeros(self.out_shape)
        for f, (w, b) in enumerate(zip(self.w[:], self.b[:])):
            for j in range(0, x.shape[1], self.stride):
                for i in range(0, x.shape[2], self.stride):
                    a = x[:, j:j+self.kernel, i:i+self.kernel]
                    if a.shape == w.shape:
                        out[f,j,i] = self.activate(np.sum(w * a) + b)

        return out

    def conv2D(self, x, w, stride, padding):
        ow = (((x.shape[0] - w.shape[0]) + 2 * padding) // stride) + 1
        oh = (((x.shape[1] - w.shape[1]) + 2 * padding) // stride) + 1
        out = np.zeros((oh, ow))
        k = w.shape[0]
        x = np.pad(x, padding)
        for j in range(oh):
            for i in range(ow):
                a = x[j:j+k,i:i+k]
                if a.shape == w.shape:
                    out[j,i] = np.sum(w * a)
        return out

    def conv3D(self, x, w, stride, padding):
        ow = (((x.shape[1] - w.shape[1]) + 2 * padding) // stride) + 1
        oh = (((x.shape[2] - w.shape[2]) + 2 * padding) // stride) + 1
        out = np.zeros((oh, ow))
        k = w.shape[1]
        x = np.pad(x, padding, axis=(1,2))
        for j in range(oh):
            for i in range(ow):
                a = x[:,j:j+k,i:i+k]
                if a.shape == w.shape:
                    out[j,i] = np.sum(w * a)
        return out

    def backprop(self, x, delta):
        x = x.reshape(self.in_shape)
        dy = delta.reshape(self.out_shape)
        dx = np.zeros(self.in_shape)
        dw = np.zeros(self.w.shape)
        # gradient of bias
        db = np.sum(dy, axis=(1,2)).reshape(self.b.shape)
        # for every feature map
        for f in range(self.features):
            for n in range(x.shape[0]):
                # gradient of weights
                dw[f,n] = self.conv2D(x[n], dy[f], 1, 0)
                w_p = np.rot90(self.w[f,n], 2, axes=(0,1))#.reshape(self.w.shape[2], self.w.shape[3])
                # accumulate gradient of input
                dx[n] += self.conv2D(dy[f], w_p, 1, self.kernel - 1)

        return db, dw, dx



class MaxPool:
    def __init__(self, inputs, size=2):
        self.in_shape = inputs
        self.out_shape = (inputs[0], inputs[1]//size, inputs[2]//size)
        self.size = size

    def feed(self, x: np.ndarray):
        sz = self.size
        out = np.zeros(self.out_shape)
        indices = np.zeros(self.out_shape, dtype=int)
        for f in range(x.shape[0]):
            for j in range(0, x.shape[1], sz):
                for i in range(0, x.shape[2], sz):
                    mx = x[f, j:j+sz, i:i+sz].flatten()
                    out[f, j//sz, i//sz] = np.max(mx)
                    indices[f, j//sz, i//sz] = np.argmax(mx)
        return out, indices

    def backprop(self, delta, indices):
        sz = self.size
        dy = delta.reshape(self.out_shape)
        dx = np.zeros(self.in_shape)
        for f in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                for i in range(indices.shape[2]):
                    x_j, x_i = np.unravel_index(indices[f, i, j], (sz,sz))
                    dx[f, j*sz+x_j, i*sz+x_i] = dy[f,j,i]
        return dx


#######################################################################################

def test():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("input")
    print("x:",x_train[0].shape, "y:", y_train[0])

    x = x_train[0]
    conv = ConvolutionLayer(inputs=(1,28,28), kernel=5, features=8)
    pool = MaxPool(inputs=conv.out_shape, size=2)

    features, _ = pool.feed(conv.convolve(x))
    print("feature shape:" + features.shape)

    for f in features[:]:
        print(f.shape)
        plt.imshow(np.clip(np.round(f * 255), 0, 255), cmap='gray', vmax=255, interpolation="nearest")
        plt.show()

def test2():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # setup
    conv = ConvolutionLayer(inputs=(1,28,28), kernel=5, features=4)
    maxp = MaxPool(inputs=conv.out_shape, size=2)
    nn = NN([np.prod(maxp.out_shape), 100, 10])

    def encode(tx, ty):
        x = tx / 255
        y = np.zeros(10)
        y[ty] = 1
        return x, y

    dataset = [encode(tx, ty) for (tx, ty) in zip(x_train, y_train)]
    testset = [encode(tx, ty) for (tx, ty) in zip(x_test, y_test)]

    alpha = 0.1
    regul = 1 - (alpha * 5 / len(dataset))

    def perf():
        print("testing...")
        sys.stdout.flush()

        cost = 0
        matches = 0
        n = len(testset)        
        for x, y in testset:
            cx = conv.convolve(x)
            cx, _ = maxp.feed(cx)
            ay = nn.feedforward(cx.flatten())

            if np.argmax(ay) == np.argmax(y):
                matches += 1
            
            # quadratic cost
            cost += (0.5 * np.linalg.norm(ay - y)**2)

        cost = cost / n
        print("tested:")
        print("cost:", cost)
        print("accuracy:", matches, "/", n)
        sys.stdout.flush()

    for e in range(30):
        print("epoch", e)
        sys.stdout.flush()
        np.random.shuffle(dataset)

        for i, (x, y) in enumerate(dataset):
            if i % 100 == 0:
                print(i)
                sys.stdout.flush()

            # convolution layer
            cx = conv.convolve(x)
            cx, imax = maxp.feed(cx)
            # fully connected layers
            deltas = nn.backprop(cx.flatten(), y)

            delta_cx = np.dot(nn.layers[0].w.T, deltas[0][0])
            delta_cx = maxp.backprop(delta_cx, imax)
            db, dw, dx = conv.backprop(x, delta_cx)

            # adjust parameters
            conv.w = (regul * conv.w) - (alpha * dw)
            conv.b = conv.b - (alpha * db)

            for layer, (d_b, d_w) in zip(nn.layers, deltas):
                layer.b = layer.b - (alpha * d_b)
                layer.w = (regul * layer.w) - (alpha * d_w)

        perf()
        

if __name__ == "__main__":
    sys.stdout = open("cnn.log", "w")
    test2()