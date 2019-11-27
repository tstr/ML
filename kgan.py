import os
import sys
import imageio
import numpy as np
import keras

from matplotlib import pyplot as plt
from tqdm import tqdm

import matplotlib.animation as anim
from keras.datasets import mnist
from keras.models import Sequential, Model, Input
from keras.layers import Dense, ReLU, LeakyReLU, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.initializers import random_uniform

z_size = 10
image_width = 28
image_height = 28
image_size = 28*28

def build_model():
    opt_g = Adam(lr=0.0002, beta_1=0.5)
    opt_d = Adam(lr=0.0002, beta_1=0.5)

    G = Sequential(name="Generator")
    G.add(Dense(256, input_dim=z_size))
    G.add(LeakyReLU(0.2))
    G.add(Dense(512))
    G.add(LeakyReLU(0.2))
    G.add(Dense(1024))
    G.add(LeakyReLU(0.2))
    G.add(Dense(image_size, activation='tanh'))
    G.compile(loss="categorical_crossentropy", optimizer=opt_g)

    D = Sequential(name="Discriminator")
    D.add(Dense(1024, input_dim=image_size))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.3))
    D.add(Dense(512))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.3))
    D.add(Dense(256))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.3))
    D.add(Dense(1, activation="sigmoid"))
    D.compile(loss="binary_crossentropy", optimizer=opt_d)

    D.trainable = False
    inputs = Input(shape=(z_size, ))
    hidden = G(inputs)
    output = D(hidden)
    
    gan = Model(inputs, output)
    gan.compile(loss='binary_crossentropy', optimizer=opt_g)
    return gan, G, D


def train_main():
    print("GPUs:", keras.backend.tensorflow_backend._get_available_gpus())

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], image_size) / 255
    x_train = (x_train * 2) - 1

    if not os.path.exists("mnist_gen"):
        os.makedirs("mnist_gen")

    epochs = 300
    batch_size = 128
    gan, G, D = build_model()

    z_test = np.random.randn(100, z_size)
    g_cost = []
    d_cost = []

    def plotGeneratedImages(epoch):
        dim=(10, 10)
        figsize=(10, 10)
        gimages = G.predict(z_test)
        gimages = gimages.reshape(gimages.shape[0], image_width, image_height)
        plt.figure(figsize=figsize)
        for i in range(gimages.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(gimages[i], interpolation='nearest', cmap="gray")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig("mnist_gen/gen%d.png" % epoch)
        plt.close()

    def plotCost():
        plt.figure(figsize=(10,8))
        plt.plot(d_cost, label="Discriminator cost")
        plt.plot(g_cost, label="Generator cost")
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.legend()
        plt.savefig("mnist_gen/cost.png")

    # training
    for e in range(epochs):
        print("epoch (%d / %d)" % (e+1, epochs))
        dcost = 0
        gcost = 0
        for _ in tqdm(range(x_train.shape[0] // batch_size)):
            xbatch = x_train[np.random.choice(x_train.shape[0], batch_size, replace=False)]
            zbatch = np.random.randn(batch_size, z_size)
            gbatch = G.predict(zbatch)

            # label 0 means fake data, 0.9 means real data 
            xd = np.concatenate((xbatch, gbatch), axis=0)
            yd = np.zeros(batch_size*2)
            yd[0:batch_size] = 0.9

            # train discriminator on sample
            D.trainable = True
            dcost = D.train_on_batch(xd, yd)

            zg = np.random.randn(batch_size, z_size)
            yg = np.ones(batch_size)

            # train generator
            D.trainable = False
            gcost = gan.train_on_batch(zg, yg)

        print("G cost:", gcost, "D cost:", dcost)
        d_cost.append(dcost)
        g_cost.append(gcost)

        G.save("mnist_gen/mnist_gen.hdf5")
        D.save("mnist_gen/mnist_dsc.hdf5")
        gan.save("mnist_gen/mnist_gan.hdf5")

        #if e  == 0:
        plotGeneratedImages(e)
        #for c in range(10):
        #    gimage = G.predict(ztest[c].reshape(1, generator_input_size)).reshape(image_width, image_height)
        #    gimage = ((gimage + 1) / 2) * 255
        #    plt.imsave("mnist_gen/gen%d-%d.png" % (e, c), gimage, cmap='gray', vmax=255)
    plotCost()

#######################################################################################################################################

def save_training_gif():
    epochs=300
    #images = [ for e in range(0,epochs)]
    images = []

    for e in range(0, epochs):
        img = imageio.imread("mnist_gen/gen%d.png" % e)
        plt.imshow(img)
        plt.xlabel("epoch %d" % e)
        plt.yticks([])
        plt.xticks([])
        plt.tight_layout()
        plt.savefig("_temp.png")
        images.append(imageio.imread("_temp.png"))

    os.remove("_temp.png")
    imageio.mimsave("mnist_epoch.gif", images, 'GIF', duration=0.2)

def save_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    xsample = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]

    dim=(10, 10)
    plt.figure(figsize=(10, 10))
    for i in range(xsample.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(xsample[i], interpolation='nearest', cmap="gray")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("mnist_training_sample.png")
    plt.close()

def save_interpolant_gif(G, z_size):
    z0_4 = [0.53928658, 0.65982907, -0.53464781, 0.27090545, -0.24086383, -0.43261732,  0.9701266, 0.87231195, -1.7823505, -0.02450712]
    z1_4 = [1.12309223, 1.80309363,  0.44081664, 0.26745522, -0.16855255, -0.95319326, -0.73860731, 1.17009558, -1.34056898, -0.09573973]
    z0_6 = [ 0.08868714, 0.30986944, 0.49607959, 1.30878289, 0.23699604, -1.33046484, -0.96221701, -0.34803292, 0.29662048, 1.96215788]
    z0_3 = [-0.66338076, -1.76482085, -0.24729566, -0.23468077, -0.76729455, 0.17772756,-1.29608243, 1.00358173, 2.34672588, -0.80023919]
    z0_0 = [ 0.44904647, -1.31786746, 1.13539311, 0.61316969, 0.22178287, -0.50743093, -1.02231253, 0.69842373, -0.90191282, -0.48529115]
    z0_7 = [-1.03097292, -0.26629681, -0.6250876, -1.50605386, 0.64643196, 2.52357352, -0.35692954, -2.04354313, -1.3244745,  -0.83361366]

    def interp(a, b, f):
        return a + ((b - a) * f)

    ims = []
    gimgs = []
    fig = plt.figure()
    plt.axis("off")
    for i in range(100):
        zNew = interp(np.array(z0_0), np.array(z0_7), i / 100)
        #zNew = interp(np.array([-1,0,0,0,0,0,0,0,0,0]), np.array([0,0,0,0,0,0,0,0,0,1]), i / 100)
        print(zNew)
        g = G.predict(zNew.reshape(1,z_size)).reshape(28,28)
        gimgs.append(g)
        ims.append([plt.imshow(g, animated=True, cmap="gray")])

    ani = anim.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save("aaa.gif")
    plt.show()
    #imageio.mimsave("aaaaaaaa2.gif", gimgs, 'GIF', duration=0.1)
    
    return 

    z = np.interp(z0_3, z0_6, np.array([0]*z_size))

    x = G.predict(z)
    x = ((x + 1) / 2).reshape(image_height, image_width)
    plt.imshow(x, cmap="gray")
    plt.show()

    return 

    while True:
        z = np.random.randn(1, z_size)
        print("z:", z)
        x = G.predict(z)
        x = ((x + 1) / 2).reshape(image_height, image_width)
        plt.imshow(x, cmap="gray")
        plt.show()

    return

#######################################################################################################################################

def test_main():
    G = Sequential(name="Generator")
    G.add(Dense(256, input_dim=z_size))
    G.add(LeakyReLU(0.2))
    G.add(Dense(512))
    G.add(LeakyReLU(0.2))
    G.add(Dense(1024))
    G.add(LeakyReLU(0.2))
    G.add(Dense(image_size, activation='tanh'))

    G.load_weights("mnist_generator.hdf5")

if __name__ == "__main__":
    #train_main()
    test_main()
    #save_mnist()

#######################################################################################################################################

def main2():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28*28) / 255
    y_train = to_categorical(y_train, 10)

    x_test = x_test.reshape(-1, 28*28) / 255
    y_test = to_categorical(y_test, 10)

    model = Sequential()
    model.add(Dense(784, activation="sigmoid"))
    model.add(Dense(200, activation="sigmoid"))
    model.add(Dense(90, activation="sigmoid"))
    model.add(Dense(10, activation="sigmoid"))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=10, epochs=1)

    model.summary()

    score = model.evaluate(x_test, y_test, verbose=0)
    print(score)

