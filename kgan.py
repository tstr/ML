import os
import sys
import numpy as np
import keras

from matplotlib import pyplot as plt
from tqdm import tqdm

from keras.datasets import mnist
from keras.models import Sequential, Model, Input
from keras.layers import Dense, ReLU, LeakyReLU
from keras.utils import to_categorical

generator_input_size = 40
image_width = 28
image_height = 28
image_size = 28*28

def build_generator():
    model = Sequential(name="Generator")
    model.add(Dense(generator_input_size))
    model.add(ReLU())
    model.add(Dense(140))
    model.add(ReLU())
    model.add(Dense(80))
    model.add(ReLU())
    model.add(Dense(image_size, activation="sigmoid"))

    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model

def build_discriminator():
    model = Sequential(name="Discriminator")
    model.add(Dense(image_size))
    model.add(LeakyReLU())
    model.add(Dense(200))
    model.add(LeakyReLU())
    model.add(Dense(90))
    model.add(LeakyReLU())
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model

def build_GAN(G, D):
    gan_input = Input(shape=(generator_input_size,))
    x = G(gan_input)
    gan_output = D(x)
    
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

def main():
    print("tf gpus", keras.backend.tensorflow_backend._get_available_gpus())
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if not os.path.exists("mnist_gen"):
        os.makedirs("mnist_gen")

    epochs = 400
    batch_size = 10

    D = build_discriminator()
    G = build_generator()
    gan = build_GAN(G, D)

    ztest = np.random.randn(10, generator_input_size)

    # training
    for e in range(epochs):
        print("epoch", e)
        for _ in tqdm(range(x_train.shape[0] // batch_size)):
            xbatch = x_train[np.random.choice(x_train.shape[0], batch_size, replace=False)].reshape(batch_size, image_size) / 255
            zbatch = np.random.randn(batch_size, generator_input_size)
            gbatch = G.predict(zbatch)

            x = np.concatenate((xbatch, gbatch), axis=0)
            y = np.zeros(x.shape[0])
            y[:batch_size] = 0.9

            # train discriminator on sample
            D.trainable = True
            D.train_on_batch(x, y)

            z = np.random.randn(batch_size, generator_input_size)
            y = np.ones(z.shape[0])

            # train generator
            D.trainable = False
            gan.train_on_batch(z, y)

        gan.save("mnist_gen/mnist_model.hdf5")

        for c in range(10):
            gen = G.predict(ztest[c].reshape(1, generator_input_size)).reshape(image_width, image_height)
            plt.imsave("mnist_gen/gen%d-%d.png" % (e, c), gen[c] * 255, cmap='gray', vmax=255)

if __name__ == "__main__":
    main()



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

