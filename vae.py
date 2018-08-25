from keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dense,
    Input,
    Flatten,
    Lambda,
    MaxPool2D,
    UpSampling2D,
    Reshape,
)
from keras.callbacks import LambdaCallback
from keras.models import Model
from keras.datasets import mnist
import keras.backend as K
import tensorflow as tf
import numpy as np
import os
import matplotlib
try:
    if os.environ["DISPLAY"]:
        print("Found display. Using ffmpeg backend.")
except KeyError:
    matplotlib.use("Agg")


import matplotlib.pyplot as plt


class VariationalAutoencoder:
    def __init__(self, input_shape, z_dim):
        self.input_shape = input_shape
        self.z_dim = z_dim
        inputs = Input(shape=self.input_shape, name="input")
        encoder, vae_loss = self.create_encoder(inputs)
        self.decoder = self.create_decoder()
        outputs = self.decoder(encoder(inputs)[2])
        self.model = Model(inputs, outputs, name="vae")

        self.model.compile(optimizer="adam", loss=vae_loss)
        print(self.model.summary())

    def create_encoder(self, inputs):
        net = Conv2D(
            filters=32, kernel_size=3, strides=2, activation="elu", padding="same"
        )(inputs)
        net = Conv2D(
            filters=64, kernel_size=3, strides=2, activation="elu", padding="same"
        )(net)
        net = Flatten()(net)
        net = Dense(16, activation="elu")(net)
        mu = Dense(self.z_dim)(net)
        log_sigma = Dense(self.z_dim)(net)

        def sample_z(args):
            mu, log_sigma = args
            epsilon = tf.random_normal(
                shape=(tf.shape(mu)[0], self.z_dim), mean=0., stddev=1.
            )
            return mu + tf.exp(0.5 * log_sigma) * epsilon

        z = Lambda(sample_z, output_shape=(self.z_dim,))([mu, log_sigma])

        def vae_loss(y_true, y_pred):
            reconstruction_error = K.binary_crossentropy(y_true, y_pred)
            kl_divergence = 1 + log_sigma - K.square(mu) - K.exp(log_sigma)
            kl_divergence = K.sum(kl_divergence, axis=-1)
            kl_divergence *= -0.5
            return K.mean(reconstruction_error + kl_divergence)

        encoder = Model(inputs, [mu, log_sigma, z], name="encoder")
        return encoder, vae_loss

    def create_decoder(self):
        inputs = Input(shape=(self.z_dim,))
        net = Dense(np.product(self.input_shape), activation="elu")(inputs)
        net = Reshape(self.input_shape)(net)
        net = Conv2DTranspose(
            filters=64, kernel_size=3, activation="elu", padding="same"
        )(net)
        net = Conv2DTranspose(
            filters=32, kernel_size=3, activation="elu", padding="same"
        )(net)
        outputs = Conv2DTranspose(
            filters=1, kernel_size=3, activation="sigmoid", padding="same"
        )(net)
        decoder = Model(inputs, outputs, name="decoder")
        return decoder


def plot_latent_space(epoch, vae):
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig(f"latent-space-epoch-{epoch}.png")


def main():
    z_dim = 2
    epochs = 15
    vae = VariationalAutoencoder((28, 28, 1), z_dim)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    vae.model.fit(
        x=x_train,
        y=x_train,
        validation_data=[x_test, x_test],
        batch_size=128,
        epochs=epochs,
        callbacks=[
            LambdaCallback(
                on_epoch_end=lambda epoch, logs: plot_latent_space(epoch, vae)
            )
        ],
    )
    vae.model.save_weights("vae_cnn_mnist.h5")


if __name__ == "__main__":
    main()
