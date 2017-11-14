from keras.datasets import mnist
import numpy as np

# loading the dataset
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)


"""Building the model"""
from keras.layers import Input, Dense
from keras.models import Model


# the size of encoded representation
encoding_dim = 32

# input placeholder
input_img = Input(shape=(784, ))

# the encoded representation of input
encoded = Dense(encoding_dim, activation='relu')(input_img)

# the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# model for mapping input to its reconstruction
autoencoder = Model(input_img, decoded)

# model for mapping input to its encoded representation
encoder = Model(input_img, encoded)

# Create placeholder for an encoder (32-dim) input
encoded_input = Input(shape=(encoding_dim, ))
# retrice the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))


# Compiling the model
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
