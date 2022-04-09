import numpy as np
# import matplotlib.pyplot as plt

import random
import cv2
import os
import tensorflow as tf
from tqdm import tqdm

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras import backend as K

random.seed = 21
np.random.seed = 21

class autoencoder:
  def __init__(self,
              input_shape = (128, 128, 3),
              batch_size = 32,
              kernel_size = 3,
              latent_dim = 256,
              layer_filters = [64, 128, 256],
               ):
    self.input_shape = input_shape
    self.batch_size = batch_size
    self.kernel_size = kernel_size
    self.latent_dim = latent_dim
    self.layer_filters = layer_filters

  def build_encoder(self):
    inputs = Input(shape = self.input_shape, name = 'encoder_input')
    x = inputs

    for filters in self.layer_filters:
        x = Conv2D(filters=filters,
                  kernel_size=self.kernel_size,
                  strides=2,
                  activation='relu',
                  padding='same')(x)
                  
    shape = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(self.latent_dim, name='latent_vector')(x)
    encoder = Model(inputs, latent, name='encoder')
    return encoder, inputs, shape

  def build_decoder(self, shape):
    latent_inputs = Input(shape=(self.latent_dim,), name='decoder_input')
    x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    for filters in self.layer_filters[::-1]:
        x = Conv2DTranspose(filters=filters,
                            kernel_size=self.kernel_size,
                            strides=2,
                            activation='relu',
                            padding='same')(x)

    outputs = Conv2DTranspose(filters=3,
                              kernel_size=self.kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)
                          
    decoder = Model(latent_inputs, outputs, name='decoder')
    return decoder

  def build_model(self, inputs, encoder, decoder):
    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    return autoencoder

  def compile_and_train(self,autoencoder,x_train,x_test,y_train,y_test):
    autoencoder.compile(loss='mse', optimizer='adam',metrics=["acc"])
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               verbose=1,
                               min_lr=0.5e-6)
    callbacks = [lr_reducer]

    history = autoencoder.fit(x_train,
                          y_train,
                          validation_data=(x_test, y_test),
                          epochs=20,
                          batch_size=self.batch_size,
                          callbacks=callbacks)
    
  def load_model(self,weights):
    encoder, inputs, shape = self.build_encoder()
    decoder = self.build_decoder(shape)
    autoencoder = self.build_model(inputs,encoder,decoder)
    autoencoder.load_weights(weights)

    return autoencoder