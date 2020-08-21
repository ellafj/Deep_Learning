import numpy as np
import random
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
#from skimage.transform import resize
#import cv2
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.layers import Activation, Dense, Input, Dropout, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Reshape, Concatenate, Lambda, Layer
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mean_squared_error, binary_crossentropy
from keras import metrics
from keras.applications.inception_v3 import preprocess_input, InceptionV3
import keras.backend as K


## Constants
OLD_IMAGE_DIMS = (218, 178, 3)
NEW_IMAGE_DIMS = (64, 64, 3)
CROP_IMAGE_DIMS = (25, 45, 153, 173)
BATCH_SIZE = 128 # Hva er dette?
N = 6
NUM_ATTRIBUTES = 40
LATENT_DIM = 64
TOT_IMAGES = 202599
EPOCHS = 10

## Constants needed to run in Google Colab
PATH = './'
IMAGES = 'img_align_celeba/'
ATTRIBUTES = 'list_attr_celeba.txt'

## Loading dataset
def get_attributes(filename):
  f = open(filename, 'r')
  lines = f.readlines()
  attributes = []
  i = 0
  for line in lines:
      if i != 0 and i != 1:
        line = line.split()
        #line.pop(0)
        attributes.append(line)
      i += 1
  return attributes

def initialize_training_set(): # N or batch size??
    all_attributes = get_attributes(PATH + ATTRIBUTES)
    chosen_info = random.sample(all_attributes, N)
    chosen_attributes = [[info[1:]] for info in chosen_info]
    chosen_images = [info[0] for info in chosen_info]
    resized_images = []

    for name in chosen_images:
        print(name)
        img = np.array((Image.open(PATH + IMAGES + name).crop(CROP_IMAGE_DIMS)).resize((NEW_IMAGE_DIMS[0], NEW_IMAGE_DIMS[1])))
        resized_images.append(img)
    resized_images = np.array(resized_images)
    resized_images = np.array([imgs.astype('float32') for imgs in resized_images])

    return [resized_images, chosen_attributes]

training_set = initialize_training_set()
print(training_set)

class Sampling(Layer):
  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = K.shape(z_mean)[0]
    dim = K.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
"""
#Taken directly from Keras documentation
class Sampling(layers.Layer):
#Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
"""

"""
def Encode_Decode(x):
  #Encoding
  x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu')(x)
  x = Conv2D(filters = 64, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu')(x)
  x = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu')(x)
  x = Conv2D(filters = 256, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu')(x)
  x = Conv2D(filters = 512, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu')(x)
  #Decoding
  x_ = Conv2DTranspose(filters = 256, kernel_size = 3, strides = 2,  padding = 'same', activation = 'relu')(x)
  x_ = Conv2DTranspose(filters = 128, kernel_size = 3, strides = 2,  padding = 'same', activation = 'relu')(x_)
  x_ = Conv2DTranspose(filters = 64, kernel_size = 3, strides = 2,  padding = 'same', activation = 'relu')(x_)
  x_ = Conv2DTranspose(filters = 32, kernel_size = 3, strides = 2,  padding = 'same', activation = 'relu')(x_)
  x_ = Conv2DTranspose(filters = 3, kernel_size = 3, strides = 2,  padding = 'same', activation = 'sigmoid')(x_) 
  return [x,x_] 
"""
"""
def Encoding(x):
  #Downsampling
  x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu')(x)
  x = Conv2D(filters = 64, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu')(x)
  x = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu')(x)
  x = Conv2D(filters = 256, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu')(x)
  return x

def Decoding(x):
  #Upsampling
  x = Conv2DTranspose(filters = 128, kernel_size = 3, strides = 2,  padding = 'same', activation = 'relu')(x)
  x = Conv2DTranspose(filters = 64, kernel_size = 3, strides = 2,  padding = 'same', activation = 'relu')(x)
  x = Conv2DTranspose(filters = 32, kernel_size = 3, strides = 2,  padding = 'same', activation = 'relu')(x)
  x = Conv2DTranspose(filters = 3, kernel_size = 3, strides = 2,  padding = 'same', activation = 'sigmoid')(x)
  return x
"""

def VAE():
  #Define encoder model.
  input_img = Input(shape = NEW_IMAGE_DIMS, name='input_img')
  labels = Input(shape = (NUM_ATTRIBUTES,), name='labels')

  x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu')(input_img)
  x = Conv2D(filters = 64, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu')(x)
  x = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu')(x)
  x = Conv2D(filters = 256, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu')(x)
  x = Conv2D(filters = 512, kernel_size = 3, strides = 2, padding = 'same', activation = 'relu')(x)

  #encode = Encode_Decode(input_img) #Done

  shape_before_flattening = K.int_shape(x)[1:]
  #shape_before_flattening = K.int_shape(encode)[1:]

  x = Flatten()(x)

  z_mean = Dense(LATENT_DIM, name='z_mean')(x)
  z_log_sigma = Dense(LATENT_DIM, name='z_log_sigma')(x)
  z = Sampling()([z_mean, z_log_sigma])

  zy = Concatenate()([z, labels])

  inputs_embedding = Input(shape=(Latent_Dim + Attributes_length,))
  embedding = Dense(np.prod(shape_before_flattening))(inputs_embedding)
  embedding = Reshape(shape_before_flattening)(embedding)

  #Decoding
  x_ = Conv2DTranspose(filters = 256, kernel_size = 3, strides = 2,  padding = 'same', activation = 'relu')(embedding)
  x_ = Conv2DTranspose(filters = 128, kernel_size = 3, strides = 2,  padding = 'same', activation = 'relu')(x_)
  x_ = Conv2DTranspose(filters = 64, kernel_size = 3, strides = 2,  padding = 'same', activation = 'relu')(x_)
  x_ = Conv2DTranspose(filters = 32, kernel_size = 3, strides = 2,  padding = 'same', activation = 'relu')(x_)
  x_ = Conv2DTranspose(filters = 3, kernel_size = 3, strides = 2,  padding = 'same', activation = 'sigmoid')(x_)

  #x_hat = Encode_Decode(embedding)

  encoder = Model(inputs = [input_img, labels], outputs = zy, name="encoder")
  decoder = Model(inputs = inputs_embedding, outputs = x_, name="decoder")

  vae_out = decoder(encoder([input_img, labels]))

  rec_loss =  np.prod(NEW_IMAGE_DIMS) * binary_crossentropy(Flatten()(input_img), Flatten()(vae_out))
  kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
  vae_loss = K.mean(rec_loss + kl_loss)

  vae = Model(inputs = [input_img, labels], outputs = vae_out, name="vae")

  vae.add_loss(vae_loss)

  optimizer = Adam(lr=0.0005, beta_1 = 0.5)
  vae.compile(optimizer)

  return vae, encoder, decoder

vae, encoder, decoder = VAE()

encoder.summary()
decoder.summary()

