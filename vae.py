import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from skimage.transform import resize
import cv2
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
N = 60000
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
        line.pop(0)
        attributes.append(line)
      i += 1
  return attributes

def initialize_training_set(): # N or batch size??
    all_attributes = get_attributes(PATH + ATTRIBUTES)
    chosen_info = np.sample(all_attributes, N)
    chosen_attributes = [[info[1:]] for info in chosen_info]
    chosen_images = [info[0] for info in chosen_info]

    images = np.array([np.array((Image.open(PATH + IMAGES + name).crop(CROP_IMAGE_DIMS)).resize((NEW_IMAGE_DIMS[0], NEW_IMAGE_DIMS[1]))) for name in image_path])
    images = np.array([imgs.astype('float32') for imgs in images])

    return [images, chosen_attributes]

#training_set = initizalixe_training_set()

# Showing possible attributes
attributes = get_attribute_names(PATH + ATTRIBUTES)

for i in range(len(attributes)):
    print(i, attributes[i])
