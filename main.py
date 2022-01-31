import tensorflow as tf
import numpy as np
from tensorflow import keras

model = keras.models.load_model('cnn_digits_recognition.h5')
model.summary()
