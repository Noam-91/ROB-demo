import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l1

class CNN2D(Model):
    def __init__(self):
        super(CNN2D, self).__init__()
        self.C1 = layers.Conv2D(filters=8, kernel_size=(11,11), strides=1, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv1_relu',
               activation = 'relu')
        self.P1 = layers.MaxPooling2D((2, 2))
        self.C2 = layers.Conv2D(filters=4, kernel_size=(5,5), strides=2, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv2_relu',
               activation = 'relu')
        self.P2 = layers.MaxPooling2D((2, 2))
        self.C3 = layers.Conv2D(filters=2, kernel_size=(3,3), strides=2, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv3_relu',
               activation = 'relu')
        self.F1 = layers.Flatten()
        self.D1 = layers.Dense(32, activation='relu', kernel_initializer='lecun_uniform', 
              name='fc1_relu', kernel_regularizer=l1(1e-4))
        self.D2 = layers.Dense(5, activation='softmax', kernel_initializer='lecun_uniform', 
                        name='output_softmax', kernel_regularizer=l1(1e-4))

    def forward(self, x):
        y = self.C1(x)
        y = self.P1(y)
        y = self.C2(y)
        y = self.P2(y)
        y = self.C3(y)
        y = self.F1(y)
        y = self.D1(y)
        y = self.D2(y)

        return y
