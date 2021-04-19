import logging

import numpy as np
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint

from sklearn import metrics

from data import data_utils


class Model:
    data = {
        "x_train": [],
        "y_train": [],
        "x_test": [],
        "y_test": [],
        "label_encoder": None,
        "label_binary_matrix": None,
    }
    model = None

    def __init__(self):
        self._build()
        self._compile()

    def __prepare_model_data(self, dataframe):

        (
            self.data["x_train"],
            self.data["x_test"],
            self.data["y_train"],
            self.data["y_test"],
            self.data["label_encoder"],
            self.data["label_binary_matrix"],
        ) = data_utils.split_data(dataframe)

        def tf_convert(x):
            return tf.convert_to_tensor(x, dtype=tf.float32)

        for k, v in self.data.items():
            tf_convert(v)

    def _build(self):
        """
        Here we will use a Convolutional Neural Network (CNN).
        CNN’s typically make good classifiers and perform
        particular well with image classification tasks
        due to their feature extraction and classification parts.

        I believe that this will be very effective at finding patterns
        within the MFCC’s much like they are effective at finding patterns
        within images.

        We will use a sequential model, starting with a simple model architecture,
        consisting of four Conv2D convolution layers, with our final output layer
        being a dense layer.

        Our output layer will have 10 nodes (num_labels) which matches the number
        of possible classifications.

        https://medium.com/@mikesmales/sound-classification-using-deep-learning-8bc2aa1990b7
        """

        # DATA FORMAT
        input_shape = (193,)
        num_classes = 10
        keras.backend.clear_session()

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(256, activation="relu", input_shape=input_shape))
        self.model.add(keras.layers.Dense(128, activation="relu", input_shape=input_shape))
        self.model.add(keras.layers.Dense(64, activation="relu", input_shape=input_shape))
        self.model.add(keras.layers.Dense(num_classes, activation = "softmax"))

        self._compile()

    def _build_model_layers(self):
        input_shape = (193,)
        num_clases = 10
        keras.backend.clear_session()

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(
            256, activation='relu', input_shape=input_shape))
        self.model.add(keras.layers.Dense(
            128, activation='relu', input_shape=input_shape))
        self.model.add(keras.layers.Dense(
            64, activation='relu', input_shape=input_shape))
        self.model.add(keras.layers.Dense(
            num_clases, activation='softmax', input_shape=input_shape))

    def _compile(self):
        """
        Compiles model based on test dataset
        """
        try:
            self.model.compile(optimizer=keras.optimizers.Adam(1e-4),
                               loss=keras.losses.SparseCategoricalCrossentropy(),
                               metrics=["accuracy"])

            # Display model architecture summary
            self.model.summary()
            print("Model compiled successfully!")

        except Exception as e:
            print("Error while compiling model")
            logging.exception(e)
            return None

    def accuracy(self):
        score = self.model.evaluate(
            self.data["x_test"], self.data["y_test"], verbose=1
        )
        acc = 100 * score[1]

        print("Pre-training accuracy: %.4f%%" % acc)
        return acc or None

    def train(self, x_train, y_train, test_data=None):
        """
        Train model based
        """

        self.model.fit(x_train, y_train, epochs = 50, batch_size = 24, verbose = 0)
        if test_data:
            l, a = self.model.evaluate(test_data["x_test"], test_data["y_test"], verbose = 0)
            return l, a
