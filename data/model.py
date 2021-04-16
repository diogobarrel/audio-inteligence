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

    def __init__(self, data=None, labels=None, dataframe=None):
        if not dataframe.empty:
            self.__prepare_model_data(dataframe)
        elif data and labels:
            self.data["x_train"] = data
            self.data["y_train"] = labels

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
        num_rows = 40
        num_columns = 174
        num_channels = 1

        x_train = self.data["x_train"].reshape(
            self.data["x_train"].shape[0]
        )  # size = 6985
        x_test = self.data["x_test"].reshape(
            self.data["x_test"].shape[0]
        )  # size = 6985
        model_input_shape = (num_rows, num_columns,
                             num_channels)  # [40, 174, 1]

        num_labels = self.data["label_binary_matrix"].shape[1]

        # Construct model
        new_model = Sequential()

        # First layer
        new_model.add(
            Conv2D(
                filters=16,
                kernel_size=2,
                input_shape=model_input_shape,
                activation="relu",
            )
        )

        new_model.add(MaxPooling2D(pool_size=2))
        new_model.add(Dropout(0.2))
        # Second layer
        new_model.add(Conv2D(filters=32, kernel_size=2, activation="relu"))
        new_model.add(MaxPooling2D(pool_size=2))
        new_model.add(Dropout(0.2))

        # Third layer
        new_model.add(Conv2D(filters=64, kernel_size=2, activation="relu"))
        new_model.add(MaxPooling2D(pool_size=2))
        new_model.add(Dropout(0.2))

        # Forth layer
        new_model.add(Conv2D(filters=128, kernel_size=2, activation="relu"))
        new_model.add(MaxPooling2D(pool_size=2))
        new_model.add(Dropout(0.2))
        new_model.add(GlobalAveragePooling2D())

        new_model.add(Dense(num_labels, activation="softmax"))
        self.model = new_model

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

    def train(self):
        """
        Train model based on train and test datasets
        """

        num_epochs = 72
        num_batch_size = 256

        checkpointer = ModelCheckpoint(
            filepath="saved_models/weights.best.basic_cnn.hdf5",
            verbose=1,
            save_best_only=True,
        )

        self.model.fit(
            self.data["x_train"],
            self.data["y_train"],
            batch_size=num_batch_size,
            epochs=num_epochs,
            validation_data=(
                self.data["x_test"],
                self.data["y_test"],
            ),
            callbacks=[checkpointer],
            verbose=1,
        )

        return model
