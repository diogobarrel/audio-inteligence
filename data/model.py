import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint

from sklearn import metrics


def build_model(x_train: list, x_test, yy):
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

    ## DATA FORMAT
    num_rows = 40
    num_columns = 174
    num_channels = 1

    x_train = x_train.reshape(
        x_train.shape[0])
    x_test = x_test.reshape(
        x_test.shape[0])

    num_labels = yy.shape[1]

    # Construct model
    model = Sequential()

    # First layer
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(
        num_rows, num_columns, num_channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    # Second layer
    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    # Third layer
    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    # Forth layer
    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(num_labels, activation='softmax'))

    return model


def compile_model(model, x_test, y_test):
    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'], optimizer='adam')

    # Display model architecture summary
    model.summary()
    return model

def model_accuracy(model, x_test, y_test):
    # Calculate pre-training accuracy
    score = model.evaluate(x_test, y_test, verbose=1)
    accuracy = 100*score[1]

    print("Pre-training accuracy: %.4f%%" % accuracy)
    return accuracy or None

def train_model(model, x_train, y_train, x_test, y_test):

    num_epochs = 72
    num_batch_size = 256

    checkpointer = ModelCheckpoint(
        filepath='saved_models/weights.best.basic_cnn.hdf5',
        verbose=1, save_best_only=True)

    model.fit(x_train, y_train, batch_size=num_batch_size,
              epochs=num_epochs, validation_data=(x_test, y_test),
              callbacks=[checkpointer], verbose=1)

    return model


def cnn(model, data):
    """ data: data library used  """
    num_rows = 40
    num_columns = 174
    num_channels = 1

    featdf = data.read_dataframe('./featdf.pkl')
    x_train, x_test, y_train, y_test, le, yy = data.split_data(featdf)
    x_train = np.reshape(x_train, x_train.shape[0])
    x_test = np.reshape(x_test, x_test.shape[0])

    cnn_model = model.build_model(x_train, x_test, yy)
    compiled_cnn_model = model.compile_model(cnn_model, x_train, y_train)

    return compiled_cnn_model, x_test
