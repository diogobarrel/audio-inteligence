"""
Will be using keras and sklearn to load a pandas dataframe and
build the model
"""
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical, np_utils

import numpy
import pandas

## loads feature data_frame
features_df = pandas.read_pickle('feat.pkl')

x = numpy.array(features_df.feature.tolist())
y = numpy.array(features_df.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    yy,
                                                    test_size=0.2,
                                                    random_state=42)
"""
We will use a sequential model, starting with a simple model architecture,
consisting of four Conv2D convolution layers, with our final output
layer being a dense layer. Our output layer will have 10 nodes (num_labels)
which matches the number of possible classifications.

See the full report for an in-depth breakdown of the chosen layers,
we also compare the performance of the CNN with a more traditional MLP.
@mikesmales
"""

num_rows = 40
num_columns = 175
num_channels = 1

x_train = x_train.reshape(num_rows, num_columns, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

num_labels = yy.shape[1]
filter_size = 2

# Construct model
model = Sequential()
model.add(
    Conv2D(filters=16,
           kernel_size=2,
           input_shape=(num_rows, num_columns, num_channels),
           activation='relu'))

model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Display model architecture summary 
model.summary()

# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy) 