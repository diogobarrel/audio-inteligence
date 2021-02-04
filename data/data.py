"""
data manipulation and feature extraction
"""

from sklearn.model_selection import train_test_split
import os

import pandas as pd
import numpy as np

from audio.audio import AudioFile
from util.host import URBANSOUND_8K_AUDIO, URBANSOUND_8K_METADATA
from util.utils import timeit

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


@timeit
def build_features_dataframe(pkl: str):
    """
    Uses metadata info to load all dataset files and build
    a pandas dataframe of those audio features
    """
    features = []
    # Iterate through each sound file and extract the features
    metadata = pd.read_csv(URBANSOUND_8K_METADATA)
    for _, row in metadata.iterrows():

        file_name = os.path.join(
            os.path.abspath(URBANSOUND_8K_AUDIO),
            'fold'+str(row["fold"])+'/',
            str(row["slice_file_name"]))

        class_label = row["class"]
        print(file_name, class_label)
        audio_file = AudioFile(file_name)

        data = audio_file.extract_mfcc_features()
        print(data)

        features.append([data, class_label])

    # Convert into a Panda dataframe
    featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

    print('Finished feature extraction from ', len(featuresdf), ' files')

    featuresdf.to_pickle(pkl)


@timeit
def read_dataframe(audio_file: str):
    return pd.read_pickle(audio_file)


@timeit
def split_data(features_dataframe: str):
    """
    Converts features and corresponding classification
    labels into numpy arrays
    """
    X = np.array(features_dataframe.feature.tolist())
    y = np.array(features_dataframe.class_label.tolist())

    # Encode the classification labels
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))

    # split the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        X, yy, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test, le, yy
