"""
data manipulation and feature extraction
"""

import os
import glob
import logging

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold, train_test_split

from audio.audio import AudioFile
from util.utils import timeit


@timeit
def build_features_dataframe(df_name: str, dataset, metadata):
    """
    Uses metadata info to load all datas    et files and build
    a pandas dataframe of those audio features
    """
    features, labels, data = np.empty((0, 193)), np.empty(0), np.empty(0)  # 193 => total features
    # Iterate through each sound file and extract the features
    metadata = pd.read_csv(metadata)
    for _, row in metadata.iterrows():

        file_name = os.path.join(
            os.path.abspath(dataset),
            "fold" + str(row["fold"]) + "/",
            str(row["slice_file_name"]),
        )

        audio_file = AudioFile(file_name)

        label_from_audio = int(file_name.split('/')[9].split('-')[1])
        label_from_metadata = row["class"]
        print(f'{label_from_metadata}/{label_from_audio}: {file_name}')

        audio_file.extract_features()
        for k, v in audio_file._feat.items():
            print(f'{k}: {v.shape}')
        print('*************************************************************\n')
        audio_feat_list = list(audio_file._feat.values())
        broken_audio_files = []
        if not audio_feat_list:
            broken_audio_files.append(
                {'fn': file_name, 'cl': label_from_metadata})
            continue  # ignore problematic audios

        ext_features3 = np.hstack([audio_feat_list, label_from_audio, label_from_metadata])
        ext_features = np.hstack(audio_feat_list)
        features = np.vstack([features, ext_features])
        labels = np.append(labels, label_from_audio)
        data = np.append(data, ext_features3)

        features, labels = np.array(
            features, dtype=np.float32), np.array(labels, dtype=np.int8)

    # Convert into a Panda dataframe
    featuresdf = pd.DataFrame(data, columns=["mfccs", "chroma", "mel", "contrast", "tonnetz", "label", "label_title"])

    print("Finished feature extraction from ", len(featuresdf), " files")

    featuresdf.to_pickle(df_name)


def parse_audio_files(parent_dir, sub_dir, file_ext='*.wav'):
    features, labels = np.empty((0, 193)), np.empty(0)  # 193 => total features

    for file_name in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
        audio_file = AudioFile(file_name)
        audio_file.extract_features()
        ext_features = np.hstack(list(audio_file._feat.values()))

        features = np.vstack([features, ext_features])
        labels = np.append(labels, int(file_names.split('/')[2].split('-')[1]))

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int8)


@timeit
def read_dataframe(dataframe_picke: str):
    try:
        return pd.read_pickle(dataframe_picke)
    except FileNotFoundError:
        logging.exception(f'{dataframe_picke} not found')
        return None


@timeit
def split_data(features_dataframe: str):
    """
    Converts features and corresponding classification
    labels into numpy arrays
    """
    data_array = np.array(features_dataframe.feature.tolist())
    labels_array = np.array(features_dataframe.class_label.tolist())

    """
    Encode target labels with value between 0 and n_classes-1.
    This transformer should be used to encode target values, i.e. y
    and not the input X
    """
    label_encoder = LabelEncoder()

    """ Converts a class vector (integers) to binary class matrix """
    label_binary_matrix = to_categorical(
        label_encoder.fit_transform(labels_array))

    # splits the dataset
    (
        train_data_array,
        test_data_array,
        train_labels_array,
        test_labels_array,
    ) = train_test_split(
        data_array, label_binary_matrix, test_size=0.2, random_state=42
    )

    return (
        train_data_array,
        test_data_array,
        train_labels_array,
        test_labels_array,
        label_encoder,
        label_binary_matrix,
    )
