"""
Driver
"""
from concurrent.futures import ThreadPoolExecutor
import sys
import logging
import glob
import os

from sklearn.model_selection import KFold
from data import data_utils, model
import numpy as np
import pandas as pd

from audio import signal
from audio.audio import AudioFile
from util.host import get_samples, CHOSEN_DATASET, CHOSEN_METADATA, DATAFRAME, ESC_50_META
from util.utils import timeit


def plot_samples():
    audio_samples = get_samples()
    ROW, COL = 1, 1
    for sample in audio_samples:
        # audio_file = AudioFile(sample)
        # print(audio_file.extract_mfcc_features())
        signal.spectogram(sample, index=f"{ROW}{COL}")
        signal.stft_spectogram(sample, index=f"{ROW+1}{COL}")
        # signal.draw_spec3(sample, index=f'{ROW+2}{COL}')
        COL += 1


def proccess_dataset():
    """ Proccess whole datased and and creates .npz files on a new folder """

    def process_audio_file(fn):
        features, labels = np.empty((0, 193)), np.empty(
            0)  # 193 => total features

        audio_file = AudioFile(fn)
        audio_file.extract_features()

        if not audio_file._feat.values():
            logging.warning(f'broken file: {fn}.')
            return False, False  # ignore problematic audios

        ext_features = np.hstack(list(audio_file._feat.values()))
        ext_label = int(fn.split('/')[9].split('-')[1])

        return [ext_features, ext_label]

    def parse_audio_folder(parent_dir, sub_dir, file_ext='*.wav'):
        folder_features, folder_labels = np.empty(
            (0, 193)), np.empty(0)  # 193 => total features

        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            ext_features, ext_labels = process_audio_file(fn)
            if isinstance(ext_features, bool) and not ext_features:
                continue

            folder_features = np.vstack([folder_features, ext_features])
            folder_labels = np.append(folder_labels, ext_labels)

        return np.array(folder_features, dtype=np.float32), np.array(folder_labels, dtype=np.int8)

    # Pre-process and extract feature from the data
    parent_dir = f'{CHOSEN_DATASET}/audio'
    save_dir = f'{CHOSEN_DATASET}/processed'
    sub_dirs = np.array(['fold1', 'fold2', 'fold3', 'fold4',
                         'fold5', 'fold6', 'fold7', 'fold8',
                         'fold9', 'fold10'])

    for sub_dir in sub_dirs:
        features, labels = parse_audio_folder(parent_dir, sub_dir)
        np.savez("{0}/{1}".format(save_dir, sub_dir), features=features,
                 labels=labels)


def get_data_from_dataframe(dataframe, build=False):
    data = data_utils.read_dataframe(f"./{DATAFRAME}")
    if not data or data.empty and build:
        logging.warn(
            f'Building dataframe {DATAFRAME} from dataset {CHOSEN_DATASET}')
        df = data_utils.build_features_dataframe(
            DATAFRAME, CHOSEN_DATASET, CHOSEN_METADATA)
        data = data_utils.read_dataframe(f"./{DATAFRAME}")

    return data


def get_data_from_processed_files():
    ## Train and evaluate via 10-Folds cross-validation ###
    accuracies = []
    folds = np.array(['fold1', 'fold2', 'fold3', 'fold4',
                      'fold5', 'fold6', 'fold7', 'fold8',
                     'fold9', 'fold10'])

    load_dir = f"{CHOSEN_DATASET}/processed"
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(folds):
        x_train, y_train = [], []
        for ind in train_index:
            data = np.load("{0}/{1}.npz".format(load_dir, folds[ind]))
            x_train.append(data["features"])
            y_train.append(data["labels"])
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        data = np.load("{0}/{1}.npz".format(load_dir, folds[test_index][0]))
        x_test = data["features"]
        y_test = data["labels"]

        return ((x_test, y_test), (x_train, y_train))

    print("Average 10 Folds Accuracy: {0}".format(np.mean(accuracies)))

@timeit
def train_and_evaluate_model():
    CnnModel = model.Model
    cnn_model = CnnModel()

    accuracies = []
    folds = np.array(['fold1', 'fold2', 'fold3', 'fold4',
                      'fold5', 'fold6', 'fold7', 'fold8',
                      'fold9', 'fold10'])

    load_dir = f"{CHOSEN_DATASET}/processed"
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(folds):
        x_train, y_train = [], []
        for ind in train_index:
            data = np.load("{0}/{1}.npz".format(load_dir, folds[ind]))
            x_train.append(data["features"])
            y_train.append(data["labels"])

        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        data = np.load("{0}/{1}.npz".format(load_dir, folds[test_index][0]))
        x_test = data["features"]
        y_test = data["labels"]

        cnn_model.train(x_train, y_train)
        l, a = cnn_model.accuracy(x_test, y_test)
        accuracies.append(a)
        print("Loss: {0} | Accuracy: {1}".format(l, a))

    print("Average 10 Folds Accuracy: {0}".format(np.mean(accuracies)))


# train_and_evaluate_model()

def read_esc_metadata():
    metadata = pd.read_csv(f'{ESC_50_META}/esc50.csv')
    print(metadata.head())
    # Making a new df with the necesary info
    metadata["verbose_category"] = metadata["category"]
    metadata["category"] = pd.Categorical(metadata["category"]).codes
    metadata = metadata.drop(columns=["fold", "target", "esc10", "src_file", "take"])
    print(metadata.head())
    count_category = metadata.groupby('category').count()
    # count_category.plot(kind="bar")
