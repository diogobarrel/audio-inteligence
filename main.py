"""
Driver
"""
import logging
import glob
import os

from data import data_utils
from data.model import Model as CnnModel

from audio import signal
from audio.audio import AudioFile
from util.host import get_samples, CHOSEN_DATASET, CHOSEN_METADATA, DATAFRAME

import numpy as np


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

    def parse_audio_files(parent_dir, sub_dir, file_ext='*.wav'):
        features, labels = np.empty((0, 193)), np.empty(
            0)  # 193 => total features
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            audio_file = AudioFile(fn)
            audio_file.extract_features()
            ext_features = np.hstack(list(audio_file._feat.values()))
            features = np.vstack([features, ext_features])
            logging.warn(fn)
            labels = np.append(labels, int(fn.split('/')[9].split('-')[1]))
        return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int8)

    # Pre-process and extract feature from the data
    parent_dir = f'{CHOSEN_DATASET}audio'
    save_dir = f'{CHOSEN_DATASET}processed/'
    sub_dirs = np.array(['fold1', 'fold2', 'fold3', 'fold4',
                         'fold5', 'fold6', 'fold7', 'fold8',
                         'fold9', 'fold10'])

    for sub_dir in sub_dirs:
        features, labels = parse_audio_files(parent_dir, sub_dir)
        np.savez("{0}{1}".format(save_dir, sub_dir), features=features,
                 labels=labels)


def build_dataframe(dataframe):
    return data_utils.build_features_dataframe(
        dataframe, CHOSEN_DATASET, CHOSEN_METADATA
    )


"""
Justs gonna build dataframe and compile model
"""
# try:
# data = data_utils.read_dataframe(f"./{DATAFRAME}")
# if not data or data.empty:
#     df = build_dataframe(DATAFRAME)
#     data = data_utils.read_dataframe(f"./{DATAFRAME}")
# cnn_model = CnnModel(dataframe=data)
# cnn_model.train()
# cnn_model.accuracy()
# process_audio_files()
proccess_dataset()
