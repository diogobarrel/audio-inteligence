"""
Driver
"""
import logging

from data import data_utils
from data.model import Model as CnnModel

from audio import signal
from util.host import get_samples, URBANSOUND_8K, CHOSEN_DATASET, CHOSEN_METADATA, DATAFRAME

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


def build_dataframe(dataframe):
    return data_utils.build_features_dataframe(
        dataframe, CHOSEN_DATASET, CHOSEN_METADATA
    )


"""
Justs gonna build dataframe and compile model
"""
# try:
data = data_utils.read_dataframe(f"./{DATAFRAME}")
if not data or data.empty:
    df = build_dataframe(DATAFRAME)
    data = data_utils.read_dataframe(f"./{DATAFRAME}")
# cnn_model = CnnModel(dataframe=data)
# cnn_model.train()
# cnn_model.accuracy()
#process_audio_files()
