"""
Driver
"""
from data import data, model

from audio import signal
from util.host import get_samples


def plot_samples():
    audio_samples = get_samples()
    ROW, COL = 1, 1
    for sample in audio_samples:
        # audio_file = AudioFile(sample)
        # print(audio_file.extract_mfcc_features())
        signal.spectogram(sample, index=f'{ROW}{COL}')
        signal.stft_spectogram(sample, index=f'{ROW+1}{COL}')
        # signal.draw_spec3(sample, index=f'{ROW+2}{COL}')
        COL += 1


def cnn():
    featdf = data.read_dataframe('./featdf.pkl')
    x_train, x_test, y_train, y_test, le, yy = data.split_data(featdf)
    cnn_model = model.build_model(x_train, x_test, yy)
    compiled_cnn_model = model.compile_model(cnn_model, x_test, y_test)


cnn()
