"""
Driver
"""


def plot_samples():
    from audio.audio import AudioFile
    from audio import signal
    from util.host import get_samples
    ROW, COL = 1, 1

    audio_samples = get_samples()
    for sample in audio_samples:
        # audio_file = AudioFile(sample)
        # print(audio_file.extract_mfcc_features())
        signal.spectogram(sample, index=f'{ROW}{COL}')
        # signal.stft_spectogram(sample, index=f'{ROW+1}{COL}')
        # signal.draw_spec3(sample, index=f'{ROW+2}{COL}')
    COL += 1

from data.data import build_features_dataframe

build_features_dataframe('featdf.pkl')
