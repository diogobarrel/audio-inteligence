"""
Audio File basic handling
"""
import struct
import logging
import librosa
import librosa.display
import IPython.display as ipd
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


class AudioFile:
    """
    Base AudioFile class
    """
    file = None
    name = ''
    __file_type = 'wav'
    num_channels = 1
    sample_rate = 0
    bit_depth = 0
    audio = None
    stft = None
    _feat = {}
    __STFT_FRAME_SIZE = 1024
    __STFT_HOP_SIZE = 256

    def __init__(self, file: str):
        """
            By default, Librosa’s load function will convert the sampling rate to 22.05khz,
            as well as reducing the number of channels to 1(mono),
            and normalise the data so that the values will range from -1 to 1.
        """
        self.file = file
        self.name = file.split('/')[-1]
        self.audio, self.sample_rate = librosa.load(file)
        self.stft = librosa.stft(
            self.audio,
            n_fft=self.__STFT_FRAME_SIZE,
            hop_length=self.__STFT_HOP_SIZE)

        print(f'new AudioFile {self} {self.audio.shape} {self.sample_rate}')

    def extract_features(self):
        """
        Extracts handcrafted audio features from audio file
        """

        try:
            mfccs = librosa.feature.mfcc(y=self.audio, sr=self.sample_rate, n_mfcc=40)
            chroma = librosa.feature.chroma_stft(S=self.stft, sr=self.sample_rate)
            mel = librosa.feature.melspectrogram(self.audio, sr=self.sample_rate)
            contrast = librosa.feature.spectral_contrast(S=self.stft, sr=self.sample_rate)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(self.audio), sr=self.sample_rate)

            _mean = lambda x: np.mean(x, axis=0)
            self._feat = {
                'mfccs': _mean(mfccs.T),
                'chroma': _mean(chroma.T),
                'mel': _mean(mel.T),
                'contrast': _mean(contrast.T),
                'tonnetz': _mean(tonnetz.T),
            }

        except IOError as io_error:
            logging.exception(
                "Exception while parsing file: %s", self.file)
            logging.exception(io_error)

        except Exception as e:
            logging.exception(f'Failed parsing audio {self.name}')
            logging.exception(e)

    def plot_audio(self, index):
        fig, plots = plt.subplots(nrows=3, ncols=1, sharex=True)

        wave_plot = plots[0]
        wave_plot.set(title=f'{self.name} waveform')
        wave_plot.set(xlabel='Sample')
        wave_plot.set(ylabel='Amplitude')
        librosa.display.waveplot(self.audio, self.sample_rate, ax=wave_plot)
        wave_plot.label_outer()

        y_scale = np.abs(self.stft) ** 2
        D = librosa.amplitude_to_db(y_scale, ref=np.max)

        linear_spec_plot = plots[1]
        linear_spec_plot.set(
            title=f'{self.name} Linear-frequency power spectrogram')
        img = librosa.display.specshow(
            D, sr=self.sample_rate,
            hop_length=self.__STFT_HOP_SIZE,
            x_axis='time',
            y_axis='linear',
            ax=linear_spec_plot)
        linear_spec_plot.label_outer()

        log_spec_plot = plots[2]
        log_spec_plot.set(title=f'{self.name} Log-frequency power spectrogram')
        librosa.display.specshow(D, y_axis='log',
                                 sr=self.sample_rate,
                                 hop_length=self.__STFT_HOP_SIZE,
                                 x_axis='time',
                                 ax=log_spec_plot)
        log_spec_plot.label_outer()

        fig.tight_layout()
        fig.colorbar(img, ax=plots, format="%+2.f dB")
        fig.savefig(f'{index}_{self.name}.png')

    def display(self):
        return ipd.Audio(self.audio, rate=self.sample_rate)
