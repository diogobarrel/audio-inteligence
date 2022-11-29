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
    __STFT_FRAME_SIZE = 512
    __STFT_HOP_SIZE = 128

    def __init__(self, file: str):
        """
            By default, Librosa’s load function will convert the sampling rate to 22.05khz,
            as well as reducing the number of channels to 1(mono),
            and normalise the data so that the values will range from -1 to 1.
        """
        self.file = file
        self.name = file.split('/')[-1]
        self.data, self.sample_rate = librosa.load(file)
        self.stft = librosa.stft(
            self.data,
            n_fft=self.__STFT_FRAME_SIZE,
            hop_length=self.__STFT_HOP_SIZE)

        print(f'new AudioFile {self} {self.data.shape} {self.sample_rate}')

    def extract_features(self):
        """
        Extracts handcrafted audio features from audio file
        """

        try:
            mfccs = librosa.feature.mfcc(y=self.data, sr=self.sample_rate, n_mfcc=40)
            chroma = librosa.feature.chroma_stft(S=self.stft, sr=self.sample_rate)
            mel = librosa.feature.melspectrogram(self.data, sr=self.sample_rate)
            contrast = librosa.feature.spectral_contrast(S=self.stft, sr=self.sample_rate)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(self.data), sr=self.sample_rate)

            _mean = lambda x: np.mean(x.T, axis=0)
            self._feat = {
                'mfccs': _mean(mfccs),
                'chroma': _mean(chroma),
                'mel': _mean(mel),
                'contrast': _mean(contrast),
                'tonnetz': _mean(tonnetz),
            }

        except IOError as io_error:
            logging.exception(
                "Exception while parsing file: %s", self.file)
            logging.exception(io_error)

        except Exception as e:
            logging.exception(f'Failed parsing audio {self.name}')
            logging.exception(e)

    def display(self):
        return ipd.Audio(self.data, rate=self.sample_rate)
