"""
Audio File basic handling
"""
import struct
import logging
import librosa
import numpy as np


class AudioFile:
    """
    Base AudioFile class
    """
    file = None
    __file_type = 'wav'
    audio = None
    sample_rate = 0
    stft = None
    _feat = {}

    def __init__(self, file: str):
        self.file = file
        print(f'new AudioFile {self}')

    def read_props(self) -> (int, int, int):
        """
        Reads audio file and return its number of channels,
        sample_rate and bit_depth.
        """
        audio_file = open(self.file, "rb")

        # riff = audio_file.read(12)
        fmt = audio_file.read(36)

        num_channels_string = fmt[10:12]
        num_channels = struct.unpack('<H', num_channels_string)[0]

        sample_rate_string = fmt[12:16]
        sample_rate = struct.unpack("<I", sample_rate_string)[0]

        bit_depth_string = fmt[22:24]
        bit_depth = struct.unpack("<H", bit_depth_string)[0]

        return (num_channels, sample_rate, bit_depth)

    def extract_features(self) -> list:
        """
        Extracts mfcc and mfccs from audio file
        """

        try:
            # converts the sampling rate to 22.05 KHz
            # normalise the data so the bit-depth values range between -1 and 1
            # flattens the audio channels into mono
            audio, sample_rate = librosa.load(self.file)
            stft = np.abs(librosa.stft(audio))
            mfccs = np.mean(librosa.feature.mfcc(
                y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate).T, axis=0)
            mel = np.mean(librosa.feature.melspectrogram(
                audio, sr=sample_rate).T, axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(
                S=stft, sr=sample_rate).T, axis=0)
            tonnetz = np.mean(librosa.feature.tonnetz(
                y=librosa.effects.harmonic(audio), sr=sample_rate).T, axis=0)

            self.audio = audio
            self.sample_rate = sample_rate
            self.stft = stft
            self._feat = {
                'mfccs': mfccs,
                'chroma': chroma,
                'mel': mel,
                'contrast': contrast,
                'tonnetz': tonnetz,
            }

        except IOError as io_error:
            logging.exception(
                "Exception while parsing file: %s", self.file)
            logging.exception(io_error)
            return None
        except Exception as e:
            logging.exception(e)
