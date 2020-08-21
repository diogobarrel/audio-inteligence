"""
Audio File base
"""
import struct
import logging

import librosa
import numpy as np


class AudioFile:
    file = None

    def __init__(self, file: str):
        self.file = file
        print(self)

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

    def extract_features(self) -> list(int):
        """
        Extracts mfcc and mfccs from audio file
        """

        try:

            # converts the sampling rate to 22.05 KHz
            # normalise the data so the bit-depth values range between -1 and 1
            # flattens the audio channels into mono
            audio, sample_rate = librosa.load(
                self.file, res_type='kaiser_fast')

            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccsscaled = np.mean(mfccs.T, axis=0)

        except Exception as e:
            logging.exception(
                "Error encountered while parsing file: %s", self.file)
            return None

        return mfccsscaled
