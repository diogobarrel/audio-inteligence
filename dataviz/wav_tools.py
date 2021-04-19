"""
Class created to handle wav files
"""
import os
import struct
import wave

import matplotlib.pyplot as plt
import numpy
import pandas

from util import utils


class WavFileHelper():
    """
    Main class created to manage audio: read sound file proprieties, play sound
    and transform it in digital signal
    """

    def read_file_props(self: WavFileHelper, file: str):
        """
        read file
        """
        wave_file = open(file, "rb")

        riff = wave_file.read(12)
        fmt = wave_file.read(36)

        num_channels_string = fmt[10:12]
        num_channels = struct.unpack('<H', num_channels_string)[0]

        sample_rate_string = fmt[12:16]
        sample_rate = struct.unpack("<I", sample_rate_string)[0]

        bit_depth_string = fmt[22:24]
        bit_depth = struct.unpack("<H", bit_depth_string)[0]

        return (num_channels, sample_rate, bit_depth)

    def signal(self, filename: str, sample_frequency: int = 16000):
        """ draw signal wave file and sectrum of signal """
        signal_wave = wave.open(filename, 'r')

        data = numpy.fromstring(signal_wave.readframes(sample_frequency),
                                dtype=numpy.int16)

        sig = signal_wave.readframes(-1)
        sig = numpy.fromstring(sig, 'Int16')
        return sig

    @utils.timeit
    def plot_wave_spectrum(self, signals, plots=1):
        """ plot wave and spectrum of the signal wave file """
        if signals and not isinstance(signals, list):
            sig = signals
            plt.figure(plots)

            a = plt.subplot(211)
            a.set_xlabel('time [s]')
            a.set_ylabel('sample value [-]')
            plt.plot(sig)

            c = plt.subplot(212)
            Pxx, freqs, bins, im = c.specgram(sig,
                                              NFFT=1024,
                                              Fs=16000,
                                              noverlap=900)
            c.set_xlabel('Time')
            c.set_ylabel('Frequency')
            plt.show()

        if signals:
            samples = list(range(len(signals)))

            for sample, sig in list(zip(samples, signals)):
                plt.figure(sample)
                a = plt.subplot(211)
                a.set_xlabel('time [s]')
                a.set_ylabel('sample value [-]')
                plt.plot(sig)

                c = plt.subplot(212)
                Pxx, freqs, bins, im = c.specgram(sig,
                                                  NFFT=1024,
                                                  Fs=16000,
                                                  noverlap=900)
                c.set_xlabel('Time')
                c.set_ylabel('Frequency')

            plt.show()


@utils.timeit
def load_audio_data():
    audiodata = []
    wavfilehelper = WavFileHelper()

    for index, row in metadata.iterrows():

        file_name = os.path.join(os.path.abspath(urbansound8k_audio),
                                 'fold' + str(row["fold"]) + '/',
                                 str(row["slice_file_name"]))

        data = wavfilehelper.read_file_props(file_name)
        audiodata.append(data)

    # Convert into a Panda dataframe
    audiodf = pandas.DataFrame(
        audiodata, columns=['num_channels', 'sample_rate', 'bit_depth'])

    return audiodf


@utils.timeit
def load_features_data():
    """Load files in a list of paths and extract their audio features


    Returns:
        features_df: pandas.Dataframe
    """
    features = []
    wavfilehelper = WavFileHelper()

    # Iterate through each sound file and extract the features
    for index, row in metadata.iterrows():

        file_name = os.path.join(os.path.abspath(urbansound8k_audio),
                                 'fold' + str(row["fold"]) + '/',
                                 str(row["slice_file_name"]))

        class_label = row["class"]
        data = extract_features(file_name)

        features.append([data, class_label])

    # Convert into a Panda dataframe
    featuresdf = pandas.DataFrame(features, columns=['feature', 'class_label'])

    return featuresdf
