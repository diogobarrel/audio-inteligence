"""
Audio File basic handling
"""
import struct
import logging
import librosa
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
    __file_type = 'wav'
    num_channels = 1
    sample_rate = 0
    bit_depth = 0
    audio = None
    sound_info = None
    stft = None
    _feat = {}

    def __init__(self, file: str):
        self.file = file
        print(f'new AudioFile {self}')

    def save(self, path):
        return librosa.output.write_wav(path, self.audio, self.sample_rate)

    def extract_features(self):
        """
        Extracts handcrafted audio features from audio file
        """

        try:
            """
            By default, Librosa’s load function will convert the sampling rate to 22.05khz,
            as well as reducing the number of channels to 1(mono),
            and normalise the data so that the values will range from -1 to 1.
            """
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

            wav = wave.open(self.file, 'r')
            frames = wav.readframes(-1)
            self.sound_info = np.fromstring(frames, 'int16')
            self.frame_rate = wav.getframerate()
            wav.close()

        except IOError as io_error:
            logging.exception(
                "Exception while parsing file: %s", self.file)
            logging.exception(io_error)

        except Exception as e:
            logging.exception(f'Failed parsing audio {self.file}')
            logging.exception(e)


    def spectogram(self, index: str = 'X'):
        sound_info, frame_rate = self.sound_info, self.frame_rate
        plt.figure(num=None, figsize=(19, 12))
        plt.subplot(111)
        plt.title('spectrogram of %r' % self.file)
        plt.specgram(sound_info, Fs=frame_rate)
        plt.savefig(f'spectrogram{index}.png')

    def wave_form(self, index):
        # Plot the signal read from wav file
        plt.subplot(211)
        plt.title(f'waveform of a wav file {self.file}')
        plt.plot(self.sound_info)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

        plt.savefig(f'waveform{index}.png')
    
    def display(self):
        return ipd.Audio(self.audio)
