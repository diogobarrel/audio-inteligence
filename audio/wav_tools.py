"""
Class created to handle wav files
"""
import logging
from IPython.lib.display import Audio

import librosa
from librosa.core import audio
from audio.audio import AudioFile
import numpy as np

from util import utils


class Augment:
    audio_file: AudioFile
    def __new__(self, audio_file, f, **kw) -> AudioFile:
        try:
            self.audio_file = AudioFile(audio_file.file)
            augment = getattr(Augment, f)
            augment(self.audio_file, **kw)
            return self.audio_file

        except Exception as e:
            logging.exception(e)

    @utils.timeit
    def noise_injection(audio_file, noise_factor=None):
        # Permissible noise factor value = x > 0.004
        noise_factor = 0.02
        data = audio_file.data
        noise = np.random.randn(len(data))
        logging.warn(noise)
        augmented_data = data + noise_factor * noise
        augmented_data = augmented_data.astype(type(data[0]))

        # Cast back to same data type
        audio_file.data = augmented_data

    @utils.timeit
    def time_shifting(audio_file):
        shift = int(audio_file.sample_rate/20)
        augmented_data = np.roll(audio_file.data, shift)
        # Set to silence for heading/tailing
        audio_file.data = augmented_data

    @utils.timeit
    def time_stretching(audio_file):
        """
        The process of changing the speed/duration of sound without affecting the pitch of sound.
        This can be achieved using librosa’s time_stretch function.
        Time_stretch function takes wave samples and a factor by which to stretch as inputs.
        I found that this factor should be 0.4 since it has a small difference with original sample.
        """
        factor = 0.4  # Permissible factor values = 0 < x < 1.0
        augmented_data = librosa.effects.time_stretch(audio_file.data, factor)
        # librosa.output.write_wav('./time_stech.wav',wav_time_stch,sr)
        audio_file.data = augmented_data

    def pitch_shifting(audio_file):
        """
        Pitch Shifting
        It is an implementation of pitch scaling used in musical instruments.
        It is a process of changing the pitch of sound without affect it’s speed.
        Again we are going to use librosa’s pitch_shift function.
        It takes wave samples, sample rate and number of steps through which pitch must be shifted.
        """

        N_STEPS = -5  # Permissible factor values = -5 <= x <= 5 ???
        augmented_data = librosa.effects.pitch_shift(
            audio_file.data, audio_file.sampling_rate, n_steps=-N_STEPS)
        # librosa.output.write_wav('./pitch_shift.wav',wav_pitch_sf,sr)
        audio_file.data = augmented_data
