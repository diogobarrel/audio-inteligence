"""
Class created to handle wav files
"""
import logging

import librosa

import numpy as np
import pandas as pd

from util import utils

@utils.timeit
def noise_injection(audio_file, noise_factor=None):
    
    # Permissible noise factor value = x > 0.004
    noise_factor = 0.07
    data = audio_file.sound_info
    noise = np.random.randn(len(data))
    logging.warn(noise)
    augmented_data = data + noise_factor * noise
    augmented_data = augmented_data.astype(type(data[0]))

    # Cast back to same data type
    return augmented_data


@utils.timeit
def time_shifting(audio_file):
    shift = int(audio_file.sample_rate/20)
    augmented_wav = np.roll(audio_file.audio, shift)
    # Set to silence for heading/tailing
    return augmented_wav;


@utils.timeit
def time_stretching(wav):
    """
    The process of changing the speed/duration of sound without affecting the pitch of sound.
    This can be achieved using librosa’s time_stretch function.
    Time_stretch function takes wave samples and a factor by which to stretch as inputs.
    I found that this factor should be 0.4 since it has a small difference with original sample.
    """

    factor = 0.4 # Permissible factor values = 0 < x < 1.0
    augmented_wav = librosa.effects.time_stretch(wav.audio,factor)
    # librosa.output.write_wav('./time_stech.wav',wav_time_stch,sr)
    return augmented_wav

def pitch_shifting(wav):
    """
    Pitch Shifting
    It is an implementation of pitch scaling used in musical instruments.
    It is a process of changing the pitch of sound without affect it’s speed.
    Again we are going to use librosa’s pitch_shift function.
    It takes wave samples, sample rate and number of steps through which pitch must be shifted.
    """
    
    N_STEPS = -5 # Permissible factor values = -5 <= x <= 5 ???
    augmented_wav = librosa.effects.pitch_shift(wav.audio, wav.sampling_rate,n_steps=-N_STEPS)
    # librosa.output.write_wav('./pitch_shift.wav',wav_pitch_sf,sr)
    return augmented_wav
