"""
Class created to handle wav files
"""

import librosa

import numpy as np
import pandas as pd

from util import utils

@utils.timeit
def noise_injection(wav, noise_factor):
    noise = np.random.randn(len(wav))
    augmented_wav = wav + noise_factor * noise

    # Cast back to same data type
    return augmented_wav.astype(type(wav[0]))


@utils.timeit
def time_shifting(wav, shift_direction):
    SHIFT_MAX = wav.sampling_rate/10

    shift_seed = np.random.randint(wav.sampling_rate * SHIFT_MAX)
    if shift_direction == 'right':
        shift = -shift_seed

    elif self.shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift_seed

    augmented_wav = np.roll(wav.audio, shift)
    # Set to silence for heading/tailing
    if shift > 0:
        augmented_wav[:shift] = 0
    else:
        augmented_wav[shift:] = 0
    return augmented_wav.astype(type(wav.audio[0]))


@utils.timeit
def time_stretching(wav)
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
