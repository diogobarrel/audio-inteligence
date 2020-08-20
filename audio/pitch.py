""" lets make a pith:
https://www.youtube.com/watch?v=_FatxGN3vAM
"""
import numpy
import simpleaudio

from utils import timeit

@timeit
def pitch():
    """
    creates and play a random pitch
    """
    frequency = 440  # Our played note will be 440 Hz
    fs = 44100  # 44100 samples per second
    seconds = 3  # Note duration of 3 seconds

    # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
    t = numpy.linspace(0, seconds, seconds * fs, False)

    # Generate a 440 Hz sine wave
    note = numpy.sin(frequency * t * 2 * numpy.pi)

    # Ensure that highest value is in 16-bit range
    audio = note * (2**15 - 1) / numpy.max(numpy.abs(note))
    # Convert to 16-bit data
    audio = audio.astype(numpy.int16)

    # Start playback
    play_obj = simpleaudio.play_buffer(audio, 1, 2, fs)

    # Wait for playback to finish before exiting
    play_obj.wait_done()
