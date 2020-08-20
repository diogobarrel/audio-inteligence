import simpleaudio
from utils import timeit

@timeit
def play_sound(self, file_path):
    """
    WAV files contain a sequence of bits representing the raw audio data,
    as well as headers with metadata in RIFF (Resource Interchange File Format)
    format.

    For CD recordings, the industry standard is to store each audio
    sample (an individual audio datapoint relating to air pressure) as a
    16-bit value, at 44100 samples per second.

    To reduce file size, it may be sufficient to store some recordings
    (for example of human speech) at a lower sampling rate, such as
    8000 samples per second, although this does mean that higher sound
    frequencies may not be as accurately represented.

    A few of the libraries discussed in this tutorial play and record bytes objects
    whereas others use NumPy arrays to store raw audio data. 

    @realpython
    """
    if not file_path:
        return

    wave_obj = simpleaudio.WaveObject.from_wave_file(file_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()  # Wait untill the sound has finished playing

