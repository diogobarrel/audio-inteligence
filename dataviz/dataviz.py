"""
here i will keep stuff to plot my audio files in different ways
"""
import os
import pandas

from utils.host import URBANSOUND_8K
from wav_tools import WavFileHelper

# draw sample1 waves and spectogram


def plot_spectrum():
    """  """
    urbansound8k_sample = URBANSOUND_8K + '/sample/'
    sample_metadata = pandas.read_csv(
        URBANSOUND_8K + '/sample/sample_metadata.csv')

    wavfilehelper = WavFileHelper()
    signals = []
    for _, row in sample_metadata.iterrows():

        file_name = os.path.join(os.path.abspath(urbansound8k_sample),
                                 'sample1' + '/', str(row["slice_file_name"]))

        sig = wavfilehelper.signal(file_name)
        signals.append(sig)

    wavfilehelper.plot_wave_spectrum(signals)
