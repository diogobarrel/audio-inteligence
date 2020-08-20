"""
draw sample1 waves and spectogram
"""
import os
import pandas

from wav_tools import WavFileHelper
from host.features import host_path

urbansound8k_sample = host_path + 'UrbanSound8K/sample/'
sample_metadata = pandas.read_csv(host_path +
                                  'UrbanSound8K/sample/sample_metadata.csv')

wavfilehelper = WavFileHelper()
signals = []
for index, row in sample_metadata.iterrows():

    file_name = os.path.join(os.path.abspath(urbansound8k_sample),
                             'sample1' + '/', str(row["slice_file_name"]))

    sig = wavfilehelper.signal(file_name)
    signals.append(sig)

wavfilehelper.plot_wave_spectrum(signals)
