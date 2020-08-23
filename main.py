"""
Driver
"""

from audio.audio import AudioFile
from audio import signal
from util.host import get_samples

audio_samples = get_samples()

ROW, COL = 1, 1

for sample in audio_samples:
    # audio_file = AudioFile(sample)
    # print(audio_file.extract_mfcc_features())
    signal.spectogram(sample, index=f'{ROW}{COL}')
    # signal.stft_spectogram(sample, index=f'{ROW+1}{COL}')
    # signal.draw_spec3(sample, index=f'{ROW+2}{COL}')
    COL += 1
