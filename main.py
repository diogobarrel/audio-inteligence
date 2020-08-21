"""
Driver
"""
from audio.audio import AudioFile

from util.host import get_samples

audio_samples = get_samples()

for sample in audio_samples:
    audio_file = AudioFile(sample)
    print(audio_file.extract_features())
