""" env stuff """
import os

HOME = os.environ['HOME'] + '/'
# DATASTES
WORKSPACE = 'projects/audio-inteligence/'
DATASETS = HOME + 'tcc/datasets/'

URBANSOUND_8K = DATASETS + 'UrbanSound8K/'
URBANSOUND_8K_AUDIO = URBANSOUND_8K + 'audio/'
URBANSOUND_8K_METADATA = URBANSOUND_8K + 'metadata/UrbanSound8K.csv'

# SAMPLES
SAMPLE_DIRECTORY = HOME + WORKSPACE + 'util/samples/'
SAMPLE_FILES = ['17913-4-1-0.wav', '21684-9-0-50.wav', '7061-6-0-0.wav']

def get_samples() -> list:
    return [SAMPLE_DIRECTORY + file for file in SAMPLE_FILES]
