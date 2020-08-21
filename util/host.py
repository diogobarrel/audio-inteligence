""" env stuff """

# DATASTES
URBANSOUND_8K = '/home/barrel/tcc/datasets/UrbanSound8K'

# SAMPLES
SAMPLE_DIRECTORY = "/home/barrel/projects/audio-inteligence/util/samples/"
SAMPLE_FILES = ['17913-4-1-0.wav', '21684-9-0-50.wav', '7061-6-0-0.wav']

def get_samples() -> list:
    return [SAMPLE_DIRECTORY + file for file in SAMPLE_FILES]
