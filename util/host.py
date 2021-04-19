""" env stuff """
import os

HOME = os.environ['HOME'] + '/'
# DATASTES
WORKSPACE = 'projects/audio-inteligence/'
DATASETS = HOME + 'ufop/tcc/datasets/'

URBANSOUND_8K = DATASETS + 'UrbanSound8K'
URBANSOUND_8K_AUDIO = URBANSOUND_8K + '/audio'
URBANSOUND_8K_METADATA = URBANSOUND_8K + '/metadata/UrbanSound8K.csv'
FOLDERS = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7',
           'fold8', 'fold9', 'fold10']

# SAMPLES
SAMPLE_DIRECTORY=HOME + WORKSPACE + 'util/samples/'
SAMPLE_FILES=['17913-4-1-0.wav', '21684-9-0-50.wav', '7061-6-0-0.wav']

CHOSEN_DATASET = URBANSOUND_8K
CHOSEN_METADATA = URBANSOUND_8K_METADATA
DATAFRAME = "df2.pkl"

def get_samples() -> list:
    return [SAMPLE_DIRECTORY + file for file in SAMPLE_FILES]
