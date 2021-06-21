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

ESC_50 = DATASETS + 'ESC-50'
ESC_50_AUDIO = ESC_50 + '/audio'
ESC_50_META = ESC_50 + '/meta'

# SAMPLES
SAMPLE_DIRECTORY=HOME + WORKSPACE + 'util/samples/'
SAMPLE_FILES=['gunshots.wav', 'jackhammer.wav', 'sanfona.wav']

CHOSEN_DATASET = URBANSOUND_8K
CHOSEN_METADATA = URBANSOUND_8K_METADATA
DATAFRAME = "df2.pkl"

def get_samples() -> list:
    return [SAMPLE_DIRECTORY + file for file in SAMPLE_FILES]
