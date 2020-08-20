"""
Machine learning for sound classification attemp
@author: nagihuin
"""

from wav_tools import load_audio_data, load_features_data

audio = load_audio_data()
print(audio)
features = load_features_data()
print(features)
features.to_pickle('feat.pkl')
print('Done!')
