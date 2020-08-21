"""
data manipulation and feature extraction
"""

import os

import pandas as pd
import numpy as np

from audio.audio import AudioFile
from util.host import URBANSOUND_8K_AUDIO, URBANSOUND_8K_METADATA


def build_features_dataframe():
    features = []
    # Iterate through each sound file and extract the features
    metadata = pd.read_csv(URBANSOUND_8K_METADATA)
    for _, row in metadata.iterrows():

        file_name = os.path.join(os.path.abspath(
            URBANSOUND_8K_AUDIO), 'fold'+str(row["fold"])+'/', str(row["slice_file_name"]))

        class_label = row["class"]
        print(file_name, class_label)
        audio_file = AudioFile(file_name)

        data = audio_file.extract_features()
        print(data)

        features.append([data, class_label])

    # Convert into a Panda dataframe
    featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

    print('Finished feature extraction from ', len(featuresdf), ' files')
