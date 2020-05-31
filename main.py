"""
Driver
"""
from classify.svm import svm
from data_vis import class_vis
from test_data.prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()

print('Audio Inteligence ftw')
clf = svm.cft_custom_kernel()
class_vis.prettyPicture(clf, features_test, labels_test)
