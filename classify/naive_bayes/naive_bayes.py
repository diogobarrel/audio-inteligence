#!/usr/bin/python

""" Complete the code in ClassifyNB.py with the sklearn
    Naive Bayes classifier to classify the terrain data.
    
    The objective of this exercise is to recreate the decision 
    boundary found in the lesson video, and make a plot that
    visually shows the decision boundary """

import numpy as np
import pylab as pl

from test_data.prep_terrain_data import makeTerrainData
from data_vis.class_vis import prettyPicture, output_image
from ClassifyNB import classify

from sklearn.metrics import accuracy_score

def base_nb():
    features_train, labels_train, features_test, labels_test = makeTerrainData()
    # You will need to complete this function imported from the ClassifyNB script.
    # Be sure to change to that code tab to complete this quiz.
    clf = classify(features_train, labels_train)
    pred = clf.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    print('Naive Bayes accuracy: ', accuracy)

    ### draw the decision boundary with the text points overlaid
    prettyPicture(clf, features_test, labels_test)
    output_image("test.png", "png", open("test.png", "rb").read())