""" support vector machine """
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from test_data.prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()


def classify_fit_test():

    clf = SVC(kernel="linear")
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    accurracy = accuracy_score(pred, labels_test)
    print(accurracy)
    return clf

def cft_custom_kernel():

    clf = SVC(kernel="rbf", gamma=100.0, C=2.0)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    accurracy = accuracy_score(pred, labels_test)
    print(accurracy)
    return clf