def classify(features_train, labels_train):
    ### import the sklearn module for GaussianNB
    from pprint import pprint
    from sklearn.naive_bayes import GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    ### your code goes here!
    X, y = features_train, labels_train
    # X_train, X_test, y_train, y_test = train_test_split(X,
                                                        # y,
                                                        # test_size=0.5,
                                                        # random_state=0)
    gnb = GaussianNB()

    model = gnb.fit(X, y)
    pred = gnb.predict(X)
    # pprint(pred)
    return model