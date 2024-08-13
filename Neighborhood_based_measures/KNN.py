from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

def Classify(Data,Label,tr):

    x_train, x_test, y_train, y_test = train_test_split(Data, Label, train_size=tr,
                                                        random_state=0)  # 70% for train and 30% for test


    KNeighborsClassifier(
        n_neighbors=5,          # The number of neighbours to consider
        weights='uniform',      # How to weight distances
        algorithm='auto',       # Algorithm to compute the neighbours
        leaf_size=30,           # The leaf size to speed up searches
        p=2,                    # The power parameter for the Minkowski metric
        metric='minkowski',     # The type of distance to use
        metric_params=None,     # Keyword arguments for the metric function
        n_jobs=None             # How many parallel jobs to run
    )

    clf = KNeighborsClassifier(p=1)
    y = preprocessing.LabelEncoder()
    y= y.fit_transform(y_train)
    clf.fit(x_train, y)
    predict = clf.predict(x_test)
    return predict,y_test