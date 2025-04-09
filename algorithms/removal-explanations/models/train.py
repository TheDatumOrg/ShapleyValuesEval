import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

def train(method, X_train, y_train):
    '''
    Train machine learning models over the given input data. 

    Args:
    method(String): Type of model to train
    X_train(np.array): Feature set
    y_train(np.array): Target variable

    Returns machine learning model

    '''
    if method == 'logr':
        # Logistic Regression
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()

    elif method == 'dt':
        from sklearn import tree
        model = tree.DecisionTreeClassifier(min_samples_split=20, random_state=0)

    elif method == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0)

    elif method == 'svm':
        # Support Vector Machines
        from sklearn.svm import LinearSVC
        model = LinearSVC()

    elif method == 'knn':
        # K-Nearest Neighbors
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
    
    elif method == 'nb':
        # Naive Bayes
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
    
    elif method == 'mlp':
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier()

    else:
        raise ValueError("Invalid method name!")

    model.fit(X_train, y_train)
    return model