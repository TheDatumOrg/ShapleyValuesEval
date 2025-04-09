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

    if method == 'xg':
        import xgboost

        model = xgboost.XGBRegressor(nthread=1)
        model.fit(X_train, y_train)

    elif method == 'dt':
        from sklearn import tree

        model = tree.DecisionTreeRegressor(min_samples_split=20, random_state=0)
        model.fit(X_train, y_train)

    elif method == 'rf':
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0)
        model.fit(X_train, y_train)

    elif method == 'svm':
        from sklearn import svm
        model = svm.SVR()
        model.fit(X_train, y_train)

    elif method == 'lr':
        from sklearn import linear_model

        model = linear_model.LinearRegression()
        model.fit(X_train, y_train)

    elif method == 'mlp':
        import os

        os.environ["OMP_NUM_THREADS"] = "1"

        from sklearn.neural_network import MLPRegressor
        import mkl
        mkl.set_num_threads(1)
        model = MLPRegressor(solver="lbfgs", random_state=0)
        model.fit(X_train, y_train)

    elif method == 'keras':
        import os
        import numpy as np
        import pandas as pd
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        os.environ["OMP_NUM_THREADS"] = "1"

        model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer with one neuron for regression
         ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
       
    else:
        raise ValueError("Invalid method name!")
        

    return model