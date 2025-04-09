from rexplain import behavior, summary, removal
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import pickle
import argparse
from models.train import train
import time
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--method', type=str)
parser.add_argument('--strategy', type=str)
args = parser.parse_args()


def run(argus):
    dname, method, strategy, X_train, y_train, X_test, y_test, num_features, cat = argus[0], argus[1], argus[2], argus[3], argus[4], argus[5], argus[6], argus[7], argus[8]
    # dname, method, strategy, X_train, y_train, X_test, y_test, num_features = argus[0], argus[1], argus[2], argus[3], argus[4], argus[5], argus[6], argus[7]
    print(strategy)
    print(dname, method, num_features)
    os.makedirs('results/' + strategy + '/' + method, exist_ok = True)


    model = train(method,X_train, y_train)


    if not os.path.exists('results/' + strategy + '/' + method + '/' + dname + '.npy'):
        
        attr_dict = {} 
        
        if strategy == 'zero':
            attributions = list()
            start = time.time()

            values = np.zeros((1, num_features))
            extension = removal.DefaultExtension(values, model.predict)

            for sample in tqdm(X_test[:100, :]):
                game = behavior.PredictionGame(extension, sample)
                # attr = summary.exact(game)
                attr = summary.ShapleyValue(game)
                attributions.append(attr)

            end = time.time()

            attr_dict = np.array({
                'shap_values': np.array(attributions),
                'data': X_test[:100, :],
                'pred': model.predict(X_test[:100, :]),
                'compute_time': end - start
            })


        elif strategy == 'mean':
            attributions = list()
            start = time.time()
            values = np.array([np.mean(X_train, axis=0)])
            extension = removal.DefaultExtension(values, model.predict)

            for sample in tqdm(X_test[:100, :]):
                game = behavior.PredictionGame(extension, sample)
                attr = summary.ShapleyValue(game)
                attributions.append(attr)

            end = time.time()

            attr_dict = np.array({
                'shap_values': np.array(attributions),
                'data': X_test[:100, :],
                'pred': model.predict(X_test[:100, :]),
                'compute_time': end - start
            })

        elif strategy == 'marginal':
            attributions = list()
            start = time.time()
            data = X_train
            extension = removal.MarginalExtension(data, model.predict)
            for sample in tqdm(X_test[:100, :]):
                game = behavior.PredictionGame(extension, sample)
                attr = summary.ShapleyValue(game)
                attributions.append(attr)

            end = time.time()

            attr_dict = np.array({
                'shap_values': np.array(attributions),
                'data': X_test[:100, :],
                'pred': model.predict(X_test[:100, :]),
                'compute_time': end - start
            })


        elif strategy == 'uniform':
            attributions = list()
            start = time.time()
            data = X_train
            extension = removal.UniformExtension(data, cat, len(data), model.predict)
            for sample in tqdm(X_test[:100, :]):
                game = behavior.PredictionGame(extension, sample)
                attr = summary.ShapleyValue(game)
                attributions.append(attr)

            end = time.time()

            attr_dict = np.array({
                'shap_values': np.array(attributions),
                'data': X_test,
                'pred': model.predict(X_test[:100, :]),
                'compute_time': end - start
            })

        elif strategy == 'product':
            attributions = list()
            start = time.time()
            data = X_train
            extension = removal.ProductMarginalExtension(data, len(data), model.predict)
            for sample in tqdm(X_test[:100, :]):
                game = behavior.PredictionGame(extension, sample)
                attr = summary.ShapleyValue(game)
                attributions.append(attr)

            end = time.time()

            attr_dict = np.array({
                'shap_values': np.array(attributions),
                'data': X_test,
                'pred': model.predict(X_test[:100, :]),
                'compute_time': end - start
            })

        
        elif strategy == 'surrogate':
            attributions = list()

            # Split data
            X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

            # Normalize
            mean = X_t.mean(axis=0)
            std = X_t.std(axis=0)
            X_t = (X_t - mean) / std
            X_v = (X_v - mean) / std
            X_test_scaled = (X_test - mean) / std
            # model = train(method, X_t, y_t)


            start1 = time.time()

            from train_surrogate import train_surr
            model_lam = train_surr(X_t, X_v, model, num_features)

            start2 = time.time()

            from rexplain.utils import mseloss
            # Model extension
            conditional_extension = removal.ConditionalSupervisedExtension(model_lam)

            for ind in range(X_test_scaled.shape[0]):
                # Cooperative game
                game = behavior.PredictionLossGame(conditional_extension, X_test_scaled[ind:ind+1], y_test[ind:ind+1], mseloss)
                attr = summary.ShapleyValue(game)
                attributions.append(attr)
            
            end = time.time()

            attr_dict = np.array({
                'shap_values': np.array(attributions),
                'data': X_test_scaled,
                'pred': model.predict(X_test_scaled),
                'compute_time': end - start1,
                'compute_time_wo_surr': end - start2
            })


        elif strategy == 'remove_individual':
            attributions = list()
            start = time.time()
            data = X_train
            extension = removal.MarginalExtension(data, model.predict)
            start = time.time()

            for sample in tqdm(X_test[:100, :]):
                game = behavior.PredictionGame(extension, sample)
                attr = summary.RemoveIndividual(game)
                attributions.append(attr)

            end = time.time()
        
            attr_dict = np.array({
                'shap_values': np.array(attributions),
                'data': X_test[:100, :],
                'pred': model.predict(X_test[:100, :]),
                'compute_time': end - start
            })


        elif strategy == 'include_individual':
            attributions = list()
            start = time.time()
            data = X_train
            extension = removal.MarginalExtension(data, model.predict)
            for sample in tqdm(X_test[:100, :]):
                game = behavior.PredictionGame(extension, sample)
                attr = summary.IncludeIndividual(game)
                attributions.append(attr)

            end = time.time()
        
            attr_dict = np.array({
                'shap_values': np.array(attributions),
                'data': X_test[:100, :],
                'pred': model.predict(X_test[:100, :]),
                'compute_time': end - start
            })


        else:
            raise ValueError("Invalid shap method or Shap method not yet implemented")
        
        np.save('results/' + strategy + '/' + method + '/' + dname + '.npy', attr_dict)

    else:
        print("Required results already exist! Please check the results directory")


def identify_categorical_columns(df):
    # Create a boolean array where True represents the column is categorical
    categorical_columns = df.dtypes == 'object'
    
    # Convert the boolean array to integer (1 for categorical, 0 for non-categorical)
    boolean_array = categorical_columns.astype(int).to_numpy()
    
    return boolean_array




if __name__=="__main__":
    method = args.method
    argus = []

    dname = args.dataset
    data = pd.read_csv('github/datasets/classification/' + dname + '.csv')
    df = data.copy(deep=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    X = df.loc[:, df.columns != 'class']
    y = df.loc[:, 'class']

    raw_data = pd.read_csv("github/datasets/raw/classification/raw/" + dname + '.csv')
    # Identify categorical columns

    categorical = identify_categorical_columns(raw_data.loc[:, raw_data.columns != 'class'])
    # print(categorical_boolean_array)
    
    from sklearn.preprocessing import StandardScaler
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=0)
    num_features = X_train.shape[1]
    # argus.append([dname, method, args.shap_method, X_train, y_train, X_test, y_test, num_features, categorical])
    # argus.append([dname, method, args.shap_method, X_train, y_train, X_test, y_test, num_features])
    if num_features <= 16:
        argus.append([dname, method, args.strategy, X_train, y_train, X_test, y_test, num_features, categorical])
        run(argus[0])
    # else:
    #     print()
        

