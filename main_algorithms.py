from models.classification.train import train
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import multiprocessing
import pandas as pd
import argparse
from algorithms._exact import Exact


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--method', type=str)
parser.add_argument('--shap_method', type=str)
args = parser.parse_args()



def run(argus):
    dname, method, shap_method, X_train, y_train, X_test, y_test, num_features, cat = argus[0], argus[1], argus[2], argus[3], argus[4], argus[5], argus[6], argus[7], argus[8]
    print(shap_method)
    print(dname, method, num_features)
    os.makedirs('github/results/classification/' + shap_method + '/' + method, exist_ok = True)

    model = train(method, X_train, y_train)
        
    if not os.path.exists('github/results/classification/' + shap_method + '/' + method + '/' + dname + '.npy'):

        if shap_method == 'sampling':
            from algorithms._sampling import Sampling
            explainer = Sampling(model.predict, num_features)
            attributions = explainer(X_test, X_train, num_evals=1000)
        
        elif shap_method == 'raw':
            from algorithms._raw import Raw
            explainer = Raw(method, model, num_features)
            attributions = explainer(X_train, y_train, X_test, y_test[:100])

        elif shap_method == 'sampling_anti':
            from algorithms._permutation import Permutation
            explainer = Permutation(model.predict, num_features)
            attributions = explainer(X_test, X_train, num_evals=1000)

        elif shap_method == 'mle':
            from algorithms._mle import Multilinear
            explainer = Multilinear(model.predict, num_features)
            attributions = explainer(X_test, X_train, num_evals=1000)

        elif shap_method == 'mle_anti':
            from algorithms._mle import Multilinear
            explainer = Multilinear(model.predict, num_features)
            attributions = explainer(X_test, X_train, num_evals=1000, is_antithetic=True)

        elif shap_method == 'ces':
            from algorithms._ces import CES
            explainer = CES(model.predict, num_features)
            attributions = explainer(X_test, X_train, smoothing=0.0000001, max_perms=1000)

        elif shap_method == 'cohort':
            from algorithms._cohortShapley import CohortShapley
            explainer = CohortShapley(model.predict, X_train, cat)
            attributions = explainer(X_test)

        elif shap_method == 'kernel':
            from algorithms._kernel import Kernel
            explainer = Kernel(model.predict, num_features)
            attributions = explainer(X_test, X_train, num_evals=1000)

        elif shap_method == 'kernel_anti':
            from algorithms._kernel import Kernel
            explainer = Kernel(model.predict, num_features)
            attributions = explainer(X_test, X_train, num_evals=1000, is_antithetic=True)

        elif shap_method == 'sgd':
            from algorithms._kernelSGD import SGDShapley
            explainer = SGDShapley(model.predict, num_features, y_train.max())
            attributions = explainer(X_test, np.array([np.mean(X_train, axis=0)]), num_evals=1000)

        elif shap_method == 'sgd_anti':
            from algorithms._kernelSGD import SGDShapley
            explainer = SGDShapley(model.predict, num_features, y_train.max())
            attributions = explainer(X_test, np.array([np.mean(X_train, axis=0)]), num_evals=1000)

        # To do: Implement model specific approaches
            
        elif shap_method == 'tree_path_dependent':
            from algorithms._tree import Tree
            explainer = Tree(model, num_features)
            attributions = explainer(X_test, X_train, fp='correlation_dependent')

        elif shap_method == 'tree_interventional':
            from algorithms._tree import Tree
            explainer = Tree(model, num_features)
            attributions = explainer(X_test, X_train)
        
        elif shap_method == 'exact_zero':
            from algorithms._exact import Exact
            explainer = Exact(model.predict, num_features)
            attributions = explainer(X_train, y_train, X_test, y_test, np.zeros(X_train[0].shape), categorical=cat, flag=True)

        elif shap_method == 'exact_mean':
            from algorithms._exact import Exact
            explainer = Exact(model.predict, num_features)
            attributions = explainer(X_train, y_train, X_test, y_test, np.mean(X_train, axis=0), categorical=cat, flag=True)

        elif shap_method == 'exact_rand':
            from algorithms._exact import Exact
            explainer = Exact(model.predict, num_features)
            attributions = explainer(X_train, y_train, X_test, y_test, X_train, categorical=cat, flag=True)

        
        elif shap_method == 'exact_cond':
            from algorithms._exact import Exact
            explainer = Exact(model.predict, num_features)
            attributions = explainer(X_train, y_train, X_test, y_test, X_train, categorical=cat, flag=True)


        elif shap_method == 'linear_independent':
            from algorithms._linear import Linear
            explainer = Linear(model, num_features)
            attributions = explainer(X_test, X_train)
        
        elif shap_method == 'linear_correlated':
            from algorithms._linear import Linear
            explainer = Linear(model, num_features)
            attributions = explainer(X_test, X_train, fp='correlation_dependent')
        
        elif shap_method == 'deep':
            import time, shap
            explainer = shap.DeepExplainer(model, X_train)
            start = time.time()
            attr = explainer.shap_values(X_test)
            end = time.time()

            attributions = np.array([{
                'shap_values': np.array(attr),
                'data': X_test,
                'preds': model.predict(X_test),
                'compute_time': end - start
            }])
        

        else:
            raise ValueError("Invalid shap method or Shap method not yet implemented")
        
        np.save('github/results/classification/' + shap_method + '/' + method + '/' + dname + '.npy', attributions)

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

    if args.shap_method in ['raw', 'exact_zero', 'exact_mean', 'exact_rand', 'exact_cond']:
        if num_features <= 16:
            argus.append([dname, method, args.shap_method, X_train, y_train, X_test, y_test, num_features, categorical])
        else:
            print("Raw and exact explainers cannot work for datasets with more than 16 features")
    elif args.shap_method in ['tree_path_dependent', 'tree_interventional']:
        if method in ['dt', 'rf']:
            argus.append([dname, method, args.shap_method, X_train, y_train, X_test, y_test, num_features, categorical])
        else:
            print("Unsupported model type!")
    else:
        argus.append([dname, method, args.shap_method, X_train, y_train, X_test, y_test, num_features, categorical])



    run(argus[0])


    
    
    
        


