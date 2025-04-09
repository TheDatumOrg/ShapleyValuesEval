from algorithms._explainer import Explainer
from typing import Any
from itertools import combinations, chain
import math
import numpy as np
from tqdm import tqdm
import time
from models.classification.train import train
import scipy


import warnings
warnings.filterwarnings('ignore')

class Raw(Explainer):
    '''
    Raw Shap: True estimations of Shapley values
    Training exponential number of models
    '''

    def __init__(self, method, model, num_features):
        super().__init__(model, num_features)
        self.method = method
        self.features = set(range(num_features))
        self.max_features = 16
    
        if self.num_features > self.max_features: 
            raise RuntimeError(
                f"Explaining {self.num_features} features is too slow, "
                f"please only use this estimator for <= {self.max_features}"
            )
    
    
    def _powerset(self, iterable):
        s = list(iterable)
        return chain.from_iterable(
            combinations(s, r) for r in range(len(s) + 1)
        )

    def _subset_on_off(self, subset, feature):

        subset_off = np.zeros(self.num_features)
        if subset:
            subset_off[np.array(subset)] = 1

        subset_on = np.copy(subset_off)
        subset_on[feature] = 1

        return (subset_on.astype("bool"), subset_off.astype("bool"))
    

    def _single_feature(self, feature, X_train, y_train, explicand):
        sizes = []
        subsets = self._powerset(self.features - set([feature]))
        preds_on = np.empty(shape=[explicand.shape[0], 0])
        preds_off = np.empty(shape=[explicand.shape[0], 0])
        # print(preds_off)
        for i, subset in enumerate(subsets):
            # print(subset)
            subset_on, subset_off = self._subset_on_off(subset, feature)
            size = subset_off.sum()
            sizes.append(size)

            X_train_off = X_train[:, subset_off]
            X_train_on = X_train[:, subset_on]
            
            explicand_off = explicand[:, subset_off]
            explicand_on = explicand[:, subset_on]

            if size == 0:
                y_avg = np.mean(self.model.predict(X_train))
                y_off = np.repeat(y_avg, explicand.shape[0]).reshape(-1,1)
                # print(y_off)
                
            else:
                model_off = train(self.method, X_train_off, y_train)
                y_off = model_off.predict(explicand_off).reshape(-1,1)

            model_on = train(self.method, X_train_on, y_train)
            y_on = model_on.predict(explicand_on).reshape(-1,1)

            # preds_on = np.append(preds_on, y_on)
            # preds_off = np.append(preds_off, y_off)
            preds_on = np.hstack((preds_on, y_on))
            preds_off = np.hstack((preds_off, y_off))

        # Compute marginal contributions
        weights = 1 / scipy.special.comb(
            self.num_features - 1, np.array(sizes)
        )
        weights /= self.num_features
        diff = np.array(preds_on)-np.array(preds_off)
  
        return np.dot(diff,weights)

    



    def _explain(self, X_train, y_train, explicands, y_test):
        start = time.time()
        phis = []
        
        for ind in tqdm(range(explicands.shape[0])):

            # explicand = explicands
            explicand = explicands[ind:ind+1, :]
            
            phi = np.zeros(explicand.shape)

            # for i in tqdm(range(self.num_features)):
            for i in range(self.num_features):
                phi[:, i] = self._single_feature(i, X_train, y_train, explicand)
                # print(i, phi)                
        
            phis.append(phi)
        end = time.time()
        

        attributions = np.array([{
            'shap_values': np.array(phis),
            'num_features': self.num_features,
            'train': {'X': X_train, 'y': y_train},
            'test':  {'X': explicands, 'y': y_test},
            'pred': self.model.predict(explicands),
            'compute_time': end - start
        }])

        return attributions

    def __call__(self, X_train, y_train, X_test, y_test):
        return self._explain(X_train, y_train, X_test, y_test)

 
        

