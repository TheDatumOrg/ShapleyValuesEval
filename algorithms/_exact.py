from typing import Any
from algorithms._explainer import Explainer
import numpy as np
from itertools import chain, combinations
import scipy
import time
from algorithms.helper import similarity_ces
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class Exact(Explainer):
    """Exact estimation (expoential in the number of features)"""
 
 
    def __init__(self, model, num_features):

        super().__init__(model, num_features)
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

    def _single_feature(self, feature, explicand, baselines):
        # print("SC 0: explicand", explicand)
    
        masked_samples = np.repeat(baselines, 2 ** self.num_features, 0)
        sizes = []

        subsets = self._powerset(self.features - set([feature]))

        eval = 0
        for i, subset in enumerate(subsets):
            eval += 1

            subset_on, subset_off = self._subset_on_off(subset, feature)

            sizes.append(subset_off.sum())

            masked_samples[i, subset_on] = explicand[subset_on]
            # print("SC 3: masked_samples", masked_samples)
            
            masked_samples[
                2 ** (self.num_features - 1) + i, subset_off
            ] = explicand[subset_off]
            

        # print("SANITY CHECK: explicand", explicand)
        # print("SANITY CHECK: baseline", baselines)
        # print("SANITY CHECK: masked_samples", masked_samples)
        
        # Compute marginal contributions
        weights = 1 / scipy.special.comb(
            self.num_features - 1, np.array(sizes)
        )
        weights /= self.num_features
        preds = self.model(masked_samples)
        preds_on = preds[: 2 ** (self.num_features - 1)]
        preds_off = preds[2 ** (self.num_features - 1) :]
        deltas = weights * (preds_on - preds_off)

        return deltas.sum()

    def _explain(self, X_train, y_train, explicands, y_test, baselines, categorical, flag):
        start = time.time()

        phis = []
        for explicand in tqdm(explicands):

            phi = np.zeros(explicand.shape)
            
            for i in range(self.num_features):
                if flag:
                    if categorical is not None:
                        baselines = similarity_ces(explicand, X_train, categorical)
                    
                    # Randomly choose a baseline sample
                    baseline_ind = np.random.randint(baselines.shape[0])
                    baseline = [baselines[baseline_ind]]


                phi[i] = self._single_feature(i, explicand, baseline)
            
            phis.append(phi)

        end = time.time()

        attributions = {
            'shap_values': np.array(phis), 
            'num_features': self.num_features,
            'train': {'X': X_train, 'y': y_train},
            'test':  {'X': explicands, 'y': y_test},
            'pred': self.model(explicands),
            'compute_time': end - start
        }
        
        return attributions

    def __call__(self, X_train, y_train, explicands, y_test, baselines, categorical=None, flag=True):

        return self._explain(X_train, y_train, explicands, y_test, baselines, categorical, flag)
