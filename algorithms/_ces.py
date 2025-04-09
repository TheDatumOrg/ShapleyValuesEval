from typing import Any
from algorithms._explainer import Explainer
import time
import numpy as np
from itertools import permutations
from algorithms.helper import similarity_ces
import random
from tqdm import tqdm

class CES(Explainer):
    def __init__(self, model, num_features):
        super().__init__(model, num_features)
        self.features = set(range(num_features))
    
    def _permutations(self, iterable):
        s = list(iterable)
        return permutations(s)

    def _explain(self, explicands, baselines, smoothing, max_perms):
        start = time.time()
        permutations = list(self._permutations(self.features))
        # print(len(permutations))
        phis = []

        for explicand in tqdm(explicands):
            pred = self.model(np.array([explicand]))
            phi = [0.0] * self.num_features
            for i in range(max_perms):
                index = random.randint(0, len(permutations) - 1)
                permutation = permutations[index]

                v_new = np.mean(self.model(baselines))
                # print(v_new)
                T = baselines
                # print(T.shape)
                T_preds = self.model(T)
                # print(T_preds.shape)
                # print(T[np.logical_and(T_preds >= pred[0] - smoothing, T_preds <= pred[0] + smoothing)])
            
                for j in range(self.num_features):
                    v_old = v_new
                    if smoothing is not None:
                        T_updated = T[np.where(np.logical_and(T_preds >= (pred[0] - smoothing), T_preds <= (pred[0] + smoothing)))]
                    else:
                        smoothing = (T_preds.max() - T_preds.min())/10000
                        # print(smoothing)
                        T_updated = T[np.where(np.logical_and(T_preds >= (pred[0] - smoothing), T_preds <= (pred[0] + smoothing)))]

                    # print(permutation)
                    T_prime = list()
                    for t in T_updated:
                        # print(explicand[j], t[permutation[j]])
                        if explicand[j] == t[permutation[j]]:
                            T_prime.append(t)

                    Tprime_np = np.array(T_prime)

                    if Tprime_np.shape[0] != 0:
                        v_new = np.mean(T_preds)
                    # else:
                    #     v_new = np.mean(self.model(Tprime_np))
                        
                    print((v_new - v_old))
                    
                    phi[j] = phi[j] + (1/np.math.factorial(self.num_features)) * (v_new - v_old)
                
                print(phi)
            phis.append(phi)
             
        end = time.time()

        attributions = {
            'shap_values': np.array(phis),
            'data': explicands,
            'pred': self.model(explicands),
            'compute_time': end - start
        }
        return attributions

    def __call__(self, explicands, baselines, smoothing = None, max_perms = None):

        if baselines.shape[0] == 1:
            raise ValueError
        
        return self._explain(explicands, baselines, smoothing, max_perms=1000)