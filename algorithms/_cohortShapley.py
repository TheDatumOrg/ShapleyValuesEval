import numpy as np
from typing import Any
from algorithms._explainer import Explainer
from algorithms.helper import similarity_cohort
import math
import time
from tqdm import tqdm

class CohortShapley(Explainer):
    '''
    CohortShapley: https://github.com/cohortshapley/cohortshapley 
    '''
    def __init__(self, model, data, categorical, func=np.average, y=None,):
        self.model = model
        self.data = data
        self.func = func
        self.num_features = data.shape[-1]
        self.categorical = categorical

        if y is None:
            self.y = model(data)
        else:
            self.y = y

        self.grand_mean = self.func(self.y)

    def CohortValue(self, data, y, explicand, vertex):
        cohort = similarity_cohort(explicand, data, vertex, self.categorical)
        n = np.count_nonzero(cohort)
        if n == 0:
            avgv = self.func(y)
        else:
            avgv = self.func(np.extract(cohort, y))
        return (n, avgv, cohort)

    def CohortShapleyOne(self, y, explicand, data):
        shapley_values = np.zeros(self.num_features)
        phi_set = np.zeros(self.num_features)
        u_k = {}
        u_k[tuple(phi_set)] = self.CohortValue(data, y, explicand, phi_set)
        for k in range(self.num_features):
            coef =  math.factorial(k) * math.factorial(self.num_features - k - 1) / math.factorial(self.num_features - 1) / self.num_features
            u_k_base = u_k
            u_k = {}
            for j in range(self.num_features):
                gain = 0
                for sett in u_k_base.keys():
                    set = np.array(sett)
                    if set[j] == 1:
                        pass
                    elif u_k_base[sett][0] == 1:
                        pass
                    else:
                        set_j = set.copy()
                        set_j[j] = 1
                        if tuple(set_j) not in u_k.keys():
                            u_k[tuple(set_j)] = self.CohortValue(data, y, explicand, set_j)
                        gain_temp = u_k[tuple(set_j)][1] - u_k_base[sett][1]
                        gain += gain_temp
                shapley_values[j] += gain * coef

        return shapley_values
    
    

    def compute_cohort_shapley(self, explicands):
        start = time.time()
        y = self.y
        data = self.data
        shap_values = []
        for explicand in tqdm(explicands):
            ret1 = self.CohortShapleyOne(y, explicand, data)
            self.shapley_values = np.array([ret1])
            shap_values.append(np.array(ret1))

        end = time.time()
        attributions = {
            'shapley_values': np.array(shap_values),
            'data': explicands,
            'pred': self.model(explicands),
            'compute_time': end-start
        }
        return attributions
    
    def __call__(self, explicands):
        return self.compute_cohort_shapley(explicands)