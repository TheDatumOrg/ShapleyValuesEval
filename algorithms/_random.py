from typing import Any
from algorithms._explainer import Explainer
import time
import numpy as np

class Random(Explainer):
    def __init__(self, model, num_features):
        super().__init__(model, num_features)

    def _explain(self, explicands, baselines):
        start = time.time()
        phis = []
        for explicand in explicands:    
            # generate random feature attributions
            # we produce small values so our explanation errors are similar to a constant function
            phi = np.random.randn(*(self.num_features,)) * 0.001
            phis.append(phi)

        end = time.time()

        attributions = {
            'shap_values': np.array(phis), 
            'compute_time': end - start,
        }
        return attributions

    def __call__(self, explicands, baselines):

        return self._explain(explicands, baselines)