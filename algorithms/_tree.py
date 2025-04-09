import shap
from typing import Any
from algorithms._explainer import Explainer
import numpy as np
import time
from tqdm import tqdm 

class Tree(Explainer):
    """
    Tree Shap approach: Model-specific(only treee models)
    """

    def __init__(self, model, num_features):
        self.model = model
        self.num_features = num_features
    


    def _explain(self, explicands, baselines, fp):

        explainer = shap.explainers.Tree(model=self.model, masker=baselines, feature_perturbation=fp)
        start = time.time()

        phis = []
        for explicand in tqdm(explicands):

            phi = explainer.shap_values(np.array([explicand]))
            phis.append(phi)

        end = time.time()

        attributions = {
            'shapley_values': np.array(phis), 
            'data': explicands,
            'baseline': baselines,
            'pred': self.model.predict(explicands),
            'compute_time': end - start,
        }
        return attributions

    def __call__(self,explicands, baselines, fp='interventional'):

        return self._explain(explicands, baselines, fp)
