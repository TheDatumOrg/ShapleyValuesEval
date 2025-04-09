from typing import Any
from algorithms._explainer import Explainer
from shapkit.sgd_shapley import SGDshapley
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

class SGDShapley(Explainer):
    """WLS sampling approach using SGD:

    *Only works for baseline shapley.

    Using pre-existing package for projected gradient descent.  Each iteration
    evaluates model for subset, takes gradient step and then projects based on
    an additive efficient normalization.

    https://hal.inria.fr/hal-03414720/
    """

    def __init__(self, model, num_features, max_label):
        """
        model: model being explained
        num_features: number of features
        max_label: maximum label value
        """

        super().__init__(model, num_features)

        def model_predict(x):
            if len(x.shape) == 1:
                return self.model(x[None, :])
            elif len(x.shape) == 2:
                return self.model(x)

        self.model_predict = model_predict
        self.estimator = SGDshapley(num_features, C=max_label)

    def _explain(self, explicands, baseline, num_evals, is_antithetic=False):
        start = time.time()
        phis = []
        for explicand in tqdm(explicands):

            phi = self.estimator.sgd(
                x=pd.DataFrame(explicand[None, :]).iloc[0],
                fc=self.model_predict,
                ref=pd.DataFrame(baseline).iloc[0],
                n_iter=num_evals,
            ).values

            phis.append(phi)

        end = time.time()
        attributions = {
            'shap_values': np.array(phis),
            'explicands': explicands,
            'baseline': baseline,
            'pred': self.model(explicands),
            'compute_time': end - start,
            'num_evals': num_evals
        }
        return attributions

    def __call__(self, explicands, baseline, num_evals=1000):

        if baseline.shape[0] != 1:
            raise NotImplementedError(
                "LeastSquaresSGD only supports a single baseline"
            )

        return self._explain(explicands, baseline, num_evals)