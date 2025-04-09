from typing import Any
from algorithms._explainer import Explainer
import numpy as np
from itertools import chain, combinations
import scipy
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class Removal(Explainer):
    """
    Exact estimation (expoential in the number of features)
    Get results of different removal strategies
    """
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
    
    def _single_feature(self, feature, game):

        subsets = self._powerset(self.features - set([feature]))
        phi_i = 0

        for i, subset in enumerate(subsets):
            S_off = np.zeros((self.num_features,), dtype=bool)
            S_on = np.zeros((self.num_features,), dtype=bool)

            f_off = list(subset)
            f_on = list(subset + (feature,))
            f_on.sort()
            S_off[f_off] = True
            S_on[f_on] = True

            preds_off = game(S_off)
            preds_on = game(S_on)

            # Compute marginal contributions
            weight = 1 / scipy.special.comb(
                self.num_features - 1, len(f_off)
            )
            weight /= self.num_features
            delta = weight * (preds_on - preds_off)
            phi_i += delta
        return phi_i
    
    def _explain(self, X_train, y_train, explicands, y_test, removal):
        start = time.time()

        phis = []
        # ind = 0
        for explicand in tqdm(explicands):

            phi = np.zeros(explicand.shape)
            
            for i in range(self.num_features):
                if removal == 'zero':
    
                    from rexplain.removal import DefaultExtension
                    from rexplain.behavior import PredictionGame
                    baseline = np.zeros((1, self.num_features))
                    extension = DefaultExtension(baseline, self.model)
                    game = PredictionGame(extension, explicand)

                    phi[i] = self._single_feature(i, game)
                elif removal == 'mean':
                    
                    from rexplain.removal import DefaultExtension
                    from rexplain.behavior import PredictionGame
                    baseline = np.array([np.mean(X_train, axis=0)])
                    extension = DefaultExtension(baseline, self.model)
                    game = PredictionGame(extension, explicand)

                    phi[i] = self._single_feature(i, game)
                
                elif removal == 'marginal':
                    
                    from rexplain.removal import MarginalExtension
                    from rexplain.behavior import PredictionGame
                    extension = MarginalExtension(X_train, self.model)
                    game = PredictionGame(extension, explicand)

                    phi[i] = self._single_feature(i, game)

                # elif removal == 'conditional':
                #     from rexplain.removal import ConditionalSupervisedExtension
                #     from rexplain.behavior import PredictionLossGame
                #     from rexplain.utils import crossentropyloss
                #     print(explicands.shape)
                #     extension = ConditionalSupervisedExtension(self.model)
                #     game = PredictionLossGame(extension, explicand, y_test[ind:ind+1], crossentropyloss)

                #     phi[i] = self._single_feature(i, game)
                #     ind += 1
                else:
                    print("Blah")

                # phi[i] = self._single_feature(i, explicand, baseline)
            
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
        # print(attributions)
        
        return attributions
    
    def __call__(self, X_train, y_train, explicands, y_test, removal):

        return self._explain(X_train, y_train, explicands, y_test, removal)