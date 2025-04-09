from typing import Any
from algorithms._explainer import Explainer
import numpy as np
import scipy
import time
from tqdm import tqdm
class Permutation(Explainer):
    """Sampling explainer with antithetic sampling approach (walking through permutationin forward and reverse directions):

    https://www.sciencedirect.com/science/article/pii/S0305054808000804
    """

    def __init__(self, model, num_features):

        super().__init__(model, num_features)
        self.num_features = num_features
        self.features = np.arange(num_features)

    def _explain(self, explicands, baselines, num_evals, is_antithetic=False):

        start = time.time()
        phis = []

        for explicand in tqdm(explicands):
            # Bookkeeping total number of evaluations
            evals_count = 0

            # Determine the appropriate number of permutations based on num_evals
            # num_permutations = num_evals - 1
            num_permutations = num_evals

            # Halve number of subset sizes if we are using paired sampling
            num_permutations = num_evals
            if is_antithetic:
                num_permutations = num_evals // 2
            
            # if is_antithetic:
            #     num_permutations //= 2 * (self.num_features - 1) + 1
            # else:
            #     num_permutations //= self.num_features

            phi = np.zeros(explicand.shape)
            explicand_pred = self.model(explicand[None, :])
            evals_count += 1
            explicand_tiled = np.tile(explicand, self.num_features - 1)
            explicand_tiled = explicand_tiled.reshape(self.num_features - 1, -1)

            for _ in range(num_permutations):

                # Shuffle indices and split subsets based on feature position
                np.random.shuffle(self.features)

                # Randomly choose a baseline sample
                baseline_ind = np.random.randint(baselines.shape[0])
                baseline = baselines[baseline_ind]
                baseline_pred = self.model(baseline[None, :])
                evals_count += 1

                # Create masked samples to evaluate
                masked_samples1 = np.copy(explicand_tiled)

                if is_antithetic:
                    masked_samples2 = np.copy(explicand_tiled)

                for i in range(self.num_features - 1):
                    subset_forward = self.features[(i + 1) :]
                    masked_samples1[i, subset_forward] = baseline[subset_forward]

                    if is_antithetic:
                        subset_back = self.features[: -(i + 1)]
                        masked_samples2[i, subset_back] = baseline[subset_back]

                # Get output arrays for both forward and backward
                def _add_subsets(preds):
                    return np.hstack([baseline_pred, preds, explicand_pred])

                forward_preds = _add_subsets(self.model(masked_samples1))
                evals_count += len(masked_samples1)
                if is_antithetic:
                    backward_preds = _add_subsets(self.model(masked_samples2))
                    evals_count += len(masked_samples2)

                # Update estimates of feature attributions
                for i in range(self.num_features):
                    feature = self.features[i]
                    phi[feature] += forward_preds[i + 1] - forward_preds[i]

                    if is_antithetic:
                        feature = self.features[-(i + 1)]
                        phi[feature] += backward_preds[i + 1] - backward_preds[i]

            # Normalize according to whether we used antithetic sampling
            if is_antithetic:
                phi = phi / (num_permutations * 2)
            else:
                phi = phi / num_permutations

            phis.append(phi)

        end = time.time()

        attributions = {
            'shap_values': np.array(phis), 
            'explicands': explicands,
            'baseline': baselines,
            'pred': self.model(explicands),
            'compute_time': end - start,
        }
        return attributions

    def __call__(
        self, explicands, baselines, num_evals=1000, is_antithetic=True
    ):
        
        return self._explain(explicands, baselines, num_evals, is_antithetic)