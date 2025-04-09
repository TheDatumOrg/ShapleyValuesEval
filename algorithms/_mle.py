from typing import Any
from algorithms._explainer import Explainer
import numpy as np
import time
from tqdm import tqdm 

class Multilinear(Explainer):
    """Multilinear extension sampling approach:

    https://arxiv.org/abs/2010.12082
    """

    def __init__(self, model, num_features):

        self.probs = None

        super().__init__(model, num_features)
        self.features = np.arange(num_features)

    def _sample_subset(self):

        # Random sample if no probs
        prob = np.random.uniform()

        # Trapezoid rule by default
        if self.probs:
            prob = self.probs.pop()

        subset = np.random.binomial(1, prob, size=(self.num_features))

        return subset.astype("bool")

    def _invert_subset(self, subset):

        return np.invert(subset)

    def _explain(self, explicands, baselines, num_evals, is_antithetic=False):

        start = time.time()

        phis = []
        for explicand in tqdm(explicands):
            # Bookkeeping total number of evaluations
            evals_count = 0

            # Determine the appropriate number of subsets
            num_subsets = num_evals
            # if is_antithetic:
            #     num_subsets //= 2 * (self.num_features + 1)
            # else:
            #     num_subsets //= self.num_features + 1

            # Keep stack of probabilities to draw from for Multilinear
            if self.samples_per_prob:

                adj_num_subsets = num_subsets // self.samples_per_prob

                if adj_num_subsets > 1:

                    self.probs = np.arange(0, adj_num_subsets) / (
                        adj_num_subsets - 1
                    )
                    self.probs = list(np.repeat(self.probs, self.samples_per_prob))

            # Estimate the unnormalized attribution
            phi = np.zeros(explicand.shape)
            for _ in range(num_subsets):

                subset = self._sample_subset()

                # Randomly choose a baseline sample
                baseline_ind = np.random.randint(baselines.shape[0])
                baseline = baselines[baseline_ind]

                # Add inverse subset if antithetic
                subsets = [subset]
                if is_antithetic:
                    subsets.append(self._invert_subset(subset))

                for curr_subset in subsets:

                    # Evaluate game for known subset
                    subset_sample = np.copy(baseline)
                    subset_sample[curr_subset] = explicand[curr_subset]
                    subset_pred = self.model(subset_sample[None, :])
                    evals_count += 1

                    subset_samples = np.tile(
                        subset_sample, self.num_features
                    ).reshape(self.num_features, -1)

                    for i in range(self.num_features):

                        if curr_subset[i]:
                            subset_samples[i, i] = baseline[i]
                        else:
                            subset_samples[i, i] = explicand[i]

                    # Compute model predictions at the same time
                    subset_preds = self.model(subset_samples)
                    evals_count += self.num_features

                    for i in range(self.num_features):
                        if curr_subset[i]:
                            phi[i] += subset_pred - subset_preds[i]
                        else:
                            phi[i] += subset_preds[i] - subset_pred

            # Normalize
            if is_antithetic:
                phi = phi / (num_subsets * 2)
            else:
                phi = phi / num_subsets

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
        self,
        explicands,
        baselines,
        num_evals=1000,
        is_antithetic=False,
        samples_per_prob=2,
    ):

        # Number of samples per probability (if none, random sampling)
        self.samples_per_prob = samples_per_prob
        return self._explain(explicands, baselines, num_evals, is_antithetic)
