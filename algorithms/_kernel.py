from typing import Any
from algorithms._explainer import Explainer
import numpy as np
import time

class Kernel(Explainer):
    """WLS sampling approach (KernelSHAP):

    https://arxiv.org/abs/1705.07874
    http://proceedings.mlr.press/v130/covert21a/covert21a.pdf
    """

    def __init__(self, model, num_features):

        super().__init__(model, num_features)
        self.features = np.arange(self.num_features)

    def _solve_wls(self, A, b, total):
        """Calculate the regression coefficients."""
        try:
            if len(b.shape) == 2:
                A_inv_one = np.linalg.solve(A, np.ones((self.num_features, 1)))
            else:
                A_inv_one = np.linalg.solve(A, np.ones(self.num_features))
            A_inv_vec = np.linalg.solve(A, b)
            values = A_inv_vec - A_inv_one * (
                np.sum(A_inv_vec, axis=0, keepdims=True) - total
            ) / np.sum(A_inv_one)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Singular matrix inversion, use larger variance_batches"
            )

        return values

    def _explain(self, explicands, baselines, num_evals, is_antithetic=False):

        start = time.time()

        phis = []
        for explicand in explicands:

            # Round to the nearest even integer
            num_evals = int(2 * (num_evals // 2))

            # Bookkeeping total number of evaluations
            evals_count = 0

            # Probability of each subset size (from Shapley weighting kernel)
            size_probs = np.arange(1, self.num_features)
            size_probs = 1 / (size_probs * (self.num_features - size_probs))
            size_probs /= np.sum(size_probs)

            # Halve number of subset sizes if we are using paired sampling
            num_subsets = num_evals
            if is_antithetic:
                num_subsets = num_evals // 2

            # Sample appropriate subsets
            subsets = np.zeros((num_evals, self.num_features), dtype=bool)
            subset_sizes = np.random.choice(
                range(1, self.num_features), size=num_subsets, p=size_probs
            )

            # Create masked samples to evaluate output of model
            masked_samples = np.tile(explicand, num_evals).reshape(num_evals, -1)

            # Keep track of explicand output for empty and full game
            explicand_output = self.model(explicand[None, :])
            #     evals_count += 1 # Ignore counts for calculating full and null game
            mean_baseline_output = self.model(baselines).mean()
            #     evals_count += len(baselines)
            # @TODO(hughchen): Figure out how to account for these evaluations

            # Generate masked samples based on random subsets and baselines
            for i, size in enumerate(subset_sizes):
                baseline_ind = np.random.randint(baselines.shape[0])
                baseline = baselines[baseline_ind]
                explicand_inds = np.random.choice(
                    self.num_features, size=size, replace=False
                )
                baseline_inds = np.setdiff1d(
                    np.arange(self.num_features), explicand_inds
                )

                subsets[i, explicand_inds] = True
                masked_samples[i, baseline_inds] = baseline[baseline_inds]

                if is_antithetic:
                    subsets[-(i + 1), baseline_inds] = True
                    masked_samples[-(i + 1), explicand_inds] = baseline[
                        explicand_inds
                    ]

            outputs = self.model(masked_samples)
            evals_count += len(masked_samples)

            # Calculate intermediate matrices for final calculation
            A_matrix = np.matmul(
                subsets[:, :, np.newaxis].astype(float),
                subsets[:, np.newaxis, :].astype(float),
            )
            b_matrix = (
                subsets.astype(float).T
                * (outputs - mean_baseline_output)[:, np.newaxis].T
            ).T
            A = np.mean(A_matrix, axis=0)
            b = np.mean(b_matrix, axis=0)

            # Calculate shapley value feature attributions based on WLS formulation
            phi = self._solve_wls(A, b, explicand_output - mean_baseline_output)

            phis.append(phi)
            
        end = time.time()

        attributions = {
            'shap_values': np.array(phis), 
            'explicands': explicands,
            'baseline': baselines,
            'pred': self.model(explicands),
            'compute_time': end - start
        }
        return attributions


    def __call__(
        self, explicands, baselines, num_evals=100, is_antithetic=False
    ):

        return self._explain(explicands, baselines, num_evals, is_antithetic)
