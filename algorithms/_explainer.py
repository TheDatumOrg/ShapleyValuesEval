from typing import Any


class Explainer():
    """Class for any explainer."""

    def __init__(self, model, num_features):
        """Initialize with model."""
        self.model = model
        self.num_features = num_features

    def _explain(self, explicand, num_evals):

        raise NotImplementedError()

    def __call__(self, explicand, num_evals=100):

        raise NotImplementedError()