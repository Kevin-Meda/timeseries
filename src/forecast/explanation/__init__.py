"""Feature explanation module."""

from forecast.explanation.shap_explainer import ShapExplainer, is_shap_available
from forecast.explanation.permutation_importance import (
    PermutationImportance,
    compute_permutation_importance,
)
from forecast.explanation.decomposition import ProphetDecomposition

__all__ = [
    "ShapExplainer",
    "is_shap_available",
    "PermutationImportance",
    "compute_permutation_importance",
    "ProphetDecomposition",
]
