"""Evaluation metrics and ensemble methods."""

from .metrics import rmse, mape, mae_score, evaluate_all
from .ensemble import create_ensemble, inverse_mape_weights
from .feature_importance import (
    compute_permutation_importance,
    compute_shap_importance,
    compute_feature_importance,
    compute_model_feature_importance,
    compute_ensemble_importance,
    write_project_feature_importance,
    write_ensemble_fi_with_weights,
)

__all__ = [
    "rmse",
    "mape",
    "mae_score",
    "evaluate_all",
    "create_ensemble",
    "inverse_mape_weights",
    "compute_permutation_importance",
    "compute_shap_importance",
    "compute_feature_importance",
    "compute_model_feature_importance",
    "compute_ensemble_importance",
    "write_project_feature_importance",
    "write_ensemble_fi_with_weights",
]
