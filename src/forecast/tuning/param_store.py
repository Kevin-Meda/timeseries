"""Parameter store for saving and loading model parameters."""

import json
from pathlib import Path
from typing import Any

from forecast.utils.project_manager import ProjectManager
from forecast.utils import get_logger


# Mapping from model names to config keys
MODEL_CONFIG_KEYS = {
    "SARIMA": "sarima",
    "HoltWinters": "holt_winters",
    "Chronos": "chronos",
    "XGBoost": "xgboost",
    "Prophet": "prophet",
}


class ParamStore:
    """Store and retrieve model parameters."""

    def __init__(self, project_manager: ProjectManager):
        """Initialize parameter store.

        Args:
            project_manager: ProjectManager instance for path resolution.
        """
        self.project_manager = project_manager

    def save_params(
        self,
        model_name: str,
        product: str,
        params: dict[str, Any],
    ) -> Path:
        """Save model parameters.

        Args:
            model_name: Name of the model.
            product: Product name.
            params: Parameters to save.

        Returns:
            Path to saved parameters file.
        """
        return self.project_manager.save_params(model_name, product, params)

    def load_latest_params(
        self,
        model_name: str,
        product: str,
    ) -> dict[str, Any] | None:
        """Load the most recent parameters for a model and product.

        Args:
            model_name: Name of the model.
            product: Product name.

        Returns:
            Parameters dictionary if found, None otherwise.
        """
        return self.project_manager.find_latest_params(model_name, product)

    def params_exist(
        self,
        model_name: str,
        product: str,
    ) -> bool:
        """Check if parameters exist for a model and product.

        Args:
            model_name: Name of the model.
            product: Product name.

        Returns:
            True if parameters exist, False otherwise.
        """
        return self.load_latest_params(model_name, product) is not None


def get_model_config_key(model_name: str) -> str:
    """Get the config key for a model name.

    Args:
        model_name: Name of the model (e.g., "SARIMA", "XGBoost").

    Returns:
        Config key (e.g., "sarima", "xgboost").
    """
    return MODEL_CONFIG_KEYS.get(model_name, model_name.lower())


def should_optimize(
    models_config: dict[str, Any],
    project_manager: ProjectManager,
    model_name: str,
    product: str,
) -> tuple[bool, dict[str, Any] | None]:
    """Determine if a model should run optimization or use existing/default params.

    Logic:
    - If model has optimize_params=true: run optimization, return (True, None)
    - If model has optimize_params=false:
        - Try to load saved params for project-model-product
        - If found: return (False, loaded_params)
        - If not found: return (False, defaults from config)

    Args:
        models_config: Models configuration dictionary.
        project_manager: ProjectManager for finding existing params.
        model_name: Name of the model (e.g., "SARIMA", "XGBoost").
        product: Product name.

    Returns:
        Tuple of (should_optimize: bool, params: dict | None).
        If should_optimize is False, params contains loaded or default parameters.
    """
    logger = get_logger()
    config_key = get_model_config_key(model_name)
    model_config = models_config.get(config_key, {})

    # Get optimize_params flag (default to False if not present)
    optimize_flag = model_config.get("optimize_params", False)

    if optimize_flag:
        # Optimization mode: will run Optuna
        logger.debug(f"Optimization enabled for {model_name}/{product}")
        return True, None

    # Load params mode: try to find existing params
    param_store = ParamStore(project_manager)
    existing_params = param_store.load_latest_params(model_name, product)

    if existing_params is not None:
        logger.info(f"Loading existing params for {model_name}/{product}")
        return False, existing_params

    # No existing params found, use defaults from config
    defaults = model_config.get("defaults", {})
    if defaults:
        logger.info(
            f"No saved params for {model_name}/{product}, using defaults from config"
        )
        return False, defaults.copy()

    # No defaults either, return None (model will use its internal defaults)
    logger.debug(
        f"No saved params or defaults for {model_name}/{product}, "
        f"using model internal defaults"
    )
    return False, None


# Keep old function for backward compatibility during transition
def should_retrain(
    config: dict[str, Any],
    project_manager: ProjectManager,
    model_name: str,
    product: str,
) -> tuple[bool, dict[str, Any] | None]:
    """Deprecated: Use should_optimize() instead.

    This function is kept for backward compatibility.
    """
    return should_optimize(config, project_manager, model_name, product)
