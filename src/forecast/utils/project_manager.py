"""Project manager for organizing output paths by project and timestamp."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ProjectManager:
    """Manages project-based output directory structure.

    Output structure:
        output/{project_name}/{results|models|plots}/{timestamp}/
        logs/{project_name}/
    """

    def __init__(
        self,
        project_name: str = "default",
        base_output_dir: str = "output",
        base_log_dir: str = "logs",
        timestamp: str | None = None,
    ):
        """Initialize project manager.

        Args:
            project_name: Name of the project for organizing outputs.
            base_output_dir: Base directory for outputs.
            base_log_dir: Base directory for logs.
            timestamp: Optional timestamp string. If None, generates new one.
        """
        self.project_name = project_name
        self.base_output_dir = Path(base_output_dir)
        self.base_log_dir = Path(base_log_dir)
        self._timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def timestamp(self) -> str:
        """Get the timestamp for this run."""
        return self._timestamp

    @property
    def project_output_dir(self) -> Path:
        """Get the project-specific output directory."""
        return self.base_output_dir / self.project_name

    @property
    def results_dir(self) -> Path:
        """Get the timestamped results directory."""
        path = self.project_output_dir / "results" / self._timestamp
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def models_dir(self) -> Path:
        """Get the timestamped models directory."""
        path = self.project_output_dir / "models" / self._timestamp
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def plots_dir(self) -> Path:
        """Get the timestamped plots directory."""
        path = self.project_output_dir / "plots" / self._timestamp
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def logs_dir(self) -> Path:
        """Get the project-specific logs directory."""
        path = self.base_log_dir / self.project_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def find_latest_params(
        self,
        model_name: str,
        product: str,
    ) -> dict[str, Any] | None:
        """Find the most recent saved parameters for a model and product.

        Searches through all timestamp directories in the models folder
        to find the most recent parameters file.

        Args:
            model_name: Name of the model (e.g., "XGBoost", "Prophet").
            product: Product name (e.g., "Product_A").

        Returns:
            Dictionary of parameters if found, None otherwise.
        """
        models_base = self.project_output_dir / "models"
        if not models_base.exists():
            return None

        # Get all timestamp directories, sorted descending (most recent first)
        timestamp_dirs = sorted(
            [d for d in models_base.iterdir() if d.is_dir()],
            key=lambda x: x.name,
            reverse=True,
        )

        # Search for the params file
        param_filename = f"{product}_{model_name}_params.json"
        for ts_dir in timestamp_dirs:
            param_file = ts_dir / param_filename
            if param_file.exists():
                try:
                    with open(param_file, "r") as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError):
                    continue

        return None

    def save_params(
        self,
        model_name: str,
        product: str,
        params: dict[str, Any],
    ) -> Path:
        """Save model parameters to the models directory.

        Args:
            model_name: Name of the model.
            product: Product name.
            params: Dictionary of parameters to save.

        Returns:
            Path to the saved parameters file.
        """
        param_file = self.models_dir / f"{product}_{model_name}_params.json"
        with open(param_file, "w") as f:
            json.dump(params, f, indent=2, default=str)
        return param_file

    def get_all_timestamps(self) -> list[str]:
        """Get all existing timestamps for this project.

        Returns:
            List of timestamp strings, sorted descending (most recent first).
        """
        models_base = self.project_output_dir / "models"
        if not models_base.exists():
            return []

        timestamps = sorted(
            [d.name for d in models_base.iterdir() if d.is_dir()],
            reverse=True,
        )
        return timestamps
