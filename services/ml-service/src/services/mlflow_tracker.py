"""
MLflow Experiment Tracker for ML operations.

@TENSOR @SENTRY - MLflow integration for experiment tracking and model registry.
"""

import os
from datetime import datetime
from typing import Any, Optional

import mlflow
from mlflow.tracking import MlflowClient
import structlog

from src.config import settings
from src.models.training import (
    TrainingJob,
    TrainingConfig,
    TrainingMetrics,
    ExperimentResult,
)

logger = structlog.get_logger()


class MLflowTracker:
    """
    MLflow experiment tracker for NEURECTOMY.
    
    @TENSOR @SENTRY - Comprehensive ML experiment tracking:
    - Experiment organization
    - Parameter logging
    - Metric tracking
    - Artifact storage
    - Model registry
    """
    
    def __init__(self):
        self._client: Optional[MlflowClient] = None
        self._tracking_uri = settings.mlflow_tracking_uri
    
    async def initialize(self) -> None:
        """Initialize MLflow connection."""
        mlflow.set_tracking_uri(self._tracking_uri)
        self._client = MlflowClient(tracking_uri=self._tracking_uri)
        
        # Test connection
        try:
            self._client.search_experiments()
            logger.info(f"✅ MLflow connected: {self._tracking_uri}")
        except Exception as e:
            logger.warning(f"MLflow connection failed (will use local): {e}")
            # Fall back to local tracking
            mlflow.set_tracking_uri("file:./mlruns")
            self._client = MlflowClient()
            logger.info("Using local MLflow tracking")
    
    @property
    def client(self) -> MlflowClient:
        if not self._client:
            raise RuntimeError("MLflow not initialized")
        return self._client
    
    # =========================================================================
    # Experiment Management
    # =========================================================================
    
    def get_or_create_experiment(self, name: str) -> str:
        """Get or create an MLflow experiment."""
        experiment = mlflow.get_experiment_by_name(name)
        if experiment:
            return experiment.experiment_id
        
        experiment_id = mlflow.create_experiment(
            name,
            tags={"created_by": "neurectomy", "module": "intelligence_foundry"},
        )
        logger.info(f"Created experiment: {name} (ID: {experiment_id})")
        return experiment_id
    
    def list_experiments(self) -> list[dict]:
        """List all experiments."""
        experiments = self.client.search_experiments()
        return [
            {
                "id": exp.experiment_id,
                "name": exp.name,
                "lifecycle_stage": exp.lifecycle_stage,
                "tags": exp.tags,
            }
            for exp in experiments
        ]
    
    # =========================================================================
    # Run Management
    # =========================================================================
    
    def start_run(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Start a new MLflow run.
        
        Returns the run ID.
        """
        experiment_id = self.get_or_create_experiment(experiment_name)
        
        run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=tags,
        )
        
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run.info.run_id
    
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""
        mlflow.end_run(status=status)
    
    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters for current run."""
        # Flatten nested dicts
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)
    
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics for current run."""
        mlflow.log_metrics(metrics, step=step)
    
    def log_training_config(self, config: TrainingConfig) -> None:
        """Log complete training configuration."""
        # Log hyperparameters
        self.log_params({
            "model_name": config.model_name,
            "model_type": config.model_type,
            **config.hyperparameters.model_dump(),
        })
        
        # Log data configuration
        self.log_params({
            "dataset": config.data.dataset_name,
            "max_length": config.data.max_length,
        })
        
        # Log tags
        mlflow.set_tags(config.tags)
    
    def log_training_metrics(self, metrics: TrainingMetrics) -> None:
        """Log training metrics at a step."""
        metric_dict = {
            "train_loss": metrics.train_loss,
            "learning_rate": metrics.learning_rate,
        }
        
        if metrics.eval_loss is not None:
            metric_dict["eval_loss"] = metrics.eval_loss
        
        if metrics.accuracy is not None:
            metric_dict["accuracy"] = metrics.accuracy
        
        if metrics.f1_score is not None:
            metric_dict["f1_score"] = metrics.f1_score
        
        # Add additional metrics
        metric_dict.update(metrics.additional)
        
        self.log_metrics(metric_dict, step=metrics.step)
    
    # =========================================================================
    # Artifact Management
    # =========================================================================
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a file or directory as an artifact."""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Log a model to MLflow.
        
        @TENSOR - Model logging with framework detection.
        """
        # Detect model type and log appropriately
        model_type = type(model).__module__.split(".")[0]
        
        if model_type == "torch":
            mlflow.pytorch.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                **kwargs,
            )
        elif model_type == "transformers":
            mlflow.transformers.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                **kwargs,
            )
        elif model_type == "sklearn":
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                **kwargs,
            )
        else:
            # Generic pickle logging
            mlflow.pyfunc.log_model(
                artifact_path,
                python_model=model,
                registered_model_name=registered_model_name,
                **kwargs,
            )
    
    def log_figure(self, figure: Any, artifact_path: str) -> None:
        """Log a matplotlib or plotly figure."""
        mlflow.log_figure(figure, artifact_path)
    
    # =========================================================================
    # Model Registry
    # =========================================================================
    
    def register_model(
        self,
        run_id: str,
        artifact_path: str,
        model_name: str,
    ) -> str:
        """Register a model from a run."""
        model_uri = f"runs:/{run_id}/{artifact_path}"
        result = mlflow.register_model(model_uri, model_name)
        
        logger.info(f"Registered model: {model_name} v{result.version}")
        return result.version
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,  # Staging, Production, Archived
    ) -> None:
        """Transition model to a different stage."""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
        )
        logger.info(f"Model {model_name} v{version} → {stage}")
    
    def get_latest_model(
        self,
        model_name: str,
        stage: str = "Production",
    ) -> Optional[str]:
        """Get the latest model version in a stage."""
        versions = self.client.get_latest_versions(model_name, stages=[stage])
        if versions:
            return versions[0].version
        return None
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> Any:
        """
        Load a model from the registry.
        
        @TENSOR - Model loading with version/stage support.
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
        elif stage:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            model_uri = f"models:/{model_name}/latest"
        
        return mlflow.pyfunc.load_model(model_uri)
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def _flatten_dict(
        self,
        d: dict,
        parent_key: str = "",
        sep: str = ".",
    ) -> dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def get_run_metrics(self, run_id: str) -> dict:
        """Get all metrics for a run."""
        run = self.client.get_run(run_id)
        return run.data.metrics
    
    def compare_runs(self, run_ids: list[str]) -> list[dict]:
        """Compare metrics across multiple runs."""
        comparisons = []
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            comparisons.append({
                "run_id": run_id,
                "run_name": run.info.run_name,
                "params": run.data.params,
                "metrics": run.data.metrics,
                "status": run.info.status,
            })
        return comparisons
