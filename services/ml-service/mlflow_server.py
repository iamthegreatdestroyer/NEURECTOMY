"""
MLflow Server

FastAPI wrapper for MLflow Tracking Server with REST API endpoints.
Handles experiment tracking, run management, metrics logging, and model registry.
"""

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus, ViewType
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from pathlib import Path
from datetime import datetime
import logging

from config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Initialize MLflow
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow_client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

# Create router
router = APIRouter(prefix="/api/mlflow", tags=["mlflow"])

# ============================================================================
# Request/Response Models
# ============================================================================

class CreateExperimentRequest(BaseModel):
    name: str = Field(..., description="Experiment name")
    artifact_location: Optional[str] = Field(None, description="Artifact storage location")
    tags: Optional[Dict[str, str]] = Field(default_factory=dict, description="Experiment tags")


class CreateExperimentResponse(BaseModel):
    experiment_id: str
    name: str
    artifact_location: str
    lifecycle_stage: str
    tags: Dict[str, str]


class StartRunRequest(BaseModel):
    experiment_id: str = Field(..., description="Experiment ID to create run in")
    user_id: Optional[str] = Field(None, description="User ID starting the run")
    start_time: Optional[int] = Field(None, description="Start timestamp (ms)")
    tags: Optional[Dict[str, str]] = Field(default_factory=dict, description="Run tags")
    run_name: Optional[str] = Field(None, description="Run name")


class StartRunResponse(BaseModel):
    run_id: str
    run_uuid: str  # MLflow compatibility alias for run_id
    experiment_id: str
    status: str
    start_time: int
    artifact_uri: str


class LogMetricsRequest(BaseModel):
    run_id: str = Field(..., description="Run ID to log metrics for")
    metrics: List[Dict[str, Any]] = Field(..., description="List of metric dicts with key, value, timestamp, step")


class LogParamsRequest(BaseModel):
    run_id: str = Field(..., description="Run ID to log parameters for")
    params: List[Dict[str, str]] = Field(..., description="List of param dicts with key, value")


class LogMetricRequest(BaseModel):
    """Request model for logging a single metric"""
    key: str = Field(..., description="Metric key/name")
    value: float = Field(..., description="Metric value")
    timestamp: Optional[int] = Field(None, description="Timestamp (ms)")
    step: int = Field(0, description="Step number")


class LogParameterRequest(BaseModel):
    """Request model for logging a single parameter"""
    key: str = Field(..., description="Parameter key/name")
    value: str = Field(..., description="Parameter value")


class LogBatchRequest(BaseModel):
    """Request model for batch logging"""
    metrics: Optional[List[Dict[str, Any]]] = Field(None, description="List of metrics")
    params: Optional[List[Dict[str, str]]] = Field(None, description="List of parameters")
    tags: Optional[List[Dict[str, str]]] = Field(None, description="List of tags")


class UpdateRunRequest(BaseModel):
    """Request model for updating run status"""
    status: str = Field(..., description="Run status: FINISHED, FAILED, KILLED, RUNNING")
    end_time: Optional[int] = Field(None, description="End timestamp (ms)")


class EndRunRequest(BaseModel):
    run_id: str = Field(..., description="Run ID to end")
    status: str = Field("FINISHED", description="Final status: FINISHED, FAILED, KILLED")
    end_time: Optional[int] = Field(None, description="End timestamp (ms)")


class RegisterModelRequest(BaseModel):
    run_id: str = Field(..., description="Run ID containing the model")
    model_name: str = Field(..., description="Model name in registry")
    description: Optional[str] = Field(None, description="Model description")
    tags: Optional[Dict[str, str]] = Field(default_factory=dict, description="Model tags")


# ============================================================================
# Experiment Endpoints
# ============================================================================

@router.post("/experiments/create", response_model=CreateExperimentResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(request: CreateExperimentRequest):
    """
    Create a new MLflow experiment
    
    Creates an experiment for organizing related training runs.
    Experiments provide logical grouping and can have associated tags.
    """
    try:
        logger.info(f"Creating experiment: {request.name}")
        
        # Create experiment
        experiment_id = mlflow_client.create_experiment(
            name=request.name,
            artifact_location=request.artifact_location,
            tags=request.tags,
        )
        
        # Get experiment details
        experiment = mlflow_client.get_experiment(experiment_id)
        
        logger.info(f"Created experiment {experiment_id}: {request.name}")
        
        return CreateExperimentResponse(
            experiment_id=experiment.experiment_id,
            name=experiment.name,
            artifact_location=experiment.artifact_location,
            lifecycle_stage=experiment.lifecycle_stage,
            tags=dict(experiment.tags) if experiment.tags else {},
        )
        
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create experiment: {str(e)}",
        )


@router.get("/experiments/list")
async def list_experiments(
    view_type: str = "ACTIVE_ONLY",
    max_results: int = 100,
):
    """
    List all experiments with optional filtering
    
    Returns experiments based on lifecycle stage filter.
    """
    try:
        logger.info(f"Listing experiments (view_type={view_type})")
        
        # Map view type to lowercase - MLflow expects "active_only", not "ACTIVE_ONLY"
        view_type_str = view_type.lower()  # Keep underscores, just lowercase
        view_type_enum = ViewType.from_string(view_type_str)
        
        # List experiments
        experiments = mlflow_client.search_experiments(
            view_type=view_type_enum,
            max_results=max_results,
        )
        
        return {
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "tags": dict(exp.tags) if exp.tags else {},
                    "creation_time": exp.creation_time,
                    "last_update_time": exp.last_update_time,
                }
                for exp in experiments
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list experiments: {str(e)}",
        )


@router.get("/experiments/{experiment_id}")
async def get_experiment_by_id(experiment_id: str):
    """
    Get experiment by ID
    
    Returns detailed information about a specific experiment.
    """
    try:
        logger.info(f"Getting experiment by ID: {experiment_id}")
        
        exp = mlflow_client.get_experiment(experiment_id)
        if exp is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        
        return {
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "artifact_location": exp.artifact_location,
            "lifecycle_stage": exp.lifecycle_stage,
            "tags": dict(exp.tags) if exp.tags else {},
            "creation_time": exp.creation_time,
            "last_update_time": exp.last_update_time,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get experiment {experiment_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get experiment: {str(e)}",
        )


@router.get("/experiments/by-name/{name}")
async def get_experiment_by_name(name: str):
    """
    Get experiment by name
    
    Returns experiment matching the specified name.
    """
    try:
        logger.info(f"Getting experiment by name: {name}")
        
        exp = mlflow_client.get_experiment_by_name(name)
        if exp is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment '{name}' not found"
            )
        
        return {
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "artifact_location": exp.artifact_location,
            "lifecycle_stage": exp.lifecycle_stage,
            "tags": dict(exp.tags) if exp.tags else {},
            "creation_time": exp.creation_time,
            "last_update_time": exp.last_update_time,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get experiment by name '{name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get experiment: {str(e)}",
        )


class ExperimentSearchRequest(BaseModel):
    """Request model for searching experiments"""
    filter_string: Optional[str] = Field(None, description="MLflow filter string")
    max_results: int = Field(100, ge=1, le=10000)
    order_by: Optional[List[str]] = Field(None, description="Order by clauses")
    view_type: str = Field("ACTIVE_ONLY", description="ACTIVE_ONLY, DELETED_ONLY, or ALL")


@router.post("/experiments/search")
async def search_experiments(request: ExperimentSearchRequest):
    """
    Search experiments with filters
    
    Advanced search with filtering and ordering capabilities.
    """
    try:
        logger.info(f"Searching experiments (filter={request.filter_string})")
        
        # Map view type to lowercase - Keep underscores
        view_type_str = request.view_type.lower()  # Keep underscores, just lowercase
        view_type_enum = ViewType.from_string(view_type_str)
        
        experiments = mlflow_client.search_experiments(
            view_type=view_type_enum,
            max_results=request.max_results,
            filter_string=request.filter_string,
            order_by=request.order_by,
        )
        
        return {
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "tags": dict(exp.tags) if exp.tags else {},
                    "creation_time": exp.creation_time,
                    "last_update_time": exp.last_update_time,
                }
                for exp in experiments
            ],
            "total": len(experiments)
        }
        
    except Exception as e:
        logger.error(f"Failed to search experiments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search experiments: {str(e)}",
        )


@router.post("/experiments/{experiment_id}/delete")
async def delete_experiment(experiment_id: str):
    """Delete an experiment (soft delete)"""
    try:
        logger.info(f"Deleting experiment: {experiment_id}")
        mlflow_client.delete_experiment(experiment_id)
        return {"message": f"Experiment {experiment_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete experiment: {str(e)}",
        )


# ============================================================================
# Run Endpoints
# ============================================================================

@router.post("/runs/create", response_model=StartRunResponse, status_code=status.HTTP_200_OK)
@router.post("/runs/start", response_model=StartRunResponse, status_code=status.HTTP_201_CREATED)
async def start_run(request: StartRunRequest):
    """
    Start a new training run
    
    Creates a run within an experiment to track metrics, parameters, and artifacts.
    """
    try:
        logger.info(f"Starting run in experiment {request.experiment_id}")
        
        # Create run
        run = mlflow_client.create_run(
            experiment_id=request.experiment_id,
            start_time=request.start_time or int(datetime.now().timestamp() * 1000),
            tags=request.tags,
            run_name=request.run_name,
        )
        
        logger.info(f"Started run {run.info.run_id}")
        
        return StartRunResponse(
            run_id=run.info.run_id,
            run_uuid=run.info.run_id,  # MLflow compatibility - same as run_id
            experiment_id=run.info.experiment_id,
            status=run.info.status,
            start_time=run.info.start_time,
            artifact_uri=run.info.artifact_uri,
        )
        
    except Exception as e:
        logger.error(f"Failed to start run: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start run: {str(e)}",
        )


@router.post("/runs/log-metrics")
async def log_metrics(request: LogMetricsRequest):
    """
    Log metrics for a training run
    
    Supports batch logging of multiple metrics at different steps.
    """
    try:
        logger.debug(f"Logging {len(request.metrics)} metrics for run {request.run_id}")
        
        # Log each metric
        for metric in request.metrics:
            mlflow_client.log_metric(
                run_id=request.run_id,
                key=metric["key"],
                value=metric["value"],
                timestamp=metric.get("timestamp", int(datetime.now().timestamp() * 1000)),
                step=metric.get("step", 0),
            )
        
        return {"message": f"Logged {len(request.metrics)} metrics successfully"}
        
    except Exception as e:
        logger.error(f"Failed to log metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log metrics: {str(e)}",
        )


@router.post("/runs/log-params")
async def log_params(request: LogParamsRequest):
    """Log parameters for a training run"""
    try:
        logger.debug(f"Logging {len(request.params)} params for run {request.run_id}")
        
        # Log each parameter
        for param in request.params:
            mlflow_client.log_param(
                run_id=request.run_id,
                key=param["key"],
                value=param["value"],
            )
        
        return {"message": f"Logged {len(request.params)} parameters successfully"}
        
    except Exception as e:
        logger.error(f"Failed to log parameters: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log parameters: {str(e)}",
        )


@router.post("/runs/end")
async def end_run(request: EndRunRequest):
    """End a training run and set final status"""
    try:
        logger.info(f"Ending run {request.run_id} with status {request.status}")
        
        # Map status string to enum
        status_enum = RunStatus.from_string(request.status)
        
        # End run
        mlflow_client.set_terminated(
            run_id=request.run_id,
            status=request.status,
            end_time=request.end_time or int(datetime.now().timestamp() * 1000),
        )
        
        return {"message": f"Run {request.run_id} ended successfully"}
        
    except Exception as e:
        logger.error(f"Failed to end run: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to end run: {str(e)}",
        )


@router.post("/runs/{run_id}/log-metric")
async def log_single_metric(run_id: str, request: LogMetricRequest):
    """Log a single metric for a run"""
    try:
        mlflow_client.log_metric(
            run_id=run_id,
            key=request.key,
            value=request.value,
            timestamp=request.timestamp or int(datetime.now().timestamp() * 1000),
            step=request.step,
        )
        return {"message": "Metric logged successfully"}
    except Exception as e:
        logger.error(f"Failed to log metric: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log metric: {str(e)}")


@router.post("/runs/{run_id}/log-parameter")
async def log_single_parameter(run_id: str, request: LogParameterRequest):
    """Log a single parameter for a run"""
    try:
        mlflow_client.log_param(run_id=run_id, key=request.key, value=request.value)
        return {"message": "Parameter logged successfully"}
    except Exception as e:
        logger.error(f"Failed to log parameter: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log parameter: {str(e)}")


@router.post("/runs/{run_id}/log-batch")
async def log_batch(run_id: str, request: LogBatchRequest):
    """Log batch of metrics, parameters, and tags"""
    try:
        if request.metrics:
            for metric in request.metrics:
                mlflow_client.log_metric(
                    run_id=run_id,
                    key=metric["key"],
                    value=metric["value"],
                    timestamp=metric.get("timestamp", int(datetime.now().timestamp() * 1000)),
                    step=metric.get("step", 0),
                )
        if request.params:
            for param in request.params:
                mlflow_client.log_param(run_id=run_id, key=param["key"], value=param["value"])
        if request.tags:
            for tag in request.tags:
                mlflow_client.set_tag(run_id=run_id, key=tag["key"], value=tag["value"])
        return {"message": "Batch logged successfully"}
    except Exception as e:
        logger.error(f"Failed to log batch: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log batch: {str(e)}")


@router.post("/runs/{run_id}/update")
async def update_run(run_id: str, request: UpdateRunRequest):
    """Update run status"""
    try:
        # Map status string to RunStatus enum
        status_map = {
            "FINISHED": "FINISHED",
            "FAILED": "FAILED",
            "KILLED": "KILLED",
            "RUNNING": "RUNNING",
        }
        run_status = status_map.get(request.status.upper(), "FINISHED")
        
        mlflow_client.set_terminated(
            run_id=run_id,
            status=run_status,
            end_time=request.end_time or int(datetime.now().timestamp() * 1000),
        )
        return {"message": f"Run updated to {request.status}"}
    except Exception as e:
        logger.error(f"Failed to update run: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update run: {str(e)}")


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Get detailed information about a specific run"""
    try:
        logger.debug(f"Getting run details: {run_id}")
        
        run = mlflow_client.get_run(run_id)
        
        return {
            "info": {
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "artifact_uri": run.info.artifact_uri,
            },
            "data": {
                "metrics": {k: v for k, v in run.data.metrics.items()},
                "params": {k: v for k, v in run.data.params.items()},
                "tags": {k: v for k, v in run.data.tags.items()},
            },
        }
        
    except Exception as e:
        logger.error(f"Failed to get run: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )


@router.get("/runs/{run_id}/metrics")
async def get_run_metrics(run_id: str, metric_key: Optional[str] = None):
    """Get metrics history for a specific run"""
    try:
        logger.debug(f"Getting metrics for run {run_id}")
        
        if metric_key:
            # Get specific metric history
            metric_history = mlflow_client.get_metric_history(run_id, metric_key)
            metrics = [
                {
                    "key": m.key,
                    "value": m.value,
                    "timestamp": m.timestamp,
                    "step": m.step,
                }
                for m in metric_history
            ]
        else:
            # Get all metrics
            run = mlflow_client.get_run(run_id)
            metrics = [
                {"key": k, "value": v, "timestamp": run.info.start_time, "step": 0}
                for k, v in run.data.metrics.items()
            ]
        
        return {"metrics": metrics}
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Failed to get metrics for run {run_id}",
        )


@router.post("/runs/search")
async def search_runs(
    experiment_ids: List[str],
    filter: Optional[str] = None,
    order_by: Optional[List[str]] = None,
    max_results: int = 100,
):
    """Search for runs across experiments with advanced filtering"""
    try:
        logger.info(f"Searching runs in {len(experiment_ids)} experiments")
        
        runs = mlflow_client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter,
            order_by=order_by,
            max_results=max_results,
        )
        
        return {
            "runs": [
                {
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "data": {
                        "metrics": [{"key": k, "value": v} for k, v in run.data.metrics.items()],
                        "params": [{"key": k, "value": v} for k, v in run.data.params.items()],
                        "tags": [{"key": k, "value": v} for k, v in run.data.tags.items()],
                    },
                }
                for run in runs
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to search runs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search runs: {str(e)}",
        )


# ============================================================================
# Model Registry Endpoints
# ============================================================================

@router.post("/models/register", status_code=status.HTTP_201_CREATED)
async def register_model(request: RegisterModelRequest):
    """Register a trained model to the MLflow Model Registry"""
    try:
        logger.info(f"Registering model {request.model_name} from run {request.run_id}")
        
        # Register model
        model_version = mlflow_client.create_model_version(
            name=request.model_name,
            source=f"runs:/{request.run_id}/model",
            run_id=request.run_id,
            description=request.description,
            tags=request.tags,
        )
        
        logger.info(f"Registered model version {model_version.version}")
        
        return {
            "name": model_version.name,
            "version": model_version.version,
            "creation_timestamp": model_version.creation_timestamp,
            "current_stage": model_version.current_stage,
            "source": model_version.source,
            "run_id": model_version.run_id,
            "status": model_version.status,
        }
        
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register model: {str(e)}",
        )


@router.get("/models/{model_name}")
async def get_model(model_name: str):
    """Get registered model by name"""
    try:
        logger.debug(f"Getting model: {model_name}")
        
        model = mlflow_client.get_registered_model(model_name)
        
        return {
            "name": model.name,
            "creation_timestamp": model.creation_timestamp,
            "last_updated_timestamp": model.last_updated_timestamp,
            "description": model.description,
            "latest_versions": [
                {
                    "version": v.version,
                    "current_stage": v.current_stage,
                    "creation_timestamp": v.creation_timestamp,
                    "source": v.source,
                    "run_id": v.run_id,
                    "status": v.status,
                }
                for v in model.latest_versions
            ],
            "tags": dict(model.tags) if model.tags else {},
        }
        
    except Exception as e:
        logger.error(f"Failed to get model: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_name} not found",
        )


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test MLflow connection
        experiments = mlflow_client.search_experiments(max_results=1)
        
        return {
            "status": "healthy",
            "service": "mlflow-server",
            "tracking_uri": settings.mlflow_tracking_uri,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MLflow service unhealthy: {str(e)}",
        )


# ============================================================================
# Artifact Endpoints
# ============================================================================

@router.post("/runs/{run_id}/log-artifact")
async def log_artifact(
    run_id: str,
    file: UploadFile = File(...),
    artifact_path: Optional[str] = Form(None)
):
    """Upload artifact file for a run"""
    try:
        import tempfile
        import os
        
        # Create temp file with original filename
        temp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(temp_dir, file.filename)
        
        content = await file.read()
        with open(tmp_path, 'wb') as f:
            f.write(content)
        
        try:
            # Log artifact to MLflow
            logger.info(f"Logging artifact '{file.filename}' to run {run_id}, artifact_path='{artifact_path}'")
            mlflow_client.log_artifact(
                run_id=run_id,
                local_path=tmp_path,
                artifact_path=artifact_path
            )
            logger.info(f"Successfully logged artifact '{file.filename}'")
            
            return {
                "message": "Artifact logged successfully",
                "filename": file.filename,
                "artifact_path": artifact_path or ""
            }
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
                
    except Exception as e:
        logger.error(f"Failed to log artifact: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log artifact: {str(e)}"
        )


@router.get("/runs/{run_id}/artifacts/list")
async def list_artifacts(run_id: str, path: Optional[str] = None):
    """List artifacts for a run, recursively including files in subdirectories"""
    try:
        def list_recursive(base_path: Optional[str] = None):
            """Recursively list all artifacts"""
            items = []
            artifacts = mlflow_client.list_artifacts(run_id=run_id, path=base_path)
            
            for artifact in artifacts:
                items.append({
                    "path": artifact.path,
                    "is_dir": artifact.is_dir,
                    "file_size": artifact.file_size
                })
                
                # If directory, recurse into it
                if artifact.is_dir:
                    items.extend(list_recursive(artifact.path))
            
            return items
        
        result = list_recursive(path)
        logger.info(f"Listed {len(result)} artifacts for run {run_id}: {[a['path'] for a in result]}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to list artifacts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list artifacts: {str(e)}"
        )


@router.get("/runs/{run_id}/artifacts/download")
async def download_artifact(run_id: str, path: str):
    """Download artifact file"""
    try:
        logger.info(f"Downloading artifact '{path}' from run {run_id}")
        
        # Download artifact to temp location
        local_path = mlflow_client.download_artifacts(run_id=run_id, path=path)
        logger.info(f"Downloaded to local path: {local_path}")
        
        # Return file
        from fastapi.responses import FileResponse
        return FileResponse(
            local_path,
            filename=Path(path).name,
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"Failed to download artifact '{path}' from run {run_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact not found: {str(e)}"
        )


@router.post("/model-versions/register")
async def register_model_version(request: dict):
    """Register model from run artifacts"""
    try:
        name = request.get("name")
        run_id = request.get("run_id")
        artifact_path = request.get("artifact_path", "model")
        description = request.get("description")
        tags = request.get("tags")
        
        if not name or not run_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="name and run_id are required"
            )
        
        # Register model
        import mlflow
        model_uri = f"runs:/{run_id}/{artifact_path}"
        model_version = mlflow.register_model(model_uri=model_uri, name=name)
        
        # Update description if provided
        if description:
            mlflow_client.update_model_version(
                name=name,
                version=model_version.version,
                description=description
            )
        
        # Set tags if provided
        if tags:
            for key, value in tags.items():
                mlflow_client.set_model_version_tag(
                    name=name,
                    version=model_version.version,
                    key=key,
                    value=value
                )
        
        return {
            "name": model_version.name,
            "version": model_version.version,
            "creation_timestamp": model_version.creation_timestamp,
            "last_updated_timestamp": model_version.last_updated_timestamp,
            "source": model_version.source,
            "run_id": model_version.run_id,
            "status": model_version.status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register model: {str(e)}"
        )
