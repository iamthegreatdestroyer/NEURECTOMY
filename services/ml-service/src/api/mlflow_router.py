"""
MLflow API Router

Provides REST API endpoints for MLflow experiment tracking, run management,
artifact storage, and model registry integration.

Endpoints:
- Experiment Management: create, list, get, search experiments
- Run Management: create, update, log metrics/params/tags, search runs
- Artifact Management: upload, download, list artifacts
- Model Registry: register models, manage versions

@TENSOR @VELOCITY - ML experiment tracking and optimization
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Depends
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
import structlog
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

logger = structlog.get_logger()

router = APIRouter()

# Initialize MLflow client
mlflow_client = MlflowClient()


# ==============================================================================
# Request/Response Models
# ==============================================================================

class ExperimentCreate(BaseModel):
    """Request model for creating an experiment."""
    name: str = Field(..., description="Unique experiment name")
    artifact_location: Optional[str] = Field(None, description="S3 path for artifacts")
    tags: Optional[Dict[str, str]] = Field(default_factory=dict, description="Experiment tags")


class ExperimentResponse(BaseModel):
    """Response model for experiment data."""
    experiment_id: str
    name: str
    artifact_location: Optional[str]
    lifecycle_stage: str
    creation_time: Optional[int]
    last_update_time: Optional[int]
    tags: Dict[str, str] = Field(default_factory=dict)


class ExperimentListResponse(BaseModel):
    """Response model for listing experiments."""
    experiments: List[ExperimentResponse]
    total: int


class ExperimentSearchRequest(BaseModel):
    """Request model for searching experiments."""
    filter_string: Optional[str] = Field(None, description="MLflow filter string")
    max_results: int = Field(100, ge=1, le=10000)
    order_by: Optional[List[str]] = Field(None, description="Order by clauses")
    view_type: str = Field("ACTIVE_ONLY", description="ACTIVE_ONLY, DELETED_ONLY, or ALL")


class RunCreate(BaseModel):
    """Request model for creating a run."""
    experiment_id: str = Field(..., description="Experiment ID")
    run_name: Optional[str] = Field(None, description="Run name")
    start_time: Optional[int] = Field(None, description="Start time in milliseconds")
    tags: Optional[Dict[str, str]] = Field(default_factory=dict)


class RunResponse(BaseModel):
    """Response model for run data."""
    run_id: str
    experiment_id: str
    status: str
    start_time: Optional[int]
    end_time: Optional[int]
    artifact_uri: Optional[str]
    lifecycle_stage: str


class MetricLog(BaseModel):
    """Request model for logging a single metric."""
    key: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    timestamp: Optional[int] = Field(None, description="Timestamp in milliseconds")
    step: Optional[int] = Field(0, description="Step number")


class ParameterLog(BaseModel):
    """Request model for logging a parameter."""
    key: str = Field(..., description="Parameter name")
    value: str = Field(..., description="Parameter value")


class TagLog(BaseModel):
    """Request model for logging a tag."""
    key: str = Field(..., description="Tag key")
    value: str = Field(..., description="Tag value")


class BatchLog(BaseModel):
    """Request model for batch logging."""
    metrics: Optional[List[MetricLog]] = Field(default_factory=list)
    params: Optional[List[ParameterLog]] = Field(default_factory=list)
    tags: Optional[List[TagLog]] = Field(default_factory=list)


class RunUpdate(BaseModel):
    """Request model for updating a run."""
    status: Optional[str] = Field(None, description="Run status: RUNNING, SCHEDULED, FINISHED, FAILED, KILLED")
    end_time: Optional[int] = Field(None, description="End time in milliseconds")
    run_name: Optional[str] = Field(None, description="Update run name")


class RunSearchRequest(BaseModel):
    """Request model for searching runs."""
    experiment_ids: Optional[List[str]] = Field(None, description="List of experiment IDs")
    filter_string: Optional[str] = Field(None, description="MLflow filter string")
    max_results: int = Field(100, ge=1, le=10000)
    order_by: Optional[List[str]] = Field(None, description="Order by clauses")


class ArtifactListResponse(BaseModel):
    """Response model for listing artifacts."""
    artifacts: List[Dict[str, Any]]
    root_uri: str


class ModelVersionRegister(BaseModel):
    """Request model for registering a model version."""
    name: str = Field(..., description="Model name")
    source: str = Field(..., description="Source artifact path")
    run_id: Optional[str] = Field(None, description="Run ID")
    tags: Optional[Dict[str, str]] = Field(default_factory=dict)
    description: Optional[str] = Field(None, description="Model description")


# ==============================================================================
# Experiment Endpoints
# ==============================================================================

@router.post("/experiments/create", status_code=201, response_model=ExperimentResponse)
async def create_experiment(experiment: ExperimentCreate):
    """
    Create a new MLflow experiment.
    
    Returns 201 with experiment_id on success.
    Returns 400 if experiment with same name already exists.
    """
    try:
        # Check if experiment exists
        existing = mlflow_client.get_experiment_by_name(experiment.name)
        if existing is not None:
            raise HTTPException(
                status_code=400,
                detail=f"Experiment '{experiment.name}' already exists with ID: {existing.experiment_id}"
            )
        
        # Create experiment
        experiment_id = mlflow_client.create_experiment(
            name=experiment.name,
            artifact_location=experiment.artifact_location,
            tags=experiment.tags or {}
        )
        
        # Fetch and return created experiment
        exp = mlflow_client.get_experiment(experiment_id)
        
        logger.info(
            "experiment_created",
            experiment_id=experiment_id,
            name=experiment.name
        )
        
        return ExperimentResponse(
            experiment_id=exp.experiment_id,
            name=exp.name,
            artifact_location=exp.artifact_location,
            lifecycle_stage=exp.lifecycle_stage,
            creation_time=exp.creation_time,
            last_update_time=exp.last_update_time,
            tags=exp.tags or {}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("experiment_create_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create experiment: {str(e)}")


@router.get("/experiments/list", response_model=ExperimentListResponse)
async def list_experiments(
    view_type: str = Query("ACTIVE_ONLY", description="ACTIVE_ONLY, DELETED_ONLY, or ALL"),
    max_results: int = Query(100, ge=1, le=10000)
):
    """List all experiments."""
    try:
        # Map view_type string to ViewType enum
        view_type_map = {
            "ACTIVE_ONLY": ViewType.ACTIVE_ONLY,
            "DELETED_ONLY": ViewType.DELETED_ONLY,
            "ALL": ViewType.ALL
        }
        view = view_type_map.get(view_type, ViewType.ACTIVE_ONLY)
        
        experiments = mlflow_client.search_experiments(
            view_type=view,
            max_results=max_results
        )
        
        exp_list = [
            ExperimentResponse(
                experiment_id=exp.experiment_id,
                name=exp.name,
                artifact_location=exp.artifact_location,
                lifecycle_stage=exp.lifecycle_stage,
                creation_time=exp.creation_time,
                last_update_time=exp.last_update_time,
                tags=exp.tags or {}
            )
            for exp in experiments
        ]
        
        return ExperimentListResponse(
            experiments=exp_list,
            total=len(exp_list)
        )
        
    except Exception as e:
        logger.error("list_experiments_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list experiments: {str(e)}")


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str):
    """Get experiment by ID."""
    try:
        exp = mlflow_client.get_experiment(experiment_id)
        
        if exp is None:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
        
        return ExperimentResponse(
            experiment_id=exp.experiment_id,
            name=exp.name,
            artifact_location=exp.artifact_location,
            lifecycle_stage=exp.lifecycle_stage,
            creation_time=exp.creation_time,
            last_update_time=exp.last_update_time,
            tags=exp.tags or {}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_experiment_failed", experiment_id=experiment_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get experiment: {str(e)}")


@router.get("/experiments/by-name/{name}", response_model=ExperimentResponse)
async def get_experiment_by_name(name: str):
    """Get experiment by name."""
    try:
        exp = mlflow_client.get_experiment_by_name(name)
        
        if exp is None:
            raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found")
        
        return ExperimentResponse(
            experiment_id=exp.experiment_id,
            name=exp.name,
            artifact_location=exp.artifact_location,
            lifecycle_stage=exp.lifecycle_stage,
            creation_time=exp.creation_time,
            last_update_time=exp.last_update_time,
            tags=exp.tags or {}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_experiment_by_name_failed", name=name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get experiment: {str(e)}")


@router.post("/experiments/search", response_model=ExperimentListResponse)
async def search_experiments(request: ExperimentSearchRequest):
    """Search experiments with filters."""
    try:
        # Map view_type string to ViewType enum
        view_type_map = {
            "ACTIVE_ONLY": ViewType.ACTIVE_ONLY,
            "DELETED_ONLY": ViewType.DELETED_ONLY,
            "ALL": ViewType.ALL
        }
        view = view_type_map.get(request.view_type, ViewType.ACTIVE_ONLY)
        
        experiments = mlflow_client.search_experiments(
            view_type=view,
            max_results=request.max_results,
            filter_string=request.filter_string,
            order_by=request.order_by
        )
        
        exp_list = [
            ExperimentResponse(
                experiment_id=exp.experiment_id,
                name=exp.name,
                artifact_location=exp.artifact_location,
                lifecycle_stage=exp.lifecycle_stage,
                creation_time=exp.creation_time,
                last_update_time=exp.last_update_time,
                tags=exp.tags or {}
            )
            for exp in experiments
        ]
        
        return ExperimentListResponse(
            experiments=exp_list,
            total=len(exp_list)
        )
        
    except Exception as e:
        logger.error("search_experiments_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to search experiments: {str(e)}")


@router.post("/experiments/{experiment_id}/delete", status_code=200)
async def delete_experiment(experiment_id: str):
    """Delete (soft delete) an experiment."""
    try:
        mlflow_client.delete_experiment(experiment_id)
        
        logger.info("experiment_deleted", experiment_id=experiment_id)
        
        return {"status": "success", "experiment_id": experiment_id}
        
    except Exception as e:
        logger.error("delete_experiment_failed", experiment_id=experiment_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete experiment: {str(e)}")


# ==============================================================================
# Run Endpoints
# ==============================================================================

@router.post("/runs/create", status_code=201, response_model=RunResponse)
async def create_run(run: RunCreate):
    """Create a new MLflow run."""
    try:
        # Verify experiment exists
        exp = mlflow_client.get_experiment(run.experiment_id)
        if exp is None:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment {run.experiment_id} not found"
            )
        
        # Create run
        mlflow_run = mlflow_client.create_run(
            experiment_id=run.experiment_id,
            start_time=run.start_time or int(datetime.now().timestamp() * 1000),
            tags=run.tags or {}
        )
        
        # Set run name if provided
        if run.run_name:
            mlflow_client.set_tag(mlflow_run.info.run_id, "mlflow.runName", run.run_name)
        
        logger.info(
            "run_created",
            run_id=mlflow_run.info.run_id,
            experiment_id=run.experiment_id
        )
        
        return RunResponse(
            run_id=mlflow_run.info.run_id,
            experiment_id=mlflow_run.info.experiment_id,
            status=mlflow_run.info.status,
            start_time=mlflow_run.info.start_time,
            end_time=mlflow_run.info.end_time,
            artifact_uri=mlflow_run.info.artifact_uri,
            lifecycle_stage=mlflow_run.info.lifecycle_stage
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("create_run_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create run: {str(e)}")


@router.post("/runs/{run_id}/log-metric", status_code=200)
async def log_metric(run_id: str, metric: MetricLog):
    """Log a single metric to a run."""
    try:
        mlflow_client.log_metric(
            run_id=run_id,
            key=metric.key,
            value=metric.value,
            timestamp=metric.timestamp or int(datetime.now().timestamp() * 1000),
            step=metric.step or 0
        )
        
        return {"status": "success", "run_id": run_id, "metric": metric.key}
        
    except Exception as e:
        logger.error("log_metric_failed", run_id=run_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to log metric: {str(e)}")


@router.post("/runs/{run_id}/log-parameter", status_code=200)
async def log_parameter(run_id: str, param: ParameterLog):
    """Log a parameter to a run."""
    try:
        mlflow_client.log_param(
            run_id=run_id,
            key=param.key,
            value=param.value
        )
        
        return {"status": "success", "run_id": run_id, "parameter": param.key}
        
    except Exception as e:
        logger.error("log_parameter_failed", run_id=run_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to log parameter: {str(e)}")


@router.post("/runs/{run_id}/log-batch", status_code=200)
async def log_batch(run_id: str, batch: BatchLog):
    """Log metrics, parameters, and tags in batch."""
    try:
        # Log metrics
        for metric in batch.metrics:
            mlflow_client.log_metric(
                run_id=run_id,
                key=metric.key,
                value=metric.value,
                timestamp=metric.timestamp or int(datetime.now().timestamp() * 1000),
                step=metric.step or 0
            )
        
        # Log parameters
        for param in batch.params:
            mlflow_client.log_param(
                run_id=run_id,
                key=param.key,
                value=param.value
            )
        
        # Log tags
        for tag in batch.tags:
            mlflow_client.set_tag(
                run_id=run_id,
                key=tag.key,
                value=tag.value
            )
        
        logger.info(
            "batch_logged",
            run_id=run_id,
            metrics_count=len(batch.metrics),
            params_count=len(batch.params),
            tags_count=len(batch.tags)
        )
        
        return {
            "status": "success",
            "run_id": run_id,
            "logged": {
                "metrics": len(batch.metrics),
                "parameters": len(batch.params),
                "tags": len(batch.tags)
            }
        }
        
    except Exception as e:
        logger.error("log_batch_failed", run_id=run_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to log batch: {str(e)}")


@router.post("/runs/{run_id}/update", status_code=200)
async def update_run(run_id: str, update: RunUpdate):
    """Update run status and metadata."""
    try:
        # Update status if provided
        if update.status:
            mlflow_client.set_terminated(
                run_id=run_id,
                status=update.status,
                end_time=update.end_time or int(datetime.now().timestamp() * 1000)
            )
        
        # Update run name if provided
        if update.run_name:
            mlflow_client.set_tag(run_id, "mlflow.runName", update.run_name)
        
        logger.info("run_updated", run_id=run_id, status=update.status)
        
        return {"status": "success", "run_id": run_id}
        
    except Exception as e:
        logger.error("update_run_failed", run_id=run_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update run: {str(e)}")


@router.post("/runs/search")
async def search_runs(request: RunSearchRequest):
    """Search runs with filters."""
    try:
        runs = mlflow_client.search_runs(
            experiment_ids=request.experiment_ids or [],
            filter_string=request.filter_string,
            max_results=request.max_results,
            order_by=request.order_by
        )
        
        run_list = [
            {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "artifact_uri": run.info.artifact_uri,
                "lifecycle_stage": run.info.lifecycle_stage,
                "metrics": {k: v for k, v in run.data.metrics.items()},
                "params": {k: v for k, v in run.data.params.items()},
                "tags": {k: v for k, v in run.data.tags.items()}
            }
            for run in runs
        ]
        
        return {
            "runs": run_list,
            "total": len(run_list)
        }
        
    except Exception as e:
        logger.error("search_runs_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to search runs: {str(e)}")


# ==============================================================================
# Artifact Endpoints
# ==============================================================================

@router.post("/runs/{run_id}/log-artifact", status_code=200)
async def log_artifact(
    run_id: str,
    file: UploadFile = File(...),
    artifact_path: Optional[str] = Form(None)
):
    """Upload an artifact file to a run."""
    try:
        # Get run info to find artifact location
        run = mlflow_client.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        
        # Create temp directory for artifact
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save uploaded file
            if artifact_path:
                file_path = temp_path / artifact_path / file.filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                file_path = temp_path / file.filename
            
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Log artifact to MLflow
            if artifact_path:
                mlflow_client.log_artifact(run_id, str(file_path), artifact_path)
            else:
                mlflow_client.log_artifact(run_id, str(file_path))
        
        logger.info(
            "artifact_logged",
            run_id=run_id,
            filename=file.filename,
            artifact_path=artifact_path
        )
        
        return {
            "status": "success",
            "run_id": run_id,
            "filename": file.filename,
            "artifact_path": artifact_path
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("log_artifact_failed", run_id=run_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to log artifact: {str(e)}")


@router.get("/runs/{run_id}/artifacts/list", response_model=ArtifactListResponse)
async def list_artifacts(run_id: str, path: Optional[str] = Query(None)):
    """List artifacts in a run."""
    try:
        artifacts = mlflow_client.list_artifacts(run_id, path=path)
        
        artifact_list = [
            {
                "path": artifact.path,
                "is_dir": artifact.is_dir,
                "file_size": artifact.file_size
            }
            for artifact in artifacts
        ]
        
        run = mlflow_client.get_run(run_id)
        
        return ArtifactListResponse(
            artifacts=artifact_list,
            root_uri=run.info.artifact_uri
        )
        
    except Exception as e:
        logger.error("list_artifacts_failed", run_id=run_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list artifacts: {str(e)}")


@router.get("/runs/{run_id}/artifacts/download")
async def download_artifact(run_id: str, path: str = Query(..., description="Artifact path")):
    """Download an artifact file."""
    try:
        # Download artifact to temp location
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = mlflow_client.download_artifacts(run_id, path, dst_path=temp_dir)
            
            # Return file
            if Path(local_path).is_file():
                return FileResponse(
                    path=local_path,
                    filename=Path(path).name,
                    media_type="application/octet-stream"
                )
            else:
                raise HTTPException(status_code=404, detail=f"Artifact '{path}' not found or is a directory")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("download_artifact_failed", run_id=run_id, path=path, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to download artifact: {str(e)}")


# ==============================================================================
# Model Registry Endpoints
# ==============================================================================

@router.post("/model-versions/register", status_code=201)
async def register_model(model: ModelVersionRegister):
    """Register a model version in the MLflow Model Registry."""
    try:
        # Create registered model if doesn't exist
        try:
            mlflow_client.create_registered_model(
                name=model.name,
                tags=model.tags,
                description=model.description
            )
        except Exception:
            # Model already exists, that's fine
            pass
        
        # Create model version
        model_version = mlflow_client.create_model_version(
            name=model.name,
            source=model.source,
            run_id=model.run_id,
            tags=model.tags,
            description=model.description
        )
        
        logger.info(
            "model_registered",
            name=model.name,
            version=model_version.version,
            run_id=model.run_id
        )
        
        return {
            "status": "success",
            "name": model.name,
            "version": model_version.version,
            "source": model.source,
            "run_id": model.run_id
        }
        
    except Exception as e:
        logger.error("register_model_failed", name=model.name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to register model: {str(e)}")
