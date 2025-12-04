"""
Training API Routes.

@TENSOR @PRISM - Training orchestration and experiment tracking endpoints.
"""

from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from src.models.training import (
    TrainingJob,
    TrainingConfig,
    TrainingStatus,
    HyperparameterConfig,
    ExperimentResult,
)
from src.services.training import TrainingOrchestrator
from src.services.mlflow_tracker import MLflowTracker

router = APIRouter()

# Service dependencies
_training_service: Optional[TrainingOrchestrator] = None
_mlflow_tracker: Optional[MLflowTracker] = None


async def get_training_service() -> TrainingOrchestrator:
    """Dependency to get training orchestrator."""
    global _training_service
    if _training_service is None:
        _training_service = TrainingOrchestrator()
        await _training_service.initialize()
    return _training_service


async def get_mlflow() -> MLflowTracker:
    """Dependency to get MLflow tracker."""
    global _mlflow_tracker
    if _mlflow_tracker is None:
        _mlflow_tracker = MLflowTracker()
        await _mlflow_tracker.initialize()
    return _mlflow_tracker


class CreateJobRequest(BaseModel):
    """Request to create a training job."""
    name: str
    description: str = ""
    config: TrainingConfig
    hyperparameters: Optional[HyperparameterConfig] = None
    tags: list[str] = Field(default_factory=list)


class JobResponse(BaseModel):
    """Training job response."""
    job: TrainingJob


class JobsListResponse(BaseModel):
    """List of training jobs."""
    jobs: list[TrainingJob]
    total: int


class ExperimentResponse(BaseModel):
    """Experiment details response."""
    experiment_id: str
    name: str
    runs: list[dict]


class ExperimentsListResponse(BaseModel):
    """List of experiments."""
    experiments: list[dict]


class MetricsRequest(BaseModel):
    """Log metrics request."""
    run_id: str
    metrics: dict[str, float]
    step: Optional[int] = None


class HPORequest(BaseModel):
    """Hyperparameter optimization request."""
    job_id: str
    n_trials: int = 20
    timeout: Optional[int] = None  # seconds
    search_space: dict = Field(default_factory=dict)


class HPOResponse(BaseModel):
    """HPO result response."""
    best_params: dict
    best_value: float
    n_trials_completed: int


@router.post("/jobs", response_model=JobResponse)
async def create_job(
    request: CreateJobRequest,
    background_tasks: BackgroundTasks,
    training: TrainingOrchestrator = Depends(get_training_service),
) -> JobResponse:
    """
    Create a new training job.
    
    @TENSOR - Creates and optionally starts a training job.
    
    Example:
        ```json
        {
            "name": "fine-tune-llama",
            "description": "Fine-tune Llama 3.2 on custom data",
            "config": {
                "model_type": "llm",
                "base_model": "llama3.2:latest",
                "dataset_path": "/data/training/custom.jsonl",
                "batch_size": 4,
                "learning_rate": 2e-5,
                "epochs": 3
            }
        }
        ```
    """
    try:
        job = await training.create_job(
            name=request.name,
            description=request.description,
            config=request.config,
            hyperparameters=request.hyperparameters,
            tags=request.tags,
        )
        return JobResponse(job=job)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/start", response_model=JobResponse)
async def start_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    training: TrainingOrchestrator = Depends(get_training_service),
) -> JobResponse:
    """
    Start a training job.
    
    @TENSOR @FLUX - Launches training in background.
    """
    job = await training.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Start training in background
    background_tasks.add_task(training.run_training, job_id)
    
    job.status = TrainingStatus.RUNNING
    return JobResponse(job=job)


@router.post("/jobs/{job_id}/stop", response_model=JobResponse)
async def stop_job(
    job_id: str,
    training: TrainingOrchestrator = Depends(get_training_service),
) -> JobResponse:
    """
    Stop a running training job.
    
    Gracefully stops the training process.
    """
    success = await training.stop_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or not running")
    
    job = await training.get_job(job_id)
    return JobResponse(job=job)


@router.get("/jobs", response_model=JobsListResponse)
async def list_jobs(
    status: Optional[TrainingStatus] = None,
    limit: int = 50,
    offset: int = 0,
    training: TrainingOrchestrator = Depends(get_training_service),
) -> JobsListResponse:
    """
    List training jobs.
    
    Optionally filter by status.
    """
    jobs = await training.list_jobs(status=status, limit=limit, offset=offset)
    return JobsListResponse(jobs=jobs, total=len(jobs))


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    training: TrainingOrchestrator = Depends(get_training_service),
) -> JobResponse:
    """
    Get training job details.
    """
    job = await training.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(job=job)


@router.get("/jobs/{job_id}/status")
async def get_job_status(
    job_id: str,
    training: TrainingOrchestrator = Depends(get_training_service),
) -> TrainingStatus:
    """
    Get real-time training status.
    
    @SENTRY - Returns current epoch, loss, and metrics.
    """
    status = await training.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str,
    training: TrainingOrchestrator = Depends(get_training_service),
) -> dict:
    """
    Delete a training job.
    
    Cannot delete running jobs - stop first.
    """
    success = await training.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"deleted": True, "job_id": job_id}


# =============================================================================
# Hyperparameter Optimization
# =============================================================================

@router.post("/optimize", response_model=HPOResponse)
async def optimize_hyperparameters(
    request: HPORequest,
    background_tasks: BackgroundTasks,
    training: TrainingOrchestrator = Depends(get_training_service),
) -> HPOResponse:
    """
    Run hyperparameter optimization.
    
    @TENSOR @PRISM @VELOCITY - Uses Optuna for Bayesian optimization.
    
    Finds optimal hyperparameters for a training configuration.
    """
    try:
        result = await training.optimize_hyperparameters(
            job_id=request.job_id,
            n_trials=request.n_trials,
            timeout=request.timeout,
        )
        return HPOResponse(
            best_params=result["best_params"],
            best_value=result["best_value"],
            n_trials_completed=result["n_trials"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MLflow Integration
# =============================================================================

@router.get("/experiments", response_model=ExperimentsListResponse)
async def list_experiments(
    mlflow: MLflowTracker = Depends(get_mlflow),
) -> ExperimentsListResponse:
    """
    List MLflow experiments.
    
    @SENTRY - Returns all tracked experiments.
    """
    experiments = await mlflow.list_experiments()
    return ExperimentsListResponse(experiments=experiments)


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str,
    mlflow: MLflowTracker = Depends(get_mlflow),
) -> ExperimentResponse:
    """
    Get experiment details with runs.
    """
    experiment = await mlflow.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return ExperimentResponse(**experiment)


@router.post("/experiments/{experiment_id}/runs")
async def create_run(
    experiment_id: str,
    mlflow: MLflowTracker = Depends(get_mlflow),
) -> dict:
    """
    Create a new MLflow run.
    """
    run_id = await mlflow.start_run(experiment_id)
    return {"run_id": run_id, "experiment_id": experiment_id}


@router.post("/runs/{run_id}/metrics")
async def log_metrics(
    run_id: str,
    request: MetricsRequest,
    mlflow: MLflowTracker = Depends(get_mlflow),
) -> dict:
    """
    Log metrics to an MLflow run.
    
    @SENTRY - Track training metrics.
    """
    await mlflow.log_metrics(
        run_id=run_id,
        metrics=request.metrics,
        step=request.step,
    )
    return {"logged": True, "run_id": run_id}


@router.post("/runs/{run_id}/end")
async def end_run(
    run_id: str,
    status: str = "FINISHED",
    mlflow: MLflowTracker = Depends(get_mlflow),
) -> dict:
    """
    End an MLflow run.
    """
    await mlflow.end_run(run_id, status)
    return {"ended": True, "run_id": run_id, "status": status}


# =============================================================================
# Model Registry
# =============================================================================

@router.post("/models/register")
async def register_model(
    run_id: str,
    model_name: str,
    model_path: str = "model",
    mlflow: MLflowTracker = Depends(get_mlflow),
) -> dict:
    """
    Register a trained model.
    
    @TENSOR - Registers model in MLflow Model Registry.
    """
    version = await mlflow.register_model(
        run_id=run_id,
        model_name=model_name,
        model_path=model_path,
    )
    return {
        "registered": True,
        "model_name": model_name,
        "version": version,
    }


@router.get("/models")
async def list_registered_models(
    mlflow: MLflowTracker = Depends(get_mlflow),
) -> dict:
    """
    List registered models.
    """
    models = await mlflow.list_registered_models()
    return {"models": models}


@router.post("/models/{model_name}/stage")
async def transition_model_stage(
    model_name: str,
    version: int,
    stage: str,
    mlflow: MLflowTracker = Depends(get_mlflow),
) -> dict:
    """
    Transition model to a new stage.
    
    Stages: None, Staging, Production, Archived
    """
    await mlflow.transition_model_stage(
        model_name=model_name,
        version=version,
        stage=stage,
    )
    return {
        "transitioned": True,
        "model_name": model_name,
        "version": version,
        "stage": stage,
    }
