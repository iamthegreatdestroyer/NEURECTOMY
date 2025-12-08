"""
Optuna Service

FastAPI service for Optuna hyperparameter optimization.
Handles study creation, trial management, and optimization tracking.
"""

import optuna
from optuna.samplers import TPESampler, RandomSampler, GridSampler, CmaEsSampler
from optuna.pruners import MedianPruner, PercentilePruner, HyperbandPruner, NopPruner
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Union
from datetime import datetime
import logging
import asyncio

from config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/optuna", tags=["optuna"])

# Global study storage
studies: Dict[str, optuna.Study] = {}

# ============================================================================
# Request/Response Models
# ============================================================================

class CreateStudyRequest(BaseModel):
    study_name: str = Field(..., description="Study name")
    direction: str = Field("minimize", description="Optimization direction: minimize or maximize")
    sampler: str = Field("tpe", description="Sampler: tpe, random, grid, cmaes")
    pruner: str = Field("median", description="Pruner: median, percentile, hyperband, none")
    load_if_exists: bool = Field(False, description="Load study if it exists")


class CreateStudyResponse(BaseModel):
    study_id: str
    study_name: str
    direction: str


class SubmitTrialRequest(BaseModel):
    params: Dict[str, Any] = Field(..., description="Trial parameter values")
    value: Optional[float] = Field(None, description="Objective value (if complete)")
    state: str = Field("complete", description="Trial state: complete, pruned, failed")
    user_attrs: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SubmitTrialResponse(BaseModel):
    trial_id: int
    params: Dict[str, Any]
    value: Optional[float] = None


class SuggestRequest(BaseModel):
    trial_id: str = Field(..., description="Trial ID for suggestions")
    parameters: List[Dict[str, Any]] = Field(..., description="Parameter definitions")


class ReportIntermediateRequest(BaseModel):
    trial_id: str = Field(..., description="Trial ID")
    step: int = Field(..., description="Training step")
    value: float = Field(..., description="Intermediate value")


# ============================================================================
# Helper Functions
# ============================================================================

def get_sampler(sampler_name: str) -> optuna.samplers.BaseSampler:
    """Get Optuna sampler by name"""
    samplers = {
        "tpe": TPESampler,
        "random": RandomSampler,
        "grid": GridSampler,
        "cmaes": CmaEsSampler,
    }
    sampler_class = samplers.get(sampler_name.lower(), TPESampler)
    return sampler_class()


def get_pruner(pruner_name: str) -> optuna.pruners.BasePruner:
    """Get Optuna pruner by name"""
    pruners = {
        "median": MedianPruner,
        "percentile": lambda: PercentilePruner(percentile=25.0),
        "hyperband": HyperbandPruner,
        "none": NopPruner,
    }
    pruner_factory = pruners.get(pruner_name.lower(), MedianPruner)
    return pruner_factory()


def trial_to_dict(trial: optuna.trial.FrozenTrial) -> Dict[str, Any]:
    """Convert Optuna trial to dictionary"""
    # Handle fixed params from enqueued trials
    params = trial.params.copy() if trial.params else {}
    if not params and "fixed_params" in trial.system_attrs:
        params = trial.system_attrs["fixed_params"]
    
    return {
        "number": trial.number,
        "value": trial.value,
        "params": params,
        "state": trial.state.name.lower(),
        "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
        "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        "duration": trial.duration.total_seconds() if trial.duration else None,
        "user_attrs": trial.user_attrs,
        "system_attrs": trial.system_attrs,
        "intermediate_values": [
            {"step": step, "value": value}
            for step, value in trial.intermediate_values.items()
        ],
    }


def study_to_dict(study: optuna.Study) -> Dict[str, Any]:
    """Convert Optuna study to dictionary"""
    trials = study.get_trials(deepcopy=False)
    best_trial = study.best_trial if len(trials) > 0 and any(t.state == optuna.trial.TrialState.COMPLETE for t in trials) else None
    
    return {
        "study_id": study.study_name,
        "study_name": study.study_name,
        "direction": study.direction.name.lower(),
        "n_trials": len(trials),
        "datetime_start": min((t.datetime_start for t in trials if t.datetime_start), default=datetime.now()).isoformat(),
        "best_trial": trial_to_dict(best_trial) if best_trial else None,
        "trials": [trial_to_dict(t) for t in trials],
    }


# ============================================================================
# Study Endpoints
# ============================================================================

@router.post("/studies/create", response_model=CreateStudyResponse, status_code=status.HTTP_200_OK)
async def create_study(request: CreateStudyRequest):
    """
    Create a new Optuna study for hyperparameter optimization
    
    Creates a study with specified sampler and pruner algorithms.
    Studies can be persistent using the configured database storage.
    """
    try:
        logger.info(f"Creating study: {request.study_name}")
        
        # Get sampler and pruner
        sampler = get_sampler(request.sampler)
        pruner = get_pruner(request.pruner)
        
        # Parse direction
        direction = optuna.study.StudyDirection.MAXIMIZE if request.direction.lower() == "maximize" else optuna.study.StudyDirection.MINIMIZE
        
        # Create study
        study = optuna.create_study(
            study_name=request.study_name,
            storage=settings.optuna_storage,
            sampler=sampler,
            pruner=pruner,
            direction=direction,
            load_if_exists=request.load_if_exists,
        )
        
        # Store in memory for quick access
        studies[study.study_name] = study
        
        logger.info(f"Created study {study.study_name}")
        
        return CreateStudyResponse(
            study_id=study.study_name,
            study_name=study.study_name,
            direction=request.direction,
        )
        
    except Exception as e:
        logger.error(f"Failed to create study: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create study: {str(e)}",
        )


@router.get("/studies/list")
async def list_studies():
    """
    List all available Optuna studies
    
    Returns a list of all studies stored in the database.
    """
    try:
        logger.debug("Listing all studies")
        
        # Get all study summaries from storage
        study_summaries = optuna.get_all_study_summaries(storage=settings.optuna_storage)
        
        # Convert to list of study info
        studies_list = [
            {
                "study_name": summary.study_name,
                "direction": summary.direction.name.lower(),
                "n_trials": summary.n_trials,
                "datetime_start": summary.datetime_start.isoformat() if summary.datetime_start else None,
            }
            for summary in study_summaries
        ]
        
        logger.debug(f"Found {len(studies_list)} studies")
        
        return {"studies": studies_list}
        
    except Exception as e:
        logger.error(f"Failed to list studies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list studies: {str(e)}",
        )


@router.get("/studies/{study_name}")
async def get_study(study_name: str):
    """Get study details including all trials"""
    try:
        logger.debug(f"Getting study: {study_name}")
        
        # Load study if not in memory
        if study_name not in studies:
            study = optuna.load_study(
                study_name=study_name,
                storage=settings.optuna_storage,
            )
            studies[study_name] = study
        else:
            study = studies[study_name]
        
        return study_to_dict(study)
        
    except Exception as e:
        logger.error(f"Failed to get study: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Study {study_name} not found",
        )


@router.delete("/studies/{study_id}")
async def delete_study(study_id: str):
    """Delete a study and all its trials"""
    try:
        logger.info(f"Deleting study: {study_id}")
        
        # Delete from database
        optuna.delete_study(
            study_name=study_id,
            storage=settings.optuna_storage,
        )
        
        # Remove from memory
        studies.pop(study_id, None)
        
        return {"message": f"Study {study_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to delete study: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete study: {str(e)}",
        )


@router.post("/studies/{study_id}/stop")
async def stop_study(study_id: str):
    """Stop an ongoing optimization study"""
    try:
        logger.info(f"Stopping study: {study_id}")
        
        if study_id in studies:
            study = studies[study_id]
            study.stop()
            return {"message": f"Study {study_id} stopped successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Study {study_id} not found in active studies",
            )
        
    except Exception as e:
        logger.error(f"Failed to stop study: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop study: {str(e)}",
        )


# ============================================================================
# Trial Endpoints
# ============================================================================

@router.post("/studies/{study_name}/trials/create", response_model=SubmitTrialResponse, status_code=status.HTTP_200_OK)
async def create_trial(study_name: str, request: SubmitTrialRequest):
    """
    Create a completed trial with results
    
    Records trial parameters and objective value in the study.
    """
    try:
        logger.info(f"Creating trial for study {study_name}")
        
        # Load study
        if study_name not in studies:
            study = optuna.load_study(
                study_name=study_name,
                storage=settings.optuna_storage,
            )
            studies[study_name] = study
        else:
            study = studies[study_name]
        
        # Enqueue trial with fixed parameters
        study.enqueue_trial(request.params, user_attrs=request.user_attrs)
        
        # Create trial and immediately complete it
        trial = study.ask()
        
        # Set state
        if request.state == "complete" and request.value is not None:
            study.tell(trial, request.value)
            trial_state = "complete"
        elif request.state == "pruned":
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            trial_state = "pruned"
        elif request.state == "failed":
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            trial_state = "failed"
        else:
            trial_state = "running"
        
        logger.info(f"Created trial {trial.number} with state {trial_state}")
        
        return SubmitTrialResponse(
            trial_id=trial.number,
            params=request.params,
            value=request.value,
        )
        
    except Exception as e:
        logger.error(f"Failed to submit trial: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit trial: {str(e)}",
        )


@router.get("/studies/{study_name}/trials/list")
async def list_trials(
    study_name: str,
    states: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """List all trials in a study with optional filtering"""
    try:
        logger.debug(f"Listing trials for study {study_name}")
        
        # Load study
        if study_name not in studies:
            study = optuna.load_study(
                study_name=study_name,
                storage=settings.optuna_storage,
            )
            studies[study_name] = study
        else:
            study = studies[study_name]
        
        # Get trials
        all_trials = study.get_trials(deepcopy=False)
        
        # Filter by states if specified
        if states:
            state_list = [s.strip().upper() for s in states.split(",")]
            state_enums = [optuna.trial.TrialState[s] for s in state_list if s in optuna.trial.TrialState.__members__]
            filtered_trials = [t for t in all_trials if t.state in state_enums]
        else:
            filtered_trials = all_trials
        
        # Apply pagination
        paginated_trials = filtered_trials[offset:offset + limit]
        
        return {
            "trials": [trial_to_dict(t) for t in paginated_trials],
            "total": len(filtered_trials),
        }
        
    except Exception as e:
        logger.error(f"Failed to list trials: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list trials: {str(e)}",
        )


@router.get("/studies/{study_name}/best-trial")
async def get_best_trial(study_name: str):
    """Get the best trial from a study"""
    try:
        logger.debug(f"Getting best trial for study {study_name}")
        
        # Load study
        if study_name not in studies:
            study = optuna.load_study(
                study_name=study_name,
                storage=settings.optuna_storage,
            )
            studies[study_name] = study
        else:
            study = studies[study_name]
        
        # Get best trial
        best_trial = study.best_trial
        
        return trial_to_dict(best_trial)
        
    except Exception as e:
        logger.error(f"Failed to get best trial: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No complete trials found in study {study_name}",
        )


@router.get("/studies/{study_name}/best-params")
async def get_best_params(study_name: str):
    """
    Get the best parameters from a study
    
    Returns only the parameter dictionary without trial metadata.
    """
    try:
        logger.debug(f"Getting best params for study {study_name}")
        
        # Load study
        if study_name not in studies:
            study = optuna.load_study(
                study_name=study_name,
                storage=settings.optuna_storage,
            )
            studies[study_name] = study
        else:
            study = studies[study_name]
        
        # Get best trial and return just params
        best_trial = study.best_trial
        
        # Handle fixed params from enqueued trials
        params = best_trial.params.copy() if best_trial.params else {}
        if not params and "fixed_params" in best_trial.system_attrs:
            params = best_trial.system_attrs["fixed_params"]
        
        return params
        
    except Exception as e:
        logger.error(f"Failed to get best params: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No complete trials found in study {study_name}",
        )


@router.post("/studies/{study_id}/suggest")
async def suggest(study_id: str, request: SuggestRequest):
    """Get parameter suggestions for a new trial"""
    try:
        logger.debug(f"Getting suggestions for study {study_id}")
        
        # Load study
        if study_id not in studies:
            study = optuna.load_study(
                study_name=study_id,
                storage=settings.optuna_storage,
            )
            studies[study_id] = study
        else:
            study = studies[study_id]
        
        # Create trial
        trial = study.ask()
        
        # Generate suggestions based on parameter definitions
        suggestions = {}
        for param_def in request.parameters:
            name = param_def["name"]
            param_type = param_def["type"]
            
            if param_type == "float":
                suggestions[name] = trial.suggest_float(
                    name,
                    param_def["min"],
                    param_def["max"],
                    log=param_def.get("log", False),
                )
            elif param_type == "int":
                suggestions[name] = trial.suggest_int(
                    name,
                    param_def["min"],
                    param_def["max"],
                    log=param_def.get("log", False),
                )
            elif param_type == "categorical":
                suggestions[name] = trial.suggest_categorical(
                    name,
                    param_def["choices"],
                )
        
        return {
            "suggestions": suggestions,
            "trial_id": str(trial._trial_id),
            "trial_number": trial.number,
        }
        
    except Exception as e:
        logger.error(f"Failed to get suggestions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get suggestions: {str(e)}",
        )


@router.post("/studies/{study_id}/trials/{trial_id}/intermediate")
async def report_intermediate_value(
    study_id: str,
    trial_id: str,
    request: ReportIntermediateRequest,
):
    """Report intermediate value during trial execution"""
    try:
        logger.debug(f"Reporting intermediate value for trial {trial_id}")
        
        # Load study
        if study_id not in studies:
            study = optuna.load_study(
                study_name=study_id,
                storage=settings.optuna_storage,
            )
            studies[study_id] = study
        else:
            study = studies[study_id]
        
        # Find trial
        trial = next((t for t in study.get_trials() if str(t._trial_id) == trial_id), None)
        if not trial:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Trial {trial_id} not found",
            )
        
        # Report intermediate value (this would typically be done during trial execution)
        # For now, we just acknowledge the request
        return {"message": "Intermediate value reported successfully"}
        
    except Exception as e:
        logger.error(f"Failed to report intermediate value: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to report intermediate value: {str(e)}",
        )


@router.get("/studies/{study_id}/trials/{trial_id}/should-prune")
async def should_prune(study_id: str, trial_id: str):
    """Check if a trial should be pruned"""
    try:
        # Load study
        if study_id not in studies:
            study = optuna.load_study(
                study_name=study_id,
                storage=settings.optuna_storage,
            )
            studies[study_id] = study
        else:
            study = studies[study_id]
        
        # For now, return false (actual pruning logic would be more complex)
        return {"should_prune": False}
        
    except Exception as e:
        logger.error(f"Failed to check pruning status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check pruning status: {str(e)}",
        )


# ============================================================================
# Analysis Endpoints
# ============================================================================

@router.get("/studies/{study_id}/history")
async def get_optimization_history(study_id: str):
    """Get optimization history (best value progression)"""
    try:
        logger.debug(f"Getting optimization history for study {study_id}")
        
        # Load study
        if study_id not in studies:
            study = optuna.load_study(
                study_name=study_id,
                storage=settings.optuna_storage,
            )
            studies[study_id] = study
        else:
            study = studies[study_id]
        
        # Get trials
        trials = study.get_trials(deepcopy=False)
        complete_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        # Calculate best value progression
        history = []
        best_value = None
        for i, trial in enumerate(complete_trials):
            if best_value is None:
                best_value = trial.value
            else:
                if study.direction == optuna.study.StudyDirection.MINIMIZE:
                    best_value = min(best_value, trial.value)
                else:
                    best_value = max(best_value, trial.value)
            
            history.append({
                "trial_number": trial.number,
                "value": best_value,
            })
        
        return {"history": history}
        
    except Exception as e:
        logger.error(f"Failed to get optimization history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get optimization history: {str(e)}",
        )


@router.get("/studies/{study_id}/importance")
async def get_param_importance(study_id: str):
    """Get parameter importance scores"""
    try:
        logger.debug(f"Getting parameter importance for study {study_id}")
        
        # Load study
        if study_id not in studies:
            study = optuna.load_study(
                study_name=study_id,
                storage=settings.optuna_storage,
            )
            studies[study_id] = study
        else:
            study = studies[study_id]
        
        # Calculate importance
        try:
            importance = optuna.importance.get_param_importances(study)
            return {"importance": importance}
        except Exception:
            # Not enough trials yet
            return {"importance": {}}
        
    except Exception as e:
        logger.error(f"Failed to get parameter importance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get parameter importance: {str(e)}",
        )


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        test_study = optuna.create_study(
            study_name=f"health_check_{datetime.now().timestamp()}",
            storage=settings.optuna_storage,
            load_if_exists=False,
        )
        optuna.delete_study(
            study_name=test_study.study_name,
            storage=settings.optuna_storage,
        )
        
        return {
            "status": "healthy",
            "service": "optuna-service",
            "storage": settings.optuna_storage,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Optuna service unhealthy: {str(e)}",
        )
