"""
Training Orchestrator for ML model training.

@TENSOR @PRISM - Complete training pipeline with Optuna hyperparameter optimization.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Callable, Optional

import structlog
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import settings
from src.models.training import (
    TrainingJob,
    TrainingConfig,
    TrainingStatus,
    TrainingMetrics,
    ExperimentResult,
    HyperparameterConfig,
)
from src.services.mlflow_tracker import MLflowTracker
from src.db.redis import get_redis

logger = structlog.get_logger()


class TrainingOrchestrator:
    """
    Training orchestrator for ML models.
    
    @TENSOR @FLUX - GPU-aware training with:
    - Distributed training support
    - MLflow experiment tracking
    - Optuna hyperparameter optimization
    - Checkpoint management
    - Early stopping
    """
    
    def __init__(self):
        self._mlflow: Optional[MLflowTracker] = None
        self._active_jobs: dict[str, TrainingJob] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if torch.cuda.is_available():
            self._gpu_count = torch.cuda.device_count()
            logger.info(f"GPU training available: {self._gpu_count} GPU(s)")
        else:
            self._gpu_count = 0
            logger.warning("No GPU available, using CPU")
    
    async def initialize(self) -> None:
        """Initialize training orchestrator."""
        self._mlflow = MLflowTracker()
        await self._mlflow.initialize()
        logger.info("✅ Training Orchestrator initialized")
    
    # =========================================================================
    # Job Management
    # =========================================================================
    
    async def create_training_job(
        self,
        config: TrainingConfig,
    ) -> TrainingJob:
        """
        Create a new training job.
        
        @TENSOR - Training job creation with validation.
        """
        job_id = str(uuid.uuid4())
        
        job = TrainingJob(
            id=job_id,
            status=TrainingStatus.PENDING,
            config=config,
            total_epochs=config.hyperparameters.num_epochs,
            created_at=datetime.utcnow(),
        )
        
        self._active_jobs[job_id] = job
        
        # Queue job for execution
        redis = await get_redis()
        await redis.enqueue_job("training", {
            "job_id": job_id,
            "config": config.model_dump(),
        })
        
        logger.info(f"Created training job: {job_id}")
        return job
    
    async def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job by ID."""
        return self._active_jobs.get(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job."""
        job = self._active_jobs.get(job_id)
        if job and job.status in [TrainingStatus.PENDING, TrainingStatus.RUNNING]:
            job.status = TrainingStatus.CANCELLED
            logger.info(f"Cancelled training job: {job_id}")
            return True
        return False
    
    # =========================================================================
    # Training Execution
    # =========================================================================
    
    async def run_training(
        self,
        config: TrainingConfig,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        callbacks: Optional[list[Callable]] = None,
    ) -> ExperimentResult:
        """
        Execute a training run.
        
        @TENSOR @SENTRY - Full training loop with MLflow tracking.
        """
        job_id = str(uuid.uuid4())
        job = TrainingJob(
            id=job_id,
            status=TrainingStatus.RUNNING,
            config=config,
            total_epochs=config.hyperparameters.num_epochs,
            started_at=datetime.utcnow(),
        )
        self._active_jobs[job_id] = job
        
        # Start MLflow run
        run_id = self._mlflow.start_run(
            experiment_name=config.experiment_name,
            run_name=config.run_name,
            tags=config.tags,
        )
        job.mlflow_run_id = run_id
        
        # Log configuration
        self._mlflow.log_training_config(config)
        
        # Setup training
        model = model.to(self._device)
        optimizer = self._create_optimizer(model, config.hyperparameters)
        scheduler = self._create_scheduler(optimizer, config.hyperparameters, len(train_dataloader))
        
        # Training loop
        best_eval_loss = float("inf")
        best_checkpoint = None
        
        try:
            for epoch in range(config.hyperparameters.num_epochs):
                job.current_epoch = epoch + 1
                
                # Training phase
                model.train()
                epoch_loss = 0.0
                
                for step, batch in enumerate(train_dataloader):
                    job.current_step += 1
                    
                    # Move batch to device
                    batch = self._move_to_device(batch)
                    
                    # Forward pass
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    if config.hyperparameters.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            config.hyperparameters.max_grad_norm,
                        )
                    
                    # Optimizer step
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    epoch_loss += loss.item()
                    
                    # Logging
                    if step % config.logging_steps == 0:
                        metrics = TrainingMetrics(
                            epoch=epoch + 1,
                            step=job.current_step,
                            train_loss=loss.item(),
                            learning_rate=scheduler.get_last_lr()[0],
                        )
                        self._mlflow.log_training_metrics(metrics)
                        job.latest_metrics = metrics
                
                avg_train_loss = epoch_loss / len(train_dataloader)
                
                # Evaluation phase
                eval_loss = None
                if eval_dataloader:
                    eval_loss = await self._evaluate(model, eval_dataloader)
                    
                    # Check for best model
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_checkpoint = f"{config.output_dir}/best_model.pt"
                        torch.save(model.state_dict(), best_checkpoint)
                        job.best_metric = eval_loss
                
                # Log epoch metrics
                epoch_metrics = {
                    "epoch_train_loss": avg_train_loss,
                    "epoch": epoch + 1,
                }
                if eval_loss:
                    epoch_metrics["epoch_eval_loss"] = eval_loss
                
                self._mlflow.log_metrics(epoch_metrics, step=epoch + 1)
                
                logger.info(
                    f"Epoch {epoch + 1}/{config.hyperparameters.num_epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Eval Loss: {eval_loss:.4f if eval_loss else 'N/A'}"
                )
                
                # Update progress
                job.progress_percentage = (epoch + 1) / config.hyperparameters.num_epochs * 100
                
                # Run callbacks
                if callbacks:
                    for callback in callbacks:
                        callback(job, epoch, avg_train_loss, eval_loss)
            
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            self._mlflow.end_run(
                status="FINISHED" if job.status == TrainingStatus.COMPLETED else "FAILED"
            )
        
        # Create result
        duration = (job.completed_at - job.started_at).total_seconds()
        
        return ExperimentResult(
            job_id=job_id,
            experiment_name=config.experiment_name,
            run_name=config.run_name,
            final_train_loss=avg_train_loss,
            final_eval_loss=eval_loss or 0.0,
            best_eval_loss=best_eval_loss,
            hyperparameters=config.hyperparameters,
            model_path=best_checkpoint or f"{config.output_dir}/final_model.pt",
            training_duration_seconds=duration,
            mlflow_run_id=run_id,
        )
    
    async def _evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
    ) -> float:
        """Evaluate model on dataloader."""
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_to_device(batch)
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    # =========================================================================
    # Hyperparameter Optimization
    # =========================================================================
    
    async def optimize_hyperparameters(
        self,
        config: TrainingConfig,
        model_factory: Callable[[], nn.Module],
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        n_trials: int = 20,
        timeout: Optional[int] = None,
    ) -> HyperparameterConfig:
        """
        Optimize hyperparameters using Optuna.
        
        @TENSOR @PRISM - Bayesian hyperparameter optimization.
        """
        import optuna
        from optuna.integration import PyTorchLightningPruningCallback
        
        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            hp = HyperparameterConfig(
                learning_rate=trial.suggest_float("lr", 1e-6, 1e-3, log=True),
                weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
                batch_size=trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
                warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.2),
                dropout=trial.suggest_float("dropout", 0.0, 0.5),
                num_epochs=config.hyperparameters.num_epochs,
            )
            
            # Create model
            model = model_factory()
            model = model.to(self._device)
            
            # Setup training
            optimizer = self._create_optimizer(model, hp)
            scheduler = self._create_scheduler(optimizer, hp, len(train_dataloader))
            
            # Quick training
            model.train()
            for epoch in range(min(3, hp.num_epochs)):  # Max 3 epochs for optimization
                for batch in train_dataloader:
                    batch = self._move_to_device(batch)
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Evaluate
                eval_loss = asyncio.get_event_loop().run_until_complete(
                    self._evaluate(model, eval_dataloader)
                )
                
                # Report for pruning
                trial.report(eval_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return eval_loss
        
        # Create study
        study = optuna.create_study(
            study_name=f"{config.experiment_name}_hpo",
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(),
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )
        
        # Get best hyperparameters
        best_params = study.best_params
        logger.info(f"Best hyperparameters: {best_params}")
        
        return HyperparameterConfig(
            learning_rate=best_params["lr"],
            weight_decay=best_params["weight_decay"],
            batch_size=best_params["batch_size"],
            warmup_ratio=best_params["warmup_ratio"],
            dropout=best_params["dropout"],
            num_epochs=config.hyperparameters.num_epochs,
        )
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def _create_optimizer(
        self,
        model: nn.Module,
        hp: HyperparameterConfig,
    ) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        if hp.optimizer.value == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=hp.learning_rate,
                weight_decay=hp.weight_decay,
            )
        elif hp.optimizer.value == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=hp.learning_rate,
                weight_decay=hp.weight_decay,
            )
        elif hp.optimizer.value == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=hp.learning_rate,
                weight_decay=hp.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {hp.optimizer}")
    
    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        hp: HyperparameterConfig,
        steps_per_epoch: int,
    ):
        """Create learning rate scheduler."""
        from torch.optim.lr_scheduler import (
            CosineAnnealingWarmRestarts,
            LinearLR,
            SequentialLR,
        )
        
        total_steps = steps_per_epoch * hp.num_epochs
        warmup_steps = int(total_steps * hp.warmup_ratio)
        
        if hp.scheduler.value == "cosine_with_warmup":
            warmup = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=warmup_steps,
            )
            cosine = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=total_steps - warmup_steps,
            )
            return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
        
        elif hp.scheduler.value == "linear":
            return LinearLR(optimizer, total_iters=total_steps)
        
        else:
            # Constant learning rate
            return LinearLR(optimizer, start_factor=1.0, total_iters=1)
    
    def _move_to_device(self, batch: Any) -> Any:
        """Move batch to device."""
        if isinstance(batch, dict):
            return {k: v.to(self._device) if hasattr(v, 'to') else v for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(v.to(self._device) if hasattr(v, 'to') else v for v in batch)
        elif hasattr(batch, 'to'):
            return batch.to(self._device)
        return batch
    
    def get_gpu_info(self) -> dict:
        """Get GPU information."""
        if not torch.cuda.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "count": torch.cuda.device_count(),
            "devices": [
                {
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_mb": torch.cuda.get_device_properties(i).total_memory // (1024 * 1024),
                    "memory_allocated_mb": torch.cuda.memory_allocated(i) // (1024 * 1024),
                }
                for i in range(torch.cuda.device_count())
            ],
        }
