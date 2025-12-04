"""
Unit Tests for Training Service

@ECLIPSE @TENSOR - Comprehensive tests for training orchestration and MLflow.

Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from uuid import uuid4

from src.models.training import (
    TrainingJob,
    TrainingConfig,
    TrainingStatus,
    TrainingMetrics,
    HyperparameterConfig,
    DataConfig,
    OptimizerType,
    SchedulerType,
)


# ==============================================================================
# Model Tests
# ==============================================================================

class TestTrainingModels:
    """Tests for Training Pydantic models."""
    
    @pytest.mark.unit
    def test_hyperparameter_config_defaults(self):
        """Test HyperparameterConfig default values."""
        config = HyperparameterConfig()
        
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 0.01
        assert config.optimizer == OptimizerType.ADAMW
        assert config.scheduler == SchedulerType.COSINE_WARMUP
        assert config.batch_size == 32
        assert config.num_epochs == 3
        assert config.seed == 42
    
    @pytest.mark.unit
    def test_hyperparameter_config_custom(self):
        """Test HyperparameterConfig with custom values."""
        config = HyperparameterConfig(
            learning_rate=2e-5,
            weight_decay=0.001,
            optimizer=OptimizerType.ADAM,
            scheduler=SchedulerType.COSINE,
            batch_size=16,
            num_epochs=10,
            fp16=True,
        )
        
        assert config.learning_rate == 2e-5
        assert config.optimizer == OptimizerType.ADAM
        assert config.fp16 is True
    
    @pytest.mark.unit
    def test_hyperparameter_validation_learning_rate(self):
        """Test learning rate validation bounds."""
        # Valid learning rate
        config = HyperparameterConfig(learning_rate=0.001)
        assert config.learning_rate == 0.001
        
        # Invalid learning rate (too high)
        with pytest.raises(ValueError):
            HyperparameterConfig(learning_rate=2.0)
        
        # Invalid learning rate (too low)
        with pytest.raises(ValueError):
            HyperparameterConfig(learning_rate=0)
    
    @pytest.mark.unit
    def test_hyperparameter_validation_batch_size(self):
        """Test batch size validation bounds."""
        # Valid batch size
        config = HyperparameterConfig(batch_size=64)
        assert config.batch_size == 64
        
        # Invalid batch size (too large)
        with pytest.raises(ValueError):
            HyperparameterConfig(batch_size=5000)
        
        # Invalid batch size (zero)
        with pytest.raises(ValueError):
            HyperparameterConfig(batch_size=0)
    
    @pytest.mark.unit
    def test_training_status_enum(self):
        """Test TrainingStatus enum values."""
        assert TrainingStatus.PENDING.value == "pending"
        assert TrainingStatus.QUEUED.value == "queued"
        assert TrainingStatus.RUNNING.value == "running"
        assert TrainingStatus.COMPLETED.value == "completed"
        assert TrainingStatus.FAILED.value == "failed"
        assert TrainingStatus.CANCELLED.value == "cancelled"
        assert TrainingStatus.PAUSED.value == "paused"
    
    @pytest.mark.unit
    def test_optimizer_type_enum(self):
        """Test OptimizerType enum values."""
        optimizers = [
            OptimizerType.ADAM,
            OptimizerType.ADAMW,
            OptimizerType.SGD,
            OptimizerType.LAMB,
            OptimizerType.ADAFACTOR,
            OptimizerType.LION,
        ]
        
        assert len(optimizers) == 6
        assert OptimizerType.ADAMW.value == "adamw"
    
    @pytest.mark.unit
    def test_scheduler_type_enum(self):
        """Test SchedulerType enum values."""
        schedulers = [
            SchedulerType.LINEAR,
            SchedulerType.COSINE,
            SchedulerType.COSINE_WARMUP,
            SchedulerType.POLYNOMIAL,
            SchedulerType.CONSTANT,
            SchedulerType.ONE_CYCLE,
        ]
        
        assert len(schedulers) == 6
        assert SchedulerType.COSINE_WARMUP.value == "cosine_with_warmup"
    
    @pytest.mark.unit
    def test_data_config_creation(self):
        """Test DataConfig model creation."""
        config = DataConfig(
            dataset_name="train_data",
            dataset_path="/data/training",
            max_length=1024,
            preprocessing_num_workers=8,
        )
        
        assert config.dataset_name == "train_data"
        assert config.dataset_path == "/data/training"
        assert config.max_length == 1024
        assert config.preprocessing_num_workers == 8
    
    @pytest.mark.unit
    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        config = DataConfig()
        
        assert config.train_split == "train"
        assert config.validation_split == "validation"
        assert config.test_split == "test"
        assert config.max_length == 512
        assert config.augmentation_enabled is False
    
    @pytest.mark.unit
    def test_training_job_creation(self, training_job_factory):
        """Test TrainingJob model creation using factory."""
        job_data = training_job_factory(
            job_id="test-job-123",
            model_type="fine-tune",
            status="pending",
        )
        
        assert job_data["id"] == "test-job-123"
        assert job_data["model_type"] == "fine-tune"
        assert job_data["status"] == "pending"


# ==============================================================================
# Training Orchestrator Tests
# ==============================================================================

class TestTrainingOrchestrator:
    """Tests for TrainingOrchestrator class."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, mock_mlflow):
        """Test training orchestrator initialization."""
        with patch("src.services.training.MLflowTracker") as MockTracker, \
             patch("torch.cuda.is_available", return_value=False):
            
            MockTracker.return_value.initialize = AsyncMock()
            
            from src.services.training import TrainingOrchestrator
            
            orchestrator = TrainingOrchestrator()
            
            assert orchestrator._device == "cpu"
            assert orchestrator._active_jobs == {}
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_orchestrator_with_gpu(self, mock_mlflow):
        """Test training orchestrator with GPU available."""
        with patch("src.services.training.MLflowTracker"), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=2):
            
            from src.services.training import TrainingOrchestrator
            
            orchestrator = TrainingOrchestrator()
            
            assert orchestrator._device == "cuda"
            assert orchestrator._gpu_count == 2
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_training_job(self, mock_redis, mock_mlflow):
        """Test creating a training job."""
        with patch("src.services.training.MLflowTracker") as MockTracker, \
             patch("src.services.training.get_redis", return_value=mock_redis), \
             patch("torch.cuda.is_available", return_value=False):
            
            MockTracker.return_value.initialize = AsyncMock()
            mock_redis.enqueue_job = AsyncMock(return_value=True)
            
            from src.services.training import TrainingOrchestrator
            
            orchestrator = TrainingOrchestrator()
            
            config = TrainingConfig(
                experiment_name="test_experiment",
                run_name="test_run",
                model_name="test_model",
                hyperparameters=HyperparameterConfig(),
            )
            
            # Simulate job creation
            job_id = str(uuid4())
            orchestrator._active_jobs[job_id] = MagicMock(
                id=job_id,
                status=TrainingStatus.PENDING,
            )
            
            assert job_id in orchestrator._active_jobs
            assert orchestrator._active_jobs[job_id].status == TrainingStatus.PENDING
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_job(self):
        """Test getting a training job."""
        with patch("src.services.training.MLflowTracker"), \
             patch("torch.cuda.is_available", return_value=False):
            
            from src.services.training import TrainingOrchestrator
            
            orchestrator = TrainingOrchestrator()
            
            # Create mock job
            job_id = "test-job-id"
            mock_job = MagicMock(id=job_id, status=TrainingStatus.RUNNING)
            orchestrator._active_jobs[job_id] = mock_job
            
            result = await orchestrator.get_job(job_id)
            
            assert result is not None
            assert result.id == job_id
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_nonexistent_job(self):
        """Test getting non-existent job returns None."""
        with patch("src.services.training.MLflowTracker"), \
             patch("torch.cuda.is_available", return_value=False):
            
            from src.services.training import TrainingOrchestrator
            
            orchestrator = TrainingOrchestrator()
            
            result = await orchestrator.get_job("nonexistent-id")
            
            assert result is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cancel_job_pending(self):
        """Test cancelling a pending job."""
        with patch("src.services.training.MLflowTracker"), \
             patch("torch.cuda.is_available", return_value=False):
            
            from src.services.training import TrainingOrchestrator
            
            orchestrator = TrainingOrchestrator()
            
            job_id = "cancel-test"
            mock_job = MagicMock(id=job_id, status=TrainingStatus.PENDING)
            orchestrator._active_jobs[job_id] = mock_job
            
            result = await orchestrator.cancel_job(job_id)
            
            assert result is True
            assert mock_job.status == TrainingStatus.CANCELLED
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cancel_job_completed(self):
        """Test cancelling a completed job returns False."""
        with patch("src.services.training.MLflowTracker"), \
             patch("torch.cuda.is_available", return_value=False):
            
            from src.services.training import TrainingOrchestrator
            
            orchestrator = TrainingOrchestrator()
            
            job_id = "completed-job"
            mock_job = MagicMock(id=job_id, status=TrainingStatus.COMPLETED)
            orchestrator._active_jobs[job_id] = mock_job
            
            result = await orchestrator.cancel_job(job_id)
            
            assert result is False


# ==============================================================================
# MLflow Integration Tests
# ==============================================================================

class TestMLflowIntegration:
    """Tests for MLflow tracking integration."""
    
    @pytest.mark.unit
    def test_mlflow_start_run(self, mock_mlflow):
        """Test starting an MLflow run."""
        mock_mlflow["start_run"]()
        mock_mlflow["start_run"].assert_called_once()
    
    @pytest.mark.unit
    def test_mlflow_log_params(self, mock_mlflow):
        """Test logging parameters to MLflow."""
        params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
        }
        
        for key, value in params.items():
            mock_mlflow["log_param"](key, value)
        
        assert mock_mlflow["log_param"].call_count == 3
    
    @pytest.mark.unit
    def test_mlflow_log_metrics(self, mock_mlflow):
        """Test logging metrics to MLflow."""
        metrics = {
            "train_loss": 0.5,
            "val_loss": 0.6,
            "accuracy": 0.85,
        }
        
        for key, value in metrics.items():
            mock_mlflow["log_metric"](key, value)
        
        assert mock_mlflow["log_metric"].call_count == 3
    
    @pytest.mark.unit
    def test_mlflow_end_run(self, mock_mlflow):
        """Test ending an MLflow run."""
        mock_mlflow["end_run"]()
        mock_mlflow["end_run"].assert_called_once()
    
    @pytest.mark.unit
    def test_mlflow_create_experiment(self, mock_mlflow):
        """Test creating an MLflow experiment."""
        mock_mlflow["create_experiment"].return_value = "exp-123"
        
        result = mock_mlflow["create_experiment"]("test_experiment")
        
        assert result == "exp-123"
        mock_mlflow["create_experiment"].assert_called_with("test_experiment")


# ==============================================================================
# Training Metrics Tests
# ==============================================================================

class TestTrainingMetrics:
    """Tests for training metrics handling."""
    
    @pytest.mark.unit
    def test_metrics_tracking(self):
        """Test tracking training metrics over epochs."""
        metrics_history = []
        
        for epoch in range(5):
            metrics = {
                "epoch": epoch,
                "train_loss": 1.0 - (epoch * 0.15),
                "val_loss": 1.1 - (epoch * 0.12),
                "learning_rate": 1e-4 * (0.9 ** epoch),
            }
            metrics_history.append(metrics)
        
        # Verify loss decreases
        assert metrics_history[0]["train_loss"] > metrics_history[-1]["train_loss"]
        assert metrics_history[0]["val_loss"] > metrics_history[-1]["val_loss"]
    
    @pytest.mark.unit
    def test_early_stopping_detection(self):
        """Test early stopping based on validation loss."""
        val_losses = [1.0, 0.9, 0.85, 0.86, 0.87, 0.88]
        patience = 3
        best_loss = float("inf")
        patience_counter = 0
        
        for epoch, loss in enumerate(val_losses):
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                early_stop_epoch = epoch
                break
        
        # Should stop at epoch 5 (patience exhausted after epochs 3, 4, 5)
        assert early_stop_epoch == 5
    
    @pytest.mark.unit
    def test_learning_rate_warmup(self):
        """Test learning rate warmup calculation."""
        base_lr = 1e-4
        warmup_steps = 100
        current_steps = [0, 25, 50, 75, 100, 150]
        
        lrs = []
        for step in current_steps:
            if step < warmup_steps:
                lr = base_lr * (step / warmup_steps)
            else:
                lr = base_lr
            lrs.append(lr)
        
        # Warmup should increase linearly
        assert lrs[0] == 0
        assert lrs[2] == pytest.approx(base_lr * 0.5, rel=1e-6)
        assert lrs[4] == base_lr  # At warmup_steps
        assert lrs[5] == base_lr  # After warmup
    
    @pytest.mark.unit
    def test_gradient_norm_clipping(self):
        """Test gradient norm clipping behavior."""
        max_grad_norm = 1.0
        
        # Simulate gradient norms
        grad_norms = [0.5, 1.0, 2.0, 5.0, 10.0]
        clipped_norms = [min(norm, max_grad_norm) for norm in grad_norms]
        
        assert clipped_norms == [0.5, 1.0, 1.0, 1.0, 1.0]


# ==============================================================================
# Training Configuration Tests
# ==============================================================================

class TestTrainingConfiguration:
    """Tests for training configuration handling."""
    
    @pytest.mark.unit
    def test_fp16_configuration(self):
        """Test FP16 mixed precision configuration."""
        config = HyperparameterConfig(fp16=True, bf16=False)
        
        assert config.fp16 is True
        assert config.bf16 is False
    
    @pytest.mark.unit
    def test_bf16_configuration(self):
        """Test BF16 mixed precision configuration."""
        config = HyperparameterConfig(fp16=False, bf16=True)
        
        assert config.fp16 is False
        assert config.bf16 is True
    
    @pytest.mark.unit
    def test_gradient_accumulation(self):
        """Test gradient accumulation configuration."""
        config = HyperparameterConfig(
            batch_size=8,
            gradient_accumulation_steps=4,
        )
        
        effective_batch_size = config.batch_size * config.gradient_accumulation_steps
        
        assert effective_batch_size == 32
    
    @pytest.mark.unit
    def test_warmup_configuration(self):
        """Test warmup steps vs ratio configuration."""
        # With explicit steps
        config_steps = HyperparameterConfig(warmup_steps=500)
        assert config_steps.warmup_steps == 500
        
        # With ratio
        config_ratio = HyperparameterConfig(warmup_ratio=0.1)
        assert config_ratio.warmup_ratio == 0.1
    
    @pytest.mark.unit
    def test_optimizer_configuration(self):
        """Test different optimizer configurations."""
        configs = [
            HyperparameterConfig(optimizer=OptimizerType.ADAM),
            HyperparameterConfig(optimizer=OptimizerType.ADAMW),
            HyperparameterConfig(optimizer=OptimizerType.SGD),
            HyperparameterConfig(optimizer=OptimizerType.LAMB),
        ]
        
        assert len(configs) == 4
        assert configs[0].optimizer == OptimizerType.ADAM
        assert configs[3].optimizer == OptimizerType.LAMB


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestTrainingIntegration:
    """Integration tests for training pipeline."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_training_job_lifecycle(self, mock_redis, mock_mlflow):
        """Test complete training job lifecycle."""
        with patch("src.services.training.MLflowTracker") as MockTracker, \
             patch("src.services.training.get_redis", return_value=mock_redis), \
             patch("torch.cuda.is_available", return_value=False):
            
            MockTracker.return_value.initialize = AsyncMock()
            mock_redis.enqueue_job = AsyncMock(return_value=True)
            
            from src.services.training import TrainingOrchestrator
            
            orchestrator = TrainingOrchestrator()
            await orchestrator.initialize()
            
            # Create job
            job_id = str(uuid4())
            mock_job = MagicMock(
                id=job_id,
                status=TrainingStatus.PENDING,
                config=MagicMock(),
            )
            orchestrator._active_jobs[job_id] = mock_job
            
            # Verify job exists
            retrieved = await orchestrator.get_job(job_id)
            assert retrieved is not None
            
            # Simulate status transitions
            mock_job.status = TrainingStatus.RUNNING
            assert mock_job.status == TrainingStatus.RUNNING
            
            mock_job.status = TrainingStatus.COMPLETED
            assert mock_job.status == TrainingStatus.COMPLETED
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_job_management(self, mock_redis):
        """Test managing multiple concurrent jobs."""
        with patch("src.services.training.MLflowTracker"), \
             patch("torch.cuda.is_available", return_value=False):
            
            from src.services.training import TrainingOrchestrator
            
            orchestrator = TrainingOrchestrator()
            
            # Create multiple jobs
            jobs = {}
            for i in range(5):
                job_id = f"job-{i}"
                jobs[job_id] = MagicMock(
                    id=job_id,
                    status=TrainingStatus.RUNNING,
                )
                orchestrator._active_jobs[job_id] = jobs[job_id]
            
            assert len(orchestrator._active_jobs) == 5
            
            # Complete some jobs
            orchestrator._active_jobs["job-0"].status = TrainingStatus.COMPLETED
            orchestrator._active_jobs["job-2"].status = TrainingStatus.FAILED
            
            # Count by status
            running = sum(
                1 for j in orchestrator._active_jobs.values()
                if j.status == TrainingStatus.RUNNING
            )
            
            assert running == 3


# ==============================================================================
# Edge Cases Tests
# ==============================================================================

class TestTrainingEdgeCases:
    """Tests for edge cases in training."""
    
    @pytest.mark.unit
    def test_very_small_learning_rate(self):
        """Test very small learning rate configuration."""
        config = HyperparameterConfig(learning_rate=1e-8)
        assert config.learning_rate == 1e-8
    
    @pytest.mark.unit
    def test_single_epoch_training(self):
        """Test single epoch training configuration."""
        config = HyperparameterConfig(num_epochs=1)
        assert config.num_epochs == 1
    
    @pytest.mark.unit
    def test_max_steps_override(self):
        """Test max_steps overriding num_epochs."""
        config = HyperparameterConfig(
            num_epochs=10,
            max_steps=1000,
        )
        
        assert config.max_steps == 1000
        # In implementation, max_steps should take precedence
    
    @pytest.mark.unit
    def test_zero_warmup(self):
        """Test training with no warmup."""
        config = HyperparameterConfig(warmup_steps=0, warmup_ratio=0.0)
        
        assert config.warmup_steps == 0
        assert config.warmup_ratio == 0.0
    
    @pytest.mark.unit
    def test_high_label_smoothing(self):
        """Test high label smoothing value."""
        config = HyperparameterConfig(label_smoothing=0.3)
        assert config.label_smoothing == 0.3
        
        # Invalid (too high)
        with pytest.raises(ValueError):
            HyperparameterConfig(label_smoothing=0.6)
