"""
Training Engine

PyTorch training orchestration with MLflow logging and WebSocket broadcasting.
Supports various model architectures, optimizers, and training strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import AutoModel, AutoTokenizer
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import logging
import time
from pathlib import Path
import json

from config import settings

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model
    model_type: str  # transformer, cnn, rnn, custom
    model_name: Optional[str] = None  # For pretrained models
    input_size: Optional[int] = None
    hidden_size: Optional[int] = None
    output_size: Optional[int] = None
    
    # Training
    batch_size: int = settings.default_batch_size
    epochs: int = settings.default_epochs
    learning_rate: float = settings.default_learning_rate
    optimizer: str = "adam"  # adam, sgd, adamw
    scheduler: Optional[str] = None  # cosine, step, exponential
    loss_function: str = "cross_entropy"  # cross_entropy, mse, mae
    
    # Regularization
    weight_decay: float = 0.0
    dropout: float = 0.0
    gradient_clip: Optional[float] = None
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 0.001
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    distributed: bool = False
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_frequency: int = 1  # epochs
    
    # Misc
    seed: int = 42


# ============================================================================
# Model Registry
# ============================================================================

class SimpleTransformer(nn.Module):
    """Simple transformer model for sequence tasks"""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, num_classes: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x


class SimpleCNN(nn.Module):
    """Simple CNN for image classification"""
    
    def __init__(self, input_channels: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SimpleRNN(nn.Module):
    """Simple RNN for sequence tasks"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Get last time step
        out = self.fc(out)
        return out


def get_model(config: TrainingConfig) -> nn.Module:
    """Get model based on configuration"""
    if config.model_type == "transformer":
        if config.model_name:
            # Pretrained transformer
            return AutoModel.from_pretrained(config.model_name)
        else:
            # Simple transformer
            return SimpleTransformer(
                vocab_size=config.input_size or 30000,
                embed_dim=config.hidden_size or 256,
                num_heads=8,
                num_layers=4,
                num_classes=config.output_size or 10,
                dropout=config.dropout,
            )
    elif config.model_type == "cnn":
        return SimpleCNN(
            input_channels=config.input_size or 3,
            num_classes=config.output_size or 10,
            dropout=config.dropout,
        )
    elif config.model_type == "rnn":
        return SimpleRNN(
            input_size=config.input_size or 100,
            hidden_size=config.hidden_size or 256,
            num_layers=2,
            num_classes=config.output_size or 10,
            dropout=config.dropout,
        )
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")


def get_optimizer(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
    """Get optimizer based on configuration"""
    if config.optimizer.lower() == "adam":
        return optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer.lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def get_scheduler(optimizer: optim.Optimizer, config: TrainingConfig) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Get learning rate scheduler"""
    if not config.scheduler:
        return None
    
    if config.scheduler.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
        )
    elif config.scheduler.lower() == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.epochs // 3,
            gamma=0.1,
        )
    elif config.scheduler.lower() == "exponential":
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def get_loss_function(config: TrainingConfig) -> nn.Module:
    """Get loss function based on configuration"""
    if config.loss_function.lower() == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif config.loss_function.lower() == "mse":
        return nn.MSELoss()
    elif config.loss_function.lower() == "mae":
        return nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {config.loss_function}")


# ============================================================================
# Training Engine
# ============================================================================

class TrainingEngine:
    """
    PyTorch training engine with MLflow logging and WebSocket broadcasting
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        mlflow_run_id: str,
        websocket_broadcast: Optional[Callable] = None,
    ):
        self.config = config
        self.mlflow_run_id = mlflow_run_id
        self.websocket_broadcast = websocket_broadcast
        self.mlflow_client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        
        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize model
        self.model = get_model(config).to(config.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = get_optimizer(self.model, config)
        self.scheduler = get_scheduler(self.optimizer, config)
        
        # Initialize loss function
        self.criterion = get_loss_function(config)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        
        logger.info(f"TrainingEngine initialized for run {mlflow_run_id}")
    
    async def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """
        Main training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")
        
        # Log configuration to MLflow
        self._log_config()
        
        # Training loop
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train epoch
            train_metrics = await self._train_epoch(train_loader, epoch)
            
            # Validate epoch
            val_metrics = {}
            if val_loader:
                val_metrics = await self._validate_epoch(val_loader, epoch)
            
            # Calculate epoch duration
            epoch_duration = time.time() - epoch_start_time
            
            # Combine metrics
            all_metrics = {
                **train_metrics,
                **val_metrics,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "epoch_duration": epoch_duration,
            }
            
            # Log to MLflow
            self._log_metrics(all_metrics, epoch)
            
            # Broadcast to WebSocket
            if self.websocket_broadcast:
                await self.websocket_broadcast({
                    "type": "training:metrics",
                    "data": {
                        "run_id": self.mlflow_run_id,
                        "epoch": epoch,
                        "metrics": all_metrics,
                        "timestamp": time.time(),
                    },
                })
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Check early stopping
            if self.config.early_stopping and val_loader:
                val_loss = val_metrics.get("val_loss", float("inf"))
                if val_loss < self.best_val_loss - self.config.min_delta:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    
                    # Save best model
                    if self.config.save_checkpoints:
                        self._save_checkpoint(epoch, "best")
                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= self.config.patience:
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
            
            # Save checkpoint
            if self.config.save_checkpoints and epoch % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(epoch, "latest")
            
            logger.info(
                f"Epoch {epoch}/{self.config.epochs} - "
                f"train_loss: {train_metrics['train_loss']:.4f}, "
                f"val_loss: {val_metrics.get('val_loss', 0):.4f}, "
                f"duration: {epoch_duration:.2f}s"
            )
        
        # Save final model
        if self.config.save_checkpoints:
            self._save_checkpoint(self.current_epoch, "final")
        
        return {
            "epochs_trained": self.current_epoch + 1,
            "best_val_loss": self.best_val_loss,
        }
    
    async def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.config.device), target.to(self.config.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )
            
            self.optimizer.step()
            
            # Calculate accuracy
            with torch.no_grad():
                pred = output.argmax(dim=1)
                correct = (pred == target).sum().item()
                total_correct += correct
                total_samples += target.size(0)
            
            total_loss += loss.item()
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
        }
    
    async def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Calculate accuracy
                pred = output.argmax(dim=1)
                correct = (pred == target).sum().item()
                total_correct += correct
                total_samples += target.size(0)
                
                total_loss += loss.item()
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
        }
    
    def _log_config(self):
        """Log configuration to MLflow"""
        params = {
            "model_type": self.config.model_type,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "optimizer": self.config.optimizer,
            "epochs": self.config.epochs,
            "device": self.config.device,
        }
        
        for key, value in params.items():
            self.mlflow_client.log_param(self.mlflow_run_id, key, value)
    
    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to MLflow"""
        for key, value in metrics.items():
            self.mlflow_client.log_metric(
                self.mlflow_run_id,
                key,
                value,
                step=step,
            )
    
    def _save_checkpoint(self, epoch: int, name: str):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        
        # Save locally
        checkpoint_path = Path(f"checkpoints/checkpoint_{name}.pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Log to MLflow
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
        
        logger.info(f"Saved checkpoint: {name}")
