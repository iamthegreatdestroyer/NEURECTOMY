"""
DirectML GPU Utilities for NEURECTOMY ML Service

This module provides GPU acceleration utilities using DirectML
for AMD, Intel, and Qualcomm GPUs on Windows.

Environment: Python 3.12 (pytdml conda environment)
"""

import torch
import torch_directml
from typing import Optional, Union, List
import logging

logger = logging.getLogger(__name__)

# Global DirectML device
_dml_device: Optional[torch.device] = None


def get_dml_device(device_id: int = 0) -> torch.device:
    """
    Get the DirectML device for GPU acceleration.
    
    Args:
        device_id: GPU device index (default 0)
        
    Returns:
        torch.device: DirectML device
    """
    global _dml_device
    if _dml_device is None:
        _dml_device = torch_directml.device(device_id)
        logger.info(f"DirectML device initialized: {torch_directml.device_name(device_id)}")
    return _dml_device


def get_device_info() -> dict:
    """
    Get information about available DirectML devices.
    
    Returns:
        dict: Device information including name, count, and capabilities
    """
    device_count = torch_directml.device_count()
    devices = []
    
    for i in range(device_count):
        devices.append({
            "id": i,
            "name": torch_directml.device_name(i),
            "device": str(torch_directml.device(i))
        })
    
    return {
        "backend": "DirectML",
        "device_count": device_count,
        "devices": devices,
        "default_device": torch_directml.device_name(0) if device_count > 0 else None
    }


def to_gpu(tensor: torch.Tensor, device_id: int = 0) -> torch.Tensor:
    """
    Move a tensor to the DirectML GPU device.
    
    Args:
        tensor: PyTorch tensor to move
        device_id: GPU device index
        
    Returns:
        torch.Tensor: Tensor on GPU
    """
    dml = get_dml_device(device_id)
    return tensor.to(dml)


def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """
    Move a tensor back to CPU.
    
    Args:
        tensor: PyTorch tensor on GPU
        
    Returns:
        torch.Tensor: Tensor on CPU
    """
    return tensor.cpu()


def is_directml_available() -> bool:
    """
    Check if DirectML is available for GPU acceleration.
    
    Returns:
        bool: True if DirectML is available
    """
    try:
        return torch_directml.device_count() > 0
    except Exception:
        return False


class DirectMLModelWrapper:
    """
    Wrapper for PyTorch models to use DirectML GPU acceleration.
    
    Example:
        >>> model = MyModel()
        >>> gpu_model = DirectMLModelWrapper(model)
        >>> output = gpu_model(input_tensor)
    """
    
    def __init__(self, model: torch.nn.Module, device_id: int = 0):
        """
        Initialize the DirectML model wrapper.
        
        Args:
            model: PyTorch model to wrap
            device_id: GPU device index
        """
        self.device = get_dml_device(device_id)
        self.model = model.to(self.device)
        logger.info(f"Model moved to DirectML device: {torch_directml.device_name(device_id)}")
    
    def __call__(self, *args, **kwargs) -> Union[torch.Tensor, tuple]:
        """
        Forward pass through the model.
        
        All tensor inputs are automatically moved to GPU,
        and outputs are kept on GPU for efficiency.
        """
        # Move input tensors to GPU
        gpu_args = [
            arg.to(self.device) if isinstance(arg, torch.Tensor) else arg
            for arg in args
        ]
        gpu_kwargs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        
        return self.model(*gpu_args, **gpu_kwargs)
    
    def inference(self, *args, **kwargs) -> Union[torch.Tensor, tuple]:
        """
        Run inference with no gradient computation.
        
        Returns results on CPU for easy post-processing.
        """
        with torch.no_grad():
            result = self(*args, **kwargs)
            
            if isinstance(result, torch.Tensor):
                return result.cpu()
            elif isinstance(result, tuple):
                return tuple(r.cpu() if isinstance(r, torch.Tensor) else r for r in result)
            return result


# Quick test function
def test_directml():
    """Test DirectML functionality."""
    print("=" * 50)
    print("DirectML GPU Test")
    print("=" * 50)
    
    info = get_device_info()
    print(f"Backend: {info['backend']}")
    print(f"Device Count: {info['device_count']}")
    
    for device in info['devices']:
        print(f"  [{device['id']}] {device['name']}")
    
    # Test computation
    dml = get_dml_device()
    t1 = torch.randn(1000, 1000).to(dml)
    t2 = torch.randn(1000, 1000).to(dml)
    
    import time
    start = time.perf_counter()
    for _ in range(100):
        _ = torch.matmul(t1, t2)
    elapsed = time.perf_counter() - start
    
    print(f"\n100x Matrix Multiply (1000x1000): {elapsed:.3f}s")
    print(f"Per operation: {elapsed/100*1000:.2f}ms")
    print("\n✅ DirectML is working!")


if __name__ == "__main__":
    test_directml()
