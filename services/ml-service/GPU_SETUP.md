# NEURECTOMY ML Service - GPU Setup

## Dual Python Environment Architecture

This project uses two Python environments:

| Environment | Python | Purpose | GPU Support |
|------------|--------|---------|-------------|
| `ml-service` (Poetry) | 3.13 | Main ML service, FastAPI, latest packages | CPU only |
| `pytdml` (Conda) | 3.12 | GPU-accelerated training & inference | DirectML (AMD/Intel/Qualcomm) |

## Why Two Environments?

- **Python 3.13** has the latest packages but DirectML doesn't support it yet
- **Python 3.12** is required for `torch-directml` GPU acceleration
- The main service runs on 3.13 for compatibility with latest ML packages
- GPU-intensive tasks can be offloaded to the 3.12 environment

## Quick Start

### Main Service (CPU - Python 3.13)
```powershell
cd services/ml-service
poetry install
poetry run uvicorn main:app --reload
```

### GPU Tasks (DirectML - Python 3.12)
```powershell
# Activate GPU environment
. .\scripts\gpu_env.ps1

# Test DirectML
Test-DirectML

# Run GPU-accelerated script
Invoke-GPUPython your_script.py
```

## DirectML Device Info

Your system:
- **GPU**: AMD Radeon (TM) Graphics
- **Backend**: DirectML (Windows native)
- **Supported**: All DirectX 12 GPUs (AMD, Intel, NVIDIA, Qualcomm)

## Installed Packages (pytdml)

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.4.1 | Deep learning framework |
| torchvision | 0.19.1 | Computer vision utilities |
| torch-directml | 0.2.5.dev240914 | GPU acceleration |
| numpy | 2.3.5 | Numerical computing |
| pillow | 12.0.0 | Image processing |

## Usage in Code

```python
# In GPU-accelerated scripts (Python 3.12 environment)
from src.utils.directml_utils import (
    get_dml_device,
    to_gpu,
    to_cpu,
    DirectMLModelWrapper,
    is_directml_available
)

# Get device
dml = get_dml_device()

# Move tensors
gpu_tensor = to_gpu(cpu_tensor)
cpu_result = to_cpu(gpu_tensor)

# Wrap models
model = MyModel()
gpu_model = DirectMLModelWrapper(model)
output = gpu_model.inference(input_tensor)
```

## Performance Benchmark

100x Matrix Multiply (1000x1000):
- **DirectML (AMD Radeon)**: ~13.56ms per operation
- **CPU**: ~50-100ms per operation (depends on CPU)

## Architecture Decision

For Phase 2 (Intelligence Layer):
- **Inference**: Can use CPU (Python 3.13) for light workloads
- **Training**: Use GPU environment (Python 3.12) for heavy tasks
- **Production**: Consider cloud GPUs (Azure ML) for large-scale training

## Conda Environment Management

```powershell
# List environments
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" env list

# Activate pytdml
& "$env:USERPROFILE\miniconda3\Scripts\activate.bat" pytdml

# Install additional packages
& "$env:USERPROFILE\miniconda3\envs\pytdml\python.exe" -m pip install <package>
```

## Troubleshooting

### DirectML not found
Ensure you're using the pytdml environment:
```powershell
& "$env:USERPROFILE\miniconda3\envs\pytdml\python.exe" -c "import torch_directml; print('OK')"
```

### Version mismatch
DirectML requires specific PyTorch versions. Don't upgrade torch in pytdml:
- torch==2.4.1
- torchvision==0.19.1
- torch-directml==0.2.5.dev240914
