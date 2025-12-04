# GPU Environment Activation Script for NEURECTOMY ML Service
# Uses Python 3.12 + DirectML for AMD GPU acceleration

$env:PYTDML_PYTHON = "$env:USERPROFILE\miniconda3\envs\pytdml\python.exe"

function Invoke-GPUPython {
    param([string]$Script)
    & $env:PYTDML_PYTHON $Script
}

function Test-DirectML {
    & $env:PYTDML_PYTHON -c @"
import torch
import torch_directml
dml = torch_directml.device()
print(f'DirectML Device: {dml}')
print(f'Device Name: {torch_directml.device_name(0)}')
t1 = torch.tensor([1.0, 2.0, 3.0]).to(dml)
t2 = torch.tensor([4.0, 5.0, 6.0]).to(dml)
result = t1 + t2
print(f'GPU Result: {result.cpu().tolist()}')
print('DirectML is working!')
"@
}

Write-Host "NEURECTOMY GPU Environment Loaded" -ForegroundColor Green
Write-Host "Python: $env:PYTDML_PYTHON" -ForegroundColor Cyan
Write-Host "Commands: Invoke-GPUPython, Test-DirectML" -ForegroundColor Yellow
