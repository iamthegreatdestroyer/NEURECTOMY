#!/usr/bin/env python3
"""Phase 12 Verification - Advanced Features."""

import sys
from pathlib import Path


def verify_phase_12a():
    """Verify Phase 12A: Multi-Model Support."""
    print("\nPhase 12A: Multi-Model Support")
    print("-" * 60)
    
    passed = 0
    failed = 0
    
    # Check model registry
    try:
        from neurectomy.core.models.registry import ModelRegistry, registry
        models = registry.list_models()
        print(f"  ✓ Model registry: {len(models)} models registered")
        passed += 1
    except Exception as e:
        print(f"  ❌ Model registry: {e}")
        failed += 1
    
    # Check BitNet loader
    try:
        from neurectomy.core.models.loaders.bitnet_loader import BitNetLoader
        loader = BitNetLoader()
        print(f"  ✓ BitNet loader: Available")
        passed += 1
    except Exception as e:
        print(f"  ❌ BitNet loader: {e}")
        failed += 1
    
    # Check GGUF loader
    try:
        from neurectomy.core.models.loaders.gguf_loader import GGUFLoader
        loader = GGUFLoader()
        print(f"  ✓ GGUF loader: Available")
        passed += 1
    except Exception as e:
        print(f"  ❌ GGUF loader: {e}")
        failed += 1
    
    # Check SafeTensors loader
    try:
        from neurectomy.core.models.loaders.safetensors_loader import SafeTensorsLoader
        loader = SafeTensorsLoader()
        print(f"  ✓ SafeTensors loader: Available")
        passed += 1
    except Exception as e:
        print(f"  ❌ SafeTensors loader: {e}")
        failed += 1
    
    return passed, failed


def verify_phase_12b():
    """Verify Phase 12B: Plugin System."""
    print("\nPhase 12B: Plugin System")
    print("-" * 60)
    
    passed = 0
    failed = 0
    
    # Check plugin base
    try:
        from neurectomy.plugins.base import Plugin, ToolPlugin, PluginInfo
        print(f"  ✓ Plugin base: Available")
        passed += 1
    except Exception as e:
        print(f"  ❌ Plugin base: {e}")
        failed += 1
    
    # Check plugin loader
    try:
        from neurectomy.plugins.loader import PluginLoader
        loader = PluginLoader()
        print(f"  ✓ Plugin loader: Available")
        passed += 1
    except Exception as e:
        print(f"  ❌ Plugin loader: {e}")
        failed += 1
    
    # Check plugin registry
    try:
        from neurectomy.plugins.registry import PluginRegistry
        registry = PluginRegistry()
        print(f"  ✓ Plugin registry: Available")
        passed += 1
    except Exception as e:
        print(f"  ❌ Plugin registry: {e}")
        failed += 1
    
    # Check web search plugin
    try:
        from neurectomy.plugins.builtin.web_search import WebSearchPlugin
        plugin = WebSearchPlugin()
        info = plugin.info
        print(f"  ✓ Web search plugin: {info.name} v{info.version}")
        passed += 1
    except Exception as e:
        print(f"  ❌ Web search plugin: {e}")
        failed += 1
    
    return passed, failed


def verify_phase_12c():
    """Verify Phase 12C: Fine-Tuning Pipeline."""
    print("\nPhase 12C: Fine-Tuning Pipeline")
    print("-" * 60)
    
    passed = 0
    failed = 0
    
    # Check dataset handler
    try:
        from neurectomy.core.training.dataset import Dataset, TrainingSample
        sample = TrainingSample(prompt="test", completion="result")
        dataset = Dataset([sample])
        stats = dataset.statistics()
        print(f"  ✓ Dataset handler: {len(dataset)} samples")
        passed += 1
    except Exception as e:
        print(f"  ❌ Dataset handler: {e}")
        failed += 1
    
    # Check LoRA config
    try:
        from neurectomy.core.training.lora import LoRAConfig, LoRALayer
        config = LoRAConfig(rank=8, alpha=16.0)
        layer = LoRALayer(4096, 4096, config)
        print(f"  ✓ LoRA config: rank={config.rank}, alpha={config.alpha}")
        passed += 1
    except Exception as e:
        print(f"  ❌ LoRA config: {e}")
        failed += 1
    
    # Check LoRA model
    try:
        from neurectomy.core.training.lora import LoRAModel, LoRAConfig
        config = LoRAConfig()
        model = LoRAModel(None, config)
        adapters = model.get_adapters()
        print(f"  ✓ LoRA model: {len(adapters)} adapter modules")
        passed += 1
    except Exception as e:
        print(f"  ❌ LoRA model: {e}")
        failed += 1
    
    # Check training script
    try:
        script_path = Path("scripts/train.py")
        if script_path.exists():
            print(f"  ✓ Training script: {script_path}")
            passed += 1
        else:
            print(f"  ❌ Training script: Not found")
            failed += 1
    except Exception as e:
        print(f"  ❌ Training script: {e}")
        failed += 1
    
    return passed, failed


def main():
    """Run all verifications."""
    print("=" * 60)
    print("  PHASE 12: Advanced Features - Verification")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    
    # 12A
    passed, failed = verify_phase_12a()
    total_passed += passed
    total_failed += failed
    
    # 12B
    passed, failed = verify_phase_12b()
    total_passed += passed
    total_failed += failed
    
    # 12C
    passed, failed = verify_phase_12c()
    total_passed += passed
    total_failed += failed
    
    print()
    print("=" * 60)
    print(f"  Results: {total_passed} passed, {total_failed} failed")
    print("=" * 60)
    
    if total_failed == 0:
        print("  ✅ PHASE 12 VERIFICATION COMPLETE")
        print("=" * 60)
        return 0
    else:
        print("  ⚠️  Some verifications failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
