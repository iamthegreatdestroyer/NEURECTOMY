#!/usr/bin/env python3
"""Phase 11 Verification - Documentation & Examples."""

import sys
from pathlib import Path


def verify_documentation():
    """Verify documentation files exist."""
    required_docs = [
        "docs/index.md",
        "docs/getting-started.md",
        "docs/architecture.md",
        "docs/api/rest-api.md",
        "docs/api/python-sdk.md",
        "docs/tutorials/basic-generation.md",
        "docs/tutorials/agent-tasks.md",
    ]

    print("Documentation Files:")
    all_exist = True
    for path in required_docs:
        if Path(path).exists():
            print(f"  ✓ {path}")
        else:
            print(f"  ❌ {path}")
            all_exist = False

    return all_exist


def verify_examples():
    """Verify example files exist."""
    required_examples = [
        "examples/basic/simple_generation.py",
        "examples/basic/streaming_example.py",
        "examples/agents/multi_agent_task.py",
        "examples/integrations/fastapi_app.py",
    ]

    print("\nExample Files:")
    all_exist = True
    for path in required_examples:
        if Path(path).exists():
            print(f"  ✓ {path}")
        else:
            print(f"  ❌ {path}")
            all_exist = False

    return all_exist


def verify_config():
    """Verify configuration files exist."""
    required_configs = [
        "mkdocs.yml",
    ]

    print("\nConfiguration Files:")
    all_exist = True
    for path in required_configs:
        if Path(path).exists():
            print(f"  ✓ {path}")
        else:
            print(f"  ❌ {path}")
            all_exist = False

    return all_exist


def main():
    """Run verification."""
    print("=" * 60)
    print("  PHASE 11: Documentation & Examples - Verification")
    print("=" * 60)
    print()

    doc_ok = verify_documentation()
    example_ok = verify_examples()
    config_ok = verify_config()

    print()
    print("=" * 60)

    if doc_ok and example_ok and config_ok:
        print("  ✅ PHASE 11 VERIFICATION COMPLETE")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Install mkdocs: pip install mkdocs mkdocs-material")
        print("  2. Build docs: mkdocs build")
        print("  3. Serve locally: mkdocs serve")
        print("  4. View at: http://localhost:8000")
        print()
        print("Run examples:")
        print("  python examples/basic/simple_generation.py")
        print("  python examples/basic/streaming_example.py")
        print("  python examples/agents/multi_agent_task.py")
        print("  python examples/integrations/fastapi_app.py")
        print()
        return 0
    else:
        print("  ❌ PHASE 11 VERIFICATION FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
