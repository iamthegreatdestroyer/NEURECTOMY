# ============================================================================
# Integration Test Configuration
# ============================================================================
"""Integration test package configuration."""

import pytest


def pytest_configure(config):
    """Configure pytest for integration tests."""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests by default unless --integration flag is passed."""
    if config.getoption("--integration", default=False):
        return
    
    skip_integration = pytest.mark.skip(reason="need --integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests"
    )
