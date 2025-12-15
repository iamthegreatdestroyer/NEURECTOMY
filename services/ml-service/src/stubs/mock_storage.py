"""Mock Storage Bridge for Testing"""

import time
import uuid
from typing import Dict, Optional

from ..api.types import Artifact, ProjectContext
from ..api.interfaces import StorageBridge


class MockStorageBridge(StorageBridge):
    """Mock storage bridge for integration testing."""

    def __init__(self):
        self._storage: Dict[str, Artifact] = {}
        self._projects: Dict[str, ProjectContext] = {}
        self._locked: Dict[str, bool] = {}
        self._available = True

    def store_artifact(self, artifact: Artifact, encrypt: bool = True) -> str:
        storage_key = artifact.artifact_id or f"artifact_{uuid.uuid4().hex[:8]}"
        self._storage[storage_key] = artifact
        return storage_key

    def retrieve_artifact(self, artifact_id: str) -> Optional[Artifact]:
        return self._storage.get(artifact_id)

    def store_project(self, project: ProjectContext) -> bool:
        self._projects[project.project_id] = project
        return True

    def lock_project(self, project_id: str, passphrase: Optional[str] = None) -> bool:
        if project_id in self._projects:
            self._locked[project_id] = True
            return True
        return False

    def is_available(self) -> bool:
        return self._available
