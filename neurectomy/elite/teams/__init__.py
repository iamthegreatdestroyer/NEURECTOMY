"""Elite Agent Teams"""

from .base import EliteAgent, TeamCommander, TeamConfig, TeamRole
from .inference import create_inference_team, InferenceCommander
from .compression import create_compression_team, CompressionCommander
from .storage import create_storage_team, StorageCommander
from .analysis import create_analysis_team, AnalysisCommander
from .synthesis import create_synthesis_team, SynthesisCommander

__all__ = [
    "EliteAgent", "TeamCommander", "TeamConfig", "TeamRole",
    "create_inference_team", "InferenceCommander",
    "create_compression_team", "CompressionCommander",
    "create_storage_team", "StorageCommander",
    "create_analysis_team", "AnalysisCommander",
    "create_synthesis_team", "SynthesisCommander",
]
