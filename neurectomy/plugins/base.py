"""Plugin base class."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class PluginInfo:
    """Plugin metadata."""
    name: str
    version: str
    description: str
    author: str = ""
    requires: list = field(default_factory=list)


class Plugin(ABC):
    """Base class for all plugins."""
    
    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Plugin information."""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        pass
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute plugin functionality."""
        pass
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass


class ToolPlugin(Plugin):
    """Plugin that provides tools/capabilities."""
    
    @abstractmethod
    def get_tools(self) -> list:
        """Get list of tools provided by this plugin."""
        pass
    
    @abstractmethod
    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call a specific tool."""
        pass
