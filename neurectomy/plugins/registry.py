"""Plugin registry."""

from typing import Dict, List, Optional

from .base import Plugin, PluginInfo, ToolPlugin
from .loader import PluginLoader


class PluginRegistry:
    """Central registry for plugins."""
    
    def __init__(self):
        self._loader = PluginLoader()
        self._plugins: Dict[str, Plugin] = {}
        self._tools: Dict[str, tuple] = {}  # tool_name -> (plugin, method)
    
    def discover_plugins(self) -> List[PluginInfo]:
        """Discover all available plugins."""
        return self._loader.discover()
    
    def load_plugin(self, name: str, config: dict = None) -> bool:
        """Load and initialize a plugin."""
        plugin = self._loader.load(name)
        if not plugin:
            return False
        
        plugin.initialize(config or {})
        self._plugins[name] = plugin
        
        # Register tools if applicable
        if isinstance(plugin, ToolPlugin):
            for tool in plugin.get_tools():
                self._tools[tool] = (plugin, tool)
        
        return True
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get loaded plugin."""
        return self._plugins.get(name)
    
    def list_plugins(self) -> List[str]:
        """List loaded plugins."""
        return list(self._plugins.keys())
    
    def list_tools(self) -> List[str]:
        """List available tools."""
        return list(self._tools.keys())
    
    def call_tool(self, tool_name: str, **kwargs):
        """Call a registered tool."""
        if tool_name not in self._tools:
            raise ValueError(f"Tool not found: {tool_name}")
        
        plugin, method = self._tools[tool_name]
        return plugin.call_tool(method, **kwargs)
    
    def unload_plugin(self, name: str) -> None:
        """Unload a plugin."""
        if name in self._plugins:
            self._plugins[name].cleanup()
            del self._plugins[name]
            
            # Remove tools from this plugin
            tools_to_remove = [
                tool for tool, (p, _) in self._tools.items()
                if p == self._plugins.get(name)
            ]
            for tool in tools_to_remove:
                del self._tools[tool]
