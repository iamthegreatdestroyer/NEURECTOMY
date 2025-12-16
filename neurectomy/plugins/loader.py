"""Plugin discovery and loading."""

from typing import Dict, List, Optional, Type
from pathlib import Path
import importlib.util
import sys

try:
    import yaml
except ImportError:
    yaml = None

from .base import Plugin, PluginInfo


class PluginLoader:
    """Discovers and loads plugins."""
    
    def __init__(self, plugin_dirs: List[str] = None):
        self.plugin_dirs = plugin_dirs or ["plugins", "neurectomy/plugins/builtin"]
        self._plugins: Dict[str, Plugin] = {}
    
    def discover(self) -> List[PluginInfo]:
        """Discover available plugins."""
        discovered = []
        
        for plugin_dir in self.plugin_dirs:
            path = Path(plugin_dir)
            if not path.exists():
                continue
            
            for plugin_path in path.iterdir():
                if plugin_path.is_dir() and (plugin_path / "plugin.py").exists():
                    info = self._load_plugin_info(plugin_path)
                    if info:
                        discovered.append(info)
        
        return discovered
    
    def load(self, plugin_name: str) -> Optional[Plugin]:
        """Load a plugin by name."""
        if plugin_name in self._plugins:
            return self._plugins[plugin_name]
        
        # Find plugin directory
        for plugin_dir in self.plugin_dirs:
            plugin_path = Path(plugin_dir) / plugin_name
            if plugin_path.exists():
                plugin = self._load_plugin(plugin_path)
                if plugin:
                    self._plugins[plugin_name] = plugin
                    return plugin
        
        return None
    
    def _load_plugin_info(self, path: Path) -> Optional[PluginInfo]:
        """Load plugin info from config."""
        config_path = path / "config.yaml"
        if config_path.exists():
            try:
                if yaml is None:
                    return None
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    return PluginInfo(**config)
            except Exception:
                return None
        return None
    
    def _load_plugin(self, path: Path) -> Optional[Plugin]:
        """Load plugin module."""
        plugin_file = path / "plugin.py"
        
        try:
            spec = importlib.util.spec_from_file_location("plugin", plugin_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules["plugin"] = module
                spec.loader.exec_module(module)
                
                # Find Plugin subclass
                for name in dir(module):
                    obj = getattr(module, name)
                    if isinstance(obj, type) and issubclass(obj, Plugin) and obj != Plugin:
                        return obj()
        except Exception:
            pass
        
        return None
