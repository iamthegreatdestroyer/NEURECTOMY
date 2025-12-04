import { useState, useEffect, useMemo, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Search,
  LayoutDashboard,
  Box,
  Brain,
  Beaker,
  Compass,
  Shield,
  Settings,
  Bot,
  Plus,
  Terminal,
  FileCode,
  Zap,
  X,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useAppStore } from '@/stores/app.store';
import { useAgentsStore } from '@/stores/agents.store';

interface CommandItem {
  id: string;
  title: string;
  description?: string;
  icon: React.ReactNode;
  action: () => void;
  category: 'navigation' | 'agents' | 'actions' | 'settings';
  keywords?: string[];
}

export function CommandPalette() {
  const { commandPalette, closeCommandPalette } = useAppStore();
  const isOpen = commandPalette.isOpen;
  const onClose = closeCommandPalette;
  
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const navigate = useNavigate();
  const { agents } = useAgentsStore();

  // Navigation commands
  const navigationCommands: CommandItem[] = useMemo(() => [
    {
      id: 'nav-dashboard',
      title: 'Go to Dashboard',
      description: 'View system overview and metrics',
      icon: <LayoutDashboard className="h-4 w-4" />,
      action: () => navigate('/'),
      category: 'navigation',
      keywords: ['home', 'overview', 'main'],
    },
    {
      id: 'nav-dimensional-forge',
      title: 'Dimensional Forge',
      description: '3D visualization and spatial computing',
      icon: <Box className="h-4 w-4" />,
      action: () => navigate('/dimensional-forge'),
      category: 'navigation',
      keywords: ['3d', 'visualization', 'forge', 'spatial'],
    },
    {
      id: 'nav-container-command',
      title: 'Container Command',
      description: 'Docker and Kubernetes management',
      icon: <Terminal className="h-4 w-4" />,
      action: () => navigate('/container-command'),
      category: 'navigation',
      keywords: ['docker', 'kubernetes', 'containers', 'pods'],
    },
    {
      id: 'nav-intelligence-foundry',
      title: 'Intelligence Foundry',
      description: 'ML training and model management',
      icon: <Brain className="h-4 w-4" />,
      action: () => navigate('/intelligence-foundry'),
      category: 'navigation',
      keywords: ['ml', 'training', 'models', 'ai'],
    },
    {
      id: 'nav-discovery-engine',
      title: 'Discovery Engine',
      description: 'Research and knowledge exploration',
      icon: <Compass className="h-4 w-4" />,
      action: () => navigate('/discovery-engine'),
      category: 'navigation',
      keywords: ['research', 'knowledge', 'graph', 'explore'],
    },
    {
      id: 'nav-legal-fortress',
      title: 'Legal Fortress',
      description: 'Compliance and legal documentation',
      icon: <Shield className="h-4 w-4" />,
      action: () => navigate('/legal-fortress'),
      category: 'navigation',
      keywords: ['legal', 'compliance', 'security', 'audit'],
    },
    {
      id: 'nav-settings',
      title: 'Settings',
      description: 'Configure application preferences',
      icon: <Settings className="h-4 w-4" />,
      action: () => navigate('/settings'),
      category: 'navigation',
      keywords: ['preferences', 'config', 'options'],
    },
  ], [navigate]);

  // Agent commands
  const agentCommands: CommandItem[] = useMemo(() => [
    {
      id: 'agent-create',
      title: 'Create New Agent',
      description: 'Deploy a new AI agent',
      icon: <Plus className="h-4 w-4" />,
      action: () => navigate('/agent-editor/new'),
      category: 'agents',
      keywords: ['new', 'deploy', 'agent'],
    },
    ...agents.slice(0, 5).map(agent => ({
      id: `agent-${agent.id}`,
      title: `Open ${agent.name}`,
      description: `${agent.type} - ${agent.status}`,
      icon: <Bot className="h-4 w-4" />,
      action: () => navigate(`/agent-editor/${agent.id}`),
      category: 'agents' as const,
      keywords: [agent.type, agent.status],
    })),
  ], [agents, navigate]);

  // Action commands
  const actionCommands: CommandItem[] = useMemo(() => [
    {
      id: 'action-new-container',
      title: 'Create Container',
      description: 'Spin up a new Docker container',
      icon: <Box className="h-4 w-4" />,
      action: () => {
        navigate('/container-command');
        // Could trigger modal via store
      },
      category: 'actions',
      keywords: ['docker', 'container', 'new'],
    },
    {
      id: 'action-start-training',
      title: 'Start Training Job',
      description: 'Begin a new ML training session',
      icon: <Zap className="h-4 w-4" />,
      action: () => navigate('/intelligence-foundry'),
      category: 'actions',
      keywords: ['train', 'ml', 'model'],
    },
    {
      id: 'action-code-review',
      title: 'Code Review',
      description: 'Analyze code with AI assistance',
      icon: <FileCode className="h-4 w-4" />,
      action: () => {
        // Open code review modal
        console.log('Code review action');
      },
      category: 'actions',
      keywords: ['review', 'analyze', 'code'],
    },
  ], [navigate]);

  // Combine all commands
  const allCommands = useMemo(() => [
    ...navigationCommands,
    ...agentCommands,
    ...actionCommands,
  ], [navigationCommands, agentCommands, actionCommands]);

  // Filter commands based on query
  const filteredCommands = useMemo(() => {
    if (!query.trim()) return allCommands;
    
    const lowerQuery = query.toLowerCase();
    return allCommands.filter(cmd => 
      cmd.title.toLowerCase().includes(lowerQuery) ||
      cmd.description?.toLowerCase().includes(lowerQuery) ||
      cmd.keywords?.some(k => k.toLowerCase().includes(lowerQuery))
    );
  }, [allCommands, query]);

  // Group commands by category
  const groupedCommands = useMemo(() => {
    const groups: Record<string, CommandItem[]> = {
      navigation: [],
      agents: [],
      actions: [],
      settings: [],
    };
    
    filteredCommands.forEach(cmd => {
      groups[cmd.category].push(cmd);
    });
    
    return groups;
  }, [filteredCommands]);

  // Handle keyboard navigation
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (!isOpen) return;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex(prev => 
          Math.min(prev + 1, filteredCommands.length - 1)
        );
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex(prev => Math.max(prev - 1, 0));
        break;
      case 'Enter':
        e.preventDefault();
        if (filteredCommands[selectedIndex]) {
          filteredCommands[selectedIndex].action();
          onClose();
        }
        break;
      case 'Escape':
        e.preventDefault();
        onClose();
        break;
    }
  }, [isOpen, filteredCommands, selectedIndex, onClose]);

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  // Reset state when opened/closed
  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setSelectedIndex(0);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const categoryLabels: Record<string, string> = {
    navigation: 'Navigation',
    agents: 'Agents',
    actions: 'Actions',
    settings: 'Settings',
  };

  let flatIndex = 0;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh]">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-background/80 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Command Palette */}
      <div className="relative w-full max-w-xl bg-card border border-border rounded-xl shadow-2xl overflow-hidden">
        {/* Search input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-border">
          <Search className="h-5 w-5 text-muted-foreground" />
          <input
            type="text"
            placeholder="Type a command or search..."
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelectedIndex(0);
            }}
            className="flex-1 bg-transparent text-foreground placeholder:text-muted-foreground focus:outline-none"
            autoFocus
          />
          <kbd className="hidden sm:inline-flex h-5 items-center gap-1 rounded border border-border bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground">
            ESC
          </kbd>
          <button onClick={onClose} className="sm:hidden">
            <X className="h-5 w-5 text-muted-foreground" />
          </button>
        </div>

        {/* Results */}
        <div className="max-h-[400px] overflow-y-auto p-2">
          {filteredCommands.length === 0 ? (
            <div className="py-8 text-center text-muted-foreground">
              No commands found for "{query}"
            </div>
          ) : (
            Object.entries(groupedCommands).map(([category, commands]) => {
              if (commands.length === 0) return null;
              
              return (
                <div key={category} className="mb-2">
                  <div className="px-2 py-1.5 text-xs font-medium text-muted-foreground">
                    {categoryLabels[category]}
                  </div>
                  {commands.map((cmd) => {
                    const currentIndex = flatIndex++;
                    const isSelected = currentIndex === selectedIndex;
                    
                    return (
                      <button
                        key={cmd.id}
                        onClick={() => {
                          cmd.action();
                          onClose();
                        }}
                        onMouseEnter={() => setSelectedIndex(currentIndex)}
                        className={cn(
                          'w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-colors',
                          isSelected 
                            ? 'bg-accent text-accent-foreground' 
                            : 'hover:bg-accent/50'
                        )}
                      >
                        <div className={cn(
                          'flex items-center justify-center h-8 w-8 rounded-md',
                          isSelected ? 'bg-primary text-primary-foreground' : 'bg-muted'
                        )}>
                          {cmd.icon}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="font-medium truncate">{cmd.title}</div>
                          {cmd.description && (
                            <div className="text-xs text-muted-foreground truncate">
                              {cmd.description}
                            </div>
                          )}
                        </div>
                        {isSelected && (
                          <kbd className="hidden sm:inline-flex h-5 items-center rounded border border-border bg-muted px-1.5 font-mono text-[10px]">
                            ↵
                          </kbd>
                        )}
                      </button>
                    );
                  })}
                </div>
              );
            })
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-2 border-t border-border bg-muted/50 text-xs text-muted-foreground">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <kbd className="px-1 rounded border border-border bg-background">↑</kbd>
              <kbd className="px-1 rounded border border-border bg-background">↓</kbd>
              to navigate
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1 rounded border border-border bg-background">↵</kbd>
              to select
            </span>
          </div>
          <span>{filteredCommands.length} commands</span>
        </div>
      </div>
    </div>
  );
}

export default CommandPalette;
