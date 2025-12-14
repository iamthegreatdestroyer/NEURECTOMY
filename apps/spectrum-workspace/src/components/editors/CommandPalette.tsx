/**
 * Command Palette Component for NEURECTOMY IDE
 * Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
 *
 * A quick-access command palette similar to VS Code's Ctrl+Shift+P
 */

import React, {
  useState,
  useEffect,
  useCallback,
  useMemo,
  useRef,
} from "react";
import {
  Command,
  Search,
  FileText,
  Settings,
  Terminal,
  GitBranch,
  Zap,
  Play,
  Save,
} from "lucide-react";

export interface CommandItem {
  id: string;
  label: string;
  description?: string;
  icon?: React.ComponentType<{ className?: string }>;
  shortcut?: string;
  category?: string;
  action: () => void | Promise<void>;
}

export interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  commands?: CommandItem[];
  placeholder?: string;
}

const DEFAULT_COMMANDS: CommandItem[] = [
  {
    id: "file.save",
    label: "Save File",
    description: "Save the current file",
    icon: Save,
    shortcut: "Ctrl+S",
    category: "File",
    action: () => {
      document.dispatchEvent(new CustomEvent("neurectomy:save"));
    },
  },
  {
    id: "file.saveAll",
    label: "Save All Files",
    description: "Save all open files",
    icon: Save,
    shortcut: "Ctrl+Shift+S",
    category: "File",
    action: () => {
      document.dispatchEvent(new CustomEvent("neurectomy:saveAll"));
    },
  },
  {
    id: "terminal.toggle",
    label: "Toggle Terminal",
    description: "Show or hide the integrated terminal",
    icon: Terminal,
    shortcut: "Ctrl+`",
    category: "View",
    action: () => {
      document.dispatchEvent(new CustomEvent("neurectomy:toggleTerminal"));
    },
  },
  {
    id: "git.status",
    label: "Git: Show Status",
    description: "View current git status",
    icon: GitBranch,
    category: "Git",
    action: () => {
      document.dispatchEvent(new CustomEvent("neurectomy:gitStatus"));
    },
  },
  {
    id: "agent.run",
    label: "Run Agent",
    description: "Execute an AI agent on the current file",
    icon: Zap,
    shortcut: "Ctrl+Shift+A",
    category: "Agents",
    action: () => {
      document.dispatchEvent(new CustomEvent("neurectomy:runAgent"));
    },
  },
  {
    id: "experiment.run",
    label: "Run Experiment",
    description: "Start an A/B experiment",
    icon: Play,
    category: "Experiments",
    action: () => {
      document.dispatchEvent(new CustomEvent("neurectomy:runExperiment"));
    },
  },
  {
    id: "settings.open",
    label: "Open Settings",
    description: "Open the settings panel",
    icon: Settings,
    shortcut: "Ctrl+,",
    category: "Preferences",
    action: () => {
      document.dispatchEvent(new CustomEvent("neurectomy:openSettings"));
    },
  },
];

/**
 * CommandPalette - Quick command access modal
 */
export const CommandPalette: React.FC<CommandPaletteProps> = ({
  isOpen,
  onClose,
  commands = DEFAULT_COMMANDS,
  placeholder = "Type a command or search...",
}) => {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Filter commands based on query
  const filteredCommands = useMemo(() => {
    if (!query.trim()) return commands;

    const lowerQuery = query.toLowerCase();
    return commands.filter(
      (cmd) =>
        cmd.label.toLowerCase().includes(lowerQuery) ||
        cmd.description?.toLowerCase().includes(lowerQuery) ||
        cmd.category?.toLowerCase().includes(lowerQuery)
    );
  }, [commands, query]);

  // Group commands by category
  const groupedCommands = useMemo(() => {
    const groups: Record<string, CommandItem[]> = {};
    for (const cmd of filteredCommands) {
      const category = cmd.category || "General";
      if (!groups[category]) {
        groups[category] = [];
      }
      groups[category].push(cmd);
    }
    return groups;
  }, [filteredCommands]);

  // Reset state when opened
  useEffect(() => {
    if (isOpen) {
      setQuery("");
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [isOpen]);

  // Handle keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((i) => Math.min(i + 1, filteredCommands.length - 1));
          break;
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((i) => Math.max(i - 1, 0));
          break;
        case "Enter":
          e.preventDefault();
          if (filteredCommands[selectedIndex]) {
            filteredCommands[selectedIndex].action();
            onClose();
          }
          break;
        case "Escape":
          e.preventDefault();
          onClose();
          break;
      }
    },
    [filteredCommands, selectedIndex, onClose]
  );

  // Handle command click
  const handleCommandClick = useCallback(
    (cmd: CommandItem) => {
      cmd.action();
      onClose();
    },
    [onClose]
  );

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]"
      onClick={onClose}
    >
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />

      {/* Palette */}
      <div
        className="relative w-full max-w-xl bg-neutral-900 rounded-lg shadow-2xl border border-neutral-700 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search Input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-neutral-700">
          <Command className="w-5 h-5 text-neutral-400" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelectedIndex(0);
            }}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className="flex-1 bg-transparent text-neutral-100 placeholder-neutral-500 outline-none text-sm"
          />
          <kbd className="px-2 py-1 text-xs bg-neutral-800 text-neutral-400 rounded">
            ESC
          </kbd>
        </div>

        {/* Command List */}
        <div ref={listRef} className="max-h-80 overflow-y-auto">
          {filteredCommands.length === 0 ? (
            <div className="px-4 py-8 text-center text-neutral-500">
              <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>No commands found</p>
            </div>
          ) : (
            Object.entries(groupedCommands).map(([category, cmds]) => (
              <div key={category}>
                <div className="px-4 py-2 text-xs font-medium text-neutral-500 uppercase tracking-wide bg-neutral-800/50">
                  {category}
                </div>
                {cmds.map((cmd, i) => {
                  const globalIndex = filteredCommands.indexOf(cmd);
                  const isSelected = globalIndex === selectedIndex;
                  const Icon = cmd.icon || FileText;

                  return (
                    <button
                      key={cmd.id}
                      type="button"
                      onClick={() => handleCommandClick(cmd)}
                      className={`
                        w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors
                        ${isSelected ? "bg-indigo-500/20 text-indigo-300" : "hover:bg-neutral-800 text-neutral-300"}
                      `}
                    >
                      <Icon
                        className={`w-4 h-4 ${isSelected ? "text-indigo-400" : "text-neutral-500"}`}
                      />
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium truncate">
                          {cmd.label}
                        </div>
                        {cmd.description && (
                          <div className="text-xs text-neutral-500 truncate">
                            {cmd.description}
                          </div>
                        )}
                      </div>
                      {cmd.shortcut && (
                        <kbd className="px-2 py-0.5 text-xs bg-neutral-800 text-neutral-400 rounded">
                          {cmd.shortcut}
                        </kbd>
                      )}
                    </button>
                  );
                })}
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-2 border-t border-neutral-700 bg-neutral-800/50 flex items-center justify-between text-xs text-neutral-500">
          <div className="flex items-center gap-4">
            <span>↑↓ Navigate</span>
            <span>↵ Select</span>
            <span>ESC Close</span>
          </div>
          <span>{filteredCommands.length} commands</span>
        </div>
      </div>
    </div>
  );
};

CommandPalette.displayName = "CommandPalette";

export default CommandPalette;
