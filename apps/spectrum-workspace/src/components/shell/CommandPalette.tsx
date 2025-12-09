/**
 * Command Palette Component
 *
 * VS Code-style command palette with fuzzy search.
 * Invoked with Ctrl+Shift+P (or Cmd+Shift+P on Mac).
 *
 * Features:
 * - Fuzzy search across all commands
 * - Categories for organization
 * - Recent commands at top
 * - Keyboard navigation
 * - Keybinding display
 *
 * @module @neurectomy/components/shell/CommandPalette
 * @author @APEX @CANVAS
 */

import React, {
  useState,
  useEffect,
  useRef,
  useMemo,
  useCallback,
} from "react";
import { createPortal } from "react-dom";
import { cn } from "@/lib/utils";
import {
  Search,
  Command,
  ChevronRight,
  FileText,
  Settings,
  Terminal,
  GitBranch,
  Folder,
  Play,
  Bug,
  Zap,
  Palette,
  Layout,
  type LucideIcon,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface CommandItem {
  id: string;
  label: string;
  description?: string;
  category?: string;
  icon?: LucideIcon;
  keybinding?: string;
  action: () => void;
  when?: () => boolean;
}

export interface CommandCategory {
  id: string;
  label: string;
  icon?: LucideIcon;
}

export interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  commands: CommandItem[];
  categories?: CommandCategory[];
  recentCommandIds?: string[];
  placeholder?: string;
  className?: string;
}

// ============================================================================
// Fuzzy Search
// ============================================================================

function fuzzyMatch(
  pattern: string,
  str: string
): { match: boolean; score: number; indices: number[] } {
  const patternLower = pattern.toLowerCase();
  const strLower = str.toLowerCase();

  let patternIdx = 0;
  let strIdx = 0;
  const indices: number[] = [];
  let score = 0;
  let consecutiveBonus = 0;

  while (patternIdx < patternLower.length && strIdx < strLower.length) {
    if (patternLower[patternIdx] === strLower[strIdx]) {
      indices.push(strIdx);

      // Score calculation
      score += 1;

      // Bonus for consecutive matches
      if (
        indices.length > 1 &&
        indices[indices.length - 1] === indices[indices.length - 2] + 1
      ) {
        consecutiveBonus += 2;
      }

      // Bonus for matching at word boundaries
      if (strIdx === 0 || /[\s\-_./\\:]/.test(str[strIdx - 1])) {
        score += 3;
      }

      patternIdx++;
    }
    strIdx++;
  }

  // Check if full pattern was matched
  if (patternIdx !== patternLower.length) {
    return { match: false, score: 0, indices: [] };
  }

  // Apply consecutive bonus
  score += consecutiveBonus;

  // Penalize for string length (prefer shorter matches)
  score -= str.length * 0.1;

  return { match: true, score, indices };
}

// ============================================================================
// Highlight Component
// ============================================================================

function HighlightedText({
  text,
  indices,
  className,
}: {
  text: string;
  indices: number[];
  className?: string;
}) {
  if (indices.length === 0) {
    return <span className={className}>{text}</span>;
  }

  const chars = text.split("");
  const indexSet = new Set(indices);

  return (
    <span className={className}>
      {chars.map((char, i) =>
        indexSet.has(i) ? (
          <span
            key={i}
            className="text-accent-foreground bg-accent/30 font-semibold"
          >
            {char}
          </span>
        ) : (
          <span key={i}>{char}</span>
        )
      )}
    </span>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function CommandPalette({
  isOpen,
  onClose,
  commands,
  categories = [],
  recentCommandIds = [],
  placeholder = "Type a command or search...",
  className,
}: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Filter and sort commands
  const filteredCommands = useMemo(() => {
    // Filter by visibility
    const visibleCommands = commands.filter((cmd) => !cmd.when || cmd.when());

    if (!query.trim()) {
      // Show recent commands first, then all commands
      const recentSet = new Set(recentCommandIds);
      const recent = visibleCommands.filter((cmd) => recentSet.has(cmd.id));
      const others = visibleCommands.filter((cmd) => !recentSet.has(cmd.id));

      return [
        ...recent.map((cmd) => ({ command: cmd, score: 1000, indices: [] })),
        ...others
          .slice(0, 20)
          .map((cmd) => ({ command: cmd, score: 0, indices: [] })),
      ];
    }

    // Fuzzy search
    const results = visibleCommands
      .map((cmd) => {
        const labelMatch = fuzzyMatch(query, cmd.label);
        const descMatch = cmd.description
          ? fuzzyMatch(query, cmd.description)
          : { match: false, score: 0, indices: [] };
        const categoryMatch = cmd.category
          ? fuzzyMatch(query, cmd.category)
          : { match: false, score: 0, indices: [] };

        const bestMatch = [labelMatch, descMatch, categoryMatch]
          .filter((m) => m.match)
          .sort((a, b) => b.score - a.score)[0];

        return {
          command: cmd,
          score: bestMatch?.score || 0,
          indices: bestMatch === labelMatch ? labelMatch.indices : [],
        };
      })
      .filter((result) => result.score > 0)
      .sort((a, b) => b.score - a.score);

    return results;
  }, [commands, query, recentCommandIds]);

  // Reset selection when results change
  useEffect(() => {
    setSelectedIndex(0);
  }, [filteredCommands.length]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setQuery("");
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  // Scroll selected item into view
  useEffect(() => {
    const list = listRef.current;
    if (!list) return;

    const selectedItem = list.children[selectedIndex] as
      | HTMLElement
      | undefined;
    if (selectedItem) {
      selectedItem.scrollIntoView({ block: "nearest" });
    }
  }, [selectedIndex]);

  // Keyboard navigation
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
            filteredCommands[selectedIndex].command.action();
            onClose();
          }
          break;
        case "Escape":
          e.preventDefault();
          onClose();
          break;
        case "Tab":
          e.preventDefault();
          if (e.shiftKey) {
            setSelectedIndex((i) => Math.max(i - 1, 0));
          } else {
            setSelectedIndex((i) =>
              Math.min(i + 1, filteredCommands.length - 1)
            );
          }
          break;
      }
    },
    [filteredCommands, selectedIndex, onClose]
  );

  // Execute command
  const executeCommand = useCallback(
    (cmd: CommandItem) => {
      cmd.action();
      onClose();
    },
    [onClose]
  );

  // Global keyboard shortcut
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      // Ctrl+Shift+P or Cmd+Shift+P
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "P") {
        e.preventDefault();
        // Toggle palette if already open, otherwise handled by parent
      }

      // Escape to close
      if (e.key === "Escape" && isOpen) {
        e.preventDefault();
        onClose();
      }
    };

    window.addEventListener("keydown", handleGlobalKeyDown);
    return () => window.removeEventListener("keydown", handleGlobalKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]"
      onClick={onClose}
    >
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" />

      {/* Palette */}
      <div
        className={cn(
          "relative w-full max-w-2xl",
          "bg-card border border-border rounded-lg shadow-2xl",
          "overflow-hidden",
          className
        )}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search Input */}
        <div className="flex items-center gap-2 px-4 py-3 border-b border-border">
          <Command size={16} className="text-muted-foreground" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className={cn(
              "flex-1 bg-transparent border-none outline-none",
              "text-sm text-foreground placeholder:text-muted-foreground"
            )}
            autoComplete="off"
            autoCorrect="off"
            autoCapitalize="off"
            spellCheck={false}
          />
          <kbd className="px-2 py-0.5 text-xs bg-muted rounded border border-border">
            ESC
          </kbd>
        </div>

        {/* Results */}
        <div ref={listRef} className="max-h-80 overflow-y-auto">
          {filteredCommands.length === 0 ? (
            <div className="px-4 py-8 text-center text-sm text-muted-foreground">
              No commands found
            </div>
          ) : (
            filteredCommands.map((result, index) => (
              <CommandItemRow
                key={result.command.id}
                command={result.command}
                indices={result.indices}
                selected={index === selectedIndex}
                onClick={() => executeCommand(result.command)}
                onMouseEnter={() => setSelectedIndex(index)}
              />
            ))
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-2 border-t border-border bg-muted/30">
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <kbd className="px-1 py-0.5 bg-muted rounded">↑</kbd>
              <kbd className="px-1 py-0.5 bg-muted rounded">↓</kbd>
              to navigate
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1 py-0.5 bg-muted rounded">↵</kbd>
              to select
            </span>
          </div>
          <div className="text-xs text-muted-foreground">
            {filteredCommands.length} commands
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}

// ============================================================================
// Command Item Row
// ============================================================================

function CommandItemRow({
  command,
  indices,
  selected,
  onClick,
  onMouseEnter,
}: {
  command: CommandItem;
  indices: number[];
  selected: boolean;
  onClick: () => void;
  onMouseEnter: () => void;
}) {
  const Icon = command.icon || Command;

  return (
    <button
      className={cn(
        "w-full flex items-center gap-3 px-4 py-2.5 text-left",
        "transition-colors",
        selected ? "bg-accent text-accent-foreground" : "hover:bg-muted/50"
      )}
      onClick={onClick}
      onMouseEnter={onMouseEnter}
    >
      <Icon
        size={16}
        className={cn(
          selected ? "text-accent-foreground" : "text-muted-foreground"
        )}
      />

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          {command.category && (
            <>
              <span className="text-xs text-muted-foreground">
                {command.category}
              </span>
              <ChevronRight size={12} className="text-muted-foreground" />
            </>
          )}
          <HighlightedText
            text={command.label}
            indices={indices}
            className="text-sm truncate"
          />
        </div>
        {command.description && (
          <div className="text-xs text-muted-foreground truncate mt-0.5">
            {command.description}
          </div>
        )}
      </div>

      {command.keybinding && (
        <div className="flex items-center gap-1">
          {command.keybinding.split("+").map((key, i) => (
            <kbd
              key={i}
              className={cn(
                "px-1.5 py-0.5 text-xs rounded border",
                selected
                  ? "bg-accent-foreground/10 border-accent-foreground/20"
                  : "bg-muted border-border"
              )}
            >
              {key}
            </kbd>
          ))}
        </div>
      )}
    </button>
  );
}

// ============================================================================
// Default Commands Factory
// ============================================================================

export function createDefaultCommands(handlers: {
  openFile?: () => void;
  saveFile?: () => void;
  closeFile?: () => void;
  newFile?: () => void;
  newFolder?: () => void;
  openSettings?: () => void;
  openTerminal?: () => void;
  toggleSidebar?: () => void;
  toggleBottomPanel?: () => void;
  runProject?: () => void;
  debugProject?: () => void;
  gitCommit?: () => void;
  gitPush?: () => void;
  gitPull?: () => void;
  formatDocument?: () => void;
  findInFiles?: () => void;
  goToLine?: () => void;
  goToSymbol?: () => void;
  changeTheme?: () => void;
  changeLanguage?: () => void;
  reloadWindow?: () => void;
}): CommandItem[] {
  return [
    // File
    {
      id: "file.open",
      label: "Open File",
      description: "Open a file from the workspace",
      category: "File",
      icon: FileText,
      keybinding: "Ctrl+O",
      action: handlers.openFile || (() => {}),
    },
    {
      id: "file.save",
      label: "Save",
      description: "Save the current file",
      category: "File",
      icon: FileText,
      keybinding: "Ctrl+S",
      action: handlers.saveFile || (() => {}),
    },
    {
      id: "file.close",
      label: "Close Editor",
      description: "Close the current editor tab",
      category: "File",
      icon: FileText,
      keybinding: "Ctrl+W",
      action: handlers.closeFile || (() => {}),
    },
    {
      id: "file.newFile",
      label: "New File",
      description: "Create a new file",
      category: "File",
      icon: FileText,
      keybinding: "Ctrl+N",
      action: handlers.newFile || (() => {}),
    },
    {
      id: "file.newFolder",
      label: "New Folder",
      description: "Create a new folder",
      category: "File",
      icon: Folder,
      action: handlers.newFolder || (() => {}),
    },

    // View
    {
      id: "view.toggleSidebar",
      label: "Toggle Primary Sidebar",
      description: "Show or hide the primary sidebar",
      category: "View",
      icon: Layout,
      keybinding: "Ctrl+B",
      action: handlers.toggleSidebar || (() => {}),
    },
    {
      id: "view.togglePanel",
      label: "Toggle Panel",
      description: "Show or hide the bottom panel",
      category: "View",
      icon: Layout,
      keybinding: "Ctrl+J",
      action: handlers.toggleBottomPanel || (() => {}),
    },
    {
      id: "view.terminal",
      label: "Toggle Terminal",
      description: "Open integrated terminal",
      category: "View",
      icon: Terminal,
      keybinding: "Ctrl+`",
      action: handlers.openTerminal || (() => {}),
    },

    // Settings
    {
      id: "preferences.settings",
      label: "Open Settings",
      description: "Open user settings",
      category: "Preferences",
      icon: Settings,
      keybinding: "Ctrl+,",
      action: handlers.openSettings || (() => {}),
    },
    {
      id: "preferences.theme",
      label: "Color Theme",
      description: "Change color theme",
      category: "Preferences",
      icon: Palette,
      keybinding: "Ctrl+K Ctrl+T",
      action: handlers.changeTheme || (() => {}),
    },

    // Run
    {
      id: "run.start",
      label: "Run Without Debugging",
      description: "Start the project without debugger",
      category: "Run",
      icon: Play,
      keybinding: "Ctrl+F5",
      action: handlers.runProject || (() => {}),
    },
    {
      id: "debug.start",
      label: "Start Debugging",
      description: "Start debugging the project",
      category: "Debug",
      icon: Bug,
      keybinding: "F5",
      action: handlers.debugProject || (() => {}),
    },

    // Git
    {
      id: "git.commit",
      label: "Commit",
      description: "Commit staged changes",
      category: "Git",
      icon: GitBranch,
      action: handlers.gitCommit || (() => {}),
    },
    {
      id: "git.push",
      label: "Push",
      description: "Push commits to remote",
      category: "Git",
      icon: GitBranch,
      action: handlers.gitPush || (() => {}),
    },
    {
      id: "git.pull",
      label: "Pull",
      description: "Pull from remote",
      category: "Git",
      icon: GitBranch,
      action: handlers.gitPull || (() => {}),
    },

    // Editor
    {
      id: "editor.format",
      label: "Format Document",
      description: "Format the current document",
      category: "Editor",
      icon: FileText,
      keybinding: "Shift+Alt+F",
      action: handlers.formatDocument || (() => {}),
    },
    {
      id: "editor.goToLine",
      label: "Go to Line...",
      description: "Go to a specific line number",
      category: "Go",
      icon: FileText,
      keybinding: "Ctrl+G",
      action: handlers.goToLine || (() => {}),
    },
    {
      id: "editor.goToSymbol",
      label: "Go to Symbol in Editor",
      description: "Navigate to a symbol in the current file",
      category: "Go",
      icon: Zap,
      keybinding: "Ctrl+Shift+O",
      action: handlers.goToSymbol || (() => {}),
    },

    // Search
    {
      id: "search.findInFiles",
      label: "Find in Files",
      description: "Search across all files",
      category: "Search",
      icon: Search,
      keybinding: "Ctrl+Shift+F",
      action: handlers.findInFiles || (() => {}),
    },

    // Developer
    {
      id: "workbench.action.reloadWindow",
      label: "Reload Window",
      description: "Reload the application window",
      category: "Developer",
      icon: Command,
      action: handlers.reloadWindow || (() => window.location.reload()),
    },
  ];
}

// ============================================================================
// Hook for Command Palette
// ============================================================================

export function useCommandPalette() {
  const [isOpen, setIsOpen] = useState(false);
  const [recentCommands, setRecentCommands] = useState<string[]>([]);

  // Global keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "P") {
        e.preventDefault();
        setIsOpen((prev) => !prev);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const open = useCallback(() => setIsOpen(true), []);
  const close = useCallback(() => setIsOpen(false), []);

  const trackCommand = useCallback((commandId: string) => {
    setRecentCommands((prev) => {
      const filtered = prev.filter((id) => id !== commandId);
      return [commandId, ...filtered].slice(0, 10);
    });
  }, []);

  return {
    isOpen,
    open,
    close,
    recentCommands,
    trackCommand,
  };
}

export default CommandPalette;
