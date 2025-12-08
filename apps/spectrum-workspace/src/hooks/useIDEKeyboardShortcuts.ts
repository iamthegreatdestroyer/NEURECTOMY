/**
 * IDE Keyboard Shortcuts Hook
 * Handles keyboard shortcuts specific to the IDE view
 */

import { useEffect, useCallback } from "react";

export interface IDEShortcutHandlers {
  toggleTerminal: () => void;
  toggleAgentGraph: () => void;
  toggleSidebar?: () => void;
  switchToExplorer: () => void;
  switchToSearch: () => void;
  switchToGit: () => void;
  switchToAgents: () => void;
  switchToExperiments: () => void;
  switchToExtensions: () => void;
  switchToSettings: () => void;
  closeActiveFile?: () => void;
  saveFile?: () => void;
  openCommandPalette?: () => void;
  focusEditor?: () => void;
}

interface ShortcutDefinition {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  meta?: boolean;
  description: string;
  category: "panel" | "sidebar" | "file" | "navigation" | "command";
}

// Define all IDE shortcuts with their metadata
export const ideShortcuts: Record<
  keyof IDEShortcutHandlers,
  ShortcutDefinition
> = {
  // Panel toggles
  toggleTerminal: {
    key: "`",
    ctrl: true,
    description: "Toggle Terminal",
    category: "panel",
  },
  toggleAgentGraph: {
    key: "a",
    ctrl: true,
    shift: true,
    description: "Toggle Agent Graph",
    category: "panel",
  },
  toggleSidebar: {
    key: "b",
    ctrl: true,
    description: "Toggle Sidebar",
    category: "panel",
  },

  // Sidebar navigation
  switchToExplorer: {
    key: "e",
    ctrl: true,
    shift: true,
    description: "Explorer Panel",
    category: "sidebar",
  },
  switchToSearch: {
    key: "f",
    ctrl: true,
    shift: true,
    description: "Search Panel",
    category: "sidebar",
  },
  switchToGit: {
    key: "g",
    ctrl: true,
    shift: true,
    description: "Git Panel",
    category: "sidebar",
  },
  switchToAgents: {
    key: "i",
    ctrl: true,
    shift: true,
    description: "AI Agents Panel",
    category: "sidebar",
  },
  switchToExperiments: {
    key: "x",
    ctrl: true,
    shift: true,
    description: "Experiments Panel",
    category: "sidebar",
  },
  switchToExtensions: {
    key: "p",
    ctrl: true,
    shift: true,
    description: "Extensions Panel",
    category: "sidebar",
  },
  switchToSettings: {
    key: ",",
    ctrl: true,
    description: "Settings",
    category: "sidebar",
  },

  // File operations
  closeActiveFile: {
    key: "w",
    ctrl: true,
    description: "Close Active File",
    category: "file",
  },
  saveFile: {
    key: "s",
    ctrl: true,
    description: "Save File",
    category: "file",
  },

  // Command operations
  openCommandPalette: {
    key: "p",
    ctrl: true,
    description: "Command Palette",
    category: "command",
  },
  focusEditor: {
    key: "1",
    ctrl: true,
    description: "Focus Editor",
    category: "navigation",
  },
};

/**
 * Hook for handling IDE keyboard shortcuts
 */
export function useIDEKeyboardShortcuts(
  handlers: Partial<IDEShortcutHandlers>
) {
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs (except for Escape)
      const target = event.target as HTMLElement;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target.isContentEditable
      ) {
        // Allow certain shortcuts in inputs
        const allowedInInputs = ["Escape", "s"];
        if (!allowedInInputs.includes(event.key.toLowerCase())) {
          return;
        }
      }

      // Check each shortcut
      for (const [handlerName, shortcut] of Object.entries(ideShortcuts)) {
        const handler = handlers[handlerName as keyof IDEShortcutHandlers];
        if (!handler) continue;

        const ctrlMatch = shortcut.ctrl
          ? event.ctrlKey || event.metaKey
          : !event.ctrlKey && !event.metaKey;
        const shiftMatch = shortcut.shift ? event.shiftKey : !event.shiftKey;
        const altMatch = shortcut.alt ? event.altKey : !event.altKey;
        const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase();

        if (keyMatch && ctrlMatch && shiftMatch && altMatch) {
          event.preventDefault();
          handler();
          return;
        }
      }
    },
    [handlers]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);
}

/**
 * Format shortcut for display
 */
export function formatShortcut(shortcut: ShortcutDefinition): string {
  const isMac =
    typeof navigator !== "undefined" &&
    navigator.platform.toUpperCase().indexOf("MAC") >= 0;

  const parts: string[] = [];

  if (shortcut.ctrl) parts.push(isMac ? "⌘" : "Ctrl");
  if (shortcut.alt) parts.push(isMac ? "⌥" : "Alt");
  if (shortcut.shift) parts.push(isMac ? "⇧" : "Shift");
  if (shortcut.meta) parts.push(isMac ? "⌘" : "Win");

  // Format special keys
  const keyDisplay =
    shortcut.key === "`"
      ? "`"
      : shortcut.key === ","
        ? ","
        : shortcut.key.toUpperCase();
  parts.push(keyDisplay);

  return parts.join(isMac ? "" : "+");
}

/**
 * Get all shortcuts grouped by category
 */
export function getShortcutsByCategory(): Record<
  string,
  Array<{ name: string; shortcut: ShortcutDefinition; formatted: string }>
> {
  const grouped: Record<
    string,
    Array<{ name: string; shortcut: ShortcutDefinition; formatted: string }>
  > = {};

  for (const [name, shortcut] of Object.entries(ideShortcuts)) {
    if (!grouped[shortcut.category]) {
      grouped[shortcut.category] = [];
    }
    grouped[shortcut.category].push({
      name,
      shortcut,
      formatted: formatShortcut(shortcut),
    });
  }

  return grouped;
}
