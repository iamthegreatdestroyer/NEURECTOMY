/**
 * Keybindings Manager
 *
 * VS Code-style keyboard shortcut management system.
 * Supports customizable keybindings, contexts, and command execution.
 *
 * Features:
 * - Default keybindings
 * - User customization
 * - Context-aware bindings (when clauses)
 * - Chord shortcuts (e.g., Ctrl+K Ctrl+C)
 * - Platform-specific bindings
 * - Conflict detection
 *
 * @module @neurectomy/components/shell/Keybindings
 * @author @APEX @ARCHITECT
 */

import React, {
  createContext,
  useContext,
  useCallback,
  useEffect,
  useState,
  useMemo,
} from "react";

// ============================================================================
// Types
// ============================================================================

export interface Keybinding {
  key: string;
  command: string;
  when?: string;
  args?: Record<string, unknown>;
}

export interface KeybindingContext {
  [key: string]: boolean | string | number;
}

export interface KeybindingsState {
  /** Execute a command by ID */
  executeCommand: (commandId: string, args?: Record<string, unknown>) => void;
  /** Register a command handler */
  registerCommand: (commandId: string, handler: CommandHandler) => () => void;
  /** Get keybinding for a command */
  getKeybinding: (commandId: string) => Keybinding | undefined;
  /** Get all keybindings */
  getAllKeybindings: () => Keybinding[];
  /** Set context value */
  setContext: (key: string, value: boolean | string | number) => void;
  /** Get context value */
  getContext: (key: string) => boolean | string | number | undefined;
  /** Add custom keybindings */
  addKeybindings: (bindings: Keybinding[]) => void;
  /** Remove keybinding */
  removeKeybinding: (key: string, command: string) => void;
  /** Format keybinding for display */
  formatKeybinding: (key: string) => string;
}

export type CommandHandler = (
  args?: Record<string, unknown>
) => void | Promise<void>;

// ============================================================================
// Default Keybindings
// ============================================================================

export const DEFAULT_KEYBINDINGS: Keybinding[] = [
  // File operations
  { key: "ctrl+n", command: "file.new" },
  { key: "ctrl+o", command: "file.open" },
  { key: "ctrl+s", command: "file.save" },
  { key: "ctrl+shift+s", command: "file.saveAs" },
  { key: "ctrl+w", command: "file.close" },
  { key: "ctrl+shift+t", command: "file.reopenClosed" },

  // Edit operations
  { key: "ctrl+z", command: "edit.undo" },
  { key: "ctrl+y", command: "edit.redo" },
  { key: "ctrl+shift+z", command: "edit.redo" },
  { key: "ctrl+x", command: "edit.cut" },
  { key: "ctrl+c", command: "edit.copy" },
  { key: "ctrl+v", command: "edit.paste" },
  { key: "ctrl+a", command: "edit.selectAll" },
  { key: "ctrl+d", command: "edit.duplicateLine" },
  { key: "ctrl+shift+k", command: "edit.deleteLine" },
  { key: "alt+up", command: "edit.moveLineUp" },
  { key: "alt+down", command: "edit.moveLineDown" },
  { key: "ctrl+/", command: "edit.toggleComment" },
  { key: "ctrl+shift+/", command: "edit.toggleBlockComment" },

  // Find and replace
  { key: "ctrl+f", command: "find.open" },
  { key: "ctrl+h", command: "replace.open" },
  { key: "ctrl+shift+f", command: "search.openGlobal" },
  { key: "ctrl+shift+h", command: "replace.openGlobal" },
  { key: "f3", command: "find.next" },
  { key: "shift+f3", command: "find.previous" },
  { key: "ctrl+g", command: "go.toLine" },

  // View operations
  { key: "ctrl+shift+p", command: "commandPalette.open" },
  { key: "ctrl+p", command: "quickOpen.open" },
  { key: "ctrl+shift+e", command: "view.explorer" },
  { key: "ctrl+shift+g", command: "view.sourceControl" },
  { key: "ctrl+shift+d", command: "view.debug" },
  { key: "ctrl+shift+x", command: "view.extensions" },
  { key: "ctrl+b", command: "view.toggleSidebar" },
  { key: "ctrl+j", command: "view.togglePanel" },
  { key: "ctrl+`", command: "terminal.toggle" },
  { key: "ctrl+shift+`", command: "terminal.new" },
  { key: "f11", command: "view.toggleFullscreen" },
  { key: "ctrl+shift+m", command: "view.problems" },
  { key: "ctrl+shift+u", command: "view.output" },

  // Editor operations
  { key: "ctrl+\\", command: "editor.split" },
  { key: "ctrl+1", command: "editor.focusFirst" },
  { key: "ctrl+2", command: "editor.focusSecond" },
  { key: "ctrl+3", command: "editor.focusThird" },
  { key: "ctrl+tab", command: "editor.nextTab" },
  { key: "ctrl+shift+tab", command: "editor.previousTab" },
  { key: "ctrl+pageup", command: "editor.previousTab" },
  { key: "ctrl+pagedown", command: "editor.nextTab" },

  // Code intelligence
  {
    key: "ctrl+space",
    command: "intellisense.trigger",
    when: "editorTextFocus",
  },
  { key: "f12", command: "go.toDefinition", when: "editorTextFocus" },
  { key: "alt+f12", command: "go.peekDefinition", when: "editorTextFocus" },
  { key: "shift+f12", command: "go.toReferences", when: "editorTextFocus" },
  { key: "f2", command: "editor.rename", when: "editorTextFocus" },
  { key: "ctrl+.", command: "editor.quickFix", when: "editorTextFocus" },
  { key: "ctrl+shift+o", command: "go.toSymbol", when: "editorTextFocus" },
  { key: "ctrl+shift+\\", command: "go.toBracket", when: "editorTextFocus" },

  // Folding
  { key: "ctrl+shift+[", command: "editor.fold", when: "editorTextFocus" },
  { key: "ctrl+shift+]", command: "editor.unfold", when: "editorTextFocus" },
  { key: "ctrl+k ctrl+0", command: "editor.foldAll", when: "editorTextFocus" },
  {
    key: "ctrl+k ctrl+j",
    command: "editor.unfoldAll",
    when: "editorTextFocus",
  },

  // Debug
  { key: "f5", command: "debug.start" },
  { key: "ctrl+f5", command: "debug.runWithoutDebugging" },
  { key: "shift+f5", command: "debug.stop" },
  { key: "f9", command: "debug.toggleBreakpoint" },
  { key: "f10", command: "debug.stepOver" },
  { key: "f11", command: "debug.stepInto" },
  { key: "shift+f11", command: "debug.stepOut" },

  // AI features
  { key: "ctrl+i", command: "ai.inlineChat", when: "editorTextFocus" },
  { key: "ctrl+shift+i", command: "ai.openChat" },
  { key: "ctrl+l", command: "ai.explain", when: "editorHasSelection" },

  // Git
  { key: "ctrl+shift+g g", command: "git.openChanges" },
  { key: "ctrl+enter", command: "git.commit", when: "scmInputFocus" },
];

// ============================================================================
// Context
// ============================================================================

const KeybindingsContext = createContext<KeybindingsState | null>(null);

export function useKeybindings() {
  const context = useContext(KeybindingsContext);
  if (!context) {
    throw new Error("useKeybindings must be used within KeybindingsProvider");
  }
  return context;
}

// ============================================================================
// Provider
// ============================================================================

interface KeybindingsProviderProps {
  children: React.ReactNode;
  customBindings?: Keybinding[];
}

export function KeybindingsProvider({
  children,
  customBindings = [],
}: KeybindingsProviderProps) {
  const [bindings, setBindings] = useState<Keybinding[]>([
    ...DEFAULT_KEYBINDINGS,
    ...customBindings,
  ]);
  const [commands, setCommands] = useState<Map<string, CommandHandler>>(
    new Map()
  );
  const [context, setContextState] = useState<KeybindingContext>({});
  const [pendingChord, setPendingChord] = useState<string | null>(null);
  const [chordTimeout, setChordTimeout] = useState<NodeJS.Timeout | null>(null);

  // Platform detection
  const isMac = useMemo(() => {
    return typeof navigator !== "undefined" && /Mac/.test(navigator.platform);
  }, []);

  // Build keybinding lookup
  const keybindingMap = useMemo(() => {
    const map = new Map<string, Keybinding[]>();
    bindings.forEach((binding) => {
      const key = normalizeKey(binding.key, isMac);
      const existing = map.get(key) || [];
      existing.push(binding);
      map.set(key, existing);
    });
    return map;
  }, [bindings, isMac]);

  // Evaluate when clause
  const evaluateWhen = useCallback(
    (when: string | undefined): boolean => {
      if (!when) return true;

      // Simple expression evaluator
      const tokens = when.split(/\s*(&&|\|\|)\s*/);
      let result = true;
      let operator: "&&" | "||" | null = null;

      for (const token of tokens) {
        if (token === "&&" || token === "||") {
          operator = token as "&&" | "||";
          continue;
        }

        let value: boolean;
        if (token.startsWith("!")) {
          const key = token.slice(1);
          value = !context[key];
        } else if (token.includes("==")) {
          const [key, expected] = token.split("==").map((s) => s.trim());
          value = context[key] === expected;
        } else if (token.includes("!=")) {
          const [key, expected] = token.split("!=").map((s) => s.trim());
          value = context[key] !== expected;
        } else {
          value = !!context[token];
        }

        if (operator === "||") {
          result = result || value;
        } else {
          result = result && value;
        }
      }

      return result;
    },
    [context]
  );

  // Execute command
  const executeCommand = useCallback(
    (commandId: string, args?: Record<string, unknown>) => {
      const handler = commands.get(commandId);
      if (handler) {
        handler(args);
      } else {
        console.warn(`No handler registered for command: ${commandId}`);
      }
    },
    [commands]
  );

  // Register command
  const registerCommand = useCallback(
    (commandId: string, handler: CommandHandler) => {
      setCommands((prev) => {
        const next = new Map(prev);
        next.set(commandId, handler);
        return next;
      });

      return () => {
        setCommands((prev) => {
          const next = new Map(prev);
          next.delete(commandId);
          return next;
        });
      };
    },
    []
  );

  // Get keybinding for command
  const getKeybinding = useCallback(
    (commandId: string): Keybinding | undefined => {
      return bindings.find((b) => b.command === commandId);
    },
    [bindings]
  );

  // Get all keybindings
  const getAllKeybindings = useCallback(() => [...bindings], [bindings]);

  // Set context
  const setContext = useCallback(
    (key: string, value: boolean | string | number) => {
      setContextState((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  // Get context
  const getContext = useCallback((key: string) => context[key], [context]);

  // Add keybindings
  const addKeybindings = useCallback((newBindings: Keybinding[]) => {
    setBindings((prev) => [...prev, ...newBindings]);
  }, []);

  // Remove keybinding
  const removeKeybinding = useCallback((key: string, command: string) => {
    setBindings((prev) =>
      prev.filter((b) => !(b.key === key && b.command === command))
    );
  }, []);

  // Format keybinding for display
  const formatKeybinding = useCallback(
    (key: string): string => {
      return key
        .split("+")
        .map((part) => {
          const normalized = part.toLowerCase().trim();
          if (isMac) {
            switch (normalized) {
              case "ctrl":
                return "⌃";
              case "alt":
                return "⌥";
              case "shift":
                return "⇧";
              case "meta":
              case "cmd":
                return "⌘";
              default:
                return part.toUpperCase();
            }
          } else {
            switch (normalized) {
              case "ctrl":
                return "Ctrl";
              case "alt":
                return "Alt";
              case "shift":
                return "Shift";
              case "meta":
                return "Win";
              default:
                return part.toUpperCase();
            }
          }
        })
        .join(isMac ? "" : "+");
    },
    [isMac]
  );

  // Handle keyboard events
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Build key string
      const parts: string[] = [];
      if (e.ctrlKey || e.metaKey) parts.push("ctrl");
      if (e.altKey) parts.push("alt");
      if (e.shiftKey) parts.push("shift");

      const key = e.key.toLowerCase();
      if (!["control", "alt", "shift", "meta"].includes(key)) {
        parts.push(key);
      }

      if (
        parts.length === 0 ||
        (parts.length === 1 &&
          ["control", "alt", "shift", "meta"].includes(parts[0]))
      ) {
        return;
      }

      const keyString = parts.join("+");
      const fullKey = pendingChord ? `${pendingChord} ${keyString}` : keyString;

      // Look up binding
      const matchingBindings =
        keybindingMap.get(normalizeKey(fullKey, isMac)) || [];

      // Check for chord continuation
      const chordBindings = [...keybindingMap.entries()].filter(([k]) =>
        k.startsWith(`${keyString} `)
      );

      if (chordBindings.length > 0 && !pendingChord) {
        e.preventDefault();
        setPendingChord(keyString);

        // Clear after timeout
        if (chordTimeout) clearTimeout(chordTimeout);
        const timeout = setTimeout(() => {
          setPendingChord(null);
        }, 1500);
        setChordTimeout(timeout);
        return;
      }

      // Execute matching binding
      for (const binding of matchingBindings) {
        if (evaluateWhen(binding.when)) {
          e.preventDefault();
          executeCommand(binding.command, binding.args);
          setPendingChord(null);
          if (chordTimeout) clearTimeout(chordTimeout);
          return;
        }
      }

      // Clear pending chord if no match
      if (pendingChord) {
        setPendingChord(null);
        if (chordTimeout) clearTimeout(chordTimeout);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    keybindingMap,
    pendingChord,
    chordTimeout,
    isMac,
    evaluateWhen,
    executeCommand,
  ]);

  const value: KeybindingsState = {
    executeCommand,
    registerCommand,
    getKeybinding,
    getAllKeybindings,
    setContext,
    getContext,
    addKeybindings,
    removeKeybinding,
    formatKeybinding,
  };

  return (
    <KeybindingsContext.Provider value={value}>
      {children}
      {/* Chord indicator */}
      {pendingChord && (
        <div className="fixed bottom-20 left-1/2 -translate-x-1/2 z-50 px-4 py-2 bg-card border border-border rounded-lg shadow-lg text-sm">
          Waiting for second key... ({formatKeybinding(pendingChord)})
        </div>
      )}
    </KeybindingsContext.Provider>
  );
}

// ============================================================================
// Utilities
// ============================================================================

function normalizeKey(key: string, isMac: boolean): string {
  return key
    .toLowerCase()
    .replace(/\s+/g, " ")
    .split(" ")
    .map((part) => {
      const keys = part.split("+").map((k) => {
        const normalized = k.trim().toLowerCase();
        // Normalize cmd/meta to ctrl on non-Mac
        if (!isMac && (normalized === "cmd" || normalized === "meta")) {
          return "ctrl";
        }
        return normalized;
      });
      return keys.sort().join("+");
    })
    .join(" ");
}

// ============================================================================
// Hook for Command Registration
// ============================================================================

export function useCommand(
  commandId: string,
  handler: CommandHandler,
  deps: unknown[] = []
) {
  const { registerCommand } = useKeybindings();

  useEffect(() => {
    return registerCommand(commandId, handler);
  }, [commandId, registerCommand, ...deps]);
}

export default KeybindingsProvider;
