/**
 * NEURECTOMY Monaco Editor Keybindings Configuration
 *
 * Comprehensive keyboard shortcuts for the Spectrum Workspace IDE
 */

import * as monaco from "monaco-editor";

/**
 * Callback interface for keybinding actions
 * All callbacks are optional - if not provided, the action uses Monaco's default behavior
 */
export interface KeybindingCallbacks {
  // File operations
  onSave?: () => void | Promise<void>;
  onSaveAll?: () => void | Promise<void>;

  // Navigation
  onQuickOpen?: () => void;
  onCommandPalette?: () => void;
  onGoToLine?: () => void;
  onGoToDefinition?: () => void;
  onFindReferences?: () => void;

  // Editing
  onQuickFix?: () => void;
  onRenameSymbol?: () => void;
  onTriggerSuggestions?: () => void;

  // Custom actions
  onDeleteLine?: () => void;
  onDuplicateLine?: () => void;
  onMoveLineUp?: () => void;
  onMoveLineDown?: () => void;
  onSelectLine?: () => void;
  onToggleComment?: () => void;

  // Multi-cursor
  onAddCursorAbove?: () => void;
  onAddCursorBelow?: () => void;
}

/**
 * Keybinding definition structure
 */
interface KeybindingDefinition {
  id: string;
  label: string;
  keybinding: number;
  contextMenuGroupId?: string;
  contextMenuOrder?: number;
  precondition?: string;
  handler: (
    editor: monaco.editor.IStandaloneCodeEditor,
    callbacks: KeybindingCallbacks
  ) => void;
}

/**
 * Standard editor keybindings
 */
const standardEditorKeybindings: KeybindingDefinition[] = [
  // Ctrl+S: Save file
  {
    id: "neurectomy.save",
    label: "Save File",
    keybinding: monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS,
    handler: (editor, callbacks) => {
      if (callbacks.onSave) {
        callbacks.onSave();
      }
    },
  },

  // Ctrl+Shift+S: Save all files
  {
    id: "neurectomy.saveAll",
    label: "Save All Files",
    keybinding:
      monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.KeyS,
    handler: (editor, callbacks) => {
      if (callbacks.onSaveAll) {
        callbacks.onSaveAll();
      }
    },
  },

  // Ctrl+Z: Undo (Monaco default, but we register for potential override)
  {
    id: "neurectomy.undo",
    label: "Undo",
    keybinding: monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyZ,
    handler: (editor) => {
      editor.trigger("keyboard", "undo", null);
    },
  },

  // Ctrl+Y: Redo
  {
    id: "neurectomy.redo",
    label: "Redo",
    keybinding: monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyY,
    handler: (editor) => {
      editor.trigger("keyboard", "redo", null);
    },
  },

  // Ctrl+F: Find
  {
    id: "neurectomy.find",
    label: "Find",
    keybinding: monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyF,
    handler: (editor) => {
      editor.trigger("keyboard", "actions.find", null);
    },
  },

  // Ctrl+H: Replace
  {
    id: "neurectomy.replace",
    label: "Find and Replace",
    keybinding: monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyH,
    handler: (editor) => {
      editor.trigger("keyboard", "editor.action.startFindReplaceAction", null);
    },
  },

  // Ctrl+/: Toggle comment
  {
    id: "neurectomy.toggleComment",
    label: "Toggle Line Comment",
    keybinding: monaco.KeyMod.CtrlCmd | monaco.KeyCode.Slash,
    contextMenuGroupId: "modification",
    contextMenuOrder: 1,
    handler: (editor, callbacks) => {
      if (callbacks.onToggleComment) {
        callbacks.onToggleComment();
      } else {
        editor.trigger("keyboard", "editor.action.commentLine", null);
      }
    },
  },

  // Ctrl+Shift+K: Delete line
  {
    id: "neurectomy.deleteLine",
    label: "Delete Line",
    keybinding:
      monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.KeyK,
    contextMenuGroupId: "modification",
    contextMenuOrder: 2,
    handler: (editor, callbacks) => {
      if (callbacks.onDeleteLine) {
        callbacks.onDeleteLine();
      } else {
        editor.trigger("keyboard", "editor.action.deleteLines", null);
      }
    },
  },

  // Alt+Up: Move line up
  {
    id: "neurectomy.moveLineUp",
    label: "Move Line Up",
    keybinding: monaco.KeyMod.Alt | monaco.KeyCode.UpArrow,
    handler: (editor, callbacks) => {
      if (callbacks.onMoveLineUp) {
        callbacks.onMoveLineUp();
      } else {
        editor.trigger("keyboard", "editor.action.moveLinesUpAction", null);
      }
    },
  },

  // Alt+Down: Move line down
  {
    id: "neurectomy.moveLineDown",
    label: "Move Line Down",
    keybinding: monaco.KeyMod.Alt | monaco.KeyCode.DownArrow,
    handler: (editor, callbacks) => {
      if (callbacks.onMoveLineDown) {
        callbacks.onMoveLineDown();
      } else {
        editor.trigger("keyboard", "editor.action.moveLinesDownAction", null);
      }
    },
  },

  // Ctrl+D: Duplicate line
  {
    id: "neurectomy.duplicateLine",
    label: "Duplicate Line",
    keybinding: monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyD,
    contextMenuGroupId: "modification",
    contextMenuOrder: 3,
    handler: (editor, callbacks) => {
      if (callbacks.onDuplicateLine) {
        callbacks.onDuplicateLine();
      } else {
        editor.trigger("keyboard", "editor.action.copyLinesDownAction", null);
      }
    },
  },

  // Ctrl+L: Select line
  {
    id: "neurectomy.selectLine",
    label: "Select Line",
    keybinding: monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyL,
    handler: (editor, callbacks) => {
      if (callbacks.onSelectLine) {
        callbacks.onSelectLine();
      } else {
        editor.trigger("keyboard", "expandLineSelection", null);
      }
    },
  },
];

/**
 * IDE navigation keybindings
 */
const ideKeybindings: KeybindingDefinition[] = [
  // Ctrl+P: Quick open file
  {
    id: "neurectomy.quickOpen",
    label: "Quick Open",
    keybinding: monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyP,
    handler: (editor, callbacks) => {
      if (callbacks.onQuickOpen) {
        callbacks.onQuickOpen();
      }
    },
  },

  // Ctrl+Shift+P: Command palette
  {
    id: "neurectomy.commandPalette",
    label: "Command Palette",
    keybinding:
      monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.KeyP,
    handler: (editor, callbacks) => {
      if (callbacks.onCommandPalette) {
        callbacks.onCommandPalette();
      } else {
        editor.trigger("keyboard", "editor.action.quickCommand", null);
      }
    },
  },

  // Ctrl+G: Go to line
  {
    id: "neurectomy.goToLine",
    label: "Go to Line",
    keybinding: monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyG,
    handler: (editor, callbacks) => {
      if (callbacks.onGoToLine) {
        callbacks.onGoToLine();
      } else {
        editor.trigger("keyboard", "editor.action.gotoLine", null);
      }
    },
  },

  // F12: Go to definition
  {
    id: "neurectomy.goToDefinition",
    label: "Go to Definition",
    keybinding: monaco.KeyCode.F12,
    contextMenuGroupId: "navigation",
    contextMenuOrder: 1,
    handler: (editor, callbacks) => {
      if (callbacks.onGoToDefinition) {
        callbacks.onGoToDefinition();
      } else {
        editor.trigger("keyboard", "editor.action.revealDefinition", null);
      }
    },
  },

  // Shift+F12: Find references
  {
    id: "neurectomy.findReferences",
    label: "Find All References",
    keybinding: monaco.KeyMod.Shift | monaco.KeyCode.F12,
    contextMenuGroupId: "navigation",
    contextMenuOrder: 2,
    handler: (editor, callbacks) => {
      if (callbacks.onFindReferences) {
        callbacks.onFindReferences();
      } else {
        editor.trigger("keyboard", "editor.action.goToReferences", null);
      }
    },
  },

  // Ctrl+Space: Trigger suggestions
  {
    id: "neurectomy.triggerSuggestions",
    label: "Trigger Suggestions",
    keybinding: monaco.KeyMod.CtrlCmd | monaco.KeyCode.Space,
    handler: (editor, callbacks) => {
      if (callbacks.onTriggerSuggestions) {
        callbacks.onTriggerSuggestions();
      } else {
        editor.trigger("keyboard", "editor.action.triggerSuggest", null);
      }
    },
  },

  // Ctrl+.: Quick fix
  {
    id: "neurectomy.quickFix",
    label: "Quick Fix",
    keybinding: monaco.KeyMod.CtrlCmd | monaco.KeyCode.Period,
    contextMenuGroupId: "modification",
    contextMenuOrder: 4,
    handler: (editor, callbacks) => {
      if (callbacks.onQuickFix) {
        callbacks.onQuickFix();
      } else {
        editor.trigger("keyboard", "editor.action.quickFix", null);
      }
    },
  },

  // F2: Rename symbol
  {
    id: "neurectomy.renameSymbol",
    label: "Rename Symbol",
    keybinding: monaco.KeyCode.F2,
    contextMenuGroupId: "modification",
    contextMenuOrder: 5,
    handler: (editor, callbacks) => {
      if (callbacks.onRenameSymbol) {
        callbacks.onRenameSymbol();
      } else {
        editor.trigger("keyboard", "editor.action.rename", null);
      }
    },
  },
];

/**
 * Multi-cursor keybindings
 */
const multiCursorKeybindings: KeybindingDefinition[] = [
  // Ctrl+Alt+Up: Add cursor above
  {
    id: "neurectomy.addCursorAbove",
    label: "Add Cursor Above",
    keybinding:
      monaco.KeyMod.CtrlCmd | monaco.KeyMod.Alt | monaco.KeyCode.UpArrow,
    handler: (editor, callbacks) => {
      if (callbacks.onAddCursorAbove) {
        callbacks.onAddCursorAbove();
      } else {
        editor.trigger("keyboard", "editor.action.insertCursorAbove", null);
      }
    },
  },

  // Ctrl+Alt+Down: Add cursor below
  {
    id: "neurectomy.addCursorBelow",
    label: "Add Cursor Below",
    keybinding:
      monaco.KeyMod.CtrlCmd | monaco.KeyMod.Alt | monaco.KeyCode.DownArrow,
    handler: (editor, callbacks) => {
      if (callbacks.onAddCursorBelow) {
        callbacks.onAddCursorBelow();
      } else {
        editor.trigger("keyboard", "editor.action.insertCursorBelow", null);
      }
    },
  },
];

/**
 * All keybindings combined
 */
const allKeybindings: KeybindingDefinition[] = [
  ...standardEditorKeybindings,
  ...ideKeybindings,
  ...multiCursorKeybindings,
];

/**
 * Disposable interface for cleanup
 */
export interface KeybindingDisposable {
  dispose: () => void;
}

/**
 * Register all NEURECTOMY keybindings on a Monaco editor instance
 *
 * @param editor - The Monaco editor instance
 * @param callbacks - Optional callback handlers for keybinding actions
 * @returns Disposable object to unregister all keybindings
 *
 * @example
 * ```typescript
 * const disposable = registerNeurectomyKeybindings(editor, {
 *   onSave: () => saveCurrentFile(),
 *   onQuickOpen: () => showQuickOpenDialog(),
 *   onCommandPalette: () => showCommandPalette(),
 * });
 *
 * // Later, to cleanup:
 * disposable.dispose();
 * ```
 */
export function registerNeurectomyKeybindings(
  editor: monaco.editor.IStandaloneCodeEditor,
  callbacks: KeybindingCallbacks = {}
): KeybindingDisposable {
  const disposables: monaco.IDisposable[] = [];

  for (const binding of allKeybindings) {
    const action = editor.addAction({
      id: binding.id,
      label: binding.label,
      keybindings: [binding.keybinding],
      contextMenuGroupId: binding.contextMenuGroupId,
      contextMenuOrder: binding.contextMenuOrder,
      precondition: binding.precondition,
      run: () => {
        binding.handler(editor, callbacks);
      },
    });

    disposables.push(action);
  }

  // Enable Ctrl+Click for adding cursors (multi-cursor with mouse)
  // This is configured via editor options, not actions
  editor.updateOptions({
    multiCursorModifier: "ctrlCmd",
  });

  return {
    dispose: () => {
      for (const disposable of disposables) {
        disposable.dispose();
      }
    },
  };
}

/**
 * Get a human-readable keybinding string
 * Useful for displaying shortcuts in UI
 */
export function getKeybindingLabel(keybinding: number): string {
  const parts: string[] = [];

  if (keybinding & monaco.KeyMod.CtrlCmd) {
    parts.push("Ctrl");
  }
  if (keybinding & monaco.KeyMod.Shift) {
    parts.push("Shift");
  }
  if (keybinding & monaco.KeyMod.Alt) {
    parts.push("Alt");
  }

  // Extract the key code (lower 8 bits typically)
  const keyCode = keybinding & 0xff;
  const keyName = monaco.KeyCode[keyCode];

  if (keyName) {
    // Clean up key name (e.g., "KeyS" -> "S", "UpArrow" -> "↑")
    const cleanName = keyName
      .replace(/^Key/, "")
      .replace(/^Digit/, "")
      .replace("UpArrow", "↑")
      .replace("DownArrow", "↓")
      .replace("LeftArrow", "←")
      .replace("RightArrow", "→")
      .replace("Slash", "/")
      .replace("Period", ".");
    parts.push(cleanName);
  }

  return parts.join("+");
}

/**
 * Get all registered keybinding definitions
 * Useful for displaying a shortcuts reference
 */
export function getAllKeybindings(): ReadonlyArray<{
  id: string;
  label: string;
  keybinding: number;
  keybindingLabel: string;
  category: "editor" | "ide" | "multicursor";
}> {
  return [
    ...standardEditorKeybindings.map((k) => ({
      ...k,
      keybindingLabel: getKeybindingLabel(k.keybinding),
      category: "editor" as const,
    })),
    ...ideKeybindings.map((k) => ({
      ...k,
      keybindingLabel: getKeybindingLabel(k.keybinding),
      category: "ide" as const,
    })),
    ...multiCursorKeybindings.map((k) => ({
      ...k,
      keybindingLabel: getKeybindingLabel(k.keybinding),
      category: "multicursor" as const,
    })),
  ];
}

/**
 * Create a keybinding reference map for quick lookup
 */
export function createKeybindingMap(): Map<string, string> {
  const map = new Map<string, string>();

  for (const binding of allKeybindings) {
    map.set(binding.id, getKeybindingLabel(binding.keybinding));
  }

  return map;
}

export default registerNeurectomyKeybindings;
