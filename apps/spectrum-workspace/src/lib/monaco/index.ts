/**
 * Monaco Editor Initialization Module for NEURECTOMY
 * Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
 *
 * Central module for Monaco Editor configuration and initialization.
 */

import * as monaco from "monaco-editor";
import { registerAllThemes, applyTheme, THEME_NAMES } from "./themes";
import {
  configureAllLanguages,
  getLanguageFromPath,
  getLanguageIcon,
  getLanguageName,
} from "./languages";
import {
  registerNeurectomyKeybindings,
  type KeybindingCallbacks,
} from "./keybindings";

export type { KeybindingCallbacks } from "./keybindings";

/**
 * Monaco initialization status
 */
let isInitialized = false;
let initializationPromise: Promise<boolean> | null = null;

/**
 * Default editor configuration
 */
export const DEFAULT_EDITOR_OPTIONS: monaco.editor.IStandaloneEditorConstructionOptions =
  {
    theme: THEME_NAMES.NEURECTOMY_DARK,
    fontSize: 14,
    fontFamily:
      "'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace",
    fontLigatures: true,
    tabSize: 2,
    insertSpaces: true,
    lineNumbers: "on",
    wordWrap: "on",
    minimap: { enabled: true, scale: 1 },
    scrollBeyondLastLine: false,
    smoothScrolling: true,
    cursorBlinking: "smooth",
    cursorSmoothCaretAnimation: "on",
    padding: { top: 16, bottom: 16 },
    bracketPairColorization: { enabled: true },
    guides: {
      bracketPairs: true,
      indentation: true,
      highlightActiveIndentation: true,
    },
    suggest: {
      preview: true,
      showKeywords: true,
      showSnippets: true,
      showClasses: true,
      showFunctions: true,
      showVariables: true,
      showConstants: true,
      showInterfaces: true,
    },
    quickSuggestions: {
      other: true,
      comments: false,
      strings: true,
    },
    parameterHints: { enabled: true },
    folding: true,
    foldingStrategy: "indentation",
    showFoldingControls: "mouseover",
    glyphMargin: true,
    lightbulb: { enabled: monaco.editor.ShowLightbulbIconMode.On },
    automaticLayout: true,
    renderWhitespace: "selection",
    renderControlCharacters: false,
    renderLineHighlight: "all",
    overviewRulerBorder: false,
    hideCursorInOverviewRuler: false,
    scrollbar: {
      vertical: "auto",
      horizontal: "auto",
      verticalScrollbarSize: 12,
      horizontalScrollbarSize: 12,
      useShadows: false,
    },
    stickyScroll: { enabled: true },
    inlineSuggest: { enabled: true },
  };

/**
 * Initialize Monaco Editor with NEURECTOMY configuration
 *
 * @returns Promise<boolean> - True if initialization was successful
 */
export async function initializeMonaco(): Promise<boolean> {
  // Return cached promise if already initializing/initialized
  if (initializationPromise) {
    return initializationPromise;
  }

  initializationPromise = (async () => {
    try {
      if (isInitialized) {
        return true;
      }

      console.log("[Monaco] Initializing NEURECTOMY Monaco Editor...");

      // Register themes
      registerAllThemes(monaco);
      console.log("[Monaco] Themes registered");

      // Apply default theme
      applyTheme(monaco, THEME_NAMES.NEURECTOMY_DARK);
      console.log("[Monaco] Default theme applied");

      // Configure languages
      const disposables = configureAllLanguages(monaco);
      console.log(
        `[Monaco] Languages configured (${disposables.length} providers registered)`
      );

      // Set global editor defaults
      monaco.editor.EditorOptions.fontSize.defaultValue = 14;

      isInitialized = true;
      console.log("[Monaco] Initialization complete");

      return true;
    } catch (error) {
      console.error("[Monaco] Initialization failed:", error);
      initializationPromise = null;
      return false;
    }
  })();

  return initializationPromise;
}

/**
 * Check if Monaco is initialized
 */
export function isMonacoInitialized(): boolean {
  return isInitialized;
}

/**
 * Create a Monaco model URI from file path
 */
export function createModelUri(filePath: string): monaco.Uri {
  // Normalize path separators
  const normalizedPath = filePath.replace(/\\/g, "/");
  return monaco.Uri.file(normalizedPath);
}

/**
 * Get or create a Monaco model for a file
 */
export function getOrCreateModel(
  filePath: string,
  content: string,
  language?: string
): monaco.editor.ITextModel {
  const uri = createModelUri(filePath);
  const detectedLanguage = language || getLanguageFromPath(filePath);

  // Check if model already exists
  let model = monaco.editor.getModel(uri);

  if (model) {
    // Update content if different
    if (model.getValue() !== content) {
      model.setValue(content);
    }
    // Update language if different
    if (model.getLanguageId() !== detectedLanguage) {
      monaco.editor.setModelLanguage(model, detectedLanguage);
    }
  } else {
    // Create new model
    model = monaco.editor.createModel(content, detectedLanguage, uri);
  }

  return model;
}

/**
 * Dispose a Monaco model for a file
 */
export function disposeModel(filePath: string): void {
  const uri = createModelUri(filePath);
  const model = monaco.editor.getModel(uri);
  if (model) {
    model.dispose();
  }
}

/**
 * Get all open Monaco models
 */
export function getAllModels(): monaco.editor.ITextModel[] {
  return monaco.editor.getModels();
}

/**
 * Create an editor instance with NEURECTOMY defaults
 */
export function createEditor(
  container: HTMLElement,
  options?: Partial<monaco.editor.IStandaloneEditorConstructionOptions>
): monaco.editor.IStandaloneCodeEditor {
  return monaco.editor.create(container, {
    ...DEFAULT_EDITOR_OPTIONS,
    ...options,
  });
}

/**
 * Create a diff editor instance
 */
export function createDiffEditor(
  container: HTMLElement,
  options?: Partial<monaco.editor.IStandaloneDiffEditorConstructionOptions>
): monaco.editor.IStandaloneDiffEditor {
  return monaco.editor.createDiffEditor(container, {
    ...options,
    automaticLayout: true,
    renderSideBySide: true,
  });
}

// Re-export everything for convenience
export { monaco };
export { registerAllThemes, applyTheme, THEME_NAMES } from "./themes";
export {
  configureAllLanguages,
  getLanguageFromPath,
  getLanguageIcon,
  getLanguageName,
  LANGUAGE_EXTENSIONS,
  LANGUAGE_ICONS,
  LANGUAGE_NAMES,
} from "./languages";
export {
  registerNeurectomyKeybindings,
  getKeybindingLabel,
  getAllKeybindings,
} from "./keybindings";
