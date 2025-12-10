/**
 * Editor Types for NEURECTOMY Monaco Editor Integration
 * Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
 */

import type * as Monaco from "monaco-editor";

/**
 * Represents a file open in the editor
 */
export interface EditorFile {
  /** Unique identifier for the file */
  id: string;
  /** File path relative to workspace */
  path: string;
  /** File name with extension */
  name: string;
  /** File content */
  content: string;
  /** Programming language */
  language: string;
  /** Whether file has unsaved changes */
  isDirty: boolean;
  /** Whether file is read-only */
  isReadOnly?: boolean;
  /** Last modified timestamp */
  lastModified: number;
  /** Monaco editor model */
  model?: Monaco.editor.ITextModel;
  /** Editor view state for cursor/scroll position */
  viewState?: Monaco.editor.ICodeEditorViewState | null;
}

/**
 * Editor configuration options
 */
export interface EditorConfig {
  /** Theme name */
  theme: "neurectomy-dark" | "vs-dark" | "vs-light" | "hc-black";
  /** Font size in pixels */
  fontSize: number;
  /** Font family */
  fontFamily: string;
  /** Tab size in spaces */
  tabSize: number;
  /** Insert spaces instead of tabs */
  insertSpaces: boolean;
  /** Show line numbers */
  lineNumbers: "on" | "off" | "relative";
  /** Enable word wrap */
  wordWrap: "on" | "off" | "wordWrapColumn" | "bounded";
  /** Enable minimap */
  minimap: boolean;
  /** Auto-save delay in milliseconds (0 to disable) */
  autoSaveDelay: number;
  /** Format on save */
  formatOnSave: boolean;
  /** Format on paste */
  formatOnPaste: boolean;
  /** Enable IntelliSense */
  quickSuggestions: boolean;
  /** Render whitespace */
  renderWhitespace: "none" | "boundary" | "selection" | "all";
}

/**
 * Editor store state
 */
export interface EditorState {
  /** Currently open files */
  openFiles: EditorFile[];
  /** Active file ID */
  activeFileId: string | null;
  /** Editor configuration */
  config: EditorConfig;
  /** Monaco editor instance */
  editor: Monaco.editor.IStandaloneCodeEditor | null;
}

/**
 * Editor store actions
 */
export interface EditorActions {
  /** Open a file in the editor */
  openFile: (file: Omit<EditorFile, "id" | "isDirty" | "lastModified">) => void;
  /** Close a file */
  closeFile: (fileId: string) => void;
  /** Close all files */
  closeAllFiles: () => void;
  /** Set active file */
  setActiveFile: (fileId: string) => void;
  /** Update file content */
  updateFileContent: (fileId: string, content: string) => void;
  /** Mark file as dirty */
  markFileDirty: (fileId: string, isDirty: boolean) => void;
  /** Save file */
  saveFile: (fileId: string) => Promise<void>;
  /** Save all files */
  saveAllFiles: () => Promise<void>;
  /** Update editor configuration */
  updateConfig: (config: Partial<EditorConfig>) => void;
  /** Set Monaco editor instance */
  setEditor: (editor: Monaco.editor.IStandaloneCodeEditor | null) => void;
  /** Attach a Monaco model to a file */
  setFileModel: (fileId: string, model: Monaco.editor.ITextModel) => void;
  /** Update file view state */
  updateViewState: (
    fileId: string,
    viewState: Monaco.editor.ICodeEditorViewState | null
  ) => void;
  /** Get file by ID */
  getFile: (fileId: string) => EditorFile | undefined;
  /** Get active file */
  getActiveFile: () => EditorFile | undefined;
}

/**
 * Combined editor store
 */
export type EditorStore = EditorState & EditorActions;

/**
 * Language configuration for Monaco
 */
export interface LanguageConfig {
  /** Language identifier */
  id: string;
  /** File extensions */
  extensions: string[];
  /** Language aliases */
  aliases: string[];
  /** MIME types */
  mimeTypes?: string[];
}

/**
 * Editor theme definition
 */
export interface EditorTheme {
  /** Theme name */
  name: string;
  /** Base theme */
  base: "vs" | "vs-dark" | "hc-black";
  /** Inherit from base theme */
  inherit: boolean;
  /** Theme rules */
  rules: Monaco.editor.ITokenThemeRule[];
  /** Editor colors */
  colors: { [key: string]: string };
}

/**
 * File save event
 */
export interface FileSaveEvent {
  /** File ID */
  fileId: string;
  /** File path */
  path: string;
  /** File content */
  content: string;
  /** Timestamp */
  timestamp: number;
}

/**
 * Editor event types
 */
export type EditorEvent =
  | { type: "file:opened"; file: EditorFile }
  | { type: "file:closed"; fileId: string }
  | { type: "file:saved"; event: FileSaveEvent }
  | { type: "file:dirty"; fileId: string; isDirty: boolean }
  | { type: "editor:ready"; editor: Monaco.editor.IStandaloneCodeEditor }
  | { type: "config:updated"; config: Partial<EditorConfig> };

/**
 * Default editor configuration
 */
export const DEFAULT_EDITOR_CONFIG: EditorConfig = {
  theme: "neurectomy-dark",
  fontSize: 14,
  fontFamily: "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
  tabSize: 2,
  insertSpaces: true,
  lineNumbers: "on",
  wordWrap: "off",
  minimap: true,
  autoSaveDelay: 1000, // 1 second
  formatOnSave: true,
  formatOnPaste: true,
  quickSuggestions: true,
  renderWhitespace: "selection",
};

/**
 * Language detection utilities
 */
export const LANGUAGE_MAP: Record<string, LanguageConfig> = {
  typescript: {
    id: "typescript",
    extensions: [".ts", ".tsx"],
    aliases: ["TypeScript", "ts", "typescript"],
    mimeTypes: ["text/typescript"],
  },
  javascript: {
    id: "javascript",
    extensions: [".js", ".jsx", ".mjs", ".cjs"],
    aliases: ["JavaScript", "js", "javascript"],
    mimeTypes: ["text/javascript"],
  },
  python: {
    id: "python",
    extensions: [".py", ".pyw", ".pyi"],
    aliases: ["Python", "py", "python"],
    mimeTypes: ["text/x-python"],
  },
  rust: {
    id: "rust",
    extensions: [".rs"],
    aliases: ["Rust", "rs", "rust"],
    mimeTypes: ["text/x-rust"],
  },
  json: {
    id: "json",
    extensions: [".json", ".jsonc"],
    aliases: ["JSON", "json"],
    mimeTypes: ["application/json"],
  },
  yaml: {
    id: "yaml",
    extensions: [".yaml", ".yml"],
    aliases: ["YAML", "yaml"],
    mimeTypes: ["text/x-yaml"],
  },
  markdown: {
    id: "markdown",
    extensions: [".md", ".markdown"],
    aliases: ["Markdown", "md", "markdown"],
    mimeTypes: ["text/markdown"],
  },
  html: {
    id: "html",
    extensions: [".html", ".htm"],
    aliases: ["HTML", "html"],
    mimeTypes: ["text/html"],
  },
  css: {
    id: "css",
    extensions: [".css"],
    aliases: ["CSS", "css"],
    mimeTypes: ["text/css"],
  },
  dockerfile: {
    id: "dockerfile",
    extensions: [".dockerfile", "Dockerfile"],
    aliases: ["Dockerfile", "dockerfile"],
  },
};

/**
 * Detect language from file extension
 */
export function detectLanguage(filename: string): string {
  const ext = filename.substring(filename.lastIndexOf(".")).toLowerCase();

  for (const [langId, config] of Object.entries(LANGUAGE_MAP)) {
    if (config.extensions.some((e) => e.toLowerCase() === ext)) {
      return langId;
    }
  }

  // Check for files without extensions
  const basename = filename.substring(filename.lastIndexOf("/") + 1);
  for (const [langId, config] of Object.entries(LANGUAGE_MAP)) {
    if (config.extensions.includes(basename)) {
      return langId;
    }
  }

  return "plaintext";
}

/**
 * Generate unique file ID
 */
export function generateFileId(): string {
  return `file-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}
