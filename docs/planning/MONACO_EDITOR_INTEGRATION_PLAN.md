# Monaco Editor Integration - Action Plan

**Project:** NEURECTOMY IDE  
**Priority:** HIGH - Critical IDE Functionality  
**Estimated Effort:** 2-3 weeks  
**Target Completion:** Phase 5, Weeks 1-3  
**Status:** Not Started (0%)

---

## Executive Summary

Monaco Editor is the **core code editing component** that powers VS Code. Integrating it into NEURECTOMY is critical for providing a professional IDE experience. This plan outlines a comprehensive implementation strategy covering component architecture, language services, theming, file system integration, and advanced features.

**Dependencies Already Installed:**

- ✅ `@monaco-editor/react: ^4.6.0` (confirmed in package.json)
- ✅ React 19, TypeScript 5.5
- ✅ Zustand state management
- ✅ Tauri IPC for file system operations

---

## 📋 Implementation Phases

### Phase 1: Core Monaco Component (Week 1, Days 1-3)

#### 1.1 Create Base Monaco Wrapper Component

**File:** `packages/ui/src/components/code-editor/MonacoEditor.tsx`

```typescript
import { useEffect, useRef, useState } from 'react';
import Editor, {
  Monaco,
  OnMount,
  OnChange
} from '@monaco-editor/react';
import * as monaco from 'monaco-editor';

export interface MonacoEditorProps {
  /** File content to display */
  value: string;
  /** Programming language (typescript, python, rust, etc.) */
  language: string;
  /** Theme (neurectomy-dark, neurectomy-light) */
  theme?: string;
  /** Read-only mode */
  readOnly?: boolean;
  /** Editor options */
  options?: monaco.editor.IStandaloneEditorConstructionOptions;
  /** Callback when content changes */
  onChange?: (value: string | undefined) => void;
  /** Callback when editor is mounted */
  onMount?: OnMount;
  /** Show line numbers */
  lineNumbers?: 'on' | 'off' | 'relative' | 'interval';
  /** Enable minimap */
  minimap?: boolean;
  /** Word wrap setting */
  wordWrap?: 'off' | 'on' | 'wordWrapColumn' | 'bounded';
  /** Tab size */
  tabSize?: number;
  /** Auto-format on save */
  formatOnSave?: boolean;
  /** File path for context */
  filePath?: string;
}

export function MonacoEditor({
  value,
  language,
  theme = 'neurectomy-dark',
  readOnly = false,
  options = {},
  onChange,
  onMount,
  lineNumbers = 'on',
  minimap = true,
  wordWrap = 'off',
  tabSize = 2,
  formatOnSave = true,
  filePath,
}: MonacoEditorProps) {
  const editorRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<Monaco | null>(null);
  const [isReady, setIsReady] = useState(false);

  // Handle editor mount
  const handleEditorDidMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;
    setIsReady(true);

    // Setup custom key bindings
    editor.addCommand(
      monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS,
      () => {
        if (formatOnSave) {
          editor.getAction('editor.action.formatDocument')?.run();
        }
        // Trigger save event
        window.dispatchEvent(new CustomEvent('editor:save', {
          detail: { filePath, value: editor.getValue() }
        }));
      }
    );

    // Call user's onMount callback
    onMount?.(editor, monaco);
  };

  // Debounced onChange handler
  const handleChange: OnChange = (value) => {
    onChange?.(value);
  };

  // Configure editor options
  const editorOptions: monaco.editor.IStandaloneEditorConstructionOptions = {
    readOnly,
    lineNumbers,
    minimap: { enabled: minimap },
    wordWrap,
    tabSize,
    fontSize: 14,
    fontFamily: "'Fira Code', 'JetBrains Mono', monospace",
    fontLigatures: true,
    scrollBeyondLastLine: false,
    automaticLayout: true,
    smoothScrolling: true,
    cursorBlinking: 'smooth',
    cursorSmoothCaretAnimation: 'on',
    quickSuggestions: true,
    suggestOnTriggerCharacters: true,
    acceptSuggestionOnCommitCharacter: true,
    snippetSuggestions: 'inline',
    ...options,
  };

  return (
    <div className="monaco-editor-wrapper h-full w-full">
      <Editor
        height="100%"
        language={language}
        value={value}
        theme={theme}
        options={editorOptions}
        onChange={handleChange}
        onMount={handleEditorDidMount}
        loading={<EditorSkeleton />}
      />
    </div>
  );
}

// Loading skeleton while Monaco initializes
function EditorSkeleton() {
  return (
    <div className="flex h-full w-full items-center justify-center bg-[#0a0a0f]">
      <div className="flex flex-col items-center gap-4">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-indigo-500 border-t-transparent" />
        <p className="text-sm text-gray-400">Loading editor...</p>
      </div>
    </div>
  );
}
```

**Key Features:**

- ✅ Wrapper around @monaco-editor/react
- ✅ Custom key bindings (Ctrl/Cmd+S for save)
- ✅ Configurable options (line numbers, minimap, word wrap)
- ✅ Font ligatures support (Fira Code, JetBrains Mono)
- ✅ Loading skeleton during initialization
- ✅ Type-safe props with TypeScript

#### 1.2 Create Editor Manager Component

**File:** `packages/ui/src/components/code-editor/EditorManager.tsx`

```typescript
import { MonacoEditor } from './MonacoEditor';
import { useEditorStore } from '@neurectomy/stores';

/**
 * Multi-file editor manager with tab system
 * Manages multiple open files and active editor state
 */
export function EditorManager() {
  const {
    openFiles,
    activeFileId,
    getFileContent,
    updateFileContent,
    closeFile,
    setActiveFile,
  } = useEditorStore();

  const activeFile = openFiles.find(f => f.id === activeFileId);

  if (!activeFile) {
    return <EmptyEditorState />;
  }

  return (
    <div className="flex h-full flex-col">
      {/* Tab bar */}
      <EditorTabs
        files={openFiles}
        activeId={activeFileId}
        onSelect={setActiveFile}
        onClose={closeFile}
      />

      {/* Active editor */}
      <div className="flex-1 overflow-hidden">
        <MonacoEditor
          key={activeFile.id}
          value={getFileContent(activeFile.id)}
          language={detectLanguage(activeFile.path)}
          filePath={activeFile.path}
          onChange={(value) => updateFileContent(activeFile.id, value || '')}
          formatOnSave={true}
        />
      </div>

      {/* Editor status bar */}
      <EditorStatusBar
        filePath={activeFile.path}
        language={detectLanguage(activeFile.path)}
        line={activeFile.cursorLine}
        column={activeFile.cursorColumn}
        modified={activeFile.modified}
      />
    </div>
  );
}

function EmptyEditorState() {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-4 text-gray-400">
      <svg className="h-24 w-24" /* ... icon ... */ />
      <p className="text-lg">No file open</p>
      <p className="text-sm">Open a file to start editing</p>
    </div>
  );
}

function detectLanguage(filePath: string): string {
  const ext = filePath.split('.').pop()?.toLowerCase();
  const languageMap: Record<string, string> = {
    ts: 'typescript',
    tsx: 'typescript',
    js: 'javascript',
    jsx: 'javascript',
    py: 'python',
    rs: 'rust',
    go: 'go',
    java: 'java',
    cpp: 'cpp',
    c: 'c',
    md: 'markdown',
    json: 'json',
    yaml: 'yaml',
    yml: 'yaml',
    toml: 'toml',
    sql: 'sql',
    sh: 'shell',
    bash: 'shell',
  };
  return languageMap[ext || ''] || 'plaintext';
}
```

**Key Features:**

- ✅ Multi-file tab system
- ✅ Active editor management
- ✅ Automatic language detection
- ✅ File close/switch functionality
- ✅ Empty state for no open files
- ✅ Status bar with cursor position

#### 1.3 Create Zustand Editor Store

**File:** `apps/spectrum-workspace/src/stores/editor-store.ts`

```typescript
import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import { devtools, persist } from "zustand/middleware";

export interface OpenFile {
  id: string;
  path: string;
  content: string;
  modified: boolean;
  savedContent: string;
  cursorLine: number;
  cursorColumn: number;
  language: string;
}

interface EditorState {
  // Open files
  openFiles: OpenFile[];
  activeFileId: string | null;

  // Actions
  openFile: (filePath: string, content: string) => void;
  closeFile: (fileId: string) => void;
  setActiveFile: (fileId: string) => void;
  updateFileContent: (fileId: string, content: string) => void;
  saveFile: (fileId: string) => Promise<void>;
  saveAllFiles: () => Promise<void>;
  getFileContent: (fileId: string) => string;
  isFileModified: (fileId: string) => boolean;
  updateCursorPosition: (fileId: string, line: number, column: number) => void;

  // Editor settings
  settings: {
    fontSize: number;
    tabSize: number;
    wordWrap: "off" | "on" | "bounded";
    minimap: boolean;
    lineNumbers: "on" | "off" | "relative";
    formatOnSave: boolean;
  };
  updateSettings: (settings: Partial<EditorState["settings"]>) => void;
}

export const useEditorStore = create<EditorState>()(
  devtools(
    persist(
      immer((set, get) => ({
        openFiles: [],
        activeFileId: null,

        openFile: (filePath, content) => {
          set((state) => {
            // Check if already open
            const existing = state.openFiles.find((f) => f.path === filePath);
            if (existing) {
              state.activeFileId = existing.id;
              return;
            }

            // Create new file entry
            const fileId = `file-${Date.now()}-${Math.random()}`;
            state.openFiles.push({
              id: fileId,
              path: filePath,
              content,
              savedContent: content,
              modified: false,
              cursorLine: 1,
              cursorColumn: 1,
              language: detectLanguageFromPath(filePath),
            });
            state.activeFileId = fileId;
          });
        },

        closeFile: (fileId) => {
          set((state) => {
            const index = state.openFiles.findIndex((f) => f.id === fileId);
            if (index === -1) return;

            // Warn if modified
            const file = state.openFiles[index];
            if (file.modified) {
              // TODO: Show confirmation dialog
              console.warn(`Closing modified file: ${file.path}`);
            }

            state.openFiles.splice(index, 1);

            // Update active file
            if (state.activeFileId === fileId) {
              state.activeFileId = state.openFiles[0]?.id || null;
            }
          });
        },

        setActiveFile: (fileId) => {
          set((state) => {
            state.activeFileId = fileId;
          });
        },

        updateFileContent: (fileId, content) => {
          set((state) => {
            const file = state.openFiles.find((f) => f.id === fileId);
            if (!file) return;

            file.content = content;
            file.modified = content !== file.savedContent;
          });
        },

        saveFile: async (fileId) => {
          const file = get().openFiles.find((f) => f.id === fileId);
          if (!file) return;

          try {
            // Save via Tauri IPC
            await window.__TAURI_INVOKE__("write_file", {
              path: file.path,
              contents: file.content,
            });

            set((state) => {
              const f = state.openFiles.find((f) => f.id === fileId);
              if (f) {
                f.savedContent = f.content;
                f.modified = false;
              }
            });
          } catch (error) {
            console.error("Failed to save file:", error);
            throw error;
          }
        },

        saveAllFiles: async () => {
          const { openFiles } = get();
          const modifiedFiles = openFiles.filter((f) => f.modified);

          await Promise.all(modifiedFiles.map((f) => get().saveFile(f.id)));
        },

        getFileContent: (fileId) => {
          return get().openFiles.find((f) => f.id === fileId)?.content || "";
        },

        isFileModified: (fileId) => {
          return (
            get().openFiles.find((f) => f.id === fileId)?.modified || false
          );
        },

        updateCursorPosition: (fileId, line, column) => {
          set((state) => {
            const file = state.openFiles.find((f) => f.id === fileId);
            if (file) {
              file.cursorLine = line;
              file.cursorColumn = column;
            }
          });
        },

        settings: {
          fontSize: 14,
          tabSize: 2,
          wordWrap: "off",
          minimap: true,
          lineNumbers: "on",
          formatOnSave: true,
        },

        updateSettings: (newSettings) => {
          set((state) => {
            state.settings = { ...state.settings, ...newSettings };
          });
        },
      })),
      {
        name: "editor-storage",
        partialize: (state) => ({ settings: state.settings }),
      }
    ),
    { name: "EditorStore" }
  )
);

function detectLanguageFromPath(filePath: string): string {
  const ext = filePath.split(".").pop()?.toLowerCase() || "";
  const map: Record<string, string> = {
    ts: "typescript",
    tsx: "typescript",
    js: "javascript",
    jsx: "javascript",
    py: "python",
    rs: "rust",
    go: "go",
    md: "markdown",
    json: "json",
  };
  return map[ext] || "plaintext";
}
```

**Key Features:**

- ✅ Multi-file state management
- ✅ Modified/saved state tracking
- ✅ Cursor position persistence
- ✅ Editor settings storage
- ✅ Tauri IPC integration for file I/O
- ✅ Immer for immutable updates
- ✅ DevTools integration

---

### Phase 2: Theme Customization (Week 1, Days 4-5)

#### 2.1 Create NEURECTOMY Dark Theme

**File:** `packages/ui/src/components/code-editor/themes/neurectomy-dark.ts`

```typescript
import * as monaco from "monaco-editor";

export const neurectomyDarkTheme: monaco.editor.IStandaloneThemeData = {
  base: "vs-dark",
  inherit: true,
  rules: [
    // Keywords
    { token: "keyword", foreground: "c792ea", fontStyle: "bold" },
    { token: "keyword.control", foreground: "c792ea", fontStyle: "bold" },

    // Types
    { token: "type", foreground: "ffcb6b" },
    { token: "type.identifier", foreground: "ffcb6b" },

    // Functions
    { token: "function", foreground: "82aaff" },
    { token: "function.call", foreground: "82aaff" },

    // Variables
    { token: "variable", foreground: "f07178" },
    { token: "variable.parameter", foreground: "f78c6c" },

    // Strings
    { token: "string", foreground: "c3e88d" },
    { token: "string.escape", foreground: "89ddff" },

    // Numbers
    { token: "number", foreground: "f78c6c" },

    // Comments
    { token: "comment", foreground: "546e7a", fontStyle: "italic" },

    // Operators
    { token: "operator", foreground: "89ddff" },
    { token: "delimiter", foreground: "89ddff" },

    // Tags (JSX/HTML)
    { token: "tag", foreground: "f07178" },
    { token: "tag.id", foreground: "fad430" },
    { token: "tag.class", foreground: "fad430" },

    // Attributes
    { token: "attribute.name", foreground: "c792ea" },
    { token: "attribute.value", foreground: "c3e88d" },

    // Special
    { token: "constant", foreground: "ffcb6b" },
    { token: "entity", foreground: "82aaff" },
    { token: "meta", foreground: "ffcb6b" },
  ],
  colors: {
    // Editor colors (matching NEURECTOMY palette)
    "editor.background": "#0a0a0f",
    "editor.foreground": "#e4e4e7",
    "editor.lineHighlightBackground": "#13131a",
    "editor.selectionBackground": "#6366f144",
    "editor.inactiveSelectionBackground": "#6366f122",

    // Line numbers
    "editorLineNumber.foreground": "#52525b",
    "editorLineNumber.activeForeground": "#a1a1aa",

    // Cursor
    "editorCursor.foreground": "#6366f1",

    // Whitespace
    "editorWhitespace.foreground": "#27272a",

    // Indentation guides
    "editorIndentGuide.background": "#27272a",
    "editorIndentGuide.activeBackground": "#3f3f46",

    // Gutter
    "editorGutter.background": "#0a0a0f",
    "editorGutter.modifiedBackground": "#f59e0b",
    "editorGutter.addedBackground": "#22c55e",
    "editorGutter.deletedBackground": "#ef4444",

    // Minimap
    "minimap.background": "#0a0a0f",
    "minimap.selectionHighlight": "#6366f144",

    // Scrollbar
    "scrollbar.shadow": "#00000088",
    "scrollbarSlider.background": "#52525b44",
    "scrollbarSlider.hoverBackground": "#52525b66",
    "scrollbarSlider.activeBackground": "#52525b88",

    // Bracket matching
    "editorBracketMatch.background": "#6366f144",
    "editorBracketMatch.border": "#6366f1",

    // Find/replace
    "editor.findMatchBackground": "#f59e0b44",
    "editor.findMatchHighlightBackground": "#f59e0b22",
    "editor.findRangeHighlightBackground": "#6366f122",

    // Widgets
    "editorWidget.background": "#13131a",
    "editorWidget.border": "#27272a",
    "editorSuggestWidget.background": "#13131a",
    "editorSuggestWidget.border": "#27272a",
    "editorSuggestWidget.selectedBackground": "#1a1a24",

    // Hover
    "editorHoverWidget.background": "#13131a",
    "editorHoverWidget.border": "#6366f1",
  },
};

export function registerNeurectomyTheme(monaco: Monaco) {
  monaco.editor.defineTheme("neurectomy-dark", neurectomyDarkTheme);
  monaco.editor.setTheme("neurectomy-dark");
}
```

#### 2.2 Create Theme Manager Hook

**File:** `packages/ui/src/components/code-editor/hooks/useMonacoTheme.ts`

```typescript
import { useEffect } from "react";
import { Monaco } from "@monaco-editor/react";
import { neurectomyDarkTheme, neurectomyLightTheme } from "../themes";

export function useMonacoTheme(monaco: Monaco | null, theme: "dark" | "light") {
  useEffect(() => {
    if (!monaco) return;

    // Register themes
    monaco.editor.defineTheme("neurectomy-dark", neurectomyDarkTheme);
    monaco.editor.defineTheme("neurectomy-light", neurectomyLightTheme);

    // Apply theme
    monaco.editor.setTheme(
      theme === "dark" ? "neurectomy-dark" : "neurectomy-light"
    );
  }, [monaco, theme]);
}
```

---

### Phase 3: Language Services (Week 2, Days 1-3)

#### 3.1 TypeScript/JavaScript Language Service

**File:** `packages/ui/src/components/code-editor/language-services/typescript.ts`

```typescript
import * as monaco from "monaco-editor";

export function setupTypeScriptLanguageService(monacoInstance: Monaco) {
  // Configure TypeScript compiler options
  monaco.languages.typescript.typescriptDefaults.setCompilerOptions({
    target: monaco.languages.typescript.ScriptTarget.ES2020,
    module: monaco.languages.typescript.ModuleKind.ESNext,
    lib: ["ES2020", "DOM", "DOM.Iterable"],
    jsx: monaco.languages.typescript.JsxEmit.React,
    jsxImportSource: "react",
    allowNonTsExtensions: true,
    moduleResolution: monaco.languages.typescript.ModuleResolutionKind.NodeJs,
    esModuleInterop: true,
    allowSyntheticDefaultImports: true,
    strict: true,
    skipLibCheck: true,
  });

  // Configure diagnostics
  monaco.languages.typescript.typescriptDefaults.setDiagnosticsOptions({
    noSemanticValidation: false,
    noSyntaxValidation: false,
    diagnosticCodesToIgnore: [1375], // Ignore 'await' in non-async function
  });

  // Add extra libraries (for @neurectomy packages, etc.)
  addNeurectomyTypeDefinitions(monaco);
}

function addNeurectomyTypeDefinitions(monaco: Monaco) {
  // Add type definitions for NEURECTOMY packages
  const neurectomyTypes = `
    declare module '@neurectomy/types' {
      export interface Agent {
        id: string;
        name: string;
        type: 'research' | 'code' | 'test' | 'deploy';
        status: 'idle' | 'running' | 'paused' | 'error';
      }
      
      export interface Workspace {
        id: string;
        name: string;
        agents: Agent[];
      }
    }
    
    declare module '@neurectomy/api-client' {
      export class APIClient {
        constructor(baseUrl: string);
        query<T>(query: string, variables?: any): Promise<T>;
        subscribe<T>(subscription: string): AsyncIterableIterator<T>;
      }
    }
  `;

  monaco.languages.typescript.typescriptDefaults.addExtraLib(
    neurectomyTypes,
    "file:///node_modules/@types/neurectomy/index.d.ts"
  );
}
```

#### 3.2 Python Language Service

**File:** `packages/ui/src/components/code-editor/language-services/python.ts`

```typescript
import * as monaco from "monaco-editor";

export function setupPythonLanguageService(monacoInstance: Monaco) {
  // Configure Python-specific settings
  monaco.languages.registerCompletionItemProvider("python", {
    provideCompletionItems: (model, position) => {
      const word = model.getWordUntilPosition(position);
      const range = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn,
      };

      // Common Python keywords and built-ins
      const suggestions: monaco.languages.CompletionItem[] = [
        // Keywords
        ...[
          "def",
          "class",
          "import",
          "from",
          "if",
          "else",
          "elif",
          "for",
          "while",
          "return",
          "yield",
          "async",
          "await",
        ].map((kw) => ({
          label: kw,
          kind: monaco.languages.CompletionItemKind.Keyword,
          insertText: kw,
          range,
        })),

        // Common imports
        {
          label: "import numpy as np",
          kind: monaco.languages.CompletionItemKind.Snippet,
          insertText: "import numpy as np",
          documentation: "Import NumPy",
          range,
        },
        {
          label: "from typing import",
          kind: monaco.languages.CompletionItemKind.Snippet,
          insertText: "from typing import ${1:List}",
          insertTextRules:
            monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          documentation: "Import from typing module",
          range,
        },
      ];

      return { suggestions };
    },
  });

  // Register hover provider for Python
  monaco.languages.registerHoverProvider("python", {
    provideHover: (model, position) => {
      const word = model.getWordAtPosition(position);
      if (!word) return null;

      // Provide documentation for common Python functions
      const docs: Record<string, string> = {
        print:
          "**print**(*objects, sep=' ', end='\\n', file=sys.stdout, flush=False)\n\nPrint objects to the text stream file.",
        len: "**len**(s)\n\nReturn the length (the number of items) of an object.",
        range:
          "**range**(stop) or range(start, stop[, step])\n\nReturn an immutable sequence of numbers.",
      };

      if (word.word in docs) {
        return {
          contents: [{ value: docs[word.word] }],
        };
      }

      return null;
    },
  });
}
```

#### 3.3 Rust Language Service

**File:** `packages/ui/src/components/code-editor/language-services/rust.ts`

```typescript
import * as monaco from "monaco-editor";

export function setupRustLanguageService(monacoInstance: Monaco) {
  // Register Rust language (if not already registered)
  monaco.languages.register({ id: "rust" });

  // Configure syntax highlighting
  monaco.languages.setMonarchTokensProvider("rust", {
    keywords: [
      "as",
      "break",
      "const",
      "continue",
      "crate",
      "else",
      "enum",
      "extern",
      "false",
      "fn",
      "for",
      "if",
      "impl",
      "in",
      "let",
      "loop",
      "match",
      "mod",
      "move",
      "mut",
      "pub",
      "ref",
      "return",
      "self",
      "Self",
      "static",
      "struct",
      "super",
      "trait",
      "true",
      "type",
      "unsafe",
      "use",
      "where",
      "while",
      "async",
      "await",
      "dyn",
    ],

    typeKeywords: [
      "i8",
      "i16",
      "i32",
      "i64",
      "i128",
      "isize",
      "u8",
      "u16",
      "u32",
      "u64",
      "u128",
      "usize",
      "f32",
      "f64",
      "bool",
      "char",
      "str",
    ],

    operators: [
      "=",
      ">",
      "<",
      "!",
      "~",
      "?",
      ":",
      "==",
      "<=",
      ">=",
      "!=",
      "&&",
      "||",
      "++",
      "--",
      "+",
      "-",
      "*",
      "/",
      "&",
      "|",
      "^",
      "%",
      "<<",
      ">>",
      ">>>",
      "+=",
      "-=",
      "*=",
      "/=",
      "&=",
      "|=",
      "^=",
      "%=",
      "<<=",
      ">>=",
      ">>>=",
    ],

    tokenizer: {
      root: [
        // Identifiers and keywords
        [
          /[a-z_$][\w$]*/,
          {
            cases: {
              "@typeKeywords": "type.identifier",
              "@keywords": "keyword",
              "@default": "identifier",
            },
          },
        ],

        // Whitespace
        { include: "@whitespace" },

        // Strings
        [/"([^"\\]|\\.)*$/, "string.invalid"],
        [/"/, "string", "@string"],

        // Comments
        [/\/\/.*$/, "comment"],
        [/\/\*/, "comment", "@comment"],
      ],

      whitespace: [[/[ \t\r\n]+/, ""]],

      comment: [
        [/[^\/*]+/, "comment"],
        [/\*\//, "comment", "@pop"],
        [/[\/*]/, "comment"],
      ],

      string: [
        [/[^\\"]+/, "string"],
        [/"/, "string", "@pop"],
      ],
    },
  });

  // Completion provider
  monaco.languages.registerCompletionItemProvider("rust", {
    provideCompletionItems: (model, position) => {
      const suggestions: monaco.languages.CompletionItem[] = [
        {
          label: "fn",
          kind: monaco.languages.CompletionItemKind.Snippet,
          insertText: "fn ${1:name}(${2:args}) {\n\t$0\n}",
          insertTextRules:
            monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          documentation: "Function definition",
        },
        {
          label: "struct",
          kind: monaco.languages.CompletionItemKind.Snippet,
          insertText: "struct ${1:Name} {\n\t$0\n}",
          insertTextRules:
            monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          documentation: "Struct definition",
        },
      ];

      return { suggestions };
    },
  });
}
```

---

### Phase 4: Advanced Features (Week 2, Days 4-5)

#### 4.1 Multi-File IntelliSense

**File:** `packages/ui/src/components/code-editor/services/intellisense-service.ts`

```typescript
import * as monaco from "monaco-editor";

export class IntelliSenseService {
  private fileModels = new Map<string, monaco.editor.ITextModel>();

  /**
   * Register a file for IntelliSense across the workspace
   */
  registerFile(filePath: string, content: string, language: string) {
    // Create or update model
    const uri = monaco.Uri.file(filePath);
    let model = monaco.editor.getModel(uri);

    if (!model) {
      model = monaco.editor.createModel(content, language, uri);
    } else {
      model.setValue(content);
    }

    this.fileModels.set(filePath, model);
    return model;
  }

  /**
   * Unregister a file
   */
  unregisterFile(filePath: string) {
    const model = this.fileModels.get(filePath);
    if (model) {
      model.dispose();
      this.fileModels.delete(filePath);
    }
  }

  /**
   * Get all registered models
   */
  getAllModels() {
    return Array.from(this.fileModels.values());
  }

  /**
   * Find references across all files
   */
  async findReferences(
    position: monaco.Position,
    model: monaco.editor.ITextModel
  ): Promise<monaco.languages.Location[]> {
    const locations: monaco.languages.Location[] = [];

    // Get all models
    const models = this.getAllModels();

    for (const m of models) {
      const references = await monaco.languages.findReferences(m.uri, position);

      if (references) {
        locations.push(...references);
      }
    }

    return locations;
  }

  /**
   * Get workspace-wide symbol suggestions
   */
  getWorkspaceSymbols(query: string): monaco.languages.SymbolInformation[] {
    const symbols: monaco.languages.SymbolInformation[] = [];

    for (const model of this.fileModels.values()) {
      // Extract symbols from each model
      // This is a simplified version - real implementation would use language server
      const content = model.getValue();
      const lines = content.split("\n");

      lines.forEach((line, index) => {
        // Simple regex for function/class detection
        const functionMatch = line.match(/function\s+(\w+)/);
        const classMatch = line.match(/class\s+(\w+)/);

        if (functionMatch && functionMatch[1].includes(query)) {
          symbols.push({
            name: functionMatch[1],
            kind: monaco.languages.SymbolKind.Function,
            location: {
              uri: model.uri,
              range: new monaco.Range(index + 1, 0, index + 1, line.length),
            },
          });
        }

        if (classMatch && classMatch[1].includes(query)) {
          symbols.push({
            name: classMatch[1],
            kind: monaco.languages.SymbolKind.Class,
            location: {
              uri: model.uri,
              range: new monaco.Range(index + 1, 0, index + 1, line.length),
            },
          });
        }
      });
    }

    return symbols;
  }
}

export const intellisenseService = new IntelliSenseService();
```

#### 4.2 Diff Editor for Version Comparison

**File:** `packages/ui/src/components/code-editor/DiffEditor.tsx`

```typescript
import { DiffEditor as MonacoDiffEditor } from '@monaco-editor/react';
import * as monaco from 'monaco-editor';

export interface DiffEditorProps {
  original: string;
  modified: string;
  language: string;
  originalTitle?: string;
  modifiedTitle?: string;
  readOnly?: boolean;
}

export function DiffEditor({
  original,
  modified,
  language,
  originalTitle = 'Original',
  modifiedTitle = 'Modified',
  readOnly = true,
}: DiffEditorProps) {
  const options: monaco.editor.IStandaloneDiffEditorConstructionOptions = {
    readOnly,
    renderSideBySide: true,
    fontSize: 14,
    fontFamily: "'Fira Code', monospace",
    fontLigatures: true,
    scrollBeyondLastLine: false,
    automaticLayout: true,
  };

  return (
    <div className="flex h-full w-full flex-col">
      <div className="flex border-b border-gray-700 bg-[#13131a]">
        <div className="flex-1 px-4 py-2 text-sm text-gray-400">
          {originalTitle}
        </div>
        <div className="flex-1 border-l border-gray-700 px-4 py-2 text-sm text-gray-400">
          {modifiedTitle}
        </div>
      </div>

      <div className="flex-1">
        <MonacoDiffEditor
          original={original}
          modified={modified}
          language={language}
          theme="neurectomy-dark"
          options={options}
        />
      </div>
    </div>
  );
}
```

---

### Phase 5: File System Integration (Week 3, Days 1-2)

#### 5.1 Tauri File System Integration

**File:** `packages/ui/src/components/code-editor/services/file-system-service.ts`

```typescript
import { invoke } from "@tauri-apps/api/core";
import { readTextFile, writeTextFile } from "@tauri-apps/plugin-fs";

export class FileSystemService {
  /**
   * Read file from disk
   */
  async readFile(filePath: string): Promise<string> {
    try {
      const content = await readTextFile(filePath);
      return content;
    } catch (error) {
      console.error("Failed to read file:", error);
      throw new Error(`Failed to read file: ${filePath}`);
    }
  }

  /**
   * Write file to disk
   */
  async writeFile(filePath: string, content: string): Promise<void> {
    try {
      await writeTextFile(filePath, content);
    } catch (error) {
      console.error("Failed to write file:", error);
      throw new Error(`Failed to write file: ${filePath}`);
    }
  }

  /**
   * Open file dialog
   */
  async openFileDialog(): Promise<string | null> {
    try {
      const selected = await invoke<string | null>("open_file_dialog", {
        filters: [
          {
            name: "All Files",
            extensions: ["*"],
          },
          {
            name: "Code Files",
            extensions: ["ts", "tsx", "js", "jsx", "py", "rs", "go"],
          },
        ],
      });
      return selected;
    } catch (error) {
      console.error("Failed to open file dialog:", error);
      return null;
    }
  }

  /**
   * Save file dialog
   */
  async saveFileDialog(defaultPath?: string): Promise<string | null> {
    try {
      const selected = await invoke<string | null>("save_file_dialog", {
        defaultPath,
      });
      return selected;
    } catch (error) {
      console.error("Failed to open save dialog:", error);
      return null;
    }
  }

  /**
   * Watch file for external changes
   */
  async watchFile(
    filePath: string,
    callback: (newContent: string) => void
  ): Promise<() => void> {
    // Use Tauri's file watcher
    const unwatch = await invoke<() => void>("watch_file", {
      path: filePath,
    });

    // Setup event listener
    window.addEventListener("file-changed", (event: CustomEvent) => {
      if (event.detail.path === filePath) {
        callback(event.detail.content);
      }
    });

    return () => {
      unwatch();
      window.removeEventListener("file-changed", callback);
    };
  }
}

export const fileSystemService = new FileSystemService();
```

#### 5.2 Auto-Save Implementation

**File:** `packages/ui/src/components/code-editor/hooks/useAutoSave.ts`

```typescript
import { useEffect, useRef } from "react";
import { useEditorStore } from "@neurectomy/stores";
import { fileSystemService } from "../services/file-system-service";

export function useAutoSave(enabled: boolean, delayMs: number = 2000) {
  const { openFiles, saveFile } = useEditorStore();
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (!enabled) return;

    // Clear previous timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    // Schedule auto-save for modified files
    const modifiedFiles = openFiles.filter((f) => f.modified);

    if (modifiedFiles.length > 0) {
      timeoutRef.current = setTimeout(() => {
        modifiedFiles.forEach((file) => {
          saveFile(file.id).catch((error) => {
            console.error(`Auto-save failed for ${file.path}:`, error);
          });
        });
      }, delayMs);
    }

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [openFiles, enabled, delayMs, saveFile]);
}
```

---

### Phase 6: Integration & Testing (Week 3, Days 3-5)

#### 6.1 Integration Tests

**File:** `packages/ui/src/components/code-editor/__tests__/MonacoEditor.test.tsx`

```typescript
import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MonacoEditor } from '../MonacoEditor';

describe('MonacoEditor', () => {
  it('renders editor with initial value', async () => {
    const initialValue = 'const x = 42;';

    render(
      <MonacoEditor
        value={initialValue}
        language="typescript"
      />
    );

    // Wait for Monaco to initialize
    await waitFor(() => {
      expect(screen.queryByText('Loading editor...')).not.toBeInTheDocument();
    });
  });

  it('calls onChange when content is edited', async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();

    render(
      <MonacoEditor
        value=""
        language="typescript"
        onChange={onChange}
      />
    );

    await waitFor(() => {
      expect(screen.queryByText('Loading editor...')).not.toBeInTheDocument();
    });

    // Simulate typing
    const editor = screen.getByRole('textbox');
    await user.type(editor, 'test');

    expect(onChange).toHaveBeenCalled();
  });

  it('applies custom theme', async () => {
    render(
      <MonacoEditor
        value="test"
        language="typescript"
        theme="neurectomy-dark"
      />
    );

    await waitFor(() => {
      const wrapper = screen.getByClassName('monaco-editor-wrapper');
      expect(wrapper).toBeInTheDocument();
    });
  });
});
```

#### 6.2 Performance Tests

**File:** `packages/ui/src/components/code-editor/__tests__/performance.test.tsx`

```typescript
import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { MonacoEditor } from '../MonacoEditor';

describe('MonacoEditor Performance', () => {
  it('handles large files efficiently', async () => {
    // Generate 10,000 line file
    const largeContent = Array.from({ length: 10000 }, (_, i) =>
      `const line${i} = ${i};`
    ).join('\n');

    const startTime = performance.now();

    render(
      <MonacoEditor
        value={largeContent}
        language="typescript"
      />
    );

    const endTime = performance.now();
    const renderTime = endTime - startTime;

    // Should render in less than 500ms
    expect(renderTime).toBeLessThan(500);
  });

  it('updates quickly on content changes', async () => {
    const { rerender } = render(
      <MonacoEditor value="initial" language="typescript" />
    );

    const startTime = performance.now();

    rerender(
      <MonacoEditor value="updated content" language="typescript" />
    );

    const endTime = performance.now();
    const updateTime = endTime - startTime;

    // Should update in less than 100ms
    expect(updateTime).toBeLessThan(100);
  });
});
```

---

## 📊 Success Criteria

### Functional Requirements

- ✅ Monaco Editor renders correctly in Spectrum Workspace
- ✅ Multi-file editing with tab system
- ✅ Syntax highlighting for all supported languages
- ✅ IntelliSense/autocomplete working
- ✅ File save/load via Tauri IPC
- ✅ Theme matches NEURECTOMY design system
- ✅ Keyboard shortcuts (Ctrl+S, Ctrl+F, etc.)

### Performance Requirements

- ✅ Initial load < 500ms
- ✅ File open < 200ms
- ✅ Typing latency < 50ms
- ✅ Handle files up to 10,000 lines smoothly

### Quality Requirements

- ✅ 90%+ test coverage
- ✅ Zero TypeScript errors
- ✅ All keyboard shortcuts documented
- ✅ Accessibility (WCAG 2.2 AA)

---

## 🔄 Integration Checklist

### Week 1

- [ ] Day 1-2: Create MonacoEditor wrapper component
- [ ] Day 2-3: Implement EditorManager with tabs
- [ ] Day 3: Create editor Zustand store
- [ ] Day 4-5: Implement NEURECTOMY themes

### Week 2

- [ ] Day 1-2: Setup TypeScript language service
- [ ] Day 2: Setup Python language service
- [ ] Day 3: Setup Rust language service
- [ ] Day 4: Implement multi-file IntelliSense
- [ ] Day 5: Create diff editor component

### Week 3

- [ ] Day 1: Tauri file system integration
- [ ] Day 2: Auto-save implementation
- [ ] Day 3: Integration tests
- [ ] Day 4: Performance tests
- [ ] Day 5: Documentation and polish

---

## 📚 Documentation Required

### Developer Documentation

- [ ] Monaco Editor API reference
- [ ] Theme customization guide
- [ ] Language service extension guide
- [ ] Custom shortcuts guide

### User Documentation

- [ ] Editor keyboard shortcuts
- [ ] File management guide
- [ ] Multi-cursor editing tutorial
- [ ] Search and replace guide

---

## 🚀 Deployment Plan

### Phase 1: Development

- Implement in `feature/monaco-editor` branch
- Test locally with `pnpm dev`

### Phase 2: Testing

- Run full test suite
- Performance benchmarking
- Accessibility audit

### Phase 3: Integration

- Merge to `develop` branch
- Integration testing with other modules
- Staging deployment

### Phase 4: Production

- Merge to `main`
- Production deployment
- Monitor performance metrics

---

## 🎯 Next Steps

1. **Create feature branch:**

   ```bash
   git checkout -b feature/monaco-editor
   ```

2. **Install any additional dependencies:**

   ```bash
   cd packages/ui
   pnpm add monaco-editor
   ```

3. **Start with Phase 1, Day 1:**
   Create `packages/ui/src/components/code-editor/MonacoEditor.tsx`

4. **Test incrementally:**
   After each component, run:
   ```bash
   pnpm test packages/ui
   ```

---

**Estimated Timeline:** 2-3 weeks  
**Priority:** HIGH  
**Dependencies:** None (ready to start)  
**Owner:** Core Development Team

This plan provides a complete roadmap for Monaco Editor integration. Begin with Phase 1 and work through systematically. Each phase builds on the previous, ensuring a robust, production-ready code editor for NEURECTOMY.
