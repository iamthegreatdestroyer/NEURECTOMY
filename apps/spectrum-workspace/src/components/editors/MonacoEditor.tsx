/**
 * MonacoEditor Component - Core Monaco Editor Wrapper
 * Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
 */

import React, { useRef, useEffect, useCallback, useState } from "react";
import * as monaco from "monaco-editor";
import { useEditorStore } from "../../stores/editor-store";
import {
  initializeMonaco,
  isMonacoInitialized,
  registerNeurectomyKeybindings,
  THEME_NAMES,
  type KeybindingCallbacks,
} from "../../lib/monaco";

// Configure Monaco environment with bundler-friendly workers
const workerFactory: Record<string, () => Worker> = {
  json: () =>
    new Worker(
      new URL(
        "monaco-editor/esm/vs/language/json/json.worker?worker",
        import.meta.url
      ),
      { type: "module", name: "json" }
    ),
  css: () =>
    new Worker(
      new URL(
        "monaco-editor/esm/vs/language/css/css.worker?worker",
        import.meta.url
      ),
      { type: "module", name: "css" }
    ),
  scss: () =>
    new Worker(
      new URL(
        "monaco-editor/esm/vs/language/css/css.worker?worker",
        import.meta.url
      ),
      { type: "module", name: "scss" }
    ),
  less: () =>
    new Worker(
      new URL(
        "monaco-editor/esm/vs/language/css/css.worker?worker",
        import.meta.url
      ),
      { type: "module", name: "less" }
    ),
  html: () =>
    new Worker(
      new URL(
        "monaco-editor/esm/vs/language/html/html.worker?worker",
        import.meta.url
      ),
      { type: "module", name: "html" }
    ),
  handlebars: () =>
    new Worker(
      new URL(
        "monaco-editor/esm/vs/language/html/html.worker?worker",
        import.meta.url
      ),
      { type: "module", name: "handlebars" }
    ),
  razor: () =>
    new Worker(
      new URL(
        "monaco-editor/esm/vs/language/html/html.worker?worker",
        import.meta.url
      ),
      { type: "module", name: "razor" }
    ),
  typescript: () =>
    new Worker(
      new URL(
        "monaco-editor/esm/vs/language/typescript/ts.worker?worker",
        import.meta.url
      ),
      { type: "module", name: "typescript" }
    ),
  javascript: () =>
    new Worker(
      new URL(
        "monaco-editor/esm/vs/language/typescript/ts.worker?worker",
        import.meta.url
      ),
      { type: "module", name: "javascript" }
    ),
  default: () =>
    new Worker(
      new URL(
        "monaco-editor/esm/vs/editor/editor.worker?worker",
        import.meta.url
      ),
      { type: "module", name: "editor" }
    ),
};

self.MonacoEnvironment = {
  getWorker(_moduleId, label) {
    const factory = workerFactory[label] || workerFactory.default;
    return factory();
  },
};

export interface MonacoEditorProps {
  /** Optional className for styling */
  className?: string;
  /** Optional inline styles */
  style?: React.CSSProperties;
  /** Callback when editor is ready */
  onReady?: (editor: monaco.editor.IStandaloneCodeEditor) => void;
}

/**
 * MonacoEditor - Core editor component
 *
 * Integrates Monaco Editor with NEURECTOMY's editor store.
 * Handles file switching, content synchronization, and editor lifecycle.
 */
export const MonacoEditor: React.FC<MonacoEditorProps> = ({
  className = "",
  style,
  onReady,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null);
  const currentFileIdRef = useRef<string | null>(null);
  const [isReady, setIsReady] = useState(false);

  const {
    activeFileId,
    config,
    setEditor,
    updateFileContent,
    markFileDirty,
    updateViewState,
    getActiveFile,
    setFileModel,
    saveFile,
    saveAllFiles,
  } = useEditorStore();

  /**
   * Initialize Monaco with NEURECTOMY configuration
   */
  useEffect(() => {
    let mounted = true;

    const init = async () => {
      if (!isMonacoInitialized()) {
        await initializeMonaco();
      }
      if (mounted) {
        setIsReady(true);
      }
    };

    init();

    return () => {
      mounted = false;
    };
  }, []);

  /**
   * Initialize Monaco editor instance
   */
  useEffect(() => {
    if (!containerRef.current || !isReady) return;

    // Create editor with NEURECTOMY theme
    const editor = monaco.editor.create(containerRef.current, {
      value: "",
      language: "typescript",
      theme: THEME_NAMES.NEURECTOMY_DARK,
      fontSize: config.fontSize,
      fontFamily:
        config.fontFamily ||
        "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
      fontLigatures: true,
      tabSize: config.tabSize,
      insertSpaces: config.insertSpaces,
      lineNumbers: config.lineNumbers,
      wordWrap: config.wordWrap,
      minimap: {
        enabled: config.minimap,
      },
      quickSuggestions: config.quickSuggestions,
      renderWhitespace: config.renderWhitespace,
      formatOnPaste: config.formatOnPaste,
      automaticLayout: true,
      scrollBeyondLastLine: false,
      smoothScrolling: true,
      cursorBlinking: "smooth",
      cursorSmoothCaretAnimation: "on",
      padding: { top: 16, bottom: 16 },
      bracketPairColorization: {
        enabled: true,
      },
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
      },
      folding: true,
      foldingStrategy: "indentation",
      showFoldingControls: "mouseover",
      glyphMargin: true,
      lightbulb: {
        enabled: monaco.editor.ShowLightbulbIconMode.On,
      },
      stickyScroll: {
        enabled: true,
      },
      inlineSuggest: {
        enabled: true,
      },
    });

    editorRef.current = editor;
    setEditor(editor);

    // Register NEURECTOMY keybindings
    const keybindingCallbacks: KeybindingCallbacks = {
      onSave: async () => {
        if (currentFileIdRef.current) {
          await saveFile(currentFileIdRef.current);
        }
      },
      onSaveAll: async () => {
        await saveAllFiles();
      },
    };
    const keybindingDisposable = registerNeurectomyKeybindings(
      editor,
      keybindingCallbacks
    );

    // Listen to content changes
    const contentChangeDisposable = editor.onDidChangeModelContent(() => {
      const model = editor.getModel();
      if (!model || !currentFileIdRef.current) return;

      const content = model.getValue();
      updateFileContent(currentFileIdRef.current, content);
      markFileDirty(currentFileIdRef.current, true);
    });

    // Listen to cursor/scroll changes to save view state
    const viewStateChangeDisposable = editor.onDidChangeCursorPosition(() => {
      if (!currentFileIdRef.current) return;

      const viewState = editor.saveViewState();
      updateViewState(currentFileIdRef.current, viewState);
    });

    // Notify ready
    if (onReady) {
      onReady(editor);
    }

    // Cleanup
    return () => {
      keybindingDisposable.dispose();
      contentChangeDisposable.dispose();
      viewStateChangeDisposable.dispose();
      editor.dispose();
      setEditor(null);
    };
  }, [isReady]); // Run when Monaco is ready

  /**
   * Update editor configuration when config changes
   */
  useEffect(() => {
    if (!editorRef.current) return;

    editorRef.current.updateOptions({
      theme: config.theme,
      fontSize: config.fontSize,
      fontFamily: config.fontFamily,
      tabSize: config.tabSize,
      insertSpaces: config.insertSpaces,
      lineNumbers: config.lineNumbers,
      wordWrap: config.wordWrap,
      minimap: {
        enabled: config.minimap,
      },
      quickSuggestions: config.quickSuggestions,
      renderWhitespace: config.renderWhitespace,
      formatOnPaste: config.formatOnPaste,
    });
  }, [config]);

  /**
   * Handle active file changes
   */
  useEffect(() => {
    if (!editorRef.current) return;

    const editor = editorRef.current;
    const activeFile = getActiveFile();

    // Save view state for previous file
    if (currentFileIdRef.current) {
      const viewState = editor.saveViewState();
      updateViewState(currentFileIdRef.current, viewState);
    }

    // Switch to new file
    if (activeFile) {
      // Create or get model for this file
      let model = activeFile.model;

      if (!model) {
        // Create new model
        const uri = monaco.Uri.file(activeFile.path);
        model =
          monaco.editor.getModel(uri) ||
          monaco.editor.createModel(
            activeFile.content,
            activeFile.language,
            uri
          );

        // Store model reference
        setFileModel(activeFile.id, model);
      }

      // Set model
      editor.setModel(model);

      // Restore view state
      if (activeFile.viewState) {
        editor.restoreViewState(activeFile.viewState);
      }

      // Update read-only mode
      editor.updateOptions({
        readOnly: activeFile.isReadOnly || false,
      });

      currentFileIdRef.current = activeFile.id;
    } else {
      // No active file, clear editor
      editor.setModel(null);
      currentFileIdRef.current = null;
    }
  }, [activeFileId, getActiveFile, setFileModel]);

  // Show loading state while Monaco initializes
  if (!isReady) {
    return (
      <div
        className={`monaco-editor-container w-full h-full overflow-hidden flex items-center justify-center bg-neutral-950 ${className}`}
      >
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-neutral-400">
            Initializing editor...
          </span>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={`monaco-editor-container w-full h-full overflow-hidden ${className}`}
      style={style}
    />
  );
};

MonacoEditor.displayName = "MonacoEditor";
