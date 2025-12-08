/**
 * MonacoEditor Component - Core Monaco Editor Wrapper
 * Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
 */

import React, { useRef, useEffect, useCallback } from "react";
import * as monaco from "monaco-editor";
import { useEditorStore } from "../../stores/editor-store";
import type { EditorFile } from "@neurectomy/types/editor";

// Configure Monaco environment
self.MonacoEnvironment = {
  getWorker(_, label) {
    const getWorkerModule = (moduleUrl: string, label: string) => {
      return new Worker(
        self.MonacoEnvironment!.getWorkerUrl!(moduleUrl, label),
        {
          name: label,
          type: "module",
        }
      );
    };

    switch (label) {
      case "json":
        return getWorkerModule(
          "/monaco-editor/esm/vs/language/json/json.worker",
          label
        );
      case "css":
      case "scss":
      case "less":
        return getWorkerModule(
          "/monaco-editor/esm/vs/language/css/css.worker",
          label
        );
      case "html":
      case "handlebars":
      case "razor":
        return getWorkerModule(
          "/monaco-editor/esm/vs/language/html/html.worker",
          label
        );
      case "typescript":
      case "javascript":
        return getWorkerModule(
          "/monaco-editor/esm/vs/language/typescript/ts.worker",
          label
        );
      default:
        return getWorkerModule(
          "/monaco-editor/esm/vs/editor/editor.worker",
          label
        );
    }
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
  style = {},
  onReady,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null);
  const currentFileIdRef = useRef<string | null>(null);

  const {
    activeFileId,
    config,
    setEditor,
    updateFileContent,
    markFileDirty,
    updateViewState,
    getActiveFile,
    openFiles,
  } = useEditorStore();

  /**
   * Initialize Monaco editor instance
   */
  useEffect(() => {
    if (!containerRef.current) return;

    // Create editor
    const editor = monaco.editor.create(containerRef.current, {
      value: "",
      language: "typescript",
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
      automaticLayout: true,
      scrollBeyondLastLine: false,
      smoothScrolling: true,
      cursorBlinking: "smooth",
      cursorSmoothCaretAnimation: "on",
      padding: { top: 16, bottom: 16 },
      bracketPairColorization: {
        enabled: true,
      },
      suggest: {
        preview: true,
        showKeywords: true,
        showSnippets: true,
      },
      folding: true,
      foldingStrategy: "indentation",
      showFoldingControls: "mouseover",
      glyphMargin: true,
      lightbulb: {
        enabled: monaco.editor.ShowLightbulbIconMode.On,
      },
    });

    editorRef.current = editor;
    setEditor(editor);

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
      contentChangeDisposable.dispose();
      viewStateChangeDisposable.dispose();
      editor.dispose();
      setEditor(null);
    };
  }, []); // Only run once on mount

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
        const fileIndex = openFiles.findIndex((f) => f.id === activeFile.id);
        if (fileIndex !== -1) {
          openFiles[fileIndex].model = model;
        }
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
  }, [activeFileId, getActiveFile, openFiles]);

  return (
    <div
      ref={containerRef}
      className={`monaco-editor-container ${className}`}
      style={{
        width: "100%",
        height: "100%",
        overflow: "hidden",
        ...style,
      }}
    />
  );
};

MonacoEditor.displayName = "MonacoEditor";
