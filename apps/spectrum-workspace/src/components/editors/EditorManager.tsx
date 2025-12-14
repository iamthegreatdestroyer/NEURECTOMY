/**
 * EditorManager Component - Multi-file tab management
 * Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
 */

import React, { useCallback, useEffect } from "react";
import { X, Save, FileText, Circle } from "lucide-react";
import { MonacoEditor } from "./MonacoEditor";
import {
  useEditorStore,
  useOpenFiles,
  useActiveFile,
} from "../../stores/editor-store";
import type { EditorFile } from "@neurectomy/types";

export interface EditorManagerProps {
  /** Optional className for styling */
  className?: string;
  /** Show save all button */
  showSaveAll?: boolean;
}

/**
 * EditorManager - Manages multiple open files with tabs
 *
 * Features:
 * - Tab bar with file names
 * - Close buttons for each tab
 * - Visual indicator for unsaved changes
 * - Active tab highlighting
 * - Save all functionality
 * - Empty state when no files open
 */
export const EditorManager: React.FC<EditorManagerProps> = ({
  className = "",
  showSaveAll = true,
}) => {
  const openFiles = useOpenFiles();
  const activeFile = useActiveFile();

  const { activeFileId, setActiveFile, closeFile, saveFile, saveAllFiles } =
    useEditorStore();

  /**
   * Handle tab click
   */
  const handleTabClick = useCallback(
    (fileId: string) => {
      setActiveFile(fileId);
    },
    [setActiveFile]
  );

  /**
   * Handle close tab
   */
  const handleCloseTab = useCallback(
    (e: React.MouseEvent, fileId: string) => {
      e.stopPropagation();

      const file = openFiles.find((f) => f.id === fileId);
      if (!file) return;

      // TODO: Add confirmation dialog for unsaved changes
      if (file.isDirty) {
        const confirmed = window.confirm(
          `${file.name} has unsaved changes. Close anyway?`
        );
        if (!confirmed) return;
      }

      closeFile(fileId);
    },
    [openFiles, closeFile]
  );

  /**
   * Handle save current file
   */
  const handleSave = useCallback(async () => {
    if (!activeFile) return;

    try {
      await saveFile(activeFile.id);
    } catch (error) {
      console.error("Failed to save file:", error);
      // TODO: Show error toast
    }
  }, [activeFile, saveFile]);

  /**
   * Handle save all files
   */
  const handleSaveAll = useCallback(async () => {
    try {
      await saveAllFiles();
    } catch (error) {
      console.error("Failed to save all files:", error);
      // TODO: Show error toast
    }
  }, [saveAllFiles]);

  /**
   * Keyboard shortcuts
   */
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl/Cmd + S - Save current file
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        handleSave();
      }

      // Ctrl/Cmd + Shift + S - Save all files
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "s") {
        e.preventDefault();
        handleSaveAll();
      }

      // Ctrl/Cmd + W - Close current file
      if ((e.ctrlKey || e.metaKey) && e.key === "w" && activeFileId) {
        e.preventDefault();
        const file = openFiles.find((f) => f.id === activeFileId);
        if (file) {
          if (file.isDirty) {
            const confirmed = window.confirm(
              `${file.name} has unsaved changes. Close anyway?`
            );
            if (!confirmed) return;
          }
          closeFile(activeFileId);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [activeFileId, openFiles, closeFile, handleSave, handleSaveAll]);

  /**
   * Get language icon color
   */
  const getLanguageColor = (language: string): string => {
    const colorMap: Record<string, string> = {
      typescript: "text-blue-400",
      javascript: "text-yellow-400",
      python: "text-green-400",
      rust: "text-orange-400",
      json: "text-purple-400",
      yaml: "text-pink-400",
      markdown: "text-gray-400",
      html: "text-red-400",
      css: "text-cyan-400",
    };
    return colorMap[language] || "text-gray-400";
  };

  // Empty state
  if (openFiles.length === 0) {
    return (
      <div
        className={`flex items-center justify-center h-full bg-neutral-950 ${className}`}
      >
        <div className="text-center space-y-4">
          <FileText className="w-16 h-16 mx-auto text-neutral-600" />
          <div className="space-y-2">
            <h3 className="text-lg font-semibold text-neutral-300">
              No Files Open
            </h3>
            <p className="text-sm text-neutral-500">
              Open a file to start editing
            </p>
          </div>
        </div>
      </div>
    );
  }

  const hasUnsavedChanges = openFiles.some((f) => f.isDirty);

  return (
    <div className={`flex flex-col h-full bg-neutral-950 ${className}`}>
      {/* Tab Bar */}
      <div className="flex items-center bg-neutral-900 border-b border-neutral-800 overflow-x-auto scrollbar-thin scrollbar-thumb-neutral-700">
        <div className="flex flex-1 items-center min-w-0">
          {openFiles.map((file) => {
            const isActive = file.id === activeFileId;

            return (
              <div
                key={file.id}
                className="group flex items-center border-r border-neutral-800 min-w-0 flex-shrink-0"
              >
                <button
                  type="button"
                  onClick={() => handleTabClick(file.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      handleTabClick(file.id);
                    }
                  }}
                  className={`
                    flex items-center gap-2 px-4 py-2 hover:bg-neutral-800 transition-colors min-w-0 flex-shrink-0 text-left
                    ${isActive ? "bg-neutral-950 text-neutral-100" : "text-neutral-400"}
                  `}
                >
                  {/* Language indicator */}
                  <FileText
                    className={`w-4 h-4 flex-shrink-0 ${getLanguageColor(file.language)}`}
                  />

                  {/* File name */}
                  <span className="text-sm truncate max-w-[150px]">
                    {file.name}
                  </span>

                  {/* Dirty indicator */}
                  {file.isDirty && (
                    <Circle className="w-2 h-2 fill-cyan-400 text-cyan-400 flex-shrink-0" />
                  )}
                </button>

                {/* Close button */}
                <button
                  type="button"
                  onClick={(e) => handleCloseTab(e, file.id)}
                  className="flex-shrink-0 p-0.5 mr-2 rounded hover:bg-neutral-700 opacity-0 group-hover:opacity-100 transition-opacity"
                  title="Close"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            );
          })}
        </div>

        {/* Actions */}
        {showSaveAll && hasUnsavedChanges && (
          <button
            onClick={handleSaveAll}
            className="flex items-center gap-2 px-4 py-2 text-sm text-neutral-400 hover:text-neutral-100 hover:bg-neutral-800 transition-colors border-l border-neutral-800"
            title="Save All (Ctrl+Shift+S)"
          >
            <Save className="w-4 h-4" />
            <span>Save All</span>
          </button>
        )}
      </div>

      {/* Editor */}
      <div className="flex-1 min-h-0">
        <MonacoEditor />
      </div>

      {/* Status Bar */}
      {activeFile && (
        <div className="flex items-center justify-between px-4 py-1 bg-neutral-900 border-t border-neutral-800 text-xs text-neutral-400">
          <div className="flex items-center gap-4">
            <span className="font-mono">
              {activeFile.language.toUpperCase()}
            </span>
            <span>{activeFile.path}</span>
          </div>
          <div className="flex items-center gap-4">
            {activeFile.isDirty && (
              <span className="flex items-center gap-1 text-cyan-400">
                <Circle className="w-2 h-2 fill-current" />
                Unsaved
              </span>
            )}
            {activeFile.isReadOnly && (
              <span className="text-yellow-400">Read-Only</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

EditorManager.displayName = "EditorManager";
