/**
 * Editor Store - Zustand state management for Monaco Editor
 * Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
 */

import { create } from "zustand";
import { persist, devtools } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import type * as Monaco from "monaco-editor";
import type {
  EditorStore,
  EditorFile,
  EditorConfig,
} from "@neurectomy/types";
import { generateFileId, detectLanguage, DEFAULT_EDITOR_CONFIG } from "@neurectomy/types";

/**
 * Editor Zustand Store
 *
 * Manages all editor state including open files, active file,
 * configuration, and Monaco editor instance.
 */
export const useEditorStore = create<EditorStore>()(
  devtools(
    persist(
      immer((set, get) => ({
        // =====================================================================
        // State
        // =====================================================================
        openFiles: [],
        activeFileId: null,
        config: DEFAULT_EDITOR_CONFIG as EditorConfig,
        editor: null,

        // =====================================================================
        // File Management Actions
        // =====================================================================

        /**
         * Open a file in the editor
         */
        openFile: (file) => {
          set((state) => {
            // Check if file is already open
            const existingFile = state.openFiles.find(
              (f) => f.path === file.path
            );

            if (existingFile) {
              // File already open, just activate it
              state.activeFileId = existingFile.id;
              return;
            }

            // Create new file entry
            const newFile: EditorFile = {
              ...file,
              id: generateFileId(),
              isDirty: false,
              lastModified: Date.now(),
              language: file.language || detectLanguage(file.name),
            };

            // Add to open files
            state.openFiles.push(newFile);
            state.activeFileId = newFile.id;
          });
        },

        /**
         * Close a file
         */
        closeFile: (fileId) => {
          set((state) => {
            const index = state.openFiles.findIndex((f) => f.id === fileId);
            if (index === -1) return;

            // Dispose Monaco model if it exists
            const file = state.openFiles[index];
            if (file.model) {
              file.model.dispose();
            }

            // Remove file
            state.openFiles.splice(index, 1);

            // Update active file if needed
            if (state.activeFileId === fileId) {
              if (state.openFiles.length > 0) {
                // Activate previous file or first file
                const newIndex = Math.max(0, index - 1);
                state.activeFileId = state.openFiles[newIndex]?.id || null;
              } else {
                state.activeFileId = null;
              }
            }
          });
        },

        /**
         * Close all files
         */
        closeAllFiles: () => {
          set((state) => {
            // Dispose all Monaco models
            state.openFiles.forEach((file) => {
              if (file.model) {
                file.model.dispose();
              }
            });

            state.openFiles = [];
            state.activeFileId = null;
          });
        },

        /**
         * Set active file
         */
        setActiveFile: (fileId) => {
          set((state) => {
            const file = state.openFiles.find((f) => f.id === fileId);
            if (file) {
              state.activeFileId = fileId;
            }
          });
        },

        /**
         * Update file content
         */
        updateFileContent: (fileId, content) => {
          set((state) => {
            const file = state.openFiles.find((f) => f.id === fileId);
            if (file) {
              file.content = content;
              file.lastModified = Date.now();
              // Mark as dirty if content changed
              if (file.content !== content) {
                file.isDirty = true;
              }
            }
          });
        },

        /**
         * Mark file as dirty/clean
         */
        markFileDirty: (fileId, isDirty) => {
          set((state) => {
            const file = state.openFiles.find((f) => f.id === fileId);
            if (file) {
              file.isDirty = isDirty;
            }
          });
        },

        /**
         * Save file
         * TODO: Integrate with Tauri file system
         */
        saveFile: async (fileId) => {
          const file = get().openFiles.find((f) => f.id === fileId);
          if (!file) return;

          try {
            // TODO: Implement Tauri file save
            // await invoke('write_file', { path: file.path, content: file.content });

            console.log(`Saving file: ${file.path}`);

            set((state) => {
              const f = state.openFiles.find((f) => f.id === fileId);
              if (f) {
                f.isDirty = false;
                f.lastModified = Date.now();
              }
            });
          } catch (error) {
            console.error(`Failed to save file ${file.path}:`, error);
            throw error;
          }
        },

        /**
         * Save all files
         */
        saveAllFiles: async () => {
          const dirtyFiles = get().openFiles.filter((f) => f.isDirty);

          await Promise.all(dirtyFiles.map((file) => get().saveFile(file.id)));
        },

        // =====================================================================
        // Configuration Actions
        // =====================================================================

        /**
         * Update editor configuration
         */
        updateConfig: (config) => {
          set((state) => {
            state.config = { ...state.config, ...config };
          });
        },

        /**
         * Set Monaco editor instance
         */
        setEditor: (editor) => {
          set((state) => {
            state.editor = editor;
          });
        },

        /**
         * Update file view state (cursor position, scroll)
         */
        updateViewState: (fileId, viewState) => {
          set((state) => {
            const file = state.openFiles.find((f) => f.id === fileId);
            if (file) {
              file.viewState = viewState;
            }
          });
        },

        // =====================================================================
        // Selectors
        // =====================================================================

        /**
         * Get file by ID
         */
        getFile: (fileId) => {
          return get().openFiles.find((f) => f.id === fileId);
        },

        /**
         * Get active file
         */
        getActiveFile: () => {
          const { activeFileId, openFiles } = get();
          if (!activeFileId) return undefined;
          return openFiles.find((f) => f.id === activeFileId);
        },
      })),
      {
        name: "neurectomy-editor-storage",
        // Only persist configuration, not open files or editor instance
        partialize: (state) => ({
          config: state.config,
        }),
      }
    ),
    {
      name: "EditorStore",
      enabled: process.env.NODE_ENV === "development",
    }
  )
);

// =====================================================================
// Hooks for specific selectors
// =====================================================================

/**
 * Hook to get open files
 */
export const useOpenFiles = () => useEditorStore((state) => state.openFiles);

/**
 * Hook to get active file
 */
export const useActiveFile = () =>
  useEditorStore((state) => state.getActiveFile());

/**
 * Hook to get editor config
 */
export const useEditorConfig = () => useEditorStore((state) => state.config);

/**
 * Hook to check if any files have unsaved changes
 */
export const useHasUnsavedChanges = () =>
  useEditorStore((state) => state.openFiles.some((f) => f.isDirty));

/**
 * Hook to get dirty files count
 */
export const useDirtyFilesCount = () =>
  useEditorStore((state) => state.openFiles.filter((f) => f.isDirty).length);
