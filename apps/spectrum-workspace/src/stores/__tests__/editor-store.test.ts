/**
 * Editor Store Tests
 * Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
 */

import { describe, it, expect, beforeEach, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useEditorStore } from "../editor-store";
import type { EditorFile } from "@neurectomy/types/editor";

describe("EditorStore", () => {
  beforeEach(() => {
    // Reset store before each test
    const { result } = renderHook(() => useEditorStore());
    act(() => {
      result.current.closeAllFiles();
    });
  });

  describe("File Management", () => {
    it("should open a new file", () => {
      const { result } = renderHook(() => useEditorStore());

      act(() => {
        result.current.openFile({
          path: "/test/file.ts",
          name: "file.ts",
          content: "const x = 1;",
          language: "typescript",
        });
      });

      expect(result.current.openFiles).toHaveLength(1);
      expect(result.current.openFiles[0].name).toBe("file.ts");
      expect(result.current.openFiles[0].content).toBe("const x = 1;");
      expect(result.current.openFiles[0].language).toBe("typescript");
      expect(result.current.openFiles[0].isDirty).toBe(false);
    });

    it("should not duplicate files with same path", () => {
      const { result } = renderHook(() => useEditorStore());

      act(() => {
        result.current.openFile({
          path: "/test/file.ts",
          name: "file.ts",
          content: "const x = 1;",
          language: "typescript",
        });
        result.current.openFile({
          path: "/test/file.ts",
          name: "file.ts",
          content: "const x = 2;",
          language: "typescript",
        });
      });

      expect(result.current.openFiles).toHaveLength(1);
    });

    it("should set active file when opening", () => {
      const { result } = renderHook(() => useEditorStore());

      act(() => {
        result.current.openFile({
          path: "/test/file.ts",
          name: "file.ts",
          content: "const x = 1;",
          language: "typescript",
        });
      });

      expect(result.current.activeFileId).toBe(result.current.openFiles[0].id);
    });

    it("should close a file", () => {
      const { result } = renderHook(() => useEditorStore());

      let fileId: string;

      act(() => {
        result.current.openFile({
          path: "/test/file.ts",
          name: "file.ts",
          content: "const x = 1;",
          language: "typescript",
        });
        fileId = result.current.openFiles[0].id;
      });

      act(() => {
        result.current.closeFile(fileId!);
      });

      expect(result.current.openFiles).toHaveLength(0);
      expect(result.current.activeFileId).toBeNull();
    });

    it("should switch active file after closing current", () => {
      const { result } = renderHook(() => useEditorStore());

      let file1Id: string;
      let file2Id: string;

      act(() => {
        result.current.openFile({
          path: "/test/file1.ts",
          name: "file1.ts",
          content: "",
          language: "typescript",
        });
        file1Id = result.current.openFiles[0].id;

        result.current.openFile({
          path: "/test/file2.ts",
          name: "file2.ts",
          content: "",
          language: "typescript",
        });
        file2Id = result.current.openFiles[1].id;
      });

      // Close second file (active)
      act(() => {
        result.current.closeFile(file2Id!);
      });

      expect(result.current.openFiles).toHaveLength(1);
      expect(result.current.activeFileId).toBe(file1Id);
    });

    it("should close all files", () => {
      const { result } = renderHook(() => useEditorStore());

      act(() => {
        result.current.openFile({
          path: "/test/file1.ts",
          name: "file1.ts",
          content: "",
          language: "typescript",
        });
        result.current.openFile({
          path: "/test/file2.ts",
          name: "file2.ts",
          content: "",
          language: "typescript",
        });
      });

      act(() => {
        result.current.closeAllFiles();
      });

      expect(result.current.openFiles).toHaveLength(0);
      expect(result.current.activeFileId).toBeNull();
    });

    it("should update file content", () => {
      const { result } = renderHook(() => useEditorStore());

      let fileId: string;

      act(() => {
        result.current.openFile({
          path: "/test/file.ts",
          name: "file.ts",
          content: "const x = 1;",
          language: "typescript",
        });
        fileId = result.current.openFiles[0].id;
      });

      act(() => {
        result.current.updateFileContent(fileId!, "const x = 2;");
      });

      const file = result.current.openFiles[0];
      expect(file.content).toBe("const x = 2;");
      expect(file.isDirty).toBe(true);
    });

    it("should mark file as dirty", () => {
      const { result } = renderHook(() => useEditorStore());

      let fileId: string;

      act(() => {
        result.current.openFile({
          path: "/test/file.ts",
          name: "file.ts",
          content: "const x = 1;",
          language: "typescript",
        });
        fileId = result.current.openFiles[0].id;
      });

      act(() => {
        result.current.markFileDirty(fileId!, true);
      });

      expect(result.current.openFiles[0].isDirty).toBe(true);

      act(() => {
        result.current.markFileDirty(fileId!, false);
      });

      expect(result.current.openFiles[0].isDirty).toBe(false);
    });

    it("should save file and mark as clean", async () => {
      const { result } = renderHook(() => useEditorStore());

      let fileId: string;

      act(() => {
        result.current.openFile({
          path: "/test/file.ts",
          name: "file.ts",
          content: "const x = 1;",
          language: "typescript",
        });
        fileId = result.current.openFiles[0].id;
        result.current.markFileDirty(fileId, true);
      });

      expect(result.current.openFiles[0].isDirty).toBe(true);

      await act(async () => {
        await result.current.saveFile(fileId!);
      });

      expect(result.current.openFiles[0].isDirty).toBe(false);
    });

    it("should save all dirty files", async () => {
      const { result } = renderHook(() => useEditorStore());

      let file1Id: string;
      let file2Id: string;

      act(() => {
        result.current.openFile({
          path: "/test/file1.ts",
          name: "file1.ts",
          content: "",
          language: "typescript",
        });
        file1Id = result.current.openFiles[0].id;

        result.current.openFile({
          path: "/test/file2.ts",
          name: "file2.ts",
          content: "",
          language: "typescript",
        });
        file2Id = result.current.openFiles[1].id;

        result.current.markFileDirty(file1Id, true);
        result.current.markFileDirty(file2Id, true);
      });

      await act(async () => {
        await result.current.saveAllFiles();
      });

      expect(result.current.openFiles[0].isDirty).toBe(false);
      expect(result.current.openFiles[1].isDirty).toBe(false);
    });
  });

  describe("Active File Management", () => {
    it("should set active file", () => {
      const { result } = renderHook(() => useEditorStore());

      let file1Id: string;
      let file2Id: string;

      act(() => {
        result.current.openFile({
          path: "/test/file1.ts",
          name: "file1.ts",
          content: "",
          language: "typescript",
        });
        file1Id = result.current.openFiles[0].id;

        result.current.openFile({
          path: "/test/file2.ts",
          name: "file2.ts",
          content: "",
          language: "typescript",
        });
        file2Id = result.current.openFiles[1].id;
      });

      expect(result.current.activeFileId).toBe(file2Id);

      act(() => {
        result.current.setActiveFile(file1Id!);
      });

      expect(result.current.activeFileId).toBe(file1Id);
    });

    it("should get active file", () => {
      const { result } = renderHook(() => useEditorStore());

      act(() => {
        result.current.openFile({
          path: "/test/file.ts",
          name: "file.ts",
          content: "const x = 1;",
          language: "typescript",
        });
      });

      const activeFile = result.current.getActiveFile();
      expect(activeFile).toBeDefined();
      expect(activeFile?.name).toBe("file.ts");
    });

    it("should return undefined when no active file", () => {
      const { result } = renderHook(() => useEditorStore());

      const activeFile = result.current.getActiveFile();
      expect(activeFile).toBeUndefined();
    });
  });

  describe("Configuration", () => {
    it("should update editor configuration", () => {
      const { result } = renderHook(() => useEditorStore());

      act(() => {
        result.current.updateConfig({
          fontSize: 16,
          theme: "vs-light",
        });
      });

      expect(result.current.config.fontSize).toBe(16);
      expect(result.current.config.theme).toBe("vs-light");
    });

    it("should persist configuration", () => {
      const { result: result1 } = renderHook(() => useEditorStore());

      act(() => {
        result1.current.updateConfig({
          fontSize: 18,
          tabSize: 4,
        });
      });

      // Create new instance (simulates page reload)
      const { result: result2 } = renderHook(() => useEditorStore());

      expect(result2.current.config.fontSize).toBe(18);
      expect(result2.current.config.tabSize).toBe(4);
    });
  });

  describe("Selectors", () => {
    it("should get file by ID", () => {
      const { result } = renderHook(() => useEditorStore());

      let fileId: string;

      act(() => {
        result.current.openFile({
          path: "/test/file.ts",
          name: "file.ts",
          content: "const x = 1;",
          language: "typescript",
        });
        fileId = result.current.openFiles[0].id;
      });

      const file = result.current.getFile(fileId!);
      expect(file).toBeDefined();
      expect(file?.name).toBe("file.ts");
    });

    it("should return undefined for non-existent file", () => {
      const { result } = renderHook(() => useEditorStore());

      const file = result.current.getFile("non-existent");
      expect(file).toBeUndefined();
    });
  });
});
