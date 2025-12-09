/**
 * Quick Open Component
 *
 * VS Code-style quick file picker with fuzzy search.
 * Invoked with Ctrl+P (or Cmd+P on Mac).
 *
 * Features:
 * - Fuzzy file search
 * - Recent files prioritized
 * - Go to line with ':'
 * - Go to symbol with '@'
 * - Preview on hover
 *
 * @module @neurectomy/components/shell/QuickOpen
 * @author @APEX @CANVAS
 */

import React, {
  useState,
  useEffect,
  useRef,
  useMemo,
  useCallback,
} from "react";
import { createPortal } from "react-dom";
import { cn } from "@/lib/utils";
import {
  FileText,
  FileJson,
  FileCode2,
  Image,
  Settings,
  FileType,
  Folder,
  Clock,
  Hash,
  type LucideIcon,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface FileItem {
  path: string;
  name: string;
  folder?: string;
  lastOpened?: Date;
  icon?: LucideIcon;
}

export interface QuickOpenProps {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (item: FileItem) => void;
  files: FileItem[];
  recentFiles?: FileItem[];
  placeholder?: string;
  className?: string;
}

// ============================================================================
// File Icon Mapping
// ============================================================================

const FILE_ICONS: Record<string, LucideIcon> = {
  ".ts": FileCode2,
  ".tsx": FileCode2,
  ".js": FileCode2,
  ".jsx": FileCode2,
  ".json": FileJson,
  ".md": FileText,
  ".txt": FileText,
  ".yaml": FileText,
  ".yml": FileText,
  ".toml": Settings,
  ".png": Image,
  ".jpg": Image,
  ".jpeg": Image,
  ".gif": Image,
  ".svg": Image,
  ".webp": Image,
};

function getFileIcon(filename: string): LucideIcon {
  const ext = filename.slice(filename.lastIndexOf("."));
  return FILE_ICONS[ext.toLowerCase()] || FileType;
}

// ============================================================================
// Fuzzy Search
// ============================================================================

function fuzzyMatch(
  pattern: string,
  str: string
): { match: boolean; score: number; indices: number[] } {
  const patternLower = pattern.toLowerCase();
  const strLower = str.toLowerCase();

  let patternIdx = 0;
  let strIdx = 0;
  const indices: number[] = [];
  let score = 0;
  let consecutiveBonus = 0;

  while (patternIdx < patternLower.length && strIdx < strLower.length) {
    if (patternLower[patternIdx] === strLower[strIdx]) {
      indices.push(strIdx);
      score += 1;

      if (
        indices.length > 1 &&
        indices[indices.length - 1] === indices[indices.length - 2] + 1
      ) {
        consecutiveBonus += 2;
      }

      // Bonus for matching at filename start
      if (strIdx === 0) {
        score += 5;
      }

      // Bonus for path separators
      if (strIdx > 0 && /[/\\]/.test(str[strIdx - 1])) {
        score += 3;
      }

      patternIdx++;
    }
    strIdx++;
  }

  if (patternIdx !== patternLower.length) {
    return { match: false, score: 0, indices: [] };
  }

  score += consecutiveBonus;
  score -= str.length * 0.05;

  return { match: true, score, indices };
}

// ============================================================================
// Highlight Component
// ============================================================================

function HighlightedText({
  text,
  indices,
  className,
}: {
  text: string;
  indices: number[];
  className?: string;
}) {
  if (indices.length === 0) {
    return <span className={className}>{text}</span>;
  }

  const chars = text.split("");
  const indexSet = new Set(indices);

  return (
    <span className={className}>
      {chars.map((char, i) =>
        indexSet.has(i) ? (
          <span
            key={i}
            className="text-accent-foreground bg-accent/30 font-semibold"
          >
            {char}
          </span>
        ) : (
          <span key={i}>{char}</span>
        )
      )}
    </span>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function QuickOpen({
  isOpen,
  onClose,
  onSelect,
  files,
  recentFiles = [],
  placeholder = "Search files by name (use : for line, @ for symbol)",
  className,
}: QuickOpenProps) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [mode, setMode] = useState<"files" | "line" | "symbol">("files");
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Detect mode from query
  useEffect(() => {
    if (query.startsWith(":")) {
      setMode("line");
    } else if (query.startsWith("@")) {
      setMode("symbol");
    } else {
      setMode("files");
    }
  }, [query]);

  // Filter and sort files
  const filteredFiles = useMemo(() => {
    if (mode !== "files") {
      return [];
    }

    const searchQuery = query.trim();

    if (!searchQuery) {
      // Show recent files first
      const recentPaths = new Set(recentFiles.map((f) => f.path));
      const recent = recentFiles.map((file) => ({
        file,
        score: 1000 + (file.lastOpened?.getTime() || 0) / 1000000,
        indices: [],
      }));
      const others = files
        .filter((f) => !recentPaths.has(f.path))
        .slice(0, 20)
        .map((file) => ({ file, score: 0, indices: [] }));

      return [...recent, ...others];
    }

    // Fuzzy search
    return files
      .map((file) => {
        // Search by filename
        const nameMatch = fuzzyMatch(searchQuery, file.name);
        // Also search by full path
        const pathMatch = fuzzyMatch(searchQuery, file.path);

        const bestMatch =
          nameMatch.score > pathMatch.score ? nameMatch : pathMatch;

        return {
          file,
          score: bestMatch.score,
          indices: nameMatch.score > 0 ? nameMatch.indices : [],
        };
      })
      .filter((result) => result.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 50);
  }, [files, recentFiles, query, mode]);

  // Reset selection when results change
  useEffect(() => {
    setSelectedIndex(0);
  }, [filteredFiles.length]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setQuery("");
      setSelectedIndex(0);
      setMode("files");
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  // Scroll selected item into view
  useEffect(() => {
    const list = listRef.current;
    if (!list) return;

    const selectedItem = list.children[selectedIndex] as
      | HTMLElement
      | undefined;
    if (selectedItem) {
      selectedItem.scrollIntoView({ block: "nearest" });
    }
  }, [selectedIndex]);

  // Keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((i) => Math.min(i + 1, filteredFiles.length - 1));
          break;
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((i) => Math.max(i - 1, 0));
          break;
        case "Enter":
          e.preventDefault();
          if (mode === "line") {
            const lineNumber = parseInt(query.slice(1), 10);
            if (!isNaN(lineNumber)) {
              // TODO: Implement go to line
              console.log("Go to line:", lineNumber);
            }
            onClose();
          } else if (mode === "symbol") {
            // TODO: Implement go to symbol
            console.log("Go to symbol:", query.slice(1));
            onClose();
          } else if (filteredFiles[selectedIndex]) {
            onSelect(filteredFiles[selectedIndex].file);
            onClose();
          }
          break;
        case "Escape":
          e.preventDefault();
          onClose();
          break;
        case "Tab":
          e.preventDefault();
          if (e.shiftKey) {
            setSelectedIndex((i) => Math.max(i - 1, 0));
          } else {
            setSelectedIndex((i) => Math.min(i + 1, filteredFiles.length - 1));
          }
          break;
      }
    },
    [filteredFiles, selectedIndex, onClose, onSelect, mode, query]
  );

  // Execute selection
  const selectFile = useCallback(
    (file: FileItem) => {
      onSelect(file);
      onClose();
    },
    [onSelect, onClose]
  );

  if (!isOpen) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]"
      onClick={onClose}
    >
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" />

      {/* Quick Open */}
      <div
        className={cn(
          "relative w-full max-w-2xl",
          "bg-card border border-border rounded-lg shadow-2xl",
          "overflow-hidden",
          className
        )}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search Input */}
        <div className="flex items-center gap-2 px-4 py-3 border-b border-border">
          {mode === "files" && (
            <FileText size={16} className="text-muted-foreground" />
          )}
          {mode === "line" && (
            <Hash size={16} className="text-muted-foreground" />
          )}
          {mode === "symbol" && (
            <FileCode2 size={16} className="text-muted-foreground" />
          )}
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className={cn(
              "flex-1 bg-transparent border-none outline-none",
              "text-sm text-foreground placeholder:text-muted-foreground"
            )}
            autoComplete="off"
            autoCorrect="off"
            autoCapitalize="off"
            spellCheck={false}
          />
          <kbd className="px-2 py-0.5 text-xs bg-muted rounded border border-border">
            ESC
          </kbd>
        </div>

        {/* Mode-specific content */}
        {mode === "line" && (
          <div className="px-4 py-8 text-center text-sm text-muted-foreground">
            Type a line number and press Enter to go to that line
          </div>
        )}

        {mode === "symbol" && (
          <div className="px-4 py-8 text-center text-sm text-muted-foreground">
            Type a symbol name to navigate to it
          </div>
        )}

        {mode === "files" && (
          <>
            {/* Results */}
            <div ref={listRef} className="max-h-80 overflow-y-auto">
              {filteredFiles.length === 0 ? (
                <div className="px-4 py-8 text-center text-sm text-muted-foreground">
                  No files found
                </div>
              ) : (
                filteredFiles.map((result, index) => (
                  <FileItemRow
                    key={result.file.path}
                    file={result.file}
                    indices={result.indices}
                    selected={index === selectedIndex}
                    isRecent={recentFiles.some(
                      (f) => f.path === result.file.path
                    )}
                    onClick={() => selectFile(result.file)}
                    onMouseEnter={() => setSelectedIndex(index)}
                  />
                ))
              )}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between px-4 py-2 border-t border-border bg-muted/30">
              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  <kbd className="px-1 py-0.5 bg-muted rounded">:</kbd>
                  go to line
                </span>
                <span className="flex items-center gap-1">
                  <kbd className="px-1 py-0.5 bg-muted rounded">@</kbd>
                  go to symbol
                </span>
              </div>
              <div className="text-xs text-muted-foreground">
                {filteredFiles.length} files
              </div>
            </div>
          </>
        )}
      </div>
    </div>,
    document.body
  );
}

// ============================================================================
// File Item Row
// ============================================================================

function FileItemRow({
  file,
  indices,
  selected,
  isRecent,
  onClick,
  onMouseEnter,
}: {
  file: FileItem;
  indices: number[];
  selected: boolean;
  isRecent: boolean;
  onClick: () => void;
  onMouseEnter: () => void;
}) {
  const Icon = file.icon || getFileIcon(file.name);

  return (
    <button
      className={cn(
        "w-full flex items-center gap-3 px-4 py-2 text-left",
        "transition-colors",
        selected ? "bg-accent text-accent-foreground" : "hover:bg-muted/50"
      )}
      onClick={onClick}
      onMouseEnter={onMouseEnter}
    >
      <Icon
        size={16}
        className={cn(
          selected ? "text-accent-foreground" : "text-muted-foreground"
        )}
      />

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <HighlightedText
            text={file.name}
            indices={indices}
            className="text-sm truncate"
          />
          {isRecent && (
            <Clock size={12} className="text-muted-foreground flex-shrink-0" />
          )}
        </div>
        {file.folder && (
          <div className="text-xs text-muted-foreground truncate">
            {file.folder}
          </div>
        )}
      </div>
    </button>
  );
}

// ============================================================================
// Hook for Quick Open
// ============================================================================

export function useQuickOpen() {
  const [isOpen, setIsOpen] = useState(false);
  const [recentFiles, setRecentFiles] = useState<FileItem[]>([]);

  // Global keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "p" && !e.shiftKey) {
        e.preventDefault();
        setIsOpen((prev) => !prev);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const open = useCallback(() => setIsOpen(true), []);
  const close = useCallback(() => setIsOpen(false), []);

  const trackFile = useCallback((file: FileItem) => {
    setRecentFiles((prev) => {
      const filtered = prev.filter((f) => f.path !== file.path);
      return [{ ...file, lastOpened: new Date() }, ...filtered].slice(0, 10);
    });
  }, []);

  return {
    isOpen,
    open,
    close,
    recentFiles,
    trackFile,
  };
}

export default QuickOpen;
