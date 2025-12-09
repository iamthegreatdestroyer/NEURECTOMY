/**
 * Breadcrumb Navigation Component
 *
 * VS Code-style breadcrumb navigation showing file path hierarchy.
 * Supports navigation to parent folders and symbol navigation within files.
 *
 * Features:
 * - File path breadcrumbs with clickable segments
 * - Symbol/outline navigation (classes, functions, etc.)
 * - Dropdown menus for each segment
 * - Keyboard navigation
 * - Integration with editor context
 *
 * @module @neurectomy/components/shell/BreadcrumbNavigation
 * @author @APEX @CANVAS
 */

import React, {
  useState,
  useRef,
  useEffect,
  useMemo,
  useCallback,
} from "react";
import { cn } from "@/lib/utils";
import {
  ChevronRight,
  Folder,
  FileText,
  FileCode2,
  FileJson,
  Settings,
  Image,
  Hash,
  Braces,
  Box,
  Variable,
  type LucideIcon,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface PathSegment {
  type: "folder" | "file";
  name: string;
  path: string;
}

export interface SymbolSegment {
  type: "symbol";
  kind: SymbolKind;
  name: string;
  range?: { startLine: number; endLine: number };
  children?: SymbolSegment[];
}

export type SymbolKind =
  | "class"
  | "interface"
  | "function"
  | "method"
  | "variable"
  | "constant"
  | "property"
  | "namespace"
  | "module";

export type BreadcrumbSegment = PathSegment | SymbolSegment;

export interface BreadcrumbNavigationProps {
  filePath?: string;
  symbols?: SymbolSegment[];
  onNavigate?: (segment: BreadcrumbSegment) => void;
  onPathClick?: (path: string) => void;
  onSymbolClick?: (symbol: SymbolSegment) => void;
  siblings?: Record<string, string[]>; // path -> sibling names
  className?: string;
}

// ============================================================================
// Icon Mappings
// ============================================================================

const FILE_ICONS: Record<string, LucideIcon> = {
  ".ts": FileCode2,
  ".tsx": FileCode2,
  ".js": FileCode2,
  ".jsx": FileCode2,
  ".json": FileJson,
  ".md": FileText,
  ".yaml": Settings,
  ".yml": Settings,
  ".toml": Settings,
  ".png": Image,
  ".jpg": Image,
  ".svg": Image,
};

const SYMBOL_ICONS: Record<SymbolKind, LucideIcon> = {
  class: Box,
  interface: Box,
  function: Braces,
  method: Braces,
  variable: Variable,
  constant: Hash,
  property: Variable,
  namespace: Folder,
  module: Folder,
};

function getFileIcon(filename: string): LucideIcon {
  const ext = filename.slice(filename.lastIndexOf("."));
  return FILE_ICONS[ext.toLowerCase()] || FileText;
}

// ============================================================================
// Path Parsing
// ============================================================================

function parsePath(filePath: string): PathSegment[] {
  const normalized = filePath.replace(/\\/g, "/");
  const parts = normalized.split("/").filter(Boolean);

  if (parts.length === 0) return [];

  const segments: PathSegment[] = [];
  let currentPath = "";

  parts.forEach((part, index) => {
    currentPath += (currentPath ? "/" : "") + part;
    const isFile = index === parts.length - 1;

    segments.push({
      type: isFile ? "file" : "folder",
      name: part,
      path: currentPath,
    });
  });

  return segments;
}

// ============================================================================
// Main Component
// ============================================================================

export function BreadcrumbNavigation({
  filePath,
  symbols = [],
  onNavigate,
  onPathClick,
  onSymbolClick,
  siblings = {},
  className,
}: BreadcrumbNavigationProps) {
  const [openDropdown, setOpenDropdown] = useState<number | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Parse file path into segments
  const pathSegments = useMemo(() => {
    return filePath ? parsePath(filePath) : [];
  }, [filePath]);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setOpenDropdown(null);
      }
    };

    document.addEventListener("click", handleClick);
    return () => document.removeEventListener("click", handleClick);
  }, []);

  // Handle segment click
  const handleSegmentClick = useCallback(
    (segment: BreadcrumbSegment, index: number) => {
      if (onNavigate) {
        onNavigate(segment);
      }

      if (segment.type === "folder" || segment.type === "file") {
        if (onPathClick) {
          onPathClick(segment.path);
        }
      } else if (segment.type === "symbol") {
        if (onSymbolClick) {
          onSymbolClick(segment);
        }
      }
    },
    [onNavigate, onPathClick, onSymbolClick]
  );

  // Handle dropdown toggle
  const toggleDropdown = useCallback((index: number, e: React.MouseEvent) => {
    e.stopPropagation();
    setOpenDropdown((prev) => (prev === index ? null : index));
  }, []);

  if (!filePath && symbols.length === 0) {
    return (
      <div
        className={cn("px-3 py-1.5 text-xs text-muted-foreground", className)}
      >
        No file open
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={cn(
        "flex items-center gap-0.5 px-2 py-1 overflow-x-auto",
        "text-sm",
        "scrollbar-thin scrollbar-thumb-muted",
        className
      )}
    >
      {/* Path segments */}
      {pathSegments.map((segment, index) => (
        <React.Fragment key={`path-${index}`}>
          {index > 0 && (
            <ChevronRight
              size={14}
              className="text-muted-foreground flex-shrink-0"
            />
          )}
          <BreadcrumbSegmentItem
            segment={segment}
            index={index}
            isOpen={openDropdown === index}
            onToggle={(e) => toggleDropdown(index, e)}
            onClick={() => handleSegmentClick(segment, index)}
            siblings={siblings[segment.path]}
            onSiblingSelect={(sibling) => {
              const parentPath = segment.path.substring(
                0,
                segment.path.lastIndexOf("/")
              );
              const newPath = parentPath ? `${parentPath}/${sibling}` : sibling;
              if (onPathClick) onPathClick(newPath);
              setOpenDropdown(null);
            }}
          />
        </React.Fragment>
      ))}

      {/* Symbol segments */}
      {symbols.map((symbol, index) => (
        <React.Fragment key={`symbol-${index}`}>
          <ChevronRight
            size={14}
            className="text-muted-foreground flex-shrink-0"
          />
          <BreadcrumbSymbolItem
            symbol={symbol}
            index={pathSegments.length + index}
            isOpen={openDropdown === pathSegments.length + index}
            onToggle={(e) => toggleDropdown(pathSegments.length + index, e)}
            onClick={() =>
              handleSegmentClick(symbol, pathSegments.length + index)
            }
            onChildSelect={(child) => {
              if (onSymbolClick) onSymbolClick(child);
              setOpenDropdown(null);
            }}
          />
        </React.Fragment>
      ))}
    </div>
  );
}

// ============================================================================
// Segment Item
// ============================================================================

interface BreadcrumbSegmentItemProps {
  segment: PathSegment;
  index: number;
  isOpen: boolean;
  onToggle: (e: React.MouseEvent) => void;
  onClick: () => void;
  siblings?: string[];
  onSiblingSelect?: (name: string) => void;
}

function BreadcrumbSegmentItem({
  segment,
  index,
  isOpen,
  onToggle,
  onClick,
  siblings,
  onSiblingSelect,
}: BreadcrumbSegmentItemProps) {
  const Icon = segment.type === "folder" ? Folder : getFileIcon(segment.name);
  const dropdownRef = useRef<HTMLDivElement>(null);

  return (
    <div className="relative flex-shrink-0">
      <button
        className={cn(
          "flex items-center gap-1 px-1.5 py-0.5 rounded",
          "text-muted-foreground hover:text-foreground",
          "hover:bg-muted/50 transition-colors",
          isOpen && "bg-muted/50 text-foreground"
        )}
        onClick={onClick}
        onContextMenu={siblings?.length ? onToggle : undefined}
      >
        <Icon size={14} />
        <span className="max-w-32 truncate">{segment.name}</span>
        {siblings && siblings.length > 1 && (
          <button
            className="ml-0.5 p-0.5 rounded hover:bg-muted"
            onClick={onToggle}
            aria-label="Show siblings"
          >
            <ChevronRight
              size={10}
              className={cn("transition-transform", isOpen && "rotate-90")}
            />
          </button>
        )}
      </button>

      {/* Sibling dropdown */}
      {isOpen && siblings && siblings.length > 0 && (
        <div
          ref={dropdownRef}
          className={cn(
            "absolute top-full left-0 mt-1 z-50",
            "min-w-40 max-h-64 overflow-y-auto",
            "bg-card border border-border rounded-md shadow-lg",
            "py-1"
          )}
        >
          {siblings.map((sibling) => (
            <button
              key={sibling}
              className={cn(
                "w-full flex items-center gap-2 px-3 py-1.5 text-left",
                "text-sm hover:bg-muted/50 transition-colors",
                sibling === segment.name && "bg-accent/20"
              )}
              onClick={() => onSiblingSelect?.(sibling)}
            >
              <Icon size={14} className="text-muted-foreground" />
              <span className="truncate">{sibling}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Symbol Item
// ============================================================================

interface BreadcrumbSymbolItemProps {
  symbol: SymbolSegment;
  index: number;
  isOpen: boolean;
  onToggle: (e: React.MouseEvent) => void;
  onClick: () => void;
  onChildSelect?: (child: SymbolSegment) => void;
}

function BreadcrumbSymbolItem({
  symbol,
  index,
  isOpen,
  onToggle,
  onClick,
  onChildSelect,
}: BreadcrumbSymbolItemProps) {
  const Icon = SYMBOL_ICONS[symbol.kind] || Hash;

  return (
    <div className="relative flex-shrink-0">
      <button
        className={cn(
          "flex items-center gap-1 px-1.5 py-0.5 rounded",
          "text-muted-foreground hover:text-foreground",
          "hover:bg-muted/50 transition-colors",
          isOpen && "bg-muted/50 text-foreground"
        )}
        onClick={onClick}
      >
        <Icon size={14} />
        <span className="max-w-32 truncate">{symbol.name}</span>
        {symbol.children && symbol.children.length > 0 && (
          <button
            className="ml-0.5 p-0.5 rounded hover:bg-muted"
            onClick={onToggle}
            aria-label="Show children"
          >
            <ChevronRight
              size={10}
              className={cn("transition-transform", isOpen && "rotate-90")}
            />
          </button>
        )}
      </button>

      {/* Children dropdown */}
      {isOpen && symbol.children && symbol.children.length > 0 && (
        <div
          className={cn(
            "absolute top-full left-0 mt-1 z-50",
            "min-w-40 max-h-64 overflow-y-auto",
            "bg-card border border-border rounded-md shadow-lg",
            "py-1"
          )}
        >
          {symbol.children.map((child, childIndex) => {
            const ChildIcon = SYMBOL_ICONS[child.kind] || Hash;
            return (
              <button
                key={childIndex}
                className={cn(
                  "w-full flex items-center gap-2 px-3 py-1.5 text-left",
                  "text-sm hover:bg-muted/50 transition-colors"
                )}
                onClick={() => onChildSelect?.(child)}
              >
                <ChildIcon size={14} className="text-muted-foreground" />
                <span className="truncate">{child.name}</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Hook for Symbol Extraction
// ============================================================================

/**
 * Hook to extract symbols from document content (placeholder for LSP integration)
 */
export function useDocumentSymbols(
  content: string,
  language: string
): SymbolSegment[] {
  return useMemo(() => {
    // Placeholder - in real implementation, this would use LSP
    // For now, extract basic patterns
    const symbols: SymbolSegment[] = [];

    if (["typescript", "javascript", "tsx", "jsx"].includes(language)) {
      // Extract function declarations
      const functionRegex = /(?:export\s+)?(?:async\s+)?function\s+(\w+)/g;
      let match;
      while ((match = functionRegex.exec(content)) !== null) {
        symbols.push({
          type: "symbol",
          kind: "function",
          name: match[1],
        });
      }

      // Extract class declarations
      const classRegex = /(?:export\s+)?class\s+(\w+)/g;
      while ((match = classRegex.exec(content)) !== null) {
        symbols.push({
          type: "symbol",
          kind: "class",
          name: match[1],
        });
      }

      // Extract interface declarations
      const interfaceRegex = /(?:export\s+)?interface\s+(\w+)/g;
      while ((match = interfaceRegex.exec(content)) !== null) {
        symbols.push({
          type: "symbol",
          kind: "interface",
          name: match[1],
        });
      }
    }

    return symbols;
  }, [content, language]);
}

export default BreadcrumbNavigation;
