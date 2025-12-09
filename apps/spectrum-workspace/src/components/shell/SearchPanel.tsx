/**
 * Search Panel Component
 *
 * VS Code-style global search panel with file filtering and preview.
 * Supports regex, case sensitivity, and whole word matching.
 *
 * Features:
 * - Full-text search across workspace
 * - Include/exclude file patterns
 * - Regex support
 * - Case sensitive matching
 * - Whole word matching
 * - Search result grouping by file
 * - In-context preview
 * - Replace functionality
 *
 * @module @neurectomy/components/shell/SearchPanel
 * @author @APEX @CANVAS
 */

import React, {
  useState,
  useCallback,
  useMemo,
  useRef,
  useEffect,
} from "react";
import { cn } from "@/lib/utils";
import {
  Search,
  Replace,
  ChevronDown,
  ChevronRight,
  FileText,
  X,
  RefreshCw,
  MoreHorizontal,
  CaseSensitive,
  WholeWord,
  Regex,
  FolderOpen,
  Filter,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface SearchMatch {
  line: number;
  column: number;
  length: number;
  preview: string;
  previewStart: number;
}

export interface SearchFileResult {
  path: string;
  name: string;
  matches: SearchMatch[];
  collapsed?: boolean;
}

export interface SearchOptions {
  caseSensitive: boolean;
  wholeWord: boolean;
  useRegex: boolean;
  includePattern: string;
  excludePattern: string;
}

export interface SearchPanelProps {
  onSearch?: (
    query: string,
    options: SearchOptions
  ) => Promise<SearchFileResult[]>;
  onReplace?: (
    query: string,
    replacement: string,
    options: SearchOptions
  ) => Promise<number>;
  onReplaceAll?: (
    query: string,
    replacement: string,
    options: SearchOptions
  ) => Promise<number>;
  onOpenFile?: (path: string, line?: number, column?: number) => void;
  initialQuery?: string;
  className?: string;
}

// ============================================================================
// Main Component
// ============================================================================

export function SearchPanel({
  onSearch,
  onReplace,
  onReplaceAll,
  onOpenFile,
  initialQuery = "",
  className,
}: SearchPanelProps) {
  // Search state
  const [query, setQuery] = useState(initialQuery);
  const [replacement, setReplacement] = useState("");
  const [showReplace, setShowReplace] = useState(false);
  const [showFilters, setShowFilters] = useState(false);

  // Options state
  const [caseSensitive, setCaseSensitive] = useState(false);
  const [wholeWord, setWholeWord] = useState(false);
  const [useRegex, setUseRegex] = useState(false);
  const [includePattern, setIncludePattern] = useState("");
  const [excludePattern, setExcludePattern] = useState("**/node_modules/**");

  // Results state
  const [results, setResults] = useState<SearchFileResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [expandedFiles, setExpandedFiles] = useState<Set<string>>(new Set());

  // Refs
  const searchInputRef = useRef<HTMLInputElement>(null);
  const searchTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Computed values
  const totalMatches = useMemo(() => {
    return results.reduce((acc, file) => acc + file.matches.length, 0);
  }, [results]);

  const searchOptions: SearchOptions = useMemo(
    () => ({
      caseSensitive,
      wholeWord,
      useRegex,
      includePattern,
      excludePattern,
    }),
    [caseSensitive, wholeWord, useRegex, includePattern, excludePattern]
  );

  // Focus search on mount
  useEffect(() => {
    searchInputRef.current?.focus();
  }, []);

  // Debounced search
  useEffect(() => {
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    if (!query.trim()) {
      setResults([]);
      return;
    }

    searchTimeoutRef.current = setTimeout(() => {
      executeSearch();
    }, 300);

    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
    };
  }, [
    query,
    caseSensitive,
    wholeWord,
    useRegex,
    includePattern,
    excludePattern,
  ]);

  // Execute search
  const executeSearch = useCallback(async () => {
    if (!query.trim() || !onSearch) return;

    setIsSearching(true);
    try {
      const searchResults = await onSearch(query, searchOptions);
      setResults(searchResults);
      // Expand all files by default
      setExpandedFiles(new Set(searchResults.map((r) => r.path)));
    } catch (error) {
      console.error("Search failed:", error);
      setResults([]);
    } finally {
      setIsSearching(false);
    }
  }, [query, searchOptions, onSearch]);

  // Handle replace
  const handleReplace = useCallback(async () => {
    if (!query.trim() || !replacement || !onReplace) return;

    try {
      const count = await onReplace(query, replacement, searchOptions);
      console.log(`Replaced ${count} occurrence(s)`);
      executeSearch(); // Refresh results
    } catch (error) {
      console.error("Replace failed:", error);
    }
  }, [query, replacement, searchOptions, onReplace, executeSearch]);

  // Handle replace all
  const handleReplaceAll = useCallback(async () => {
    if (!query.trim() || !replacement || !onReplaceAll) return;

    try {
      const count = await onReplaceAll(query, replacement, searchOptions);
      console.log(`Replaced ${count} occurrence(s)`);
      executeSearch(); // Refresh results
    } catch (error) {
      console.error("Replace all failed:", error);
    }
  }, [query, replacement, searchOptions, onReplaceAll, executeSearch]);

  // Toggle file expansion
  const toggleFileExpansion = useCallback((path: string) => {
    setExpandedFiles((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  }, []);

  // Clear search
  const clearSearch = useCallback(() => {
    setQuery("");
    setResults([]);
    searchInputRef.current?.focus();
  }, []);

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Search Header */}
      <div className="flex-shrink-0 p-3 space-y-2 border-b border-border">
        {/* Search Input Row */}
        <div className="flex items-center gap-2">
          <button
            className={cn(
              "p-1 rounded transition-colors",
              showReplace ? "bg-muted" : "hover:bg-muted/50"
            )}
            onClick={() => setShowReplace(!showReplace)}
            aria-label={showReplace ? "Hide replace" : "Show replace"}
          >
            <ChevronRight
              size={14}
              className={cn("transition-transform", showReplace && "rotate-90")}
            />
          </button>

          <div className="flex-1 relative">
            <Search
              size={14}
              className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground"
            />
            <input
              ref={searchInputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search"
              className={cn(
                "w-full pl-8 pr-24 py-1.5",
                "bg-background border border-border rounded",
                "text-sm placeholder:text-muted-foreground",
                "focus:outline-none focus:ring-1 focus:ring-accent"
              )}
            />

            {/* Search options toggles */}
            <div className="absolute right-1 top-1/2 -translate-y-1/2 flex items-center gap-0.5">
              <OptionToggle
                active={caseSensitive}
                onClick={() => setCaseSensitive(!caseSensitive)}
                icon={CaseSensitive}
                tooltip="Match Case"
              />
              <OptionToggle
                active={wholeWord}
                onClick={() => setWholeWord(!wholeWord)}
                icon={WholeWord}
                tooltip="Match Whole Word"
              />
              <OptionToggle
                active={useRegex}
                onClick={() => setUseRegex(!useRegex)}
                icon={Regex}
                tooltip="Use Regular Expression"
              />
            </div>
          </div>

          {query && (
            <button
              className="p-1.5 rounded hover:bg-muted/50 text-muted-foreground"
              onClick={clearSearch}
              aria-label="Clear search"
            >
              <X size={14} />
            </button>
          )}
        </div>

        {/* Replace Input Row */}
        {showReplace && (
          <div className="flex items-center gap-2 pl-6">
            <div className="flex-1 relative">
              <Replace
                size={14}
                className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground"
              />
              <input
                type="text"
                value={replacement}
                onChange={(e) => setReplacement(e.target.value)}
                placeholder="Replace"
                className={cn(
                  "w-full pl-8 pr-2 py-1.5",
                  "bg-background border border-border rounded",
                  "text-sm placeholder:text-muted-foreground",
                  "focus:outline-none focus:ring-1 focus:ring-accent"
                )}
              />
            </div>

            <button
              className={cn(
                "px-2 py-1.5 text-xs rounded",
                "bg-muted hover:bg-muted/80 transition-colors",
                "disabled:opacity-50 disabled:cursor-not-allowed"
              )}
              onClick={handleReplace}
              disabled={!query || !replacement}
            >
              Replace
            </button>
            <button
              className={cn(
                "px-2 py-1.5 text-xs rounded",
                "bg-muted hover:bg-muted/80 transition-colors",
                "disabled:opacity-50 disabled:cursor-not-allowed"
              )}
              onClick={handleReplaceAll}
              disabled={!query || !replacement}
            >
              All
            </button>
          </div>
        )}

        {/* Filter Controls */}
        <div className="flex items-center gap-2 pl-6">
          <button
            className={cn(
              "flex items-center gap-1 px-2 py-1 text-xs rounded",
              "hover:bg-muted/50 transition-colors",
              showFilters && "bg-muted"
            )}
            onClick={() => setShowFilters(!showFilters)}
          >
            <Filter size={12} />
            <span>Filters</span>
          </button>

          {totalMatches > 0 && (
            <span className="text-xs text-muted-foreground">
              {totalMatches} result{totalMatches !== 1 ? "s" : ""} in{" "}
              {results.length} file{results.length !== 1 ? "s" : ""}
            </span>
          )}
        </div>

        {/* Filter Inputs */}
        {showFilters && (
          <div className="space-y-2 pl-6">
            <div className="relative">
              <FolderOpen
                size={14}
                className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground"
              />
              <input
                type="text"
                value={includePattern}
                onChange={(e) => setIncludePattern(e.target.value)}
                placeholder="files to include (e.g., *.ts, src/**)"
                className={cn(
                  "w-full pl-8 pr-2 py-1.5",
                  "bg-background border border-border rounded",
                  "text-xs placeholder:text-muted-foreground",
                  "focus:outline-none focus:ring-1 focus:ring-accent"
                )}
              />
            </div>
            <div className="relative">
              <X
                size={14}
                className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground"
              />
              <input
                type="text"
                value={excludePattern}
                onChange={(e) => setExcludePattern(e.target.value)}
                placeholder="files to exclude (e.g., **/node_modules/**)"
                className={cn(
                  "w-full pl-8 pr-2 py-1.5",
                  "bg-background border border-border rounded",
                  "text-xs placeholder:text-muted-foreground",
                  "focus:outline-none focus:ring-1 focus:ring-accent"
                )}
              />
            </div>
          </div>
        )}
      </div>

      {/* Results */}
      <div className="flex-1 overflow-y-auto">
        {isSearching ? (
          <div className="flex items-center justify-center py-8 text-muted-foreground">
            <RefreshCw size={16} className="animate-spin mr-2" />
            <span className="text-sm">Searching...</span>
          </div>
        ) : results.length === 0 ? (
          <div className="flex items-center justify-center py-8 text-muted-foreground text-sm">
            {query ? "No results found" : "Type to search"}
          </div>
        ) : (
          <div className="py-1">
            {results.map((file) => (
              <SearchFileResultItem
                key={file.path}
                file={file}
                expanded={expandedFiles.has(file.path)}
                onToggle={() => toggleFileExpansion(file.path)}
                onMatchClick={(match) =>
                  onOpenFile?.(file.path, match.line, match.column)
                }
                query={query}
                caseSensitive={caseSensitive}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Option Toggle
// ============================================================================

interface OptionToggleProps {
  active: boolean;
  onClick: () => void;
  icon: React.FC<{ size?: number; className?: string }>;
  tooltip: string;
}

function OptionToggle({
  active,
  onClick,
  icon: Icon,
  tooltip,
}: OptionToggleProps) {
  return (
    <button
      className={cn(
        "p-1 rounded transition-colors",
        active
          ? "bg-accent text-accent-foreground"
          : "text-muted-foreground hover:bg-muted/50"
      )}
      onClick={onClick}
      title={tooltip}
    >
      <Icon size={14} />
    </button>
  );
}

// ============================================================================
// File Result Item
// ============================================================================

interface SearchFileResultItemProps {
  file: SearchFileResult;
  expanded: boolean;
  onToggle: () => void;
  onMatchClick: (match: SearchMatch) => void;
  query: string;
  caseSensitive: boolean;
}

function SearchFileResultItem({
  file,
  expanded,
  onToggle,
  onMatchClick,
  query,
  caseSensitive,
}: SearchFileResultItemProps) {
  return (
    <div>
      {/* File header */}
      <button
        className={cn(
          "w-full flex items-center gap-2 px-3 py-1.5",
          "hover:bg-muted/50 transition-colors text-left"
        )}
        onClick={onToggle}
      >
        {expanded ? (
          <ChevronDown size={14} className="text-muted-foreground" />
        ) : (
          <ChevronRight size={14} className="text-muted-foreground" />
        )}
        <FileText size={14} className="text-muted-foreground" />
        <span className="text-sm flex-1 truncate">{file.name}</span>
        <span className="text-xs text-muted-foreground px-1.5 py-0.5 bg-muted rounded">
          {file.matches.length}
        </span>
      </button>

      {/* File path */}
      {expanded && (
        <div className="px-9 py-0.5 text-xs text-muted-foreground truncate">
          {file.path}
        </div>
      )}

      {/* Matches */}
      {expanded && (
        <div className="ml-9">
          {file.matches.map((match, index) => (
            <SearchMatchItem
              key={index}
              match={match}
              onClick={() => onMatchClick(match)}
              query={query}
              caseSensitive={caseSensitive}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Match Item
// ============================================================================

interface SearchMatchItemProps {
  match: SearchMatch;
  onClick: () => void;
  query: string;
  caseSensitive: boolean;
}

function SearchMatchItem({
  match,
  onClick,
  query,
  caseSensitive,
}: SearchMatchItemProps) {
  // Highlight the match in the preview
  const highlightedPreview = useMemo(() => {
    const flags = caseSensitive ? "g" : "gi";
    const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const regex = new RegExp(`(${escapedQuery})`, flags);

    return match.preview.split(regex).map((part, i) => {
      const isMatch = regex.test(part);
      return (
        <span
          key={i}
          className={cn(
            isMatch && "bg-yellow-500/30 text-yellow-200 font-medium"
          )}
        >
          {part}
        </span>
      );
    });
  }, [match.preview, query, caseSensitive]);

  return (
    <button
      className={cn(
        "w-full flex items-start gap-2 px-3 py-1 text-left",
        "hover:bg-muted/50 transition-colors group"
      )}
      onClick={onClick}
    >
      <span className="text-xs text-muted-foreground w-8 text-right flex-shrink-0 pt-0.5">
        {match.line}
      </span>
      <span className="text-sm font-mono truncate">{highlightedPreview}</span>
    </button>
  );
}

// ============================================================================
// Mock Search Function (for testing)
// ============================================================================

export async function mockSearch(
  query: string,
  options: SearchOptions
): Promise<SearchFileResult[]> {
  // Simulate search delay
  await new Promise((resolve) => setTimeout(resolve, 200));

  // Mock results
  return [
    {
      path: "src/components/App.tsx",
      name: "App.tsx",
      matches: [
        {
          line: 10,
          column: 5,
          length: query.length,
          preview: `  const ${query} = useState(null);`,
          previewStart: 0,
        },
        {
          line: 25,
          column: 12,
          length: query.length,
          preview: `    return <${query} />;`,
          previewStart: 0,
        },
      ],
    },
    {
      path: "src/utils/helpers.ts",
      name: "helpers.ts",
      matches: [
        {
          line: 42,
          column: 8,
          length: query.length,
          preview: `export function ${query}() {`,
          previewStart: 0,
        },
      ],
    },
  ];
}

export default SearchPanel;
