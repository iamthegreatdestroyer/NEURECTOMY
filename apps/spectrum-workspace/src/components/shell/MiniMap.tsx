/**
 * MiniMap Component
 *
 * VS Code-style code minimap showing a birds-eye view of the document.
 * Displays syntax-highlighted overview with viewport indicator.
 *
 * Features:
 * - Syntax-aware rendering
 * - Viewport indicator (slider)
 * - Click to navigate
 * - Drag to scroll
 * - Hover preview
 * - Search highlight integration
 *
 * @module @neurectomy/components/shell/MiniMap
 * @author @APEX @VELOCITY
 */

import React, {
  useRef,
  useEffect,
  useState,
  useCallback,
  useMemo,
} from "react";
import { cn } from "@/lib/utils";

// ============================================================================
// Types
// ============================================================================

export interface MiniMapProps {
  /** Lines of code to render */
  lines: string[];
  /** Language for syntax coloring */
  language?: string;
  /** Current viewport start line (0-indexed) */
  viewportStart: number;
  /** Current viewport end line (0-indexed) */
  viewportEnd: number;
  /** Total line count */
  totalLines: number;
  /** Callback when viewport position changes */
  onScroll?: (startLine: number) => void;
  /** Search matches to highlight */
  searchMatches?: { line: number; start: number; end: number }[];
  /** Current cursor line */
  cursorLine?: number;
  /** Width of minimap in pixels */
  width?: number;
  /** Character width in pixels */
  charWidth?: number;
  /** Line height in pixels */
  lineHeight?: number;
  /** Whether minimap is visible */
  visible?: boolean;
  /** Additional className */
  className?: string;
}

// ============================================================================
// Token Colors (simplified)
// ============================================================================

interface TokenColor {
  pattern: RegExp;
  color: string;
}

const TOKEN_COLORS: Record<string, TokenColor[]> = {
  typescript: [
    {
      pattern:
        /\b(import|export|from|const|let|var|function|class|interface|type|return|if|else|for|while|async|await|new|this|extends|implements)\b/g,
      color: "#c586c0",
    },
    {
      pattern:
        /\b(string|number|boolean|void|null|undefined|any|never|unknown)\b/g,
      color: "#4ec9b0",
    },
    { pattern: /"[^"]*"|'[^']*'|`[^`]*`/g, color: "#ce9178" },
    { pattern: /\/\/.*$/gm, color: "#6a9955" },
    { pattern: /\/\*[\s\S]*?\*\//g, color: "#6a9955" },
    { pattern: /\b\d+\.?\d*\b/g, color: "#b5cea8" },
    { pattern: /[{}()\[\]]/g, color: "#ffd700" },
    { pattern: /\b([A-Z][a-zA-Z0-9]*)\b/g, color: "#4ec9b0" },
  ],
  javascript: [
    {
      pattern:
        /\b(import|export|from|const|let|var|function|class|return|if|else|for|while|async|await|new|this)\b/g,
      color: "#c586c0",
    },
    { pattern: /"[^"]*"|'[^']*'|`[^`]*`/g, color: "#ce9178" },
    { pattern: /\/\/.*$/gm, color: "#6a9955" },
    { pattern: /\/\*[\s\S]*?\*\//g, color: "#6a9955" },
    { pattern: /\b\d+\.?\d*\b/g, color: "#b5cea8" },
    { pattern: /[{}()\[\]]/g, color: "#ffd700" },
  ],
  default: [
    { pattern: /"[^"]*"|'[^']*'/g, color: "#ce9178" },
    { pattern: /\/\/.*$/gm, color: "#6a9955" },
    { pattern: /#.*$/gm, color: "#6a9955" },
    { pattern: /\b\d+\.?\d*\b/g, color: "#b5cea8" },
  ],
};

// ============================================================================
// Main Component
// ============================================================================

export function MiniMap({
  lines,
  language = "default",
  viewportStart,
  viewportEnd,
  totalLines,
  onScroll,
  searchMatches = [],
  cursorLine,
  width = 80,
  charWidth = 1.2,
  lineHeight = 2,
  visible = true,
  className,
}: MiniMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [hoverLine, setHoverLine] = useState<number | null>(null);

  // Calculate dimensions
  const contentHeight = totalLines * lineHeight;
  const viewportHeight = (viewportEnd - viewportStart) * lineHeight;
  const viewportTop = viewportStart * lineHeight;

  // Get token colors for language
  const tokenColors = useMemo(() => {
    const lang = language.toLowerCase();
    if (lang.includes("typescript") || lang.includes("tsx")) {
      return TOKEN_COLORS.typescript;
    }
    if (lang.includes("javascript") || lang.includes("jsx")) {
      return TOKEN_COLORS.javascript;
    }
    return TOKEN_COLORS[lang] || TOKEN_COLORS.default;
  }, [language]);

  // Build search match lookup
  const searchMatchSet = useMemo(() => {
    const set = new Set<number>();
    searchMatches.forEach((m) => set.add(m.line));
    return set;
  }, [searchMatches]);

  // Render minimap
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = contentHeight * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${contentHeight}px`;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.fillStyle = "transparent";
    ctx.clearRect(0, 0, width, contentHeight);

    // Render each line
    lines.forEach((line, lineIndex) => {
      const y = lineIndex * lineHeight;

      // Skip if line is empty
      if (!line.trim()) return;

      // Background for search matches
      if (searchMatchSet.has(lineIndex)) {
        ctx.fillStyle = "rgba(234, 179, 8, 0.3)";
        ctx.fillRect(0, y, width, lineHeight);
      }

      // Background for cursor line
      if (cursorLine === lineIndex) {
        ctx.fillStyle = "rgba(255, 255, 255, 0.1)";
        ctx.fillRect(0, y, width, lineHeight);
      }

      // Render line content (simplified - just rectangles for characters)
      const indent = line.search(/\S|$/);
      let x = indent * charWidth;

      // Simple character rendering (colored rectangles)
      const text = line.slice(indent);
      let matched = false;

      for (const tokenColor of tokenColors) {
        const regex = new RegExp(
          tokenColor.pattern.source,
          tokenColor.pattern.flags
        );
        let match;

        while ((match = regex.exec(line)) !== null) {
          const tokenX = match.index * charWidth;
          const tokenWidth = match[0].length * charWidth;

          ctx.fillStyle = tokenColor.color;
          ctx.globalAlpha = 0.8;
          ctx.fillRect(
            Math.min(tokenX, width - 4),
            y + 0.5,
            Math.min(tokenWidth, width - tokenX),
            lineHeight - 1
          );
          matched = true;
        }
      }

      // Default rendering if no tokens matched
      if (!matched && text.length > 0) {
        ctx.fillStyle = "#d4d4d4";
        ctx.globalAlpha = 0.5;
        const textWidth = Math.min(text.length * charWidth, width - x);
        ctx.fillRect(x, y + 0.5, textWidth, lineHeight - 1);
      }

      ctx.globalAlpha = 1;
    });
  }, [
    lines,
    width,
    contentHeight,
    lineHeight,
    charWidth,
    tokenColors,
    searchMatchSet,
    cursorLine,
  ]);

  // Handle mouse down (start drag)
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!onScroll) return;

      setIsDragging(true);

      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;

      const y = e.clientY - rect.top;
      const targetLine = Math.floor(y / lineHeight);
      const viewportLines = viewportEnd - viewportStart;
      const newStart = Math.max(
        0,
        Math.min(targetLine - viewportLines / 2, totalLines - viewportLines)
      );

      onScroll(Math.floor(newStart));
    },
    [onScroll, lineHeight, viewportStart, viewportEnd, totalLines]
  );

  // Handle mouse move (drag)
  useEffect(() => {
    if (!isDragging || !onScroll) return;

    const handleMouseMove = (e: MouseEvent) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;

      const y = e.clientY - rect.top;
      const targetLine = Math.floor(y / lineHeight);
      const viewportLines = viewportEnd - viewportStart;
      const newStart = Math.max(
        0,
        Math.min(targetLine - viewportLines / 2, totalLines - viewportLines)
      );

      onScroll(Math.floor(newStart));
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [
    isDragging,
    onScroll,
    lineHeight,
    viewportStart,
    viewportEnd,
    totalLines,
  ]);

  // Handle hover
  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;

      const y = e.clientY - rect.top;
      const line = Math.floor(y / lineHeight);
      setHoverLine(line >= 0 && line < totalLines ? line : null);
    },
    [lineHeight, totalLines]
  );

  if (!visible) return null;

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative select-none cursor-pointer",
        "bg-background/50",
        className
      )}
      style={{ width, height: Math.min(contentHeight, 600) }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => setHoverLine(null)}
    >
      {/* Canvas for code rendering */}
      <div
        className="overflow-hidden"
        style={{ height: Math.min(contentHeight, 600) }}
      >
        <canvas
          ref={canvasRef}
          style={{
            transform:
              contentHeight > 600
                ? `translateY(-${(viewportStart / totalLines) * (contentHeight - 600)}px)`
                : undefined,
          }}
        />
      </div>

      {/* Viewport indicator (slider) */}
      <div
        className={cn(
          "absolute left-0 right-0",
          "bg-accent/20 border-l-2 border-accent",
          "transition-opacity",
          isDragging ? "bg-accent/30" : "hover:bg-accent/25"
        )}
        style={{
          top: Math.min(
            viewportTop,
            Math.min(contentHeight, 600) - viewportHeight
          ),
          height: viewportHeight,
        }}
      />

      {/* Hover line indicator */}
      {hoverLine !== null && (
        <div
          className="absolute left-0 right-0 bg-accent/10 pointer-events-none"
          style={{
            top: hoverLine * lineHeight,
            height: lineHeight,
          }}
        />
      )}

      {/* Hover tooltip */}
      {hoverLine !== null && lines[hoverLine] && (
        <div
          className={cn(
            "absolute right-full mr-2 px-2 py-1 max-w-xs",
            "bg-card border border-border rounded shadow-lg",
            "text-xs font-mono truncate whitespace-pre",
            "pointer-events-none"
          )}
          style={{ top: hoverLine * lineHeight }}
        >
          <span className="text-muted-foreground mr-2">{hoverLine + 1}</span>
          {lines[hoverLine].slice(0, 60)}
          {lines[hoverLine].length > 60 && "..."}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Hook for MiniMap Integration
// ============================================================================

export interface UseMiniMapOptions {
  editorRef: React.RefObject<HTMLElement>;
  lineCount: number;
  lineHeight: number;
}

export function useMiniMap({
  editorRef,
  lineCount,
  lineHeight,
}: UseMiniMapOptions) {
  const [viewportStart, setViewportStart] = useState(0);
  const [viewportEnd, setViewportEnd] = useState(30);

  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;

    const handleScroll = () => {
      const scrollTop = editor.scrollTop;
      const clientHeight = editor.clientHeight;

      const start = Math.floor(scrollTop / lineHeight);
      const visible = Math.ceil(clientHeight / lineHeight);

      setViewportStart(start);
      setViewportEnd(Math.min(start + visible, lineCount));
    };

    editor.addEventListener("scroll", handleScroll);
    handleScroll(); // Initial calculation

    return () => editor.removeEventListener("scroll", handleScroll);
  }, [editorRef, lineCount, lineHeight]);

  const scrollToLine = useCallback(
    (line: number) => {
      const editor = editorRef.current;
      if (!editor) return;

      editor.scrollTop = line * lineHeight;
    },
    [editorRef, lineHeight]
  );

  return {
    viewportStart,
    viewportEnd,
    scrollToLine,
  };
}

export default MiniMap;
