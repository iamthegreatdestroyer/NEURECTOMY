/**
 * Application Shell - Professional IDE Layout Foundation
 *
 * 5-area grid layout based on VS Code + Theia patterns:
 * - Top: Menu bar (future) + Toolbar
 * - Left: Activity Bar + Primary Sidebar
 * - Main: Editor Area with pane splits
 * - Right: Secondary Sidebar (AI Panel)
 * - Bottom: Terminal + Output + Problems
 *
 * @module @neurectomy/shell
 * @author @ARCHITECT @CANVAS
 */

import { ReactNode, useState, useCallback, useRef, useEffect } from "react";
import { cn } from "@/lib/utils";
import { useWorkspaceStore } from "@/stores/workspace-store";

export interface ApplicationShellProps {
  activityBar: ReactNode;
  primarySidebar: ReactNode;
  editorArea: ReactNode;
  secondarySidebar?: ReactNode;
  bottomPanel?: ReactNode;
  statusBar: ReactNode;
  className?: string;
}

interface ResizeState {
  isResizing: boolean;
  target: "left" | "right" | "bottom" | null;
  startPos: number;
  startSize: number;
}

/**
 * ApplicationShell provides the foundational 5-area grid layout
 * for the NEURECTOMY IDE. All panels are resizable with proper
 * constraints and persistence.
 */
export function ApplicationShell({
  activityBar,
  primarySidebar,
  editorArea,
  secondarySidebar,
  bottomPanel,
  statusBar,
  className,
}: ApplicationShellProps) {
  const { layout, setLayout } = useWorkspaceStore();

  const [resizeState, setResizeState] = useState<ResizeState>({
    isResizing: false,
    target: null,
    startPos: 0,
    startSize: 0,
  });

  const shellRef = useRef<HTMLDivElement>(null);

  // Size constraints
  const MIN_SIDEBAR_WIDTH = 170;
  const MAX_SIDEBAR_WIDTH = 600;
  const MIN_BOTTOM_HEIGHT = 100;
  const MAX_BOTTOM_HEIGHT = 500;
  const ACTIVITY_BAR_WIDTH = 48;
  const STATUS_BAR_HEIGHT = 22;

  // Start resize operation
  const startResize = useCallback(
    (target: "left" | "right" | "bottom", e: React.MouseEvent) => {
      e.preventDefault();
      const startPos = target === "bottom" ? e.clientY : e.clientX;
      const startSize =
        target === "left"
          ? layout.leftSidebarWidth
          : target === "right"
            ? layout.rightSidebarWidth
            : layout.bottomPanelHeight;

      setResizeState({
        isResizing: true,
        target,
        startPos,
        startSize,
      });
    },
    [layout]
  );

  // Handle resize movement
  useEffect(() => {
    if (!resizeState.isResizing) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeState.target) return;

      const { target, startPos, startSize } = resizeState;

      if (target === "left") {
        const delta = e.clientX - startPos;
        const newWidth = Math.min(
          MAX_SIDEBAR_WIDTH,
          Math.max(MIN_SIDEBAR_WIDTH, startSize + delta)
        );
        setLayout({ leftSidebarWidth: newWidth });
      } else if (target === "right") {
        const delta = startPos - e.clientX;
        const newWidth = Math.min(
          MAX_SIDEBAR_WIDTH,
          Math.max(MIN_SIDEBAR_WIDTH, startSize + delta)
        );
        setLayout({ rightSidebarWidth: newWidth });
      } else if (target === "bottom") {
        const delta = startPos - e.clientY;
        const newHeight = Math.min(
          MAX_BOTTOM_HEIGHT,
          Math.max(MIN_BOTTOM_HEIGHT, startSize + delta)
        );
        setLayout({ bottomPanelHeight: newHeight });
      }
    };

    const handleMouseUp = () => {
      setResizeState({
        isResizing: false,
        target: null,
        startPos: 0,
        startSize: 0,
      });
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [resizeState, setLayout]);

  // Calculate grid template
  const gridTemplateColumns = `${ACTIVITY_BAR_WIDTH}px ${
    layout.leftSidebarCollapsed ? "0px" : `${layout.leftSidebarWidth}px`
  } 1fr ${
    layout.rightSidebarCollapsed || !secondarySidebar
      ? "0px"
      : `${layout.rightSidebarWidth}px`
  }`;

  const gridTemplateRows = `1fr ${
    layout.bottomPanelCollapsed || !bottomPanel
      ? "0px"
      : `${layout.bottomPanelHeight}px`
  } ${STATUS_BAR_HEIGHT}px`;

  return (
    <div
      ref={shellRef}
      className={cn(
        "h-screen w-screen overflow-hidden bg-background text-foreground",
        resizeState.isResizing && "select-none",
        className
      )}
      style={{
        display: "grid",
        gridTemplateColumns,
        gridTemplateRows,
        gridTemplateAreas: `
          "activity sidebar main secondary"
          "activity sidebar bottom bottom"
          "status status status status"
        `,
      }}
    >
      {/* Activity Bar - Fixed width column */}
      <div
        className="bg-shell-activityBar border-r border-border"
        style={{ gridArea: "activity" }}
      >
        {activityBar}
      </div>

      {/* Primary Sidebar */}
      {!layout.leftSidebarCollapsed && (
        <div
          className="bg-shell-sidebar border-r border-border overflow-hidden relative"
          style={{ gridArea: "sidebar" }}
        >
          {primarySidebar}

          {/* Left resize handle */}
          <div
            className="absolute top-0 right-0 w-1 h-full cursor-col-resize hover:bg-primary/50 transition-colors z-10"
            onMouseDown={(e) => startResize("left", e)}
            onDoubleClick={() => setLayout({ leftSidebarWidth: 280 })}
          />
        </div>
      )}

      {/* Main Editor Area */}
      <div
        className="bg-shell-editor overflow-hidden relative"
        style={{ gridArea: "main" }}
      >
        {editorArea}
      </div>

      {/* Secondary Sidebar (AI Panel) */}
      {secondarySidebar && !layout.rightSidebarCollapsed && (
        <div
          className="bg-shell-sidebar border-l border-border overflow-hidden relative"
          style={{ gridArea: "secondary" }}
        >
          {/* Right resize handle */}
          <div
            className="absolute top-0 left-0 w-1 h-full cursor-col-resize hover:bg-primary/50 transition-colors z-10"
            onMouseDown={(e) => startResize("right", e)}
            onDoubleClick={() => setLayout({ rightSidebarWidth: 320 })}
          />

          {secondarySidebar}
        </div>
      )}

      {/* Bottom Panel */}
      {bottomPanel && !layout.bottomPanelCollapsed && (
        <div
          className="bg-shell-panel border-t border-border overflow-hidden relative"
          style={{ gridArea: "bottom" }}
        >
          {/* Bottom resize handle */}
          <div
            className="absolute top-0 left-0 w-full h-1 cursor-row-resize hover:bg-primary/50 transition-colors z-10"
            onMouseDown={(e) => startResize("bottom", e)}
            onDoubleClick={() => setLayout({ bottomPanelHeight: 240 })}
          />

          {bottomPanel}
        </div>
      )}

      {/* Status Bar - Spans full width */}
      <div
        className="bg-shell-statusBar border-t border-border"
        style={{ gridArea: "status" }}
      >
        {statusBar}
      </div>

      {/* Resize overlay to prevent iframe capture during resize */}
      {resizeState.isResizing && (
        <div
          className="fixed inset-0 z-50"
          style={{
            cursor:
              resizeState.target === "bottom" ? "row-resize" : "col-resize",
          }}
        />
      )}
    </div>
  );
}

export default ApplicationShell;
