/**
 * Pane Group Component
 *
 * Recursive component that renders split panes.
 * Follows Zed's PaneGroup pattern for flexible layouts.
 *
 * Features:
 * - Horizontal/vertical splits
 * - Drag-to-resize
 * - Minimum size constraints
 * - Recursive nesting
 *
 * @module @neurectomy/workspace
 * @author @APEX @ARCHITECT
 */

import {
  ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { cn } from "@/lib/utils";
import {
  useWorkspaceContext,
  PaneContext,
  PaneContextValue,
  PaneGroup,
  Pane,
} from "./WorkspaceContext";
import type { EntityId } from "@/interfaces/types";

// ============================================================================
// Constants
// ============================================================================

const MIN_PANE_SIZE = 100; // pixels
const RESIZE_HANDLE_SIZE = 4; // pixels

// ============================================================================
// Types
// ============================================================================

export interface PaneGroupProps {
  /** The pane group ID to render */
  groupId: EntityId;
  /** Custom class name */
  className?: string;
  /** Render function for pane content */
  renderPane?: (paneId: EntityId) => ReactNode;
}

export interface PaneRendererProps {
  /** The pane ID to render */
  paneId: EntityId;
  /** Custom class name */
  className?: string;
  /** Children to render inside the pane */
  children?: ReactNode;
}

// ============================================================================
// Pane Group Component
// ============================================================================

export function PaneGroupRenderer({
  groupId,
  className,
  renderPane,
}: PaneGroupProps) {
  const { snapshot, actions } = useWorkspaceContext();
  const group = snapshot.paneGroups.get(groupId);

  // Local state for flex ratios during resize
  const [localFlexes, setLocalFlexes] = useState<number[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);

  // Sync local flexes with group flexes
  useEffect(() => {
    if (group) {
      setLocalFlexes(group.flexes);
    }
  }, [group?.flexes.join(",")]);

  const handleResize = useCallback(
    (index: number, delta: number) => {
      if (!group || !containerRef.current) return;

      const containerSize =
        group.axis === "horizontal"
          ? containerRef.current.offsetWidth
          : containerRef.current.offsetHeight;

      // Convert delta to flex ratio change
      const totalFlex = localFlexes.reduce((a, b) => a + b, 0);
      const deltaProportion = delta / containerSize;

      setLocalFlexes((prev) => {
        const newFlexes = [...prev];
        const minFlex = (MIN_PANE_SIZE / containerSize) * totalFlex;

        // Adjust the two adjacent panes
        const newLeft = Math.max(
          minFlex,
          prev[index] + deltaProportion * totalFlex
        );
        const newRight = Math.max(
          minFlex,
          prev[index + 1] - deltaProportion * totalFlex
        );

        // Only apply if both remain above minimum
        if (newLeft >= minFlex && newRight >= minFlex) {
          newFlexes[index] = newLeft;
          newFlexes[index + 1] = newRight;
        }

        return newFlexes;
      });
    },
    [group, localFlexes]
  );

  // Commit flexes on resize end
  const handleResizeEnd = useCallback(() => {
    // TODO: Persist flex changes to state
    // For now, local state handles it
  }, []);

  if (!group) {
    return null;
  }

  const isHorizontal = group.axis === "horizontal";

  return (
    <div
      ref={containerRef}
      className={cn(
        "h-full w-full flex",
        isHorizontal ? "flex-row" : "flex-col",
        className
      )}
    >
      {group.children.map((childId, index) => {
        const isPane = snapshot.panes.has(childId);
        const isPaneGroup = snapshot.paneGroups.has(childId);
        const flex = localFlexes[index] ?? 1;

        return (
          <div key={childId} className="contents">
            {/* Child (Pane or PaneGroup) */}
            <div
              style={{
                flex: flex,
                minWidth: isHorizontal ? MIN_PANE_SIZE : undefined,
                minHeight: !isHorizontal ? MIN_PANE_SIZE : undefined,
              }}
              className="overflow-hidden"
            >
              {isPane &&
                (renderPane ? (
                  renderPane(childId)
                ) : (
                  <PaneRenderer paneId={childId} />
                ))}
              {isPaneGroup && (
                <PaneGroupRenderer groupId={childId} renderPane={renderPane} />
              )}
            </div>

            {/* Resize Handle (between children) */}
            {index < group.children.length - 1 && (
              <ResizeHandle
                axis={group.axis}
                onResize={(delta) => handleResize(index, delta)}
                onResizeEnd={handleResizeEnd}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ============================================================================
// Pane Renderer Component
// ============================================================================

export function PaneRenderer({
  paneId,
  className,
  children,
}: PaneRendererProps) {
  const { snapshot, actions } = useWorkspaceContext();
  const pane = snapshot.panes.get(paneId);
  const isActive = snapshot.activePaneId === paneId;

  const contextValue = useMemo<PaneContextValue>(
    () => ({
      paneId,
      isActive,
      items: pane?.items ?? [],
      activeItemId: pane?.activeItemId ?? null,
      split: (direction) => actions.splitPane(paneId, direction),
      close: () => actions.closePane(paneId),
      activateItem: (itemId) => actions.activateItem(itemId, paneId),
      closeItem: (itemId) => actions.closeItem(itemId, paneId),
    }),
    [paneId, isActive, pane?.items, pane?.activeItemId, actions]
  );

  if (!pane) {
    return null;
  }

  return (
    <PaneContext.Provider value={contextValue}>
      <div
        className={cn(
          "h-full w-full flex flex-col",
          isActive && "ring-1 ring-primary/30",
          className
        )}
        onClick={() => actions.activatePane(paneId)}
      >
        {children}
      </div>
    </PaneContext.Provider>
  );
}

// ============================================================================
// Resize Handle Component
// ============================================================================

interface ResizeHandleProps {
  axis: "horizontal" | "vertical";
  onResize: (delta: number) => void;
  onResizeEnd: () => void;
}

function ResizeHandle({ axis, onResize, onResizeEnd }: ResizeHandleProps) {
  const [isDragging, setIsDragging] = useState(false);
  const startPosRef = useRef(0);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(true);
      startPosRef.current = axis === "horizontal" ? e.clientX : e.clientY;

      const handleMouseMove = (e: MouseEvent) => {
        const currentPos = axis === "horizontal" ? e.clientX : e.clientY;
        const delta = currentPos - startPosRef.current;
        startPosRef.current = currentPos;
        onResize(delta);
      };

      const handleMouseUp = () => {
        setIsDragging(false);
        onResizeEnd();
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      };

      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      document.body.style.cursor =
        axis === "horizontal" ? "col-resize" : "row-resize";
      document.body.style.userSelect = "none";
    },
    [axis, onResize, onResizeEnd]
  );

  return (
    <div
      onMouseDown={handleMouseDown}
      className={cn(
        "shrink-0 bg-border hover:bg-primary/50 transition-colors z-10",
        axis === "horizontal" ? "cursor-col-resize" : "cursor-row-resize",
        isDragging && "bg-primary",
        axis === "horizontal"
          ? `w-[${RESIZE_HANDLE_SIZE}px]`
          : `h-[${RESIZE_HANDLE_SIZE}px]`
      )}
      style={{
        width: axis === "horizontal" ? RESIZE_HANDLE_SIZE : undefined,
        height: axis === "vertical" ? RESIZE_HANDLE_SIZE : undefined,
      }}
    />
  );
}

// ============================================================================
// Root Pane Group Component
// ============================================================================

export interface RootPaneGroupProps {
  /** Custom class name */
  className?: string;
  /** Render function for pane content */
  renderPane?: (paneId: EntityId) => ReactNode;
  /** Fallback when no panes exist */
  emptyState?: ReactNode;
}

export function RootPaneGroup({
  className,
  renderPane,
  emptyState,
}: RootPaneGroupProps) {
  const { snapshot } = useWorkspaceContext();

  if (!snapshot.rootPaneGroupId) {
    return (
      <div
        className={cn(
          "h-full w-full flex items-center justify-center",
          className
        )}
      >
        {emptyState || (
          <div className="text-muted-foreground">No panes available</div>
        )}
      </div>
    );
  }

  return (
    <PaneGroupRenderer
      groupId={snapshot.rootPaneGroupId}
      className={className}
      renderPane={renderPane}
    />
  );
}

export default RootPaneGroup;
