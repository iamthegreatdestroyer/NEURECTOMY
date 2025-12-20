/**
 * Dock Component
 *
 * Dockable panel container for sidebars and bottom panel.
 * Manages panel visibility, tabs, and resize behavior.
 *
 * Features:
 * - Panel tab bar with icons
 * - Drag-to-resize
 * - Collapse/expand
 * - Panel content rendering
 *
 * @module @neurectomy/workspace
 * @author @APEX @ARCHITECT
 */

import { ReactNode, useCallback, useMemo, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import {
  useWorkspaceContext,
  PanelContext,
  PanelContextValue,
} from "./WorkspaceContext";
import { useDockState, useDockControls } from "./hooks";
import type { Panel, PanelState } from "@/interfaces/Panel";
import type { EntityId, DockPosition } from "@/interfaces/types";
import {
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  MoreHorizontal,
} from "lucide-react";

// ============================================================================
// Constants
// ============================================================================

const COLLAPSE_THRESHOLD = 50; // pixels

// ============================================================================
// Types
// ============================================================================

export interface DockProps {
  /** Dock position */
  position: DockPosition;
  /** Custom class name */
  className?: string;
  /** Render function for panel content */
  renderPanel?: (panelId: EntityId) => ReactNode;
  /** Whether to show the header */
  showHeader?: boolean;
  /** Whether to allow resize */
  resizable?: boolean;
}

// ============================================================================
// Dock Component
// ============================================================================

export function Dock({
  position,
  className,
  renderPanel,
  showHeader = true,
  resizable = true,
}: DockProps) {
  const { snapshot, actions } = useWorkspaceContext();
  const dock = snapshot.docks.get(position);
  const { resize, collapse } = useDockControls(position);

  // Resize state
  const [isResizing, setIsResizing] = useState(false);
  const startSizeRef = useRef(0);
  const startPosRef = useRef(0);

  // Get active panel
  const activePanelId = dock?.activePanelId;
  const activePanel = activePanelId ? snapshot.panels.get(activePanelId) : null;
  const activePanelState = activePanelId
    ? snapshot.panelStates.get(activePanelId)
    : null;

  // Handle resize start
  const handleResizeStart = useCallback(
    (e: React.MouseEvent) => {
      if (!dock || !resizable) return;

      e.preventDefault();
      setIsResizing(true);
      startSizeRef.current = dock.size;

      if (position === "left" || position === "right") {
        startPosRef.current = e.clientX;
      } else {
        startPosRef.current = e.clientY;
      }

      const handleMouseMove = (e: MouseEvent) => {
        let delta: number;
        let newSize: number;

        if (position === "left") {
          delta = e.clientX - startPosRef.current;
          newSize = startSizeRef.current + delta;
        } else if (position === "right") {
          delta = startPosRef.current - e.clientX;
          newSize = startSizeRef.current + delta;
        } else {
          // bottom
          delta = startPosRef.current - e.clientY;
          newSize = startSizeRef.current + delta;
        }

        // Check for collapse
        if (newSize < COLLAPSE_THRESHOLD) {
          collapse(true);
        } else {
          collapse(false);
          resize(newSize);
        }
      };

      const handleMouseUp = () => {
        setIsResizing(false);
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      };

      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);

      const cursor =
        position === "left" || position === "right"
          ? "col-resize"
          : "row-resize";
      document.body.style.cursor = cursor;
      document.body.style.userSelect = "none";
    },
    [dock, position, resizable, resize, collapse]
  );

  if (!dock || !dock.isVisible) {
    return null;
  }

  // Build panel context for active panel
  const panelContextValue = useMemo<PanelContextValue | null>(() => {
    if (!activePanelId || !activePanelState) return null;

    return {
      panelId: activePanelId,
      position,
      isActive: true,
      isFocused: activePanelState.isFocused,
      state: activePanelState,
      updateState: (updates) => {
        // TODO: Implement state update
      },
      requestFocus: () => actions.focusPanel(activePanelId),
    };
  }, [activePanelId, activePanelState, position, actions]);

  // Determine flex direction and size property
  const isHorizontal = position === "left" || position === "right";
  const sizeStyle = isHorizontal
    ? { width: dock.isCollapsed ? 0 : dock.size }
    : { height: dock.isCollapsed ? 0 : dock.size };

  // Resize handle position classes
  const resizeHandleClasses = cn(
    "absolute bg-border hover:bg-primary/50 transition-colors z-20",
    position === "left" && "right-0 top-0 bottom-0 w-1 cursor-col-resize",
    position === "right" && "left-0 top-0 bottom-0 w-1 cursor-col-resize",
    position === "bottom" && "top-0 left-0 right-0 h-1 cursor-row-resize",
    isResizing && "bg-primary"
  );

  return (
    <div
      className={cn(
        "relative flex flex-col bg-background border-border overflow-hidden",
        position === "left" && "border-r",
        position === "right" && "border-l",
        position === "bottom" && "border-t",
        dock.isCollapsed && "invisible",
        className
      )}
      style={sizeStyle}
    >
      {/* Header */}
      {showHeader && (
        <DockHeader
          position={position}
          panels={dock.panels
            .map((id) => ({
              id,
              panel: snapshot.panels.get(id),
              state: snapshot.panelStates.get(id),
            }))
            .filter((p) => p.panel !== undefined)}
          activePanelId={activePanelId}
          onActivatePanel={actions.activatePanel}
          onTogglePanel={actions.togglePanel}
        />
      )}

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {panelContextValue && activePanel ? (
          <PanelContext.Provider value={panelContextValue}>
            {renderPanel ? (
              renderPanel(activePanelId!)
            ) : (
              <DefaultPanelRenderer panel={activePanel} />
            )}
          </PanelContext.Provider>
        ) : (
          <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
            No panel selected
          </div>
        )}
      </div>

      {/* Resize Handle */}
      {resizable && (
        <div className={resizeHandleClasses} onMouseDown={handleResizeStart} />
      )}
    </div>
  );
}

// ============================================================================
// Dock Header Component
// ============================================================================

interface DockHeaderProps {
  position: DockPosition;
  panels: Array<{
    id: EntityId;
    panel?: Panel;
    state?: PanelState;
  }>;
  activePanelId: EntityId | null;
  onActivatePanel: (panelId: EntityId) => void;
  onTogglePanel: (panelId: EntityId) => void;
}

function DockHeader({
  position,
  panels,
  activePanelId,
  onActivatePanel,
}: DockHeaderProps) {
  const activePanel = panels.find((p) => p.id === activePanelId);

  return (
    <div className="h-9 flex items-center gap-1 px-2 bg-muted/30 border-b border-border shrink-0">
      {/* Panel Tabs */}
      <div className="flex-1 flex items-center gap-0.5 overflow-x-auto scrollbar-none">
        {panels.map(({ id, panel }) => {
          if (!panel) return null;
          const isActive = id === activePanelId;
          const Icon = panel.config.icon;

          return (
            <button
              key={id}
              onClick={() => onActivatePanel(id)}
              className={cn(
                "flex items-center gap-1.5 px-2 py-1 rounded text-sm",
                "transition-colors",
                isActive
                  ? "bg-background text-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
              )}
              title={panel.name}
            >
              {Icon && <Icon size={14} />}
              <span className="whitespace-nowrap">{panel.name}</span>
            </button>
          );
        })}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-0.5">
        {activePanel?.panel?.headerActions?.().map((action, i) => {
          const Icon = action.icon;
          return (
            <button
              key={i}
              onClick={() => action.run()}
              className="p-1 rounded text-muted-foreground hover:text-foreground hover:bg-accent"
              title={action.label}
              aria-label={action.label}
            >
              {Icon && <Icon size={14} />}
            </button>
          );
        })}
        <button
          className="p-1 rounded text-muted-foreground hover:text-foreground hover:bg-accent"
          aria-label="More options"
          title="More options"
        >
          <MoreHorizontal size={14} />
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// Default Panel Renderer
// ============================================================================

interface DefaultPanelRendererProps {
  panel: Panel;
}

function DefaultPanelRenderer({ panel }: DefaultPanelRendererProps) {
  // Panels render their own content
  // This is a placeholder - real panels implement their own render method
  return (
    <div className="h-full p-4">
      <div className="text-sm text-muted-foreground">Panel: {panel.name}</div>
    </div>
  );
}

// ============================================================================
// Activity Bar Integration
// ============================================================================

export interface ActivityBarItemProps {
  /** Panel ID */
  panelId: EntityId;
  /** Icon component */
  icon: React.ComponentType<{ size?: number; className?: string }>;
  /** Tooltip label */
  label: string;
  /** Whether this item is active */
  active?: boolean;
  /** Click handler */
  onClick?: () => void;
}

export function ActivityBarItem({
  icon: Icon,
  label,
  active,
  onClick,
}: ActivityBarItemProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "w-12 h-12 flex items-center justify-center",
        "text-muted-foreground hover:text-foreground transition-colors",
        active && "text-foreground border-l-2 border-primary bg-accent/50"
      )}
      title={label}
      aria-label={label}
    >
      <Icon size={24} />
    </button>
  );
}

// ============================================================================
// Collapse Button Component
// ============================================================================

interface CollapseButtonProps {
  position: DockPosition;
  isCollapsed: boolean;
  onClick: () => void;
  className?: string;
}

export function CollapseButton({
  position,
  isCollapsed,
  onClick,
  className,
}: CollapseButtonProps) {
  let Icon: React.ComponentType<{ size?: number }>;

  if (position === "left") {
    Icon = isCollapsed ? ChevronRight : ChevronLeft;
  } else if (position === "right") {
    Icon = isCollapsed ? ChevronLeft : ChevronRight;
  } else {
    Icon = isCollapsed ? ChevronUp : ChevronDown;
  }

  const collapseLabel = isCollapsed ? "Expand" : "Collapse";

  return (
    <button
      onClick={onClick}
      className={cn(
        "p-1 rounded text-muted-foreground hover:text-foreground hover:bg-accent",
        className
      )}
      title={collapseLabel}
      aria-label={collapseLabel}
    >
      <Icon size={16} />
    </button>
  );
}

export default Dock;
