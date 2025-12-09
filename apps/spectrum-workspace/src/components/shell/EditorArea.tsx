/**
 * Editor Area Component
 *
 * Container for the main editor content with support for:
 * - Recursive pane splitting (horizontal/vertical)
 * - Tab bar with drag-and-drop
 * - Welcome screen when no files open
 *
 * @module @neurectomy/shell
 * @author @APEX @ARCHITECT
 */

import {
  ReactNode,
  createContext,
  useContext,
  useCallback,
  useState,
} from "react";
import { cn } from "@/lib/utils";
import {
  X,
  SplitSquareHorizontal,
  SplitSquareVertical,
  MoreHorizontal,
  Pin,
  Copy,
  FileCode2,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export type SplitDirection = "horizontal" | "vertical";

export interface EditorTab {
  id: string;
  title: string;
  icon?: ReactNode;
  dirty?: boolean;
  pinned?: boolean;
  preview?: boolean;
  path?: string;
}

export interface EditorPane {
  id: string;
  type: "pane";
  tabs: EditorTab[];
  activeTabId: string | null;
}

export interface EditorSplit {
  id: string;
  type: "split";
  direction: SplitDirection;
  children: (EditorPane | EditorSplit)[];
  sizes: number[];
}

export type EditorNode = EditorPane | EditorSplit;

// ============================================================================
// Context
// ============================================================================

interface EditorAreaContextValue {
  activePane: string | null;
  setActivePane: (id: string) => void;
  closeTab: (paneId: string, tabId: string) => void;
  activateTab: (paneId: string, tabId: string) => void;
  splitPane: (paneId: string, direction: SplitDirection) => void;
  moveTab: (fromPaneId: string, toPaneId: string, tabId: string) => void;
  pinTab: (paneId: string, tabId: string) => void;
  renderTabContent: (tab: EditorTab) => ReactNode;
}

const EditorAreaContext = createContext<EditorAreaContextValue | null>(null);

export function useEditorArea() {
  const context = useContext(EditorAreaContext);
  if (!context) {
    throw new Error("useEditorArea must be used within EditorAreaProvider");
  }
  return context;
}

// ============================================================================
// Components
// ============================================================================

export interface EditorAreaProps {
  root: EditorNode | null;
  activePane: string | null;
  onActivePaneChange: (id: string) => void;
  onCloseTab: (paneId: string, tabId: string) => void;
  onActivateTab: (paneId: string, tabId: string) => void;
  onSplitPane: (paneId: string, direction: SplitDirection) => void;
  onMoveTab: (fromPaneId: string, toPaneId: string, tabId: string) => void;
  onPinTab: (paneId: string, tabId: string) => void;
  renderTabContent: (tab: EditorTab) => ReactNode;
  welcomeScreen?: ReactNode;
  className?: string;
}

export function EditorArea({
  root,
  activePane,
  onActivePaneChange,
  onCloseTab,
  onActivateTab,
  onSplitPane,
  onMoveTab,
  onPinTab,
  renderTabContent,
  welcomeScreen,
  className,
}: EditorAreaProps) {
  const contextValue: EditorAreaContextValue = {
    activePane,
    setActivePane: onActivePaneChange,
    closeTab: onCloseTab,
    activateTab: onActivateTab,
    splitPane: onSplitPane,
    moveTab: onMoveTab,
    pinTab: onPinTab,
    renderTabContent,
  };

  return (
    <EditorAreaContext.Provider value={contextValue}>
      <div className={cn("h-full w-full", className)}>
        {root ? (
          <EditorNodeRenderer node={root} />
        ) : (
          welcomeScreen || <WelcomeScreen />
        )}
      </div>
    </EditorAreaContext.Provider>
  );
}

// ============================================================================
// Node Renderer
// ============================================================================

interface EditorNodeRendererProps {
  node: EditorNode;
}

function EditorNodeRenderer({ node }: EditorNodeRendererProps) {
  if (node.type === "pane") {
    return <EditorPaneRenderer pane={node} />;
  }
  return <EditorSplitRenderer split={node} />;
}

// ============================================================================
// Split Renderer
// ============================================================================

interface EditorSplitRendererProps {
  split: EditorSplit;
}

function EditorSplitRenderer({ split }: EditorSplitRendererProps) {
  const [sizes, setSizes] = useState(split.sizes);

  const handleResize = useCallback((index: number, delta: number) => {
    setSizes((prev) => {
      const newSizes = [...prev];
      const total = newSizes[index] + newSizes[index + 1];
      const newSize = Math.max(10, Math.min(90, newSizes[index] + delta));
      newSizes[index] = newSize;
      newSizes[index + 1] = total - newSize;
      return newSizes;
    });
  }, []);

  const isHorizontal = split.direction === "horizontal";

  return (
    <div
      className={cn(
        "h-full w-full flex",
        isHorizontal ? "flex-row" : "flex-col"
      )}
    >
      {split.children.map((child, index) => (
        <div key={child.id} className="contents">
          <div
            style={{
              [isHorizontal ? "width" : "height"]: `${sizes[index]}%`,
              flex: "none",
            }}
          >
            <EditorNodeRenderer node={child} />
          </div>
          {index < split.children.length - 1 && (
            <SplitResizer
              direction={split.direction}
              onResize={(delta) => handleResize(index, delta)}
            />
          )}
        </div>
      ))}
    </div>
  );
}

// ============================================================================
// Split Resizer
// ============================================================================

interface SplitResizerProps {
  direction: SplitDirection;
  onResize: (delta: number) => void;
}

function SplitResizer({ direction, onResize }: SplitResizerProps) {
  const [isDragging, setIsDragging] = useState(false);
  const isHorizontal = direction === "horizontal";

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      setIsDragging(true);

      const startPos = isHorizontal ? e.clientX : e.clientY;

      const handleMouseMove = (e: MouseEvent) => {
        const currentPos = isHorizontal ? e.clientX : e.clientY;
        const delta = ((currentPos - startPos) / window.innerWidth) * 100;
        onResize(delta);
      };

      const handleMouseUp = () => {
        setIsDragging(false);
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
      };

      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
    },
    [isHorizontal, onResize]
  );

  return (
    <div
      onMouseDown={handleMouseDown}
      className={cn(
        "shrink-0 bg-border hover:bg-primary/50 transition-colors",
        isHorizontal ? "w-1 cursor-col-resize" : "h-1 cursor-row-resize",
        isDragging && "bg-primary"
      )}
    />
  );
}

// ============================================================================
// Pane Renderer
// ============================================================================

interface EditorPaneRendererProps {
  pane: EditorPane;
}

function EditorPaneRenderer({ pane }: EditorPaneRendererProps) {
  const {
    activePane,
    setActivePane,
    closeTab,
    activateTab,
    splitPane,
    pinTab,
    renderTabContent,
  } = useEditorArea();
  const isActive = activePane === pane.id;
  const activeTab = pane.tabs.find((t) => t.id === pane.activeTabId);

  return (
    <div
      className={cn(
        "h-full w-full flex flex-col",
        isActive && "ring-1 ring-primary/50"
      )}
      onClick={() => setActivePane(pane.id)}
    >
      {/* Tab Bar */}
      <div className="h-9 flex items-center bg-muted/30 border-b border-border shrink-0">
        <div className="flex-1 flex items-center overflow-x-auto scrollbar-none">
          {pane.tabs.map((tab) => (
            <EditorTabComponent
              key={tab.id}
              tab={tab}
              active={tab.id === pane.activeTabId}
              onActivate={() => activateTab(pane.id, tab.id)}
              onClose={() => closeTab(pane.id, tab.id)}
              onPin={() => pinTab(pane.id, tab.id)}
            />
          ))}
        </div>
        {/* Tab Actions */}
        <div className="flex items-center gap-0.5 px-2 border-l border-border">
          <button
            onClick={() => splitPane(pane.id, "horizontal")}
            className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground"
            title="Split Right"
          >
            <SplitSquareHorizontal size={14} />
          </button>
          <button
            onClick={() => splitPane(pane.id, "vertical")}
            className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground"
            title="Split Down"
          >
            <SplitSquareVertical size={14} />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab ? renderTabContent(activeTab) : <EmptyPane />}
      </div>
    </div>
  );
}

// ============================================================================
// Tab Component
// ============================================================================

interface EditorTabComponentProps {
  tab: EditorTab;
  active: boolean;
  onActivate: () => void;
  onClose: () => void;
  onPin: () => void;
}

function EditorTabComponent({
  tab,
  active,
  onActivate,
  onClose,
  onPin,
}: EditorTabComponentProps) {
  const [showMenu, setShowMenu] = useState(false);

  return (
    <div
      onClick={onActivate}
      onContextMenu={(e) => {
        e.preventDefault();
        setShowMenu(true);
      }}
      className={cn(
        "group relative h-9 flex items-center gap-2 px-3 border-r border-border",
        "cursor-pointer select-none",
        active
          ? "bg-background text-foreground"
          : "bg-transparent text-muted-foreground hover:text-foreground",
        tab.preview && "italic"
      )}
    >
      {/* Icon */}
      {tab.icon || <FileCode2 size={14} />}

      {/* Title */}
      <span className="text-sm whitespace-nowrap">{tab.title}</span>

      {/* Dirty indicator */}
      {tab.dirty && <span className="w-2 h-2 rounded-full bg-primary" />}

      {/* Pinned indicator */}
      {tab.pinned && <Pin size={12} className="text-muted-foreground" />}

      {/* Close button */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onClose();
        }}
        className={cn(
          "p-0.5 rounded hover:bg-accent",
          "opacity-0 group-hover:opacity-100",
          active && "opacity-100"
        )}
      >
        <X size={14} />
      </button>

      {/* Context Menu */}
      {showMenu && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setShowMenu(false)}
          />
          <div
            className={cn(
              "absolute top-full left-0 mt-1 z-50",
              "bg-popover border border-border rounded shadow-lg",
              "py-1 min-w-[160px]"
            )}
          >
            <ContextMenuItem
              onClick={() => {
                onClose();
                setShowMenu(false);
              }}
            >
              Close
            </ContextMenuItem>
            <ContextMenuItem onClick={() => setShowMenu(false)}>
              Close Others
            </ContextMenuItem>
            <ContextMenuItem onClick={() => setShowMenu(false)}>
              Close All
            </ContextMenuItem>
            <div className="h-px bg-border my-1" />
            <ContextMenuItem
              onClick={() => {
                onPin();
                setShowMenu(false);
              }}
            >
              {tab.pinned ? "Unpin" : "Pin"}
            </ContextMenuItem>
            <ContextMenuItem onClick={() => setShowMenu(false)}>
              <Copy size={12} /> Copy Path
            </ContextMenuItem>
          </div>
        </>
      )}
    </div>
  );
}

function ContextMenuItem({
  onClick,
  children,
}: {
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className="w-full flex items-center gap-2 px-3 py-1.5 text-sm text-left hover:bg-accent"
    >
      {children}
    </button>
  );
}

// ============================================================================
// Empty States
// ============================================================================

function EmptyPane() {
  return (
    <div className="h-full w-full flex items-center justify-center text-muted-foreground">
      <span className="text-sm">No file selected</span>
    </div>
  );
}

function WelcomeScreen() {
  return (
    <div className="h-full w-full flex flex-col items-center justify-center gap-8 text-muted-foreground">
      <div className="text-6xl font-bold text-primary/20">NEURECTOMY</div>
      <div className="flex flex-col items-center gap-2">
        <p className="text-lg">Welcome to NEURECTOMY IDE</p>
        <p className="text-sm">Open a file or folder to get started</p>
      </div>
      <div className="grid grid-cols-2 gap-4 text-sm">
        <KeyboardShortcut keys={["Ctrl", "P"]} label="Quick Open" />
        <KeyboardShortcut
          keys={["Ctrl", "Shift", "P"]}
          label="Command Palette"
        />
        <KeyboardShortcut keys={["Ctrl", "Shift", "E"]} label="Explorer" />
        <KeyboardShortcut keys={["Ctrl", "`"]} label="Terminal" />
      </div>
    </div>
  );
}

function KeyboardShortcut({ keys, label }: { keys: string[]; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex items-center gap-1">
        {keys.map((key, i) => (
          <span key={i}>
            <kbd className="px-2 py-1 text-xs bg-muted rounded border border-border">
              {key}
            </kbd>
            {i < keys.length - 1 && <span className="mx-0.5">+</span>}
          </span>
        ))}
      </div>
      <span className="text-muted-foreground">{label}</span>
    </div>
  );
}

export default EditorArea;
