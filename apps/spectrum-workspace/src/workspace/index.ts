/**
 * Workspace Module
 *
 * Central workspace management system for Neurectomy IDE.
 * Provides pane management, dock management, and layout persistence.
 *
 * @module @neurectomy/workspace
 * @author @APEX @ARCHITECT
 *
 * @example
 * ```tsx
 * import {
 *   WorkspaceProvider,
 *   RootPaneGroup,
 *   Dock,
 *   useWorkspaceActions,
 * } from "@/workspace";
 *
 * function App() {
 *   return (
 *     <WorkspaceProvider>
 *       <div className="flex h-screen">
 *         <Dock position="left" />
 *         <RootPaneGroup />
 *         <Dock position="right" />
 *       </div>
 *       <Dock position="bottom" />
 *     </WorkspaceProvider>
 *   );
 * }
 * ```
 */

// ============================================================================
// Context & Types
// ============================================================================

export {
  WorkspaceContext,
  useWorkspaceContext,
  ItemContext,
  useItemContext,
  PanelContext,
  usePanelContext,
  PaneContext,
  usePaneContext,
} from "./WorkspaceContext";

export type {
  WorkspaceContextValue,
  WorkspaceSnapshot,
  WorkspaceActions,
  OpenItemOptions,
  ItemContextValue,
  PanelContextValue,
  PaneContextValue,
  Pane,
  PaneGroup,
  Dock as DockState,
} from "./WorkspaceContext";

// ============================================================================
// Provider
// ============================================================================

export {
  WorkspaceProvider,
  default as WorkspaceProviderDefault,
} from "./WorkspaceProvider";

export type { WorkspaceProviderProps } from "./WorkspaceProvider";

// ============================================================================
// Components
// ============================================================================

export {
  PaneGroupRenderer,
  PaneRenderer,
  RootPaneGroup,
  default as RootPaneGroupDefault,
} from "./PaneGroup";

export type {
  PaneGroupProps,
  PaneRendererProps,
  RootPaneGroupProps,
} from "./PaneGroup";

export {
  Dock,
  ActivityBarItem,
  CollapseButton,
  default as DockDefault,
} from "./Dock";

export type { DockProps, ActivityBarItemProps } from "./Dock";

// ============================================================================
// Hooks
// ============================================================================

export {
  // Workspace
  useWorkspaceActions,
  useWorkspaceSnapshot,
  useActivePaneId,
  useActivePane,
  usePane,
  usePanes,
  useRootPaneGroup,

  // Docks
  useDock,
  useDocks,
  useDockVisibility,
  useDockSizes,
  useDockControls,

  // Panels
  usePanel,
  usePanels,
  usePanelsInDock,
  useActivePanelInDock,
  usePanelState,
  usePanelControls,

  // Items
  useItem,
  useItems,
  useItemsInPane,
  useActiveItemInPane,
  useItemState,
  useItemControls,

  // Dirty Items
  useDirtyItems,
  useHasDirtyItems,

  // Layout
  useLayoutActions,

  // Focus
  useFocusActions,

  // Context Hooks
  useCurrentPane,
  useCurrentItem,
  useCurrentPanel,

  // Compound Hooks
  useShellState,
  useTabBarState,
  useDockState,
} from "./hooks";
