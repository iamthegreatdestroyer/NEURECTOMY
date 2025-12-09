/**
 * Shell Components - Export Barrel
 *
 * Professional IDE shell components for NEURECTOMY.
 * Based on patterns from VS Code, Continue, Zed, Lapce, IntelliJ, and Theia.
 *
 * @module @neurectomy/shell
 */

// Core Layout
export { ApplicationShell } from "./ApplicationShell";
export type { ApplicationShellProps } from "./ApplicationShell";

// Activity Bar
export { ActivityBar } from "./ActivityBar";
export type { ActivityBarProps, ActivityBarItem } from "./ActivityBar";

// Sidebars
export { SidebarContainer } from "./SidebarContainer";
export type {
  SidebarContainerProps,
  SidebarSection,
  SidebarAction,
} from "./SidebarContainer";

// Editor Area
export { EditorArea, useEditorArea } from "./EditorArea";
export type {
  EditorAreaProps,
  EditorTab,
  EditorPane,
  EditorSplit,
  EditorNode,
  SplitDirection,
} from "./EditorArea";

// AI Panel
export { AIPanel } from "./AIPanel";
export type {
  AIPanelProps,
  ChatMessage,
  MessageRole,
  MessageContext,
  CodeBlock,
} from "./AIPanel";

// Bottom Panel
export { BottomPanel } from "./BottomPanel";
export type {
  BottomPanelProps,
  BottomPanelTab,
  TerminalInstance,
  Problem,
  OutputMessage,
} from "./BottomPanel";

// Status Bar
export {
  StatusBar,
  StatusBarBranch,
  StatusBarLanguage,
  StatusBarEncoding,
  StatusBarLineCol,
  StatusBarIndent,
  StatusBarEOL,
  StatusBarNotifications,
} from "./StatusBar";
export type {
  StatusBarProps,
  StatusBarItem,
  PresetItemProps,
} from "./StatusBar";

// Command Palette
export { CommandPalette, useCommandPalette } from "./CommandPalette";
export type {
  CommandPaletteProps,
  Command,
  CommandCategory,
} from "./CommandPalette";

// Quick Open (Ctrl+P)
export { QuickOpen, useQuickOpen } from "./QuickOpen";
export type { QuickOpenProps, FileItem } from "./QuickOpen";

// Notifications
export {
  NotificationProvider,
  useNotifications,
  useNotify,
} from "./NotificationToast";
export type {
  Notification,
  NotificationType,
  NotificationAction,
} from "./NotificationToast";

// Breadcrumb Navigation
export {
  BreadcrumbNavigation,
  useDocumentSymbols,
} from "./BreadcrumbNavigation";
export type {
  BreadcrumbNavigationProps,
  PathSegment,
  SymbolSegment,
  SymbolKind,
  BreadcrumbSegment,
} from "./BreadcrumbNavigation";

// Search Panel
export { SearchPanel, mockSearch } from "./SearchPanel";
export type {
  SearchPanelProps,
  SearchMatch,
  SearchFileResult,
  SearchOptions,
} from "./SearchPanel";
