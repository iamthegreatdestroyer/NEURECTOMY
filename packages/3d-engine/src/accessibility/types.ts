/**
 * Accessibility Types
 *
 * Type definitions for the 3D accessibility system including
 * ARIA labels, screen reader support, and keyboard navigation.
 *
 * @module @neurectomy/3d-engine/accessibility/types
 * @agents @CANVAS @APEX
 * @phase Phase 3 - Dimensional Forge
 */

// ============================================================================
// Element Identification
// ============================================================================

/**
 * Unique identifier for accessible elements
 */
export type AccessibleElementId = string;

/**
 * Types of 3D elements that need accessibility support
 */
export type AccessibleElementType =
  | "node"
  | "edge"
  | "cluster"
  | "timeline-point"
  | "timeline-range"
  | "viewport"
  | "camera-control"
  | "selection-region"
  | "annotation"
  | "measurement"
  | "axis"
  | "grid"
  | "legend"
  | "tooltip"
  | "menu"
  | "toolbar"
  | "panel"
  | "dialog"
  | "custom";

/**
 * ARIA role mappings for 3D elements
 */
export type AriaRole =
  | "img"
  | "figure"
  | "graphics-object"
  | "graphics-symbol"
  | "button"
  | "slider"
  | "listbox"
  | "listitem"
  | "option"
  | "tree"
  | "treeitem"
  | "group"
  | "region"
  | "toolbar"
  | "menu"
  | "menuitem"
  | "dialog"
  | "alertdialog"
  | "tooltip"
  | "status"
  | "log"
  | "marquee"
  | "timer"
  | "tablist"
  | "tab"
  | "tabpanel"
  | "application"
  | "document"
  | "presentation"
  | "none";

// ============================================================================
// Accessibility Description
// ============================================================================

/**
 * Description levels for different verbosity settings
 */
export type DescriptionLevel = "minimal" | "standard" | "verbose" | "debug";

/**
 * ARIA description for a 3D element
 */
export interface AriaDescription {
  /** Unique ID of the element */
  id: AccessibleElementId;

  /** Element type */
  type: AccessibleElementType;

  /** ARIA role */
  role: AriaRole;

  /** Short label for the element */
  label: string;

  /** Extended description */
  description?: string;

  /** Value (for sliders, progress, etc.) */
  value?: string | number;

  /** Minimum value */
  valueMin?: number;

  /** Maximum value */
  valueMax?: number;

  /** Value text (human-readable) */
  valueText?: string;

  /** Is element expanded (for trees, menus) */
  expanded?: boolean;

  /** Is element selected */
  selected?: boolean;

  /** Is element disabled */
  disabled?: boolean;

  /** Is element hidden */
  hidden?: boolean;

  /** Is element busy/loading */
  busy?: boolean;

  /** Has popup */
  hasPopup?: "menu" | "listbox" | "tree" | "grid" | "dialog" | false;

  /** Is element live region */
  live?: "off" | "polite" | "assertive";

  /** Atomic update for live regions */
  atomic?: boolean;

  /** Relevant changes for live regions */
  relevant?: ("additions" | "removals" | "text" | "all")[];

  /** Reference to controlled element */
  controls?: AccessibleElementId;

  /** Reference to described-by element */
  describedBy?: AccessibleElementId;

  /** Reference to labelled-by element */
  labelledBy?: AccessibleElementId;

  /** Reference to error message element */
  errorMessage?: AccessibleElementId;

  /** Position in set (for list items) */
  posInSet?: number;

  /** Total set size */
  setSize?: number;

  /** Level (for headings, tree items) */
  level?: number;

  /** Keyboard shortcut */
  keyShortcut?: string;

  /** Roving tabindex value */
  tabIndex?: number;

  /** Contextual descriptions by verbosity level */
  levelDescriptions?: Record<DescriptionLevel, string>;

  /** Spatial location description */
  spatialDescription?: string;

  /** Relationship descriptions */
  relationships?: {
    type:
      | "connected-to"
      | "child-of"
      | "parent-of"
      | "sibling-of"
      | "associated-with";
    targetId: AccessibleElementId;
    description?: string;
  }[];
}

// ============================================================================
// Screen Reader Announcements
// ============================================================================

/**
 * Priority levels for announcements
 */
export type AnnouncementPriority = "low" | "medium" | "high" | "critical";

/**
 * Screen reader announcement
 */
export interface ScreenReaderAnnouncement {
  /** Unique ID for the announcement */
  id: string;

  /** Message to announce */
  message: string;

  /** Priority level */
  priority: AnnouncementPriority;

  /** ARIA live politeness */
  politeness: "polite" | "assertive";

  /** Should clear previous announcements */
  clearPrevious?: boolean;

  /** Delay before announcement (ms) */
  delayMs?: number;

  /** Duration to keep announcement visible (ms) */
  durationMs?: number;

  /** Source element ID */
  sourceId?: AccessibleElementId;

  /** Additional context */
  context?: Record<string, unknown>;

  /** Timestamp */
  timestamp: number;
}

/**
 * Announcement queue configuration
 */
export interface AnnouncementQueueConfig {
  /** Maximum queue size */
  maxQueueSize: number;

  /** Debounce time for rapid announcements (ms) */
  debounceMs: number;

  /** Coalesce similar announcements */
  coalesceSimilar: boolean;

  /** Default priority */
  defaultPriority: AnnouncementPriority;

  /** Default politeness */
  defaultPoliteness: "polite" | "assertive";
}

// ============================================================================
// Keyboard Navigation
// ============================================================================

/**
 * Key bindings for navigation
 */
export type KeyBinding = {
  /** Primary key */
  key: string;

  /** Modifier keys */
  modifiers?: {
    ctrl?: boolean;
    alt?: boolean;
    shift?: boolean;
    meta?: boolean;
  };

  /** Action description */
  action: string;

  /** Handler function name */
  handler: string;

  /** Is repeatable when held */
  repeatable?: boolean;

  /** Context where binding is active */
  context?: string[];
};

/**
 * Navigation mode for 3D scene
 */
export type NavigationMode =
  | "spatial"
  | "list"
  | "graph"
  | "timeline"
  | "custom";

/**
 * Focus trap configuration
 */
export interface FocusTrapConfig {
  /** Enable focus trapping */
  enabled: boolean;

  /** Element IDs to include in trap */
  includedElements?: AccessibleElementId[];

  /** Element IDs to exclude from trap */
  excludedElements?: AccessibleElementId[];

  /** Allow escape key to exit trap */
  allowEscape: boolean;

  /** Wrap focus at boundaries */
  wrapFocus: boolean;

  /** Initial focus element */
  initialFocus?: AccessibleElementId;

  /** Return focus on exit */
  returnFocus: boolean;
}

/**
 * Keyboard navigation configuration
 */
export interface KeyboardNavigationConfig {
  /** Navigation mode */
  mode: NavigationMode;

  /** Key bindings */
  bindings: KeyBinding[];

  /** Focus trap config */
  focusTrap?: FocusTrapConfig;

  /** Enable roving tabindex */
  rovingTabindex: boolean;

  /** Skip hidden elements */
  skipHidden: boolean;

  /** Skip disabled elements */
  skipDisabled: boolean;

  /** Announce on focus change */
  announceOnFocus: boolean;

  /** Focus indicator style */
  focusIndicator: "outline" | "highlight" | "glow" | "custom" | "none";

  /** Focus indicator color */
  focusIndicatorColor?: string;

  /** Focus indicator size (px) */
  focusIndicatorSize?: number;
}

// ============================================================================
// Color Accessibility
// ============================================================================

/**
 * Color blindness type
 */
export type ColorBlindnessType =
  | "normal"
  | "protanopia" // Red-blind
  | "deuteranopia" // Green-blind
  | "tritanopia" // Blue-blind
  | "achromatopsia" // Complete color blindness
  | "protanomaly" // Red-weak
  | "deuteranomaly" // Green-weak
  | "tritanomaly"; // Blue-weak

/**
 * Color palette for accessibility
 */
export interface AccessibleColorPalette {
  /** Palette name */
  name: string;

  /** Color blindness type this palette supports */
  type: ColorBlindnessType;

  /** Primary colors */
  primary: string;
  secondary: string;
  accent: string;

  /** Status colors */
  success: string;
  warning: string;
  error: string;
  info: string;

  /** Node type colors */
  nodeColors: Record<string, string>;

  /** Edge type colors */
  edgeColors: Record<string, string>;

  /** Background colors */
  background: string;
  surface: string;

  /** Text colors */
  textPrimary: string;
  textSecondary: string;
  textDisabled: string;

  /** Border colors */
  border: string;
  focusBorder: string;

  /** Selection colors */
  selection: string;
  selectionText: string;

  /** Contrast ratio compliance level */
  contrastLevel: "AA" | "AAA";
}

/**
 * High contrast theme
 */
export interface HighContrastTheme extends AccessibleColorPalette {
  /** Is high contrast mode */
  highContrast: true;

  /** Minimum contrast ratio */
  minContrastRatio: number;

  /** Border width for visibility */
  borderWidth: number;

  /** Font weight for readability */
  fontWeight: number;
}

// ============================================================================
// Accessibility Settings
// ============================================================================

/**
 * User accessibility preferences
 */
export interface AccessibilityPreferences {
  /** Description verbosity level */
  verbosityLevel: DescriptionLevel;

  /** Color blindness mode */
  colorBlindnessMode: ColorBlindnessType;

  /** High contrast enabled */
  highContrast: boolean;

  /** Reduced motion */
  reducedMotion: boolean;

  /** Keyboard navigation enabled */
  keyboardNavigation: boolean;

  /** Screen reader mode */
  screenReaderMode: boolean;

  /** Focus indicator style */
  focusIndicator: "outline" | "highlight" | "glow" | "custom" | "none";

  /** Announcement politeness */
  announcementPoliteness: "polite" | "assertive";

  /** Auto-describe new elements */
  autoDescribe: boolean;

  /** Describe spatial relationships */
  spatialDescriptions: boolean;

  /** Font size scale */
  fontScale: number;

  /** Custom key bindings */
  customKeyBindings?: KeyBinding[];
}

/**
 * Accessibility manager configuration
 */
export interface AccessibilityConfig {
  /** Default preferences */
  defaults: AccessibilityPreferences;

  /** Available color palettes */
  colorPalettes: AccessibleColorPalette[];

  /** Announcement queue config */
  announcementQueue: AnnouncementQueueConfig;

  /** Keyboard navigation config */
  keyboardNavigation: KeyboardNavigationConfig;

  /** Auto-detect system preferences */
  autoDetectPreferences: boolean;

  /** Persist preferences to storage */
  persistPreferences: boolean;

  /** Storage key for preferences */
  storageKey: string;
}

// ============================================================================
// Events
// ============================================================================

/**
 * Accessibility event types
 */
export type AccessibilityEventType =
  | "focus-changed"
  | "selection-changed"
  | "announcement"
  | "preference-changed"
  | "description-generated"
  | "keyboard-action"
  | "navigation-mode-changed"
  | "color-mode-changed";

/**
 * Accessibility event
 */
export interface AccessibilityEvent {
  type: AccessibilityEventType;
  timestamp: number;
  data: unknown;
}

/**
 * Accessibility event listener
 */
export type AccessibilityEventListener = (event: AccessibilityEvent) => void;
