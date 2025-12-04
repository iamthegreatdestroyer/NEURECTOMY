/**
 * Accessibility Module
 *
 * Complete accessibility system for the 3D engine including
 * screen reader support, keyboard navigation, and color accessibility.
 *
 * @module @neurectomy/3d-engine/accessibility
 * @agents @CANVAS @APEX
 * @phase Phase 3 - Dimensional Forge
 */

// Types
export type {
  AccessibleElementId,
  AccessibleElementType,
  AriaRole,
  DescriptionLevel,
  AriaDescription,
  AnnouncementPriority,
  ScreenReaderAnnouncement,
  AnnouncementQueueConfig,
  KeyBinding,
  NavigationMode,
  FocusTrapConfig,
  KeyboardNavigationConfig,
  ColorBlindnessType,
  AccessibleColorPalette,
  HighContrastTheme,
  AccessibilityPreferences,
  AccessibilityConfig,
  AccessibilityEventType,
  AccessibilityEvent,
  AccessibilityEventListener,
} from "./types";

// ARIA Description Generator
export {
  AriaDescriptionGenerator,
  getAriaDescriptionGenerator,
  resetAriaDescriptionGenerator,
} from "./aria-descriptions";

// Screen Reader Manager
export {
  ScreenReaderManager,
  getScreenReaderManager,
  resetScreenReaderManager,
} from "./screen-reader";

// Keyboard Navigation
export {
  KeyboardNavigationManager,
  getKeyboardNavigationManager,
  resetKeyboardNavigationManager,
  DEFAULT_KEY_BINDINGS,
  type NavigationEvent,
} from "./keyboard-navigation";

// Color Palettes
export {
  NORMAL_PALETTE,
  PROTANOPIA_PALETTE,
  DEUTERANOPIA_PALETTE,
  TRITANOPIA_PALETTE,
  MONOCHROMACY_PALETTE,
  HIGH_CONTRAST_DARK,
  HIGH_CONTRAST_LIGHT,
  COLOR_PALETTES,
  getPaletteForType,
  getHighContrastTheme,
  getAvailablePalettes,
} from "./color-palettes";

// ============================================================================
// Convenience Functions
// ============================================================================

import type {
  AccessibilityPreferences,
  ColorBlindnessType,
  DescriptionLevel,
} from "./types";
import { getAriaDescriptionGenerator } from "./aria-descriptions";
import { getScreenReaderManager } from "./screen-reader";
import { getKeyboardNavigationManager } from "./keyboard-navigation";
import { getPaletteForType, getHighContrastTheme } from "./color-palettes";

/**
 * Default accessibility preferences
 */
export const DEFAULT_ACCESSIBILITY_PREFERENCES: AccessibilityPreferences = {
  verbosityLevel: "standard",
  colorBlindnessMode: "normal",
  highContrast: false,
  reducedMotion: false,
  keyboardNavigation: true,
  screenReaderMode: false,
  focusIndicator: "outline",
  announcementPoliteness: "polite",
  autoDescribe: true,
  spatialDescriptions: true,
  fontScale: 1,
};

/**
 * Detect user's system accessibility preferences
 */
export function detectSystemPreferences(): Partial<AccessibilityPreferences> {
  const preferences: Partial<AccessibilityPreferences> = {};

  // Only run in browser
  if (typeof window === "undefined") return preferences;

  // Check for reduced motion preference
  if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
    preferences.reducedMotion = true;
  }

  // Check for high contrast preference
  if (window.matchMedia?.("(prefers-contrast: more)").matches) {
    preferences.highContrast = true;
  }

  // Check for color scheme preference (can influence high contrast)
  if (window.matchMedia?.("(prefers-color-scheme: dark)").matches) {
    // User prefers dark mode
  }

  return preferences;
}

/**
 * Initialize the accessibility system
 */
export function initializeAccessibility(
  container: HTMLElement,
  preferences: Partial<AccessibilityPreferences> = {}
): {
  descriptionGenerator: ReturnType<typeof getAriaDescriptionGenerator>;
  screenReader: ReturnType<typeof getScreenReaderManager>;
  keyboardNavigation: ReturnType<typeof getKeyboardNavigationManager>;
  currentPalette: ReturnType<typeof getPaletteForType>;
  preferences: AccessibilityPreferences;
} {
  // Merge with detected system preferences
  const systemPrefs = detectSystemPreferences();
  const finalPrefs: AccessibilityPreferences = {
    ...DEFAULT_ACCESSIBILITY_PREFERENCES,
    ...systemPrefs,
    ...preferences,
  };

  // Initialize description generator
  const descriptionGenerator = getAriaDescriptionGenerator({
    defaultLevel: finalPrefs.verbosityLevel,
    enableCache: true,
  });

  // Initialize screen reader manager
  const screenReader = getScreenReaderManager({
    defaultPoliteness: finalPrefs.announcementPoliteness,
    coalesceSimilar: true,
  });

  // Initialize keyboard navigation
  const keyboardNavigation = getKeyboardNavigationManager({
    rovingTabindex: true,
    announceOnFocus: finalPrefs.screenReaderMode,
    focusIndicator: finalPrefs.focusIndicator,
  });

  // Attach to container
  keyboardNavigation.initialize(container);

  // Get appropriate color palette
  const currentPalette = finalPrefs.highContrast
    ? getHighContrastTheme("dark")
    : getPaletteForType(finalPrefs.colorBlindnessMode);

  return {
    descriptionGenerator,
    screenReader,
    keyboardNavigation,
    currentPalette,
    preferences: finalPrefs,
  };
}

/**
 * Update accessibility preferences
 */
export function updateAccessibilityPreferences(
  preferences: Partial<AccessibilityPreferences>
): {
  descriptionGenerator: ReturnType<typeof getAriaDescriptionGenerator>;
  palette: ReturnType<typeof getPaletteForType>;
} {
  const descriptionGenerator = getAriaDescriptionGenerator();

  // Update verbosity level
  if (preferences.verbosityLevel) {
    descriptionGenerator.setDefaultLevel(preferences.verbosityLevel);
  }

  // Get updated palette
  const palette = preferences.highContrast
    ? getHighContrastTheme("dark")
    : getPaletteForType(preferences.colorBlindnessMode || "normal");

  return {
    descriptionGenerator,
    palette,
  };
}

/**
 * Clean up accessibility system
 */
export function cleanupAccessibility(): void {
  // Import reset functions
  const { resetAriaDescriptionGenerator } = require("./aria-descriptions");
  const { resetScreenReaderManager } = require("./screen-reader");
  const { resetKeyboardNavigationManager } = require("./keyboard-navigation");

  resetAriaDescriptionGenerator();
  resetScreenReaderManager();
  resetKeyboardNavigationManager();
}

/**
 * Generate accessibility report for the current scene
 */
export function generateAccessibilityReport(
  elements: Array<{
    id: string;
    type: string;
    hasLabel: boolean;
    hasDescription: boolean;
    isFocusable: boolean;
    isKeyboardAccessible: boolean;
  }>
): {
  totalElements: number;
  labeled: number;
  described: number;
  focusable: number;
  keyboardAccessible: number;
  score: number;
  issues: string[];
} {
  const totalElements = elements.length;
  const labeled = elements.filter((e) => e.hasLabel).length;
  const described = elements.filter((e) => e.hasDescription).length;
  const focusable = elements.filter((e) => e.isFocusable).length;
  const keyboardAccessible = elements.filter(
    (e) => e.isKeyboardAccessible
  ).length;

  const issues: string[] = [];

  // Check for missing labels
  const unlabeled = elements.filter((e) => !e.hasLabel);
  if (unlabeled.length > 0) {
    issues.push(`${unlabeled.length} elements missing labels`);
  }

  // Check for non-focusable interactive elements
  const nonFocusable = elements.filter((e) => !e.isFocusable);
  if (nonFocusable.length > 0) {
    issues.push(`${nonFocusable.length} elements are not focusable`);
  }

  // Calculate score (0-100)
  const labelScore = (labeled / totalElements) * 25;
  const descriptionScore = (described / totalElements) * 25;
  const focusScore = (focusable / totalElements) * 25;
  const keyboardScore = (keyboardAccessible / totalElements) * 25;
  const score = Math.round(
    labelScore + descriptionScore + focusScore + keyboardScore
  );

  return {
    totalElements,
    labeled,
    described,
    focusable,
    keyboardAccessible,
    score,
    issues,
  };
}
