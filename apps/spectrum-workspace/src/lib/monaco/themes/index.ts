/**
 * NEURECTOMY Monaco Editor Themes
 *
 * Central export point for all custom Monaco Editor themes.
 *
 * @example
 * ```typescript
 * import * as monaco from 'monaco-editor';
 * import { registerAllThemes, NEURECTOMY_DARK_THEME_NAME } from '@/lib/monaco/themes';
 *
 * // Register all available themes
 * registerAllThemes(monaco);
 *
 * // Apply the NEURECTOMY dark theme
 * monaco.editor.setTheme(NEURECTOMY_DARK_THEME_NAME);
 * ```
 */

import type * as Monaco from "monaco-editor";

// Theme imports
export {
  neurectomyDarkTheme,
  defineNeurectomyTheme,
  neurectomyColors,
  NEURECTOMY_DARK_THEME_NAME,
} from "./neurectomy-dark";

// Re-export theme data for direct access
import {
  neurectomyDarkTheme,
  defineNeurectomyTheme,
  NEURECTOMY_DARK_THEME_NAME,
} from "./neurectomy-dark";

/**
 * Available theme names in the NEURECTOMY theme collection
 */
export const THEME_NAMES = {
  NEURECTOMY_DARK: NEURECTOMY_DARK_THEME_NAME,
} as const;

/**
 * Type representing available theme names
 */
export type ThemeName = (typeof THEME_NAMES)[keyof typeof THEME_NAMES];

/**
 * Map of all available themes with their definitions
 */
export const themes = {
  [NEURECTOMY_DARK_THEME_NAME]: neurectomyDarkTheme,
} as const;

/**
 * Registers all NEURECTOMY themes with Monaco Editor
 *
 * @param monaco - Monaco editor instance
 *
 * @example
 * ```typescript
 * import * as monaco from 'monaco-editor';
 * import { registerAllThemes } from '@/lib/monaco/themes';
 *
 * registerAllThemes(monaco);
 * ```
 */
export function registerAllThemes(monaco: typeof Monaco): void {
  defineNeurectomyTheme(monaco);
  // Add additional theme registrations here as they are created
  // defineFutureTheme(monaco);
}

/**
 * Registers a specific theme by name
 *
 * @param monaco - Monaco editor instance
 * @param themeName - Name of the theme to register
 * @returns true if theme was registered, false if theme not found
 *
 * @example
 * ```typescript
 * import * as monaco from 'monaco-editor';
 * import { registerTheme, THEME_NAMES } from '@/lib/monaco/themes';
 *
 * registerTheme(monaco, THEME_NAMES.NEURECTOMY_DARK);
 * ```
 */
export function registerTheme(
  monaco: typeof Monaco,
  themeName: ThemeName
): boolean {
  switch (themeName) {
    case NEURECTOMY_DARK_THEME_NAME:
      defineNeurectomyTheme(monaco);
      return true;
    default:
      console.warn(`[NEURECTOMY] Unknown theme: ${themeName}`);
      return false;
  }
}

/**
 * Applies a theme to the Monaco editor
 *
 * @param monaco - Monaco editor instance
 * @param themeName - Name of the theme to apply
 * @param registerFirst - Whether to register the theme before applying (default: true)
 *
 * @example
 * ```typescript
 * import * as monaco from 'monaco-editor';
 * import { applyTheme, THEME_NAMES } from '@/lib/monaco/themes';
 *
 * // Automatically registers and applies the theme
 * applyTheme(monaco, THEME_NAMES.NEURECTOMY_DARK);
 * ```
 */
export function applyTheme(
  monaco: typeof Monaco,
  themeName: ThemeName,
  registerFirst = true
): void {
  if (registerFirst) {
    registerTheme(monaco, themeName);
  }
  monaco.editor.setTheme(themeName);
}

/**
 * Gets a list of all available theme names
 *
 * @returns Array of available theme names
 */
export function getAvailableThemes(): ThemeName[] {
  return Object.values(THEME_NAMES);
}

/**
 * Checks if a theme name is valid
 *
 * @param themeName - Theme name to validate
 * @returns true if the theme name is valid
 */
export function isValidTheme(themeName: string): themeName is ThemeName {
  return Object.values(THEME_NAMES).includes(themeName as ThemeName);
}

export default {
  themes,
  THEME_NAMES,
  registerAllThemes,
  registerTheme,
  applyTheme,
  getAvailableThemes,
  isValidTheme,
};
