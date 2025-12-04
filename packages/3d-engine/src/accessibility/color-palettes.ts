/**
 * Color Blindness Theme Palettes
 *
 * Accessible color palettes for various types of color vision deficiency.
 * All palettes meet WCAG 2.1 AA contrast requirements.
 *
 * @module @neurectomy/3d-engine/accessibility/color-palettes
 * @agents @CANVAS
 * @phase Phase 3 - Dimensional Forge
 */

import type {
  AccessibleColorPalette,
  ColorBlindnessType,
  HighContrastTheme,
} from "./types";

// ============================================================================
// Normal Vision Palette (Reference)
// ============================================================================

/**
 * Standard color palette for normal vision
 */
export const NORMAL_PALETTE: AccessibleColorPalette = {
  name: "Default",
  type: "normal",

  // Primary colors
  primary: "#4F46E5", // Indigo
  secondary: "#7C3AED", // Violet
  accent: "#06B6D4", // Cyan

  // Status colors
  success: "#22C55E", // Green
  warning: "#F59E0B", // Amber
  error: "#EF4444", // Red
  info: "#3B82F6", // Blue

  // Node type colors
  nodeColors: {
    agent: "#4F46E5", // Indigo
    tool: "#22C55E", // Green
    prompt: "#F59E0B", // Amber
    memory: "#7C3AED", // Violet
    output: "#06B6D4", // Cyan
    input: "#3B82F6", // Blue
    error: "#EF4444", // Red
    default: "#6B7280", // Gray
  },

  // Edge type colors
  edgeColors: {
    data: "#4F46E5", // Indigo
    control: "#22C55E", // Green
    dependency: "#F59E0B", // Amber
    reference: "#7C3AED", // Violet
    default: "#9CA3AF", // Gray
  },

  // Background colors
  background: "#0F172A", // Slate 900
  surface: "#1E293B", // Slate 800

  // Text colors
  textPrimary: "#F8FAFC", // Slate 50
  textSecondary: "#94A3B8", // Slate 400
  textDisabled: "#475569", // Slate 600

  // Border colors
  border: "#334155", // Slate 700
  focusBorder: "#4F46E5", // Indigo

  // Selection colors
  selection: "#4F46E5", // Indigo
  selectionText: "#FFFFFF",

  contrastLevel: "AA",
};

// ============================================================================
// Protanopia (Red-Blind) Palette
// ============================================================================

/**
 * Color palette optimized for protanopia (red-blind) users
 * Uses blue-yellow discrimination which is preserved in protanopia
 */
export const PROTANOPIA_PALETTE: AccessibleColorPalette = {
  name: "Protanopia (Red-Blind)",
  type: "protanopia",

  // Primary colors - avoid red, use blue/yellow
  primary: "#2563EB", // Blue
  secondary: "#7C3AED", // Violet
  accent: "#FBBF24", // Yellow

  // Status colors - optimized for red-blind
  success: "#0EA5E9", // Sky (instead of green)
  warning: "#FBBF24", // Yellow (preserved)
  error: "#F97316", // Orange (visible as different from success)
  info: "#2563EB", // Blue

  // Node type colors - blue/yellow spectrum
  nodeColors: {
    agent: "#2563EB", // Blue
    tool: "#0EA5E9", // Sky
    prompt: "#FBBF24", // Yellow
    memory: "#7C3AED", // Violet
    output: "#06B6D4", // Cyan
    input: "#3B82F6", // Blue
    error: "#F97316", // Orange
    default: "#6B7280", // Gray
  },

  // Edge type colors
  edgeColors: {
    data: "#2563EB", // Blue
    control: "#0EA5E9", // Sky
    dependency: "#FBBF24", // Yellow
    reference: "#7C3AED", // Violet
    default: "#9CA3AF", // Gray
  },

  background: "#0F172A",
  surface: "#1E293B",
  textPrimary: "#F8FAFC",
  textSecondary: "#94A3B8",
  textDisabled: "#475569",
  border: "#334155",
  focusBorder: "#2563EB",
  selection: "#2563EB",
  selectionText: "#FFFFFF",

  contrastLevel: "AA",
};

// ============================================================================
// Deuteranopia (Green-Blind) Palette
// ============================================================================

/**
 * Color palette optimized for deuteranopia (green-blind) users
 * Uses blue-yellow discrimination which is preserved in deuteranopia
 */
export const DEUTERANOPIA_PALETTE: AccessibleColorPalette = {
  name: "Deuteranopia (Green-Blind)",
  type: "deuteranopia",

  // Primary colors - avoid green, use blue/yellow
  primary: "#2563EB", // Blue
  secondary: "#A855F7", // Purple
  accent: "#FBBF24", // Yellow

  // Status colors - optimized for green-blind
  success: "#0EA5E9", // Sky (instead of green)
  warning: "#FBBF24", // Yellow
  error: "#F97316", // Orange
  info: "#2563EB", // Blue

  // Node type colors
  nodeColors: {
    agent: "#2563EB", // Blue
    tool: "#0EA5E9", // Sky
    prompt: "#FBBF24", // Yellow
    memory: "#A855F7", // Purple
    output: "#06B6D4", // Cyan
    input: "#3B82F6", // Blue
    error: "#F97316", // Orange
    default: "#6B7280", // Gray
  },

  // Edge type colors
  edgeColors: {
    data: "#2563EB", // Blue
    control: "#0EA5E9", // Sky
    dependency: "#FBBF24", // Yellow
    reference: "#A855F7", // Purple
    default: "#9CA3AF", // Gray
  },

  background: "#0F172A",
  surface: "#1E293B",
  textPrimary: "#F8FAFC",
  textSecondary: "#94A3B8",
  textDisabled: "#475569",
  border: "#334155",
  focusBorder: "#2563EB",
  selection: "#2563EB",
  selectionText: "#FFFFFF",

  contrastLevel: "AA",
};

// ============================================================================
// Tritanopia (Blue-Blind) Palette
// ============================================================================

/**
 * Color palette optimized for tritanopia (blue-blind) users
 * Uses red-green discrimination which is preserved in tritanopia
 */
export const TRITANOPIA_PALETTE: AccessibleColorPalette = {
  name: "Tritanopia (Blue-Blind)",
  type: "tritanopia",

  // Primary colors - avoid blue, use red/green/pink
  primary: "#DB2777", // Pink
  secondary: "#DC2626", // Red
  accent: "#22C55E", // Green

  // Status colors - optimized for blue-blind
  success: "#22C55E", // Green (preserved)
  warning: "#F97316", // Orange
  error: "#DC2626", // Red (preserved)
  info: "#DB2777", // Pink

  // Node type colors - red/green spectrum
  nodeColors: {
    agent: "#DB2777", // Pink
    tool: "#22C55E", // Green
    prompt: "#F97316", // Orange
    memory: "#DC2626", // Red
    output: "#10B981", // Emerald
    input: "#F472B6", // Light Pink
    error: "#B91C1C", // Dark Red
    default: "#6B7280", // Gray
  },

  // Edge type colors
  edgeColors: {
    data: "#DB2777", // Pink
    control: "#22C55E", // Green
    dependency: "#F97316", // Orange
    reference: "#DC2626", // Red
    default: "#9CA3AF", // Gray
  },

  background: "#0F172A",
  surface: "#1E293B",
  textPrimary: "#F8FAFC",
  textSecondary: "#94A3B8",
  textDisabled: "#475569",
  border: "#334155",
  focusBorder: "#DB2777",
  selection: "#DB2777",
  selectionText: "#FFFFFF",

  contrastLevel: "AA",
};

// ============================================================================
// Monochromacy (Complete Color Blindness) Palette
// ============================================================================

/**
 * Color palette for achromatopsia (complete color blindness)
 * Uses only grayscale with patterns/shapes for differentiation
 */
export const MONOCHROMACY_PALETTE: AccessibleColorPalette = {
  name: "Monochromacy (Grayscale)",
  type: "achromatopsia",

  // All colors in grayscale
  primary: "#A1A1AA", // Gray 400
  secondary: "#71717A", // Gray 500
  accent: "#E4E4E7", // Gray 200

  // Status colors - different gray intensities
  success: "#E4E4E7", // Light gray
  warning: "#A1A1AA", // Medium gray
  error: "#52525B", // Dark gray
  info: "#D4D4D8", // Light-medium gray

  // Node type colors - grayscale with sufficient contrast
  nodeColors: {
    agent: "#F4F4F5", // Very light
    tool: "#D4D4D8", // Light
    prompt: "#A1A1AA", // Medium
    memory: "#71717A", // Medium-dark
    output: "#E4E4E7", // Light
    input: "#D4D4D8", // Light
    error: "#52525B", // Dark
    default: "#A1A1AA", // Medium
  },

  // Edge type colors
  edgeColors: {
    data: "#E4E4E7", // Light
    control: "#D4D4D8", // Light-medium
    dependency: "#A1A1AA", // Medium
    reference: "#71717A", // Medium-dark
    default: "#A1A1AA", // Medium
  },

  background: "#09090B", // Nearly black
  surface: "#18181B", // Dark gray
  textPrimary: "#FAFAFA", // Nearly white
  textSecondary: "#A1A1AA", // Medium gray
  textDisabled: "#52525B", // Dark gray
  border: "#3F3F46", // Gray 700
  focusBorder: "#FAFAFA", // White
  selection: "#FAFAFA", // White
  selectionText: "#09090B", // Black

  contrastLevel: "AAA",
};

// ============================================================================
// High Contrast Themes
// ============================================================================

/**
 * High contrast dark theme
 */
export const HIGH_CONTRAST_DARK: HighContrastTheme = {
  name: "High Contrast Dark",
  type: "normal",
  highContrast: true,
  minContrastRatio: 7,
  borderWidth: 2,
  fontWeight: 600,

  primary: "#00FFFF", // Cyan
  secondary: "#FF00FF", // Magenta
  accent: "#FFFF00", // Yellow

  success: "#00FF00", // Lime
  warning: "#FFFF00", // Yellow
  error: "#FF0000", // Red
  info: "#00FFFF", // Cyan

  nodeColors: {
    agent: "#00FFFF", // Cyan
    tool: "#00FF00", // Lime
    prompt: "#FFFF00", // Yellow
    memory: "#FF00FF", // Magenta
    output: "#00FFFF", // Cyan
    input: "#FFFFFF", // White
    error: "#FF0000", // Red
    default: "#FFFFFF", // White
  },

  edgeColors: {
    data: "#00FFFF", // Cyan
    control: "#00FF00", // Lime
    dependency: "#FFFF00", // Yellow
    reference: "#FF00FF", // Magenta
    default: "#FFFFFF", // White
  },

  background: "#000000", // Black
  surface: "#000000", // Black
  textPrimary: "#FFFFFF", // White
  textSecondary: "#FFFFFF", // White
  textDisabled: "#808080", // Gray
  border: "#FFFFFF", // White
  focusBorder: "#FFFF00", // Yellow
  selection: "#FFFF00", // Yellow
  selectionText: "#000000", // Black

  contrastLevel: "AAA",
};

/**
 * High contrast light theme
 */
export const HIGH_CONTRAST_LIGHT: HighContrastTheme = {
  name: "High Contrast Light",
  type: "normal",
  highContrast: true,
  minContrastRatio: 7,
  borderWidth: 2,
  fontWeight: 600,

  primary: "#0000FF", // Blue
  secondary: "#800080", // Purple
  accent: "#008000", // Green

  success: "#008000", // Green
  warning: "#806600", // Dark yellow/olive
  error: "#CC0000", // Dark red
  info: "#0000FF", // Blue

  nodeColors: {
    agent: "#0000FF", // Blue
    tool: "#008000", // Green
    prompt: "#806600", // Dark yellow
    memory: "#800080", // Purple
    output: "#006666", // Teal
    input: "#0000FF", // Blue
    error: "#CC0000", // Dark red
    default: "#000000", // Black
  },

  edgeColors: {
    data: "#0000FF", // Blue
    control: "#008000", // Green
    dependency: "#806600", // Dark yellow
    reference: "#800080", // Purple
    default: "#000000", // Black
  },

  background: "#FFFFFF", // White
  surface: "#FFFFFF", // White
  textPrimary: "#000000", // Black
  textSecondary: "#000000", // Black
  textDisabled: "#808080", // Gray
  border: "#000000", // Black
  focusBorder: "#0000FF", // Blue
  selection: "#0000FF", // Blue
  selectionText: "#FFFFFF", // White

  contrastLevel: "AAA",
};

// ============================================================================
// Palette Registry
// ============================================================================

/**
 * All available color palettes
 */
export const COLOR_PALETTES: Record<
  ColorBlindnessType,
  AccessibleColorPalette
> = {
  normal: NORMAL_PALETTE,
  protanopia: PROTANOPIA_PALETTE,
  deuteranopia: DEUTERANOPIA_PALETTE,
  tritanopia: TRITANOPIA_PALETTE,
  achromatopsia: MONOCHROMACY_PALETTE,
  protanomaly: PROTANOPIA_PALETTE, // Use same as full protanopia
  deuteranomaly: DEUTERANOPIA_PALETTE, // Use same as full deuteranopia
  tritanomaly: TRITANOPIA_PALETTE, // Use same as full tritanopia
};

/**
 * Get palette for color blindness type
 */
export function getPaletteForType(
  type: ColorBlindnessType
): AccessibleColorPalette {
  return COLOR_PALETTES[type] || NORMAL_PALETTE;
}

/**
 * Get high contrast theme
 */
export function getHighContrastTheme(
  mode: "dark" | "light"
): HighContrastTheme {
  return mode === "dark" ? HIGH_CONTRAST_DARK : HIGH_CONTRAST_LIGHT;
}

/**
 * Get all available palette names
 */
export function getAvailablePalettes(): {
  type: ColorBlindnessType;
  name: string;
}[] {
  return [
    { type: "normal", name: "Default" },
    { type: "protanopia", name: "Protanopia (Red-Blind)" },
    { type: "deuteranopia", name: "Deuteranopia (Green-Blind)" },
    { type: "tritanopia", name: "Tritanopia (Blue-Blind)" },
    { type: "achromatopsia", name: "Monochromacy (Grayscale)" },
  ];
}
