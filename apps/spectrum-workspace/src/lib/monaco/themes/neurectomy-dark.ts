/**
 * NEURECTOMY Dark Theme for Monaco Editor
 *
 * A custom dark theme matching the NEURECTOMY design system.
 * Optimized for TypeScript, JavaScript, Python, Rust, JSON, YAML, and Markdown.
 */

import type * as Monaco from "monaco-editor";

// Design System Colors
const colors = {
  // Backgrounds
  bgPrimary: "#0a0a0f",
  bgSecondary: "#13131a",
  bgTertiary: "#1a1a24",

  // Accents
  accentPrimary: "#6366f1", // Indigo
  accentSecondary: "#8b5cf6", // Purple
  accentPrimaryDim: "#4f46e5", // Darker indigo
  accentSecondaryDim: "#7c3aed", // Darker purple

  // Text
  textPrimary: "#e4e4e7",
  textSecondary: "#a1a1aa",
  textMuted: "#71717a",

  // Semantic
  success: "#22c55e",
  successDim: "#16a34a",
  warning: "#f59e0b",
  warningDim: "#d97706",
  error: "#ef4444",
  errorDim: "#dc2626",
  info: "#3b82f6",
  infoDim: "#2563eb",

  // Syntax - Keywords & Control Flow
  keyword: "#c084fc", // Purple-400
  control: "#a78bfa", // Violet-400

  // Syntax - Types & Classes
  type: "#67e8f9", // Cyan-300
  class: "#5eead4", // Teal-300
  interface: "#2dd4bf", // Teal-400

  // Syntax - Functions & Methods
  function: "#818cf8", // Indigo-400
  method: "#a5b4fc", // Indigo-300
  parameter: "#fbbf24", // Amber-400

  // Syntax - Variables & Properties
  variable: "#e4e4e7", // Zinc-200
  property: "#93c5fd", // Blue-300
  constant: "#f472b6", // Pink-400

  // Syntax - Strings & Numbers
  string: "#86efac", // Green-300
  stringEscape: "#4ade80", // Green-400
  number: "#fdba74", // Orange-300
  regexp: "#fb923c", // Orange-400

  // Syntax - Comments & Documentation
  comment: "#6b7280", // Gray-500
  docComment: "#9ca3af", // Gray-400

  // Syntax - Operators & Punctuation
  operator: "#f9a8d4", // Pink-300
  punctuation: "#d4d4d8", // Zinc-300

  // Syntax - HTML/JSX
  tag: "#f472b6", // Pink-400
  attribute: "#fcd34d", // Amber-300

  // Syntax - Markdown
  heading: "#818cf8", // Indigo-400
  link: "#60a5fa", // Blue-400
  bold: "#f9fafb", // Gray-50
  italic: "#d1d5db", // Gray-300
  code: "#a78bfa", // Violet-400

  // Git
  gitAdded: "#22c55e",
  gitModified: "#f59e0b",
  gitDeleted: "#ef4444",
  gitUntracked: "#a1a1aa",
  gitIgnored: "#52525b",
  gitConflicting: "#f472b6",

  // Diff
  diffInserted: "#22c55e20",
  diffInsertedLine: "#22c55e10",
  diffRemoved: "#ef444420",
  diffRemovedLine: "#ef444410",

  // UI Elements
  border: "#27272a",
  borderActive: "#3f3f46",
  selection: "#6366f140",
  selectionHighlight: "#6366f125",
  wordHighlight: "#8b5cf630",
  wordHighlightStrong: "#8b5cf650",
  findMatch: "#f59e0b40",
  findMatchHighlight: "#f59e0b20",
  lineHighlight: "#1a1a2480",
  rangeHighlight: "#6366f115",
} as const;

/**
 * Monaco Editor theme definition for NEURECTOMY Dark
 */
export const neurectomyDarkTheme: Monaco.editor.IStandaloneThemeData = {
  base: "vs-dark",
  inherit: false,
  rules: [
    // ═══════════════════════════════════════════════════════════════
    // BASE TOKENS
    // ═══════════════════════════════════════════════════════════════
    { token: "", foreground: colors.textPrimary, background: colors.bgPrimary },
    { token: "invalid", foreground: colors.error },
    { token: "emphasis", fontStyle: "italic" },
    { token: "strong", fontStyle: "bold" },

    // ═══════════════════════════════════════════════════════════════
    // COMMENTS
    // ═══════════════════════════════════════════════════════════════
    { token: "comment", foreground: colors.comment, fontStyle: "italic" },
    { token: "comment.line", foreground: colors.comment, fontStyle: "italic" },
    { token: "comment.block", foreground: colors.comment, fontStyle: "italic" },
    {
      token: "comment.doc",
      foreground: colors.docComment,
      fontStyle: "italic",
    },
    {
      token: "comment.block.documentation",
      foreground: colors.docComment,
      fontStyle: "italic",
    },

    // ═══════════════════════════════════════════════════════════════
    // KEYWORDS & CONTROL FLOW
    // ═══════════════════════════════════════════════════════════════
    { token: "keyword", foreground: colors.keyword },
    { token: "keyword.control", foreground: colors.control },
    { token: "keyword.operator", foreground: colors.operator },
    { token: "keyword.other", foreground: colors.keyword },
    { token: "keyword.type", foreground: colors.type },

    // Storage
    { token: "storage", foreground: colors.keyword },
    { token: "storage.type", foreground: colors.keyword },
    { token: "storage.modifier", foreground: colors.keyword },

    // ═══════════════════════════════════════════════════════════════
    // TYPES & CLASSES
    // ═══════════════════════════════════════════════════════════════
    { token: "type", foreground: colors.type },
    { token: "type.identifier", foreground: colors.type },
    { token: "type.primitive", foreground: colors.type },
    { token: "class", foreground: colors.class },
    { token: "class.name", foreground: colors.class },
    { token: "interface", foreground: colors.interface },
    { token: "interface.name", foreground: colors.interface },
    { token: "struct", foreground: colors.class },
    { token: "enum", foreground: colors.type },
    { token: "enum.name", foreground: colors.type },
    { token: "enumMember", foreground: colors.constant },
    { token: "namespace", foreground: colors.type },

    // ═══════════════════════════════════════════════════════════════
    // FUNCTIONS & METHODS
    // ═══════════════════════════════════════════════════════════════
    { token: "function", foreground: colors.function },
    { token: "function.name", foreground: colors.function },
    { token: "function.call", foreground: colors.function },
    { token: "method", foreground: colors.method },
    { token: "method.name", foreground: colors.method },
    { token: "method.call", foreground: colors.method },
    { token: "parameter", foreground: colors.parameter },
    { token: "parameter.name", foreground: colors.parameter },
    { token: "macro", foreground: colors.function },

    // ═══════════════════════════════════════════════════════════════
    // VARIABLES & PROPERTIES
    // ═══════════════════════════════════════════════════════════════
    { token: "variable", foreground: colors.variable },
    { token: "variable.name", foreground: colors.variable },
    { token: "variable.parameter", foreground: colors.parameter },
    { token: "variable.readonly", foreground: colors.constant },
    { token: "variable.predefined", foreground: colors.constant },
    { token: "property", foreground: colors.property },
    { token: "property.name", foreground: colors.property },
    { token: "constant", foreground: colors.constant },
    { token: "constant.language", foreground: colors.constant },
    { token: "constant.numeric", foreground: colors.number },

    // ═══════════════════════════════════════════════════════════════
    // STRINGS & NUMBERS
    // ═══════════════════════════════════════════════════════════════
    { token: "string", foreground: colors.string },
    { token: "string.quoted", foreground: colors.string },
    { token: "string.template", foreground: colors.string },
    { token: "string.escape", foreground: colors.stringEscape },
    { token: "string.regexp", foreground: colors.regexp },
    { token: "string.key", foreground: colors.property },
    { token: "string.value", foreground: colors.string },
    { token: "number", foreground: colors.number },
    { token: "number.hex", foreground: colors.number },
    { token: "number.float", foreground: colors.number },

    // ═══════════════════════════════════════════════════════════════
    // OPERATORS & PUNCTUATION
    // ═══════════════════════════════════════════════════════════════
    { token: "operator", foreground: colors.operator },
    { token: "operator.arithmetic", foreground: colors.operator },
    { token: "operator.assignment", foreground: colors.operator },
    { token: "operator.comparison", foreground: colors.operator },
    { token: "operator.logical", foreground: colors.operator },
    { token: "delimiter", foreground: colors.punctuation },
    { token: "delimiter.bracket", foreground: colors.punctuation },
    { token: "delimiter.parenthesis", foreground: colors.punctuation },
    { token: "delimiter.square", foreground: colors.punctuation },
    { token: "delimiter.angle", foreground: colors.punctuation },
    { token: "punctuation", foreground: colors.punctuation },

    // ═══════════════════════════════════════════════════════════════
    // TYPESCRIPT / JAVASCRIPT SPECIFIC
    // ═══════════════════════════════════════════════════════════════
    { token: "identifier", foreground: colors.variable },
    { token: "identifier.ts", foreground: colors.variable },
    { token: "identifier.js", foreground: colors.variable },
    { token: "metatag", foreground: colors.keyword },
    { token: "metatag.content", foreground: colors.string },
    { token: "annotation", foreground: colors.docComment },
    { token: "decorator", foreground: colors.function },

    // ═══════════════════════════════════════════════════════════════
    // PYTHON SPECIFIC
    // ═══════════════════════════════════════════════════════════════
    { token: "keyword.python", foreground: colors.keyword },
    { token: "identifier.python", foreground: colors.variable },
    { token: "type.python", foreground: colors.type },
    { token: "decorator.python", foreground: colors.function },
    { token: "string.python", foreground: colors.string },
    { token: "string.escape.python", foreground: colors.stringEscape },

    // ═══════════════════════════════════════════════════════════════
    // RUST SPECIFIC
    // ═══════════════════════════════════════════════════════════════
    { token: "keyword.rust", foreground: colors.keyword },
    { token: "keyword.control.rust", foreground: colors.control },
    { token: "type.rust", foreground: colors.type },
    { token: "lifetime.rust", foreground: colors.parameter },
    { token: "macro.rust", foreground: colors.function },
    { token: "attribute.rust", foreground: colors.docComment },

    // ═══════════════════════════════════════════════════════════════
    // JSON SPECIFIC
    // ═══════════════════════════════════════════════════════════════
    { token: "string.key.json", foreground: colors.property },
    { token: "string.value.json", foreground: colors.string },
    { token: "number.json", foreground: colors.number },
    { token: "keyword.json", foreground: colors.constant },

    // ═══════════════════════════════════════════════════════════════
    // YAML SPECIFIC
    // ═══════════════════════════════════════════════════════════════
    { token: "key.yaml", foreground: colors.property },
    { token: "string.yaml", foreground: colors.string },
    { token: "number.yaml", foreground: colors.number },
    { token: "tag.yaml", foreground: colors.type },
    { token: "type.yaml", foreground: colors.type },

    // ═══════════════════════════════════════════════════════════════
    // HTML / JSX / XML
    // ═══════════════════════════════════════════════════════════════
    { token: "tag", foreground: colors.tag },
    { token: "tag.html", foreground: colors.tag },
    { token: "tag.jsx", foreground: colors.tag },
    { token: "tag.xml", foreground: colors.tag },
    { token: "tag.id", foreground: colors.tag },
    { token: "tag.class", foreground: colors.tag },
    { token: "attribute.name", foreground: colors.attribute },
    { token: "attribute.name.html", foreground: colors.attribute },
    { token: "attribute.name.jsx", foreground: colors.attribute },
    { token: "attribute.value", foreground: colors.string },
    { token: "attribute.value.html", foreground: colors.string },
    { token: "attribute.value.jsx", foreground: colors.string },

    // ═══════════════════════════════════════════════════════════════
    // MARKDOWN SPECIFIC
    // ═══════════════════════════════════════════════════════════════
    { token: "heading", foreground: colors.heading, fontStyle: "bold" },
    { token: "heading.1", foreground: colors.heading, fontStyle: "bold" },
    { token: "heading.2", foreground: colors.heading, fontStyle: "bold" },
    { token: "heading.3", foreground: colors.heading, fontStyle: "bold" },
    { token: "heading.4", foreground: colors.heading, fontStyle: "bold" },
    { token: "heading.5", foreground: colors.heading, fontStyle: "bold" },
    { token: "heading.6", foreground: colors.heading, fontStyle: "bold" },
    { token: "markup.heading", foreground: colors.heading, fontStyle: "bold" },
    { token: "markup.bold", foreground: colors.bold, fontStyle: "bold" },
    { token: "markup.italic", foreground: colors.italic, fontStyle: "italic" },
    { token: "markup.underline", fontStyle: "underline" },
    { token: "markup.strikethrough", fontStyle: "strikethrough" },
    { token: "markup.inline.raw", foreground: colors.code },
    {
      token: "markup.quote",
      foreground: colors.textSecondary,
      fontStyle: "italic",
    },
    { token: "meta.link", foreground: colors.link },
    { token: "string.link", foreground: colors.link },
    { token: "markup.link", foreground: colors.link },

    // ═══════════════════════════════════════════════════════════════
    // CSS / SCSS / LESS
    // ═══════════════════════════════════════════════════════════════
    { token: "selector", foreground: colors.tag },
    { token: "selector.css", foreground: colors.tag },
    { token: "property.css", foreground: colors.property },
    { token: "value.css", foreground: colors.string },
    { token: "unit.css", foreground: colors.number },
    { token: "variable.css", foreground: colors.constant },

    // ═══════════════════════════════════════════════════════════════
    // SQL
    // ═══════════════════════════════════════════════════════════════
    { token: "keyword.sql", foreground: colors.keyword },
    { token: "operator.sql", foreground: colors.operator },
    { token: "string.sql", foreground: colors.string },
    { token: "number.sql", foreground: colors.number },
    { token: "predefined.sql", foreground: colors.function },

    // ═══════════════════════════════════════════════════════════════
    // SHELL / BASH
    // ═══════════════════════════════════════════════════════════════
    { token: "keyword.shell", foreground: colors.keyword },
    { token: "variable.shell", foreground: colors.constant },
    { token: "string.shell", foreground: colors.string },
  ],
  colors: {
    // ═══════════════════════════════════════════════════════════════
    // EDITOR BASE
    // ═══════════════════════════════════════════════════════════════
    "editor.background": colors.bgPrimary,
    "editor.foreground": colors.textPrimary,
    "editorLineNumber.foreground": colors.textMuted,
    "editorLineNumber.activeForeground": colors.textSecondary,
    "editorCursor.foreground": colors.accentPrimary,
    "editorCursor.background": colors.bgPrimary,

    // ═══════════════════════════════════════════════════════════════
    // SELECTION & HIGHLIGHTS
    // ═══════════════════════════════════════════════════════════════
    "editor.selectionBackground": colors.selection,
    "editor.selectionForeground": colors.textPrimary,
    "editor.inactiveSelectionBackground": colors.selectionHighlight,
    "editor.selectionHighlightBackground": colors.selectionHighlight,
    "editor.selectionHighlightBorder": colors.accentPrimaryDim + "50",

    "editor.wordHighlightBackground": colors.wordHighlight,
    "editor.wordHighlightStrongBackground": colors.wordHighlightStrong,
    "editor.wordHighlightBorder": colors.accentSecondaryDim + "50",
    "editor.wordHighlightStrongBorder": colors.accentSecondary + "50",

    "editor.findMatchBackground": colors.findMatch,
    "editor.findMatchHighlightBackground": colors.findMatchHighlight,
    "editor.findMatchBorder": colors.warning,
    "editor.findMatchHighlightBorder": colors.warningDim + "50",

    "editor.hoverHighlightBackground": colors.rangeHighlight,
    "editor.lineHighlightBackground": colors.lineHighlight,
    "editor.lineHighlightBorder": "transparent",
    "editor.rangeHighlightBackground": colors.rangeHighlight,

    // ═══════════════════════════════════════════════════════════════
    // BRACKET MATCHING
    // ═══════════════════════════════════════════════════════════════
    "editorBracketMatch.background": colors.accentPrimary + "30",
    "editorBracketMatch.border": colors.accentPrimary,
    "editorBracketHighlight.foreground1": "#fbbf24", // Amber
    "editorBracketHighlight.foreground2": "#f472b6", // Pink
    "editorBracketHighlight.foreground3": "#60a5fa", // Blue
    "editorBracketHighlight.foreground4": "#34d399", // Emerald
    "editorBracketHighlight.foreground5": "#a78bfa", // Violet
    "editorBracketHighlight.foreground6": "#fb923c", // Orange
    "editorBracketHighlight.unexpectedBracket.foreground": colors.error,

    // Bracket pair guides
    "editorBracketPairGuide.activeBackground1": "#fbbf2480",
    "editorBracketPairGuide.activeBackground2": "#f472b680",
    "editorBracketPairGuide.activeBackground3": "#60a5fa80",
    "editorBracketPairGuide.activeBackground4": "#34d39980",
    "editorBracketPairGuide.activeBackground5": "#a78bfa80",
    "editorBracketPairGuide.activeBackground6": "#fb923c80",
    "editorBracketPairGuide.background1": "#fbbf2440",
    "editorBracketPairGuide.background2": "#f472b640",
    "editorBracketPairGuide.background3": "#60a5fa40",
    "editorBracketPairGuide.background4": "#34d39940",
    "editorBracketPairGuide.background5": "#a78bfa40",
    "editorBracketPairGuide.background6": "#fb923c40",

    // ═══════════════════════════════════════════════════════════════
    // DIFF EDITOR
    // ═══════════════════════════════════════════════════════════════
    "diffEditor.insertedTextBackground": colors.diffInserted,
    "diffEditor.insertedLineBackground": colors.diffInsertedLine,
    "diffEditor.insertedTextBorder": colors.success + "30",
    "diffEditor.removedTextBackground": colors.diffRemoved,
    "diffEditor.removedLineBackground": colors.diffRemovedLine,
    "diffEditor.removedTextBorder": colors.error + "30",
    "diffEditor.border": colors.border,
    "diffEditor.diagonalFill": colors.bgTertiary,
    "diffEditor.unchangedRegionBackground": colors.bgSecondary,
    "diffEditor.unchangedRegionForeground": colors.textSecondary,
    "diffEditor.unchangedCodeBackground": colors.bgSecondary + "80",
    "diffEditorGutter.insertedLineBackground": colors.success + "25",
    "diffEditorGutter.removedLineBackground": colors.error + "25",
    "diffEditorOverview.insertedForeground": colors.success,
    "diffEditorOverview.removedForeground": colors.error,

    // ═══════════════════════════════════════════════════════════════
    // GIT DECORATIONS
    // ═══════════════════════════════════════════════════════════════
    "gitDecoration.addedResourceForeground": colors.gitAdded,
    "gitDecoration.modifiedResourceForeground": colors.gitModified,
    "gitDecoration.deletedResourceForeground": colors.gitDeleted,
    "gitDecoration.renamedResourceForeground": colors.info,
    "gitDecoration.stageModifiedResourceForeground": colors.gitModified,
    "gitDecoration.stageDeletedResourceForeground": colors.gitDeleted,
    "gitDecoration.untrackedResourceForeground": colors.gitUntracked,
    "gitDecoration.ignoredResourceForeground": colors.gitIgnored,
    "gitDecoration.conflictingResourceForeground": colors.gitConflicting,
    "gitDecoration.submoduleResourceForeground": colors.accentSecondary,

    // ═══════════════════════════════════════════════════════════════
    // GUTTER & OVERVIEW RULER
    // ═══════════════════════════════════════════════════════════════
    "editorGutter.background": colors.bgPrimary,
    "editorGutter.modifiedBackground": colors.warning,
    "editorGutter.addedBackground": colors.success,
    "editorGutter.deletedBackground": colors.error,
    "editorGutter.commentRangeForeground": colors.textMuted,
    "editorGutter.commentGlyphForeground": colors.accentPrimary,
    "editorGutter.foldingControlForeground": colors.textSecondary,

    "editorOverviewRuler.border": colors.border,
    "editorOverviewRuler.background": colors.bgPrimary,
    "editorOverviewRuler.findMatchForeground": colors.warning + "80",
    "editorOverviewRuler.rangeHighlightForeground": colors.accentPrimary + "60",
    "editorOverviewRuler.selectionHighlightForeground":
      colors.accentPrimary + "80",
    "editorOverviewRuler.wordHighlightForeground":
      colors.accentSecondary + "60",
    "editorOverviewRuler.wordHighlightStrongForeground":
      colors.accentSecondary + "80",
    "editorOverviewRuler.modifiedForeground": colors.warning + "80",
    "editorOverviewRuler.addedForeground": colors.success + "80",
    "editorOverviewRuler.deletedForeground": colors.error + "80",
    "editorOverviewRuler.errorForeground": colors.error,
    "editorOverviewRuler.warningForeground": colors.warning,
    "editorOverviewRuler.infoForeground": colors.info,
    "editorOverviewRuler.bracketMatchForeground": colors.accentPrimary + "80",

    // ═══════════════════════════════════════════════════════════════
    // ERROR / WARNING / INFO MARKERS
    // ═══════════════════════════════════════════════════════════════
    "editorError.foreground": colors.error,
    "editorError.background": colors.error + "15",
    "editorError.border": "transparent",
    "editorWarning.foreground": colors.warning,
    "editorWarning.background": colors.warning + "15",
    "editorWarning.border": "transparent",
    "editorInfo.foreground": colors.info,
    "editorInfo.background": colors.info + "15",
    "editorInfo.border": "transparent",
    "editorHint.foreground": colors.success,
    "editorHint.border": "transparent",

    // ═══════════════════════════════════════════════════════════════
    // INDENTATION GUIDES
    // ═══════════════════════════════════════════════════════════════
    "editorIndentGuide.background": colors.border,
    "editorIndentGuide.background1": colors.border,
    "editorIndentGuide.background2": colors.border,
    "editorIndentGuide.background3": colors.border,
    "editorIndentGuide.background4": colors.border,
    "editorIndentGuide.background5": colors.border,
    "editorIndentGuide.background6": colors.border,
    "editorIndentGuide.activeBackground": colors.borderActive,
    "editorIndentGuide.activeBackground1": colors.borderActive,
    "editorIndentGuide.activeBackground2": colors.borderActive,
    "editorIndentGuide.activeBackground3": colors.borderActive,
    "editorIndentGuide.activeBackground4": colors.borderActive,
    "editorIndentGuide.activeBackground5": colors.borderActive,
    "editorIndentGuide.activeBackground6": colors.borderActive,

    // ═══════════════════════════════════════════════════════════════
    // RULERS
    // ═══════════════════════════════════════════════════════════════
    "editorRuler.foreground": colors.border,

    // ═══════════════════════════════════════════════════════════════
    // WHITESPACE & TRAILING SPACES
    // ═══════════════════════════════════════════════════════════════
    "editorWhitespace.foreground": colors.textMuted + "40",
    "editorUnnecessaryCode.opacity": "#00000080",
    "editorUnnecessaryCode.border": colors.warning + "60",

    // ═══════════════════════════════════════════════════════════════
    // CODE LENS
    // ═══════════════════════════════════════════════════════════════
    "editorCodeLens.foreground": colors.textMuted,

    // ═══════════════════════════════════════════════════════════════
    // LINKED EDITING
    // ═══════════════════════════════════════════════════════════════
    "editorLink.activeForeground": colors.accentPrimary,
    "editorLightBulb.foreground": colors.warning,
    "editorLightBulbAutoFix.foreground": colors.success,

    // ═══════════════════════════════════════════════════════════════
    // INLAY HINTS
    // ═══════════════════════════════════════════════════════════════
    "editorInlayHint.foreground": colors.textMuted,
    "editorInlayHint.background": colors.bgTertiary + "80",
    "editorInlayHint.typeForeground": colors.type + "90",
    "editorInlayHint.typeBackground": colors.bgTertiary + "60",
    "editorInlayHint.parameterForeground": colors.parameter + "90",
    "editorInlayHint.parameterBackground": colors.bgTertiary + "60",

    // ═══════════════════════════════════════════════════════════════
    // WIDGETS (Suggest, Hover, etc.)
    // ═══════════════════════════════════════════════════════════════
    "editorWidget.foreground": colors.textPrimary,
    "editorWidget.background": colors.bgSecondary,
    "editorWidget.border": colors.border,
    "editorWidget.resizeBorder": colors.accentPrimary,

    "editorSuggestWidget.background": colors.bgSecondary,
    "editorSuggestWidget.border": colors.border,
    "editorSuggestWidget.foreground": colors.textPrimary,
    "editorSuggestWidget.highlightForeground": colors.accentPrimary,
    "editorSuggestWidget.selectedBackground": colors.bgTertiary,
    "editorSuggestWidget.selectedForeground": colors.textPrimary,
    "editorSuggestWidget.selectedIconForeground": colors.accentPrimary,
    "editorSuggestWidget.focusHighlightForeground": colors.accentPrimary,

    "editorHoverWidget.foreground": colors.textPrimary,
    "editorHoverWidget.background": colors.bgSecondary,
    "editorHoverWidget.border": colors.border,
    "editorHoverWidget.highlightForeground": colors.accentPrimary,
    "editorHoverWidget.statusBarBackground": colors.bgTertiary,

    // ═══════════════════════════════════════════════════════════════
    // PEEK VIEW
    // ═══════════════════════════════════════════════════════════════
    "peekView.border": colors.accentPrimary,
    "peekViewEditor.background": colors.bgSecondary,
    "peekViewEditorGutter.background": colors.bgSecondary,
    "peekViewEditor.matchHighlightBackground": colors.findMatchHighlight,
    "peekViewEditor.matchHighlightBorder": colors.warning + "50",
    "peekViewResult.background": colors.bgPrimary,
    "peekViewResult.fileForeground": colors.textPrimary,
    "peekViewResult.lineForeground": colors.textSecondary,
    "peekViewResult.matchHighlightBackground": colors.findMatchHighlight,
    "peekViewResult.selectionBackground": colors.bgTertiary,
    "peekViewResult.selectionForeground": colors.textPrimary,
    "peekViewTitle.background": colors.bgTertiary,
    "peekViewTitleDescription.foreground": colors.textSecondary,
    "peekViewTitleLabel.foreground": colors.textPrimary,

    // ═══════════════════════════════════════════════════════════════
    // MERGE CONFLICT
    // ═══════════════════════════════════════════════════════════════
    "merge.currentHeaderBackground": colors.success + "40",
    "merge.currentContentBackground": colors.success + "20",
    "merge.incomingHeaderBackground": colors.info + "40",
    "merge.incomingContentBackground": colors.info + "20",
    "merge.border": colors.border,
    "merge.commonContentBackground": colors.bgTertiary,
    "merge.commonHeaderBackground": colors.bgTertiary,
    "editorOverviewRuler.currentContentForeground": colors.success,
    "editorOverviewRuler.incomingContentForeground": colors.info,
    "editorOverviewRuler.commonContentForeground": colors.textMuted,
    "mergeEditor.change.background": colors.info + "20",
    "mergeEditor.change.word.background": colors.info + "40",
    "mergeEditor.conflict.unhandledUnfocused.border": colors.warning + "80",
    "mergeEditor.conflict.unhandledFocused.border": colors.warning,
    "mergeEditor.conflict.handledUnfocused.border": colors.success + "80",
    "mergeEditor.conflict.handledFocused.border": colors.success,
    "mergeEditor.conflict.handled.minimapOverViewRuler": colors.success,
    "mergeEditor.conflict.unhandled.minimapOverViewRuler": colors.warning,

    // ═══════════════════════════════════════════════════════════════
    // SCROLLBAR
    // ═══════════════════════════════════════════════════════════════
    "scrollbar.shadow": "#00000050",
    "scrollbarSlider.background": colors.textMuted + "30",
    "scrollbarSlider.hoverBackground": colors.textMuted + "50",
    "scrollbarSlider.activeBackground": colors.textMuted + "70",

    // ═══════════════════════════════════════════════════════════════
    // MINIMAP
    // ═══════════════════════════════════════════════════════════════
    "minimap.findMatchHighlight": colors.warning + "80",
    "minimap.selectionHighlight": colors.accentPrimary + "80",
    "minimap.errorHighlight": colors.error + "80",
    "minimap.warningHighlight": colors.warning + "80",
    "minimap.background": colors.bgPrimary,
    "minimap.selectionOccurrenceHighlight": colors.accentSecondary + "60",
    "minimap.foregroundOpacity": "#000000cc",
    "minimapSlider.background": colors.textMuted + "20",
    "minimapSlider.hoverBackground": colors.textMuted + "40",
    "minimapSlider.activeBackground": colors.textMuted + "60",
    "minimapGutter.addedBackground": colors.success,
    "minimapGutter.modifiedBackground": colors.warning,
    "minimapGutter.deletedBackground": colors.error,

    // ═══════════════════════════════════════════════════════════════
    // INPUT VALIDATION
    // ═══════════════════════════════════════════════════════════════
    "inputValidation.errorBackground": colors.bgSecondary,
    "inputValidation.errorForeground": colors.error,
    "inputValidation.errorBorder": colors.error,
    "inputValidation.warningBackground": colors.bgSecondary,
    "inputValidation.warningForeground": colors.warning,
    "inputValidation.warningBorder": colors.warning,
    "inputValidation.infoBackground": colors.bgSecondary,
    "inputValidation.infoForeground": colors.info,
    "inputValidation.infoBorder": colors.info,
  },
};

/**
 * Registers the NEURECTOMY Dark theme with Monaco Editor
 *
 * @param monaco - Monaco editor instance
 *
 * @example
 * ```typescript
 * import * as monaco from 'monaco-editor';
 * import { defineNeurectomyTheme } from './themes/neurectomy-dark';
 *
 * defineNeurectomyTheme(monaco);
 * monaco.editor.setTheme('neurectomy-dark');
 * ```
 */
export function defineNeurectomyTheme(monaco: typeof Monaco): void {
  monaco.editor.defineTheme("neurectomy-dark", neurectomyDarkTheme);
}

/**
 * Theme name constant for type-safe theme references
 */
export const NEURECTOMY_DARK_THEME_NAME = "neurectomy-dark" as const;

/**
 * Design system colors exported for use in other components
 */
export { colors as neurectomyColors };

export default neurectomyDarkTheme;
