/**
 * SigmaLang Monaco Language Configuration
 *
 * Provides syntax highlighting, tokenization, and completion support
 * for .sigma files in the NEURECTOMY IDE.
 *
 * ΣLANG - Sub-Linear Algorithmic Neural Glyph Language
 *
 * @module @neurectomy/monaco
 * @author @LINGUA @CORE
 */

import * as monaco from "monaco-editor";

// ============================================================================
// Language Definition
// ============================================================================

export const SIGMALANG_LANGUAGE_ID = "sigmalang";

export const sigmaLangConfiguration: monaco.languages.LanguageConfiguration = {
  comments: {
    lineComment: "//",
    blockComment: ["/*", "*/"],
  },
  brackets: [
    ["{", "}"],
    ["[", "]"],
    ["(", ")"],
    ["⟨", "⟩"],
    ["⌈", "⌉"],
    ["⟦", "⟧"],
  ],
  autoClosingPairs: [
    { open: "{", close: "}" },
    { open: "[", close: "]" },
    { open: "(", close: ")" },
    { open: "⟨", close: "⟩" },
    { open: "⌈", close: "⌉" },
    { open: "⟦", close: "⟧" },
    { open: '"', close: '"' },
    { open: "'", close: "'" },
    { open: "«", close: "»" },
  ],
  surroundingPairs: [
    { open: "{", close: "}" },
    { open: "[", close: "]" },
    { open: "(", close: ")" },
    { open: "⟨", close: "⟩" },
    { open: '"', close: '"' },
    { open: "'", close: "'" },
  ],
  folding: {
    markers: {
      start: /^\s*\/\/\s*#region\b/,
      end: /^\s*\/\/\s*#endregion\b/,
    },
  },
  indentationRules: {
    increaseIndentPattern: /[{[(⟨⌈⟦]\s*$/,
    decreaseIndentPattern: /^\s*[})\]⟩⌉⟧]/,
  },
};

// ============================================================================
// Monarch Tokenizer
// ============================================================================

export const sigmaLangMonarch: monaco.languages.IMonarchLanguage = {
  defaultToken: "",
  tokenPostfix: ".sigma",

  // Keywords and operators from SigmaLang primitives
  keywords: [
    // Existential Primitives
    "∃",
    "∄",
    "∀",
    "∈",
    "∉",
    "⊂",
    "⊃",
    "⊆",
    "⊇",
    "∪",
    "∩",
    // Code Primitives
    "λ",
    "⊢",
    "⊣",
    "↦",
    "⟹",
    "⟸",
    "⟺",
    "⊕",
    "⊗",
    // Math Primitives
    "∑",
    "∏",
    "∫",
    "∂",
    "∇",
    "∞",
    "≈",
    "≡",
    "≠",
    "≤",
    "≥",
    // Logic Primitives
    "∧",
    "∨",
    "¬",
    "⊤",
    "⊥",
    "⊨",
    "⊬",
    // Entity Markers
    "⊛",
    "⊚",
    "⊙",
    // Action Markers
    "→",
    "←",
    "↔",
    "⇒",
    "⇐",
    "⇔",
    // Communication
    "⊲",
    "⊳",
    "⊴",
    "⊵",
    // Structure
    "⟨",
    "⟩",
    "⌈",
    "⌉",
    "⌊",
    "⌋",
    "⟦",
    "⟧",
  ],

  // SigmaLang type primitives
  typeKeywords: [
    "Σ",
    "Π",
    "Δ",
    "Ω",
    "Φ",
    "Ψ",
    "Γ",
    "Λ",
    "Θ",
    "semantic",
    "glyph",
    "tree",
    "node",
    "stream",
    "encode",
    "decode",
    "compress",
    "expand",
    "pattern",
    "codebook",
    "signature",
  ],

  // Operators
  operators: [
    "=",
    ">",
    "<",
    "!",
    "~",
    "?",
    ":",
    "==",
    "<=",
    ">=",
    "!=",
    "&&",
    "||",
    "++",
    "--",
    "+",
    "-",
    "*",
    "/",
    "&",
    "|",
    "^",
    "%",
    "<<",
    ">>",
    ">>>",
    "+=",
    "-=",
    "*=",
    "/=",
    "&=",
    "|=",
    "^=",
    "%=",
    "<<=",
    ">>=",
    ">>>=",
    "→",
    "←",
    "↔",
    "⇒",
    "⇐",
    "⇔",
    "∘",
    "⊕",
    "⊗",
    "⊖",
    "⊘",
  ],

  // Special symbols
  symbols: /[=><!~?:&|+\-*\/\^%∃∄∀∈∉⊂⊃⊆⊇∪∩λ⊢⊣↦⟹⟸⟺⊕⊗∑∏∫∂∇∞≈≡≠≤≥∧∨¬⊤⊥⊨⊬⊛⊚⊙⊲⊳⊴⊵]+/,

  // Escape sequences
  escapes:
    /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

  // Tokenizer rules
  tokenizer: {
    root: [
      // Comments
      [/\/\/.*$/, "comment"],
      [/\/\*/, "comment", "@comment"],

      // Greek letters and special primitives (keywords)
      [
        /[Σ∃∄∀∈∉⊂⊃⊆⊇∪∩λ⊢⊣↦⟹⟸⟺⊕⊗∑∏∫∂∇∞≈≡≠≤≥∧∨¬⊤⊥⊨⊬⊛⊚⊙⊲⊳⊴⊵→←↔⇒⇐⇔ΠΔΩΦΨΓΛΘα-ω]/,
        "keyword.sigma",
      ],

      // Identifiers and keywords
      [
        /[a-zA-Z_]\w*/,
        {
          cases: {
            "@typeKeywords": "keyword.type.sigma",
            "@keywords": "keyword.sigma",
            "@default": "identifier",
          },
        },
      ],

      // Whitespace
      { include: "@whitespace" },

      // Delimiters and brackets
      [/[{}()\[\]]/, "@brackets"],
      [/[⟨⟩⌈⌉⌊⌋⟦⟧«»]/, "bracket.sigma"],

      // Numbers
      [/\d*\.\d+([eE][-+]?\d+)?/, "number.float"],
      [/0[xX][0-9a-fA-F]+/, "number.hex"],
      [/0[bB][01]+/, "number.binary"],
      [/\d+/, "number"],

      // Strings
      [/"([^"\\]|\\.)*$/, "string.invalid"],
      [/"/, "string", "@string_double"],
      [/'([^'\\]|\\.)*$/, "string.invalid"],
      [/'/, "string", "@string_single"],

      // Glyph literals (special SigmaLang strings)
      [/«/, "string.glyph", "@glyph_literal"],

      // Operators
      [
        /@symbols/,
        {
          cases: {
            "@operators": "operator.sigma",
            "@default": "",
          },
        },
      ],

      // Delimiter
      [/[;,.]/, "delimiter"],
    ],

    comment: [
      [/[^/*]+/, "comment"],
      [/\/\*/, "comment", "@push"],
      [/\*\//, "comment", "@pop"],
      [/[/*]/, "comment"],
    ],

    string_double: [
      [/[^\\"]+/, "string"],
      [/@escapes/, "string.escape"],
      [/\\./, "string.escape.invalid"],
      [/"/, "string", "@pop"],
    ],

    string_single: [
      [/[^\\']+/, "string"],
      [/@escapes/, "string.escape"],
      [/\\./, "string.escape.invalid"],
      [/'/, "string", "@pop"],
    ],

    glyph_literal: [
      [/[^»\\]+/, "string.glyph"],
      [/@escapes/, "string.glyph.escape"],
      [/\\./, "string.glyph.escape.invalid"],
      [/»/, "string.glyph", "@pop"],
    ],

    whitespace: [
      [/[ \t\r\n]+/, "white"],
      [/\/\*/, "comment", "@comment"],
      [/\/\/.*$/, "comment"],
    ],
  },
};

// ============================================================================
// Completions
// ============================================================================

export const sigmaLangCompletions: monaco.languages.CompletionItemProvider = {
  provideCompletionItems: (model, position) => {
    const word = model.getWordUntilPosition(position);
    const range = {
      startLineNumber: position.lineNumber,
      endLineNumber: position.lineNumber,
      startColumn: word.startColumn,
      endColumn: word.endColumn,
    };

    const suggestions: monaco.languages.CompletionItem[] = [
      // Primitive completions
      {
        label: "Σ.encode",
        kind: monaco.languages.CompletionItemKind.Function,
        insertText: "Σ.encode(${1:text})",
        insertTextRules:
          monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
        documentation: "Encode text to SigmaLang binary representation",
        range,
      },
      {
        label: "Σ.decode",
        kind: monaco.languages.CompletionItemKind.Function,
        insertText: "Σ.decode(${1:binary})",
        insertTextRules:
          monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
        documentation: "Decode SigmaLang binary to text",
        range,
      },
      {
        label: "λ.map",
        kind: monaco.languages.CompletionItemKind.Function,
        insertText: "λ.map(${1:fn}, ${2:collection})",
        insertTextRules:
          monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
        documentation: "Apply function to each element",
        range,
      },
      {
        label: "∃.entity",
        kind: monaco.languages.CompletionItemKind.Snippet,
        insertText: "∃(${1:name}: ${2:type}) {\n\t${3:properties}\n}",
        insertTextRules:
          monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
        documentation: "Define an existential entity",
        range,
      },
      {
        label: "⟨tree⟩",
        kind: monaco.languages.CompletionItemKind.Snippet,
        insertText: "⟨${1:root}\n\t⟨${2:child}⟩\n⟩",
        insertTextRules:
          monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
        documentation: "Create a semantic tree structure",
        range,
      },
      {
        label: "pattern.define",
        kind: monaco.languages.CompletionItemKind.Function,
        insertText:
          "pattern.define(${1:name}, {\n\tsignature: ${2:sig},\n\ttransform: ${3:fn}\n})",
        insertTextRules:
          monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
        documentation: "Define a learnable pattern for the codebook",
        range,
      },
      {
        label: "codebook.add",
        kind: monaco.languages.CompletionItemKind.Function,
        insertText: "codebook.add(${1:pattern})",
        insertTextRules:
          monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
        documentation: "Add a pattern to the learned codebook",
        range,
      },
      // Type completions
      {
        label: "SemanticNode",
        kind: monaco.languages.CompletionItemKind.Class,
        insertText: "SemanticNode",
        documentation: "A node in the semantic tree",
        range,
      },
      {
        label: "SemanticTree",
        kind: monaco.languages.CompletionItemKind.Class,
        insertText: "SemanticTree",
        documentation: "The root semantic tree structure",
        range,
      },
      {
        label: "GlyphStream",
        kind: monaco.languages.CompletionItemKind.Class,
        insertText: "GlyphStream",
        documentation: "Stream of encoded glyphs",
        range,
      },
    ];

    return { suggestions };
  },
};

// ============================================================================
// Registration
// ============================================================================

export function registerSigmaLangLanguage(): void {
  // Register the language
  monaco.languages.register({
    id: SIGMALANG_LANGUAGE_ID,
    extensions: [".sigma", ".sig", ".σ"],
    aliases: ["SigmaLang", "sigma", "Σ"],
    mimetypes: ["text/x-sigmalang"],
  });

  // Set the language configuration
  monaco.languages.setLanguageConfiguration(
    SIGMALANG_LANGUAGE_ID,
    sigmaLangConfiguration
  );

  // Set the tokenizer
  monaco.languages.setMonarchTokensProvider(
    SIGMALANG_LANGUAGE_ID,
    sigmaLangMonarch
  );

  // Register completion provider
  monaco.languages.registerCompletionItemProvider(
    SIGMALANG_LANGUAGE_ID,
    sigmaLangCompletions
  );

  console.log("✓ SigmaLang language registered with Monaco");
}

// ============================================================================
// Theme Tokens
// ============================================================================

export const sigmaLangThemeTokens: monaco.editor.ITokenThemeRule[] = [
  { token: "keyword.sigma", foreground: "00ff88", fontStyle: "bold" },
  { token: "keyword.type.sigma", foreground: "a855f7" },
  { token: "operator.sigma", foreground: "22d3ee" },
  { token: "bracket.sigma", foreground: "fbbf24" },
  { token: "string.glyph", foreground: "f472b6", fontStyle: "italic" },
  { token: "string.glyph.escape", foreground: "a855f7" },
  { token: "number", foreground: "fbbf24" },
  { token: "comment", foreground: "6b7280", fontStyle: "italic" },
];

export default {
  SIGMALANG_LANGUAGE_ID,
  sigmaLangConfiguration,
  sigmaLangMonarch,
  sigmaLangCompletions,
  sigmaLangThemeTokens,
  registerSigmaLangLanguage,
};
