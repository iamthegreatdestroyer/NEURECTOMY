/**
 * @fileoverview Code Fingerprinting Engine
 * @module @neurectomy/legal-fortress/blockchain/fingerprinting
 *
 * @agents @CIPHER @APEX - Cryptographic Hashing + Implementation
 *
 * Multi-layered code fingerprinting for IP protection:
 * - Content hash (raw)
 * - Normalized hash (whitespace/comment agnostic)
 * - Structure hash (AST-based)
 * - Semantic hash (ML embedding - future)
 */

import CryptoJS from "crypto-js";
import { v4 as uuidv4 } from "uuid";
import { CodeFingerprint, HashAlgorithm } from "../types";
import { computeHash } from "./timestamping";

// ============================================================================
// LANGUAGE CONFIGURATION
// ============================================================================

/**
 * Supported languages for fingerprinting
 */
export type SupportedLanguage =
  | "typescript"
  | "javascript"
  | "python"
  | "rust"
  | "go"
  | "java"
  | "csharp"
  | "cpp"
  | "c"
  | "ruby"
  | "php"
  | "swift"
  | "kotlin";

/**
 * Language-specific configuration
 */
interface LanguageConfig {
  singleLineComment: string[];
  multiLineCommentStart: string[];
  multiLineCommentEnd: string[];
  stringDelimiters: string[];
  keywords: string[];
  operators: string[];
}

/**
 * Language configurations for normalization
 */
const LANGUAGE_CONFIGS: Record<SupportedLanguage, LanguageConfig> = {
  typescript: {
    singleLineComment: ["//"],
    multiLineCommentStart: ["/*"],
    multiLineCommentEnd: ["*/"],
    stringDelimiters: ["'", '"', "`"],
    keywords: [
      "const",
      "let",
      "var",
      "function",
      "class",
      "interface",
      "type",
      "enum",
      "import",
      "export",
      "default",
      "async",
      "await",
      "return",
      "if",
      "else",
      "for",
      "while",
      "do",
      "switch",
      "case",
      "break",
      "continue",
      "try",
      "catch",
      "finally",
      "throw",
      "new",
      "this",
      "super",
      "extends",
      "implements",
    ],
    operators: [
      "+",
      "-",
      "*",
      "/",
      "%",
      "=",
      "==",
      "===",
      "!=",
      "!==",
      "<",
      ">",
      "<=",
      ">=",
      "&&",
      "||",
      "!",
      "&",
      "|",
      "^",
      "~",
      "<<",
      ">>",
      ">>>",
      "++",
      "--",
      "+=",
      "-=",
      "*=",
      "/=",
      "?",
      ":",
      "=>",
      "...",
    ],
  },
  javascript: {
    singleLineComment: ["//"],
    multiLineCommentStart: ["/*"],
    multiLineCommentEnd: ["*/"],
    stringDelimiters: ["'", '"', "`"],
    keywords: [
      "const",
      "let",
      "var",
      "function",
      "class",
      "import",
      "export",
      "default",
      "async",
      "await",
      "return",
      "if",
      "else",
      "for",
      "while",
      "do",
      "switch",
      "case",
      "break",
      "continue",
      "try",
      "catch",
      "finally",
      "throw",
      "new",
      "this",
      "super",
      "extends",
    ],
    operators: [
      "+",
      "-",
      "*",
      "/",
      "%",
      "=",
      "==",
      "===",
      "!=",
      "!==",
      "<",
      ">",
      "<=",
      ">=",
      "&&",
      "||",
      "!",
      "&",
      "|",
      "^",
      "~",
      "<<",
      ">>",
      ">>>",
      "++",
      "--",
      "+=",
      "-=",
      "*=",
      "/=",
      "?",
      ":",
      "=>",
      "...",
    ],
  },
  python: {
    singleLineComment: ["#"],
    multiLineCommentStart: ['"""', "'''"],
    multiLineCommentEnd: ['"""', "'''"],
    stringDelimiters: ["'", '"'],
    keywords: [
      "def",
      "class",
      "import",
      "from",
      "as",
      "return",
      "if",
      "elif",
      "else",
      "for",
      "while",
      "try",
      "except",
      "finally",
      "raise",
      "with",
      "async",
      "await",
      "yield",
      "lambda",
      "pass",
      "break",
      "continue",
      "and",
      "or",
      "not",
      "in",
      "is",
      "True",
      "False",
      "None",
      "global",
      "nonlocal",
    ],
    operators: [
      "+",
      "-",
      "*",
      "/",
      "//",
      "%",
      "**",
      "=",
      "==",
      "!=",
      "<",
      ">",
      "<=",
      ">=",
      "&",
      "|",
      "^",
      "~",
      "<<",
      ">>",
      "+=",
      "-=",
      "*=",
      "/=",
      "//=",
      "%=",
      "**=",
      "&=",
      "|=",
      "^=",
      "<<=",
      ">>=",
      ":=",
    ],
  },
  rust: {
    singleLineComment: ["//"],
    multiLineCommentStart: ["/*"],
    multiLineCommentEnd: ["*/"],
    stringDelimiters: ['"'],
    keywords: [
      "fn",
      "let",
      "mut",
      "const",
      "static",
      "struct",
      "enum",
      "impl",
      "trait",
      "type",
      "mod",
      "use",
      "pub",
      "crate",
      "self",
      "super",
      "return",
      "if",
      "else",
      "match",
      "for",
      "while",
      "loop",
      "break",
      "continue",
      "async",
      "await",
      "move",
      "ref",
      "where",
      "unsafe",
      "extern",
    ],
    operators: [
      "+",
      "-",
      "*",
      "/",
      "%",
      "=",
      "==",
      "!=",
      "<",
      ">",
      "<=",
      ">=",
      "&&",
      "||",
      "!",
      "&",
      "|",
      "^",
      "<<",
      ">>",
      "+=",
      "-=",
      "*=",
      "/=",
      "%=",
      "&=",
      "|=",
      "^=",
      "<<=",
      ">>=",
      "->",
      "=>",
      "::",
    ],
  },
  go: {
    singleLineComment: ["//"],
    multiLineCommentStart: ["/*"],
    multiLineCommentEnd: ["*/"],
    stringDelimiters: ['"', "`"],
    keywords: [
      "func",
      "package",
      "import",
      "type",
      "struct",
      "interface",
      "map",
      "chan",
      "const",
      "var",
      "return",
      "if",
      "else",
      "for",
      "range",
      "switch",
      "case",
      "default",
      "break",
      "continue",
      "go",
      "defer",
      "select",
      "fallthrough",
      "goto",
    ],
    operators: [
      "+",
      "-",
      "*",
      "/",
      "%",
      "=",
      ":=",
      "==",
      "!=",
      "<",
      ">",
      "<=",
      ">=",
      "&&",
      "||",
      "!",
      "&",
      "|",
      "^",
      "<<",
      ">>",
      "&^",
      "+=",
      "-=",
      "*=",
      "/=",
      "%=",
      "&=",
      "|=",
      "^=",
      "<<=",
      ">>=",
      "&^=",
      "<-",
    ],
  },
  java: {
    singleLineComment: ["//"],
    multiLineCommentStart: ["/*"],
    multiLineCommentEnd: ["*/"],
    stringDelimiters: ['"'],
    keywords: [
      "class",
      "interface",
      "enum",
      "extends",
      "implements",
      "public",
      "private",
      "protected",
      "static",
      "final",
      "abstract",
      "synchronized",
      "volatile",
      "transient",
      "native",
      "new",
      "return",
      "if",
      "else",
      "for",
      "while",
      "do",
      "switch",
      "case",
      "default",
      "break",
      "continue",
      "try",
      "catch",
      "finally",
      "throw",
      "throws",
      "import",
      "package",
      "this",
      "super",
    ],
    operators: [
      "+",
      "-",
      "*",
      "/",
      "%",
      "=",
      "==",
      "!=",
      "<",
      ">",
      "<=",
      ">=",
      "&&",
      "||",
      "!",
      "&",
      "|",
      "^",
      "~",
      "<<",
      ">>",
      ">>>",
      "++",
      "--",
      "+=",
      "-=",
      "*=",
      "/=",
      "%=",
      "&=",
      "|=",
      "^=",
      "<<=",
      ">>=",
      ">>>=",
      "?",
      ":",
      "->",
    ],
  },
  csharp: {
    singleLineComment: ["//"],
    multiLineCommentStart: ["/*"],
    multiLineCommentEnd: ["*/"],
    stringDelimiters: ['"', '@"'],
    keywords: [
      "class",
      "interface",
      "struct",
      "enum",
      "delegate",
      "event",
      "public",
      "private",
      "protected",
      "internal",
      "static",
      "readonly",
      "const",
      "virtual",
      "override",
      "abstract",
      "sealed",
      "new",
      "return",
      "if",
      "else",
      "for",
      "foreach",
      "while",
      "do",
      "switch",
      "case",
      "default",
      "break",
      "continue",
      "try",
      "catch",
      "finally",
      "throw",
      "using",
      "namespace",
      "this",
      "base",
      "async",
      "await",
      "var",
      "dynamic",
    ],
    operators: [
      "+",
      "-",
      "*",
      "/",
      "%",
      "=",
      "==",
      "!=",
      "<",
      ">",
      "<=",
      ">=",
      "&&",
      "||",
      "!",
      "&",
      "|",
      "^",
      "~",
      "<<",
      ">>",
      "++",
      "--",
      "+=",
      "-=",
      "*=",
      "/=",
      "%=",
      "&=",
      "|=",
      "^=",
      "<<=",
      ">>=",
      "?",
      ":",
      "=>",
      "??",
      "?.",
      "?[]",
    ],
  },
  cpp: {
    singleLineComment: ["//"],
    multiLineCommentStart: ["/*"],
    multiLineCommentEnd: ["*/"],
    stringDelimiters: ['"'],
    keywords: [
      "class",
      "struct",
      "enum",
      "union",
      "namespace",
      "template",
      "typename",
      "public",
      "private",
      "protected",
      "virtual",
      "override",
      "final",
      "static",
      "const",
      "constexpr",
      "mutable",
      "volatile",
      "inline",
      "explicit",
      "friend",
      "return",
      "if",
      "else",
      "for",
      "while",
      "do",
      "switch",
      "case",
      "default",
      "break",
      "continue",
      "try",
      "catch",
      "throw",
      "new",
      "delete",
      "this",
      "nullptr",
      "auto",
      "decltype",
    ],
    operators: [
      "+",
      "-",
      "*",
      "/",
      "%",
      "=",
      "==",
      "!=",
      "<",
      ">",
      "<=",
      ">=",
      "&&",
      "||",
      "!",
      "&",
      "|",
      "^",
      "~",
      "<<",
      ">>",
      "++",
      "--",
      "+=",
      "-=",
      "*=",
      "/=",
      "%=",
      "&=",
      "|=",
      "^=",
      "<<=",
      ">>=",
      "?",
      ":",
      "->",
      "::",
      ".*",
      "->*",
    ],
  },
  c: {
    singleLineComment: ["//"],
    multiLineCommentStart: ["/*"],
    multiLineCommentEnd: ["*/"],
    stringDelimiters: ['"'],
    keywords: [
      "auto",
      "break",
      "case",
      "char",
      "const",
      "continue",
      "default",
      "do",
      "double",
      "else",
      "enum",
      "extern",
      "float",
      "for",
      "goto",
      "if",
      "int",
      "long",
      "register",
      "return",
      "short",
      "signed",
      "sizeof",
      "static",
      "struct",
      "switch",
      "typedef",
      "union",
      "unsigned",
      "void",
      "volatile",
      "while",
    ],
    operators: [
      "+",
      "-",
      "*",
      "/",
      "%",
      "=",
      "==",
      "!=",
      "<",
      ">",
      "<=",
      ">=",
      "&&",
      "||",
      "!",
      "&",
      "|",
      "^",
      "~",
      "<<",
      ">>",
      "++",
      "--",
      "+=",
      "-=",
      "*=",
      "/=",
      "%=",
      "&=",
      "|=",
      "^=",
      "<<=",
      ">>=",
      "?",
      ":",
      "->",
    ],
  },
  ruby: {
    singleLineComment: ["#"],
    multiLineCommentStart: ["=begin"],
    multiLineCommentEnd: ["=end"],
    stringDelimiters: ["'", '"', "%q", "%Q"],
    keywords: [
      "def",
      "class",
      "module",
      "end",
      "if",
      "elsif",
      "else",
      "unless",
      "case",
      "when",
      "while",
      "until",
      "for",
      "do",
      "begin",
      "rescue",
      "ensure",
      "raise",
      "return",
      "yield",
      "break",
      "next",
      "redo",
      "retry",
      "self",
      "super",
      "true",
      "false",
      "nil",
      "and",
      "or",
      "not",
      "require",
      "require_relative",
      "include",
      "extend",
      "attr_reader",
      "attr_writer",
      "attr_accessor",
      "private",
      "protected",
      "public",
    ],
    operators: [
      "+",
      "-",
      "*",
      "/",
      "%",
      "**",
      "=",
      "==",
      "===",
      "!=",
      "<",
      ">",
      "<=",
      ">=",
      "<=>",
      "&&",
      "||",
      "!",
      "&",
      "|",
      "^",
      "~",
      "<<",
      ">>",
      "+=",
      "-=",
      "*=",
      "/=",
      "%=",
      "**=",
      "&=",
      "|=",
      "^=",
      "<<=",
      ">>=",
      "..",
      "...",
      "?",
      ":",
      "=>",
    ],
  },
  php: {
    singleLineComment: ["//", "#"],
    multiLineCommentStart: ["/*"],
    multiLineCommentEnd: ["*/"],
    stringDelimiters: ["'", '"'],
    keywords: [
      "function",
      "class",
      "interface",
      "trait",
      "extends",
      "implements",
      "public",
      "private",
      "protected",
      "static",
      "final",
      "abstract",
      "const",
      "return",
      "if",
      "elseif",
      "else",
      "for",
      "foreach",
      "while",
      "do",
      "switch",
      "case",
      "default",
      "break",
      "continue",
      "try",
      "catch",
      "finally",
      "throw",
      "new",
      "use",
      "namespace",
      "require",
      "include",
      "require_once",
      "include_once",
      "echo",
      "print",
    ],
    operators: [
      "+",
      "-",
      "*",
      "/",
      "%",
      "**",
      "=",
      "==",
      "===",
      "!=",
      "!==",
      "<>",
      "<",
      ">",
      "<=",
      ">=",
      "<=>",
      "&&",
      "||",
      "!",
      "&",
      "|",
      "^",
      "~",
      "<<",
      ">>",
      "++",
      "--",
      ".",
      ".=",
      "+=",
      "-=",
      "*=",
      "/=",
      "%=",
      "**=",
      "&=",
      "|=",
      "^=",
      "<<=",
      ">>=",
      "??",
      "?:",
      "->",
    ],
  },
  swift: {
    singleLineComment: ["//"],
    multiLineCommentStart: ["/*"],
    multiLineCommentEnd: ["*/"],
    stringDelimiters: ['"'],
    keywords: [
      "func",
      "class",
      "struct",
      "enum",
      "protocol",
      "extension",
      "typealias",
      "let",
      "var",
      "static",
      "final",
      "lazy",
      "private",
      "fileprivate",
      "internal",
      "public",
      "open",
      "return",
      "if",
      "else",
      "guard",
      "for",
      "while",
      "repeat",
      "switch",
      "case",
      "default",
      "break",
      "continue",
      "fallthrough",
      "do",
      "try",
      "catch",
      "throw",
      "throws",
      "rethrows",
      "import",
      "self",
      "Self",
      "super",
      "init",
      "deinit",
      "associatedtype",
      "async",
      "await",
      "actor",
    ],
    operators: [
      "+",
      "-",
      "*",
      "/",
      "%",
      "=",
      "==",
      "!=",
      "===",
      "!==",
      "<",
      ">",
      "<=",
      ">=",
      "&&",
      "||",
      "!",
      "&",
      "|",
      "^",
      "~",
      "<<",
      ">>",
      "++",
      "--",
      "+=",
      "-=",
      "*=",
      "/=",
      "%=",
      "&=",
      "|=",
      "^=",
      "<<=",
      ">>=",
      "??",
      "?",
      "!",
      "->",
      ".",
      "...",
      "..<",
    ],
  },
  kotlin: {
    singleLineComment: ["//"],
    multiLineCommentStart: ["/*"],
    multiLineCommentEnd: ["*/"],
    stringDelimiters: ['"', '"""'],
    keywords: [
      "fun",
      "class",
      "interface",
      "object",
      "data",
      "sealed",
      "enum",
      "annotation",
      "val",
      "var",
      "const",
      "lateinit",
      "by",
      "lazy",
      "private",
      "protected",
      "internal",
      "public",
      "open",
      "final",
      "override",
      "abstract",
      "return",
      "if",
      "else",
      "when",
      "for",
      "while",
      "do",
      "try",
      "catch",
      "finally",
      "throw",
      "import",
      "package",
      "this",
      "super",
      "is",
      "as",
      "in",
      "out",
      "typealias",
      "companion",
      "init",
      "constructor",
      "suspend",
      "inline",
      "infix",
    ],
    operators: [
      "+",
      "-",
      "*",
      "/",
      "%",
      "=",
      "==",
      "!=",
      "===",
      "!==",
      "<",
      ">",
      "<=",
      ">=",
      "&&",
      "||",
      "!",
      "&",
      "|",
      "^",
      "++",
      "--",
      "+=",
      "-=",
      "*=",
      "/=",
      "%=",
      "?:",
      "?.",
      "!!",
      "->",
      "::",
      "..",
      "in",
      "!in",
      "is",
      "!is",
      "as",
      "as?",
    ],
  },
};

// ============================================================================
// NORMALIZATION UTILITIES (@CIPHER)
// ============================================================================

/**
 * Remove comments from code
 * @agent @CIPHER - Code normalization
 */
function removeComments(code: string, config: LanguageConfig): string {
  let result = code;

  // Remove multi-line comments
  for (let i = 0; i < config.multiLineCommentStart.length; i++) {
    const start = config.multiLineCommentStart[i]!;
    const end = config.multiLineCommentEnd[i]!;
    const regex = new RegExp(
      escapeRegex(start) + "[\\s\\S]*?" + escapeRegex(end),
      "g"
    );
    result = result.replace(regex, "");
  }

  // Remove single-line comments
  for (const comment of config.singleLineComment) {
    const regex = new RegExp(escapeRegex(comment) + ".*$", "gm");
    result = result.replace(regex, "");
  }

  return result;
}

/**
 * Escape special regex characters
 */
function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Normalize whitespace
 */
function normalizeWhitespace(code: string): string {
  return code
    .replace(/\s+/g, " ")
    .replace(/\s*([{}()[\];,:])\s*/g, "$1")
    .trim();
}

/**
 * Normalize identifiers to placeholders
 * Preserves keywords, replaces user-defined identifiers
 */
function normalizeIdentifiers(code: string, config: LanguageConfig): string {
  const keywordSet = new Set(config.keywords);
  let counter = 0;
  const identifierMap = new Map<string, string>();

  // Match identifiers (simplified)
  const identifierRegex = /\b([a-zA-Z_][a-zA-Z0-9_]*)\b/g;

  return code.replace(identifierRegex, (match) => {
    // Keep keywords
    if (keywordSet.has(match)) {
      return match;
    }

    // Normalize identifier
    if (!identifierMap.has(match)) {
      identifierMap.set(match, `ID${counter++}`);
    }
    return identifierMap.get(match)!;
  });
}

/**
 * Tokenize code into significant tokens
 * @agent @CIPHER - Token extraction
 */
function tokenize(code: string, config: LanguageConfig): string[] {
  const tokens: string[] = [];

  // Add keywords
  for (const keyword of config.keywords) {
    const regex = new RegExp(`\\b${keyword}\\b`, "g");
    let match;
    while ((match = regex.exec(code)) !== null) {
      tokens.push(`KW:${keyword}`);
    }
  }

  // Add operators
  for (const op of config.operators) {
    const escapedOp = escapeRegex(op);
    const regex = new RegExp(escapedOp, "g");
    let count = 0;
    while (regex.exec(code) !== null) {
      count++;
    }
    for (let i = 0; i < count; i++) {
      tokens.push(`OP:${op}`);
    }
  }

  // Add structural tokens
  const structuralTokens = ["{", "}", "(", ")", "[", "]", ";"];
  for (const token of structuralTokens) {
    const count = (code.match(new RegExp(escapeRegex(token), "g")) || [])
      .length;
    for (let i = 0; i < count; i++) {
      tokens.push(`ST:${token}`);
    }
  }

  return tokens.sort();
}

// ============================================================================
// COMPLEXITY METRICS (@APEX)
// ============================================================================

/**
 * Calculate cyclomatic complexity (simplified)
 * @agent @APEX - Complexity calculation
 */
function calculateCyclomaticComplexity(
  code: string,
  language: SupportedLanguage
): number {
  // Count decision points
  const decisionKeywords = [
    "if",
    "else if",
    "elif",
    "elseif",
    "case",
    "for",
    "while",
    "catch",
    "except",
    "&&",
    "||",
    "?",
    "and",
    "or",
  ];

  let complexity = 1; // Base complexity

  for (const keyword of decisionKeywords) {
    const regex = new RegExp(`\\b${escapeRegex(keyword)}\\b`, "g");
    const matches = code.match(regex);
    if (matches) {
      complexity += matches.length;
    }
  }

  return complexity;
}

/**
 * Calculate Halstead metrics
 * @agent @APEX - Halstead complexity
 */
function calculateHalsteadMetrics(tokens: string[]): {
  vocabulary: number;
  length: number;
  difficulty: number;
  effort: number;
} {
  const operators = tokens.filter(
    (t) => t.startsWith("OP:") || t.startsWith("KW:")
  );
  const operands = tokens.filter(
    (t) => !t.startsWith("OP:") && !t.startsWith("KW:")
  );

  const uniqueOperators = new Set(operators).size;
  const uniqueOperands = new Set(operands).size;
  const totalOperators = operators.length;
  const totalOperands = operands.length;

  const vocabulary = uniqueOperators + uniqueOperands;
  const length = totalOperators + totalOperands;
  const difficulty =
    (uniqueOperators / 2) * (totalOperands / Math.max(uniqueOperands, 1));
  const volume = length * Math.log2(Math.max(vocabulary, 1));
  const effort = difficulty * volume;

  return {
    vocabulary,
    length,
    difficulty,
    effort,
  };
}

// ============================================================================
// CODE FINGERPRINTING ENGINE (@CIPHER @APEX)
// ============================================================================

/**
 * Configuration for fingerprinting
 */
export interface FingerprintingConfig {
  hashAlgorithm: HashAlgorithm;
  includeTokens: boolean;
  calculateComplexity: boolean;
  normalizeIdentifiers: boolean;
}

/**
 * Default fingerprinting configuration
 */
export const DEFAULT_FINGERPRINTING_CONFIG: FingerprintingConfig = {
  hashAlgorithm: "sha256",
  includeTokens: true,
  calculateComplexity: true,
  normalizeIdentifiers: true,
};

/**
 * Detect language from file extension
 */
export function detectLanguage(filePath: string): SupportedLanguage | null {
  const ext = filePath.split(".").pop()?.toLowerCase();

  const extensionMap: Record<string, SupportedLanguage> = {
    ts: "typescript",
    tsx: "typescript",
    js: "javascript",
    jsx: "javascript",
    mjs: "javascript",
    cjs: "javascript",
    py: "python",
    pyw: "python",
    rs: "rust",
    go: "go",
    java: "java",
    cs: "csharp",
    cpp: "cpp",
    cxx: "cpp",
    cc: "cpp",
    c: "c",
    h: "c",
    hpp: "cpp",
    rb: "ruby",
    php: "php",
    swift: "swift",
    kt: "kotlin",
    kts: "kotlin",
  };

  return ext ? (extensionMap[ext] ?? null) : null;
}

/**
 * Generate multi-layered code fingerprint
 * @agent @CIPHER @APEX - Core fingerprinting logic
 */
export function generateCodeFingerprint(
  code: string,
  filePath: string,
  language?: SupportedLanguage,
  config: FingerprintingConfig = DEFAULT_FINGERPRINTING_CONFIG
): CodeFingerprint {
  // Detect language if not provided
  const detectedLanguage = language ?? detectLanguage(filePath);
  if (!detectedLanguage) {
    throw new Error(`Unable to detect language for file: ${filePath}`);
  }

  const langConfig = LANGUAGE_CONFIGS[detectedLanguage];

  // Content hash (raw)
  const contentHash = computeHash(code, config.hashAlgorithm);

  // Normalized hash (whitespace and comment agnostic)
  const withoutComments = removeComments(code, langConfig);
  const normalized = normalizeWhitespace(withoutComments);
  const normalizedHash = computeHash(normalized, config.hashAlgorithm);

  // Structure hash (identifier-normalized)
  const structureCode = config.normalizeIdentifiers
    ? normalizeIdentifiers(normalized, langConfig)
    : normalized;
  const structureHash = computeHash(structureCode, config.hashAlgorithm);

  // Token extraction
  const tokens = config.includeTokens
    ? tokenize(withoutComments, langConfig)
    : undefined;

  // Complexity metrics
  let complexity: CodeFingerprint["complexity"];
  if (config.calculateComplexity && tokens) {
    const halstead = calculateHalsteadMetrics(tokens);
    complexity = {
      cyclomatic: calculateCyclomaticComplexity(code, detectedLanguage),
      halstead,
    };
  }

  // Line and character counts
  const lineCount = code.split("\n").length;
  const charCount = code.length;

  return {
    id: uuidv4(),
    filePath,
    contentHash,
    structureHash,
    normalizedHash,
    language: detectedLanguage,
    lineCount,
    charCount,
    complexity,
    tokens,
    createdAt: new Date(),
    updatedAt: new Date(),
  };
}

/**
 * Compare two code fingerprints for similarity
 * @agent @CIPHER - Similarity comparison
 */
export function compareFingerprints(
  fp1: CodeFingerprint,
  fp2: CodeFingerprint
): {
  exactMatch: boolean;
  normalizedMatch: boolean;
  structureMatch: boolean;
  tokenSimilarity: number;
} {
  const exactMatch = fp1.contentHash === fp2.contentHash;
  const normalizedMatch = fp1.normalizedHash === fp2.normalizedHash;
  const structureMatch = fp1.structureHash === fp2.structureHash;

  // Jaccard similarity for tokens
  let tokenSimilarity = 0;
  if (fp1.tokens && fp2.tokens) {
    const set1 = new Set(fp1.tokens);
    const set2 = new Set(fp2.tokens);
    const intersection = new Set([...set1].filter((x) => set2.has(x)));
    const union = new Set([...set1, ...set2]);
    tokenSimilarity = intersection.size / union.size;
  }

  return {
    exactMatch,
    normalizedMatch,
    structureMatch,
    tokenSimilarity,
  };
}

/**
 * Batch fingerprint multiple files
 */
export async function batchFingerprint(
  files: Array<{ path: string; content: string }>,
  config: FingerprintingConfig = DEFAULT_FINGERPRINTING_CONFIG
): Promise<CodeFingerprint[]> {
  return files.map((file) => {
    try {
      return generateCodeFingerprint(
        file.content,
        file.path,
        undefined,
        config
      );
    } catch (error) {
      // Return partial fingerprint for unsupported files
      return {
        id: uuidv4(),
        filePath: file.path,
        contentHash: computeHash(file.content, config.hashAlgorithm),
        structureHash: "",
        normalizedHash: "",
        language: "unknown",
        lineCount: file.content.split("\n").length,
        charCount: file.content.length,
        createdAt: new Date(),
        updatedAt: new Date(),
      } as CodeFingerprint;
    }
  });
}

// Types and functions are already exported inline above
