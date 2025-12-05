/**
 * @fileoverview Code Similarity Analysis Engine
 * @module @neurectomy/legal-fortress/plagiarism/similarity
 *
 * @agents @PHANTOM @TENSOR - Reverse Engineering + ML Specialists
 *
 * Detects code similarity using multiple algorithms:
 * - Token-based comparison (winnowing)
 * - N-gram fingerprinting
 * - Jaccard similarity
 * - MinHash for approximate matching
 * - TF-IDF weighted comparison
 */

import CryptoJS from "crypto-js";
import { SimilarityMatch, SimilarityScore, CodeRegion } from "../types";

// ============================================================================
// SIMILARITY TYPES (@PHANTOM)
// ============================================================================

/**
 * Similarity analysis options
 */
export interface SimilarityOptions {
  algorithm: "winnowing" | "ngram" | "jaccard" | "minhash" | "combined";
  ngramSize: number;
  windowSize: number; // For winnowing
  threshold: number; // Minimum similarity to report
  ignoreWhitespace: boolean;
  ignoreComments: boolean;
  caseSensitive: boolean;
  numHashFunctions?: number; // For MinHash
}

/**
 * Token with position information
 */
export interface Token {
  value: string;
  type: "identifier" | "keyword" | "literal" | "operator" | "punctuation";
  line: number;
  column: number;
  index: number;
}

/**
 * Fingerprint for similarity detection
 */
export interface CodeFingerprint {
  hashes: string[];
  positions: Array<{ hash: string; start: number; end: number }>;
  tokens: Token[];
}

/**
 * Similarity result with matched regions
 */
export interface SimilarityResult {
  sourceFile: string;
  targetFile: string;
  overallSimilarity: number;
  algorithm: string;
  matchedRegions: MatchedRegion[];
  fingerprints: {
    sourceUnique: number;
    targetUnique: number;
    shared: number;
  };
  analysisTime: number;
}

/**
 * Matched region between two files
 */
export interface MatchedRegion {
  sourceStart: number;
  sourceEnd: number;
  targetStart: number;
  targetEnd: number;
  similarity: number;
  type: "exact" | "near" | "structural";
}

// ============================================================================
// DEFAULT CONFIGURATION
// ============================================================================

const DEFAULT_SIMILARITY_OPTIONS: SimilarityOptions = {
  algorithm: "combined",
  ngramSize: 5,
  windowSize: 4,
  threshold: 0.3,
  ignoreWhitespace: true,
  ignoreComments: true,
  caseSensitive: false,
  numHashFunctions: 100,
};

// ============================================================================
// TOKENIZATION (@PHANTOM)
// ============================================================================

/**
 * Simple keyword sets for common languages
 */
const KEYWORDS: Record<string, Set<string>> = {
  javascript: new Set([
    "function",
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
    "const",
    "let",
    "var",
    "class",
    "extends",
    "import",
    "export",
    "default",
    "async",
    "await",
    "try",
    "catch",
    "throw",
    "new",
    "this",
    "super",
    "typeof",
    "instanceof",
    "in",
    "of",
    "null",
    "undefined",
    "true",
    "false",
  ]),
  typescript: new Set([
    "function",
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
    "const",
    "let",
    "var",
    "class",
    "extends",
    "import",
    "export",
    "default",
    "async",
    "await",
    "try",
    "catch",
    "throw",
    "new",
    "this",
    "super",
    "typeof",
    "instanceof",
    "in",
    "of",
    "null",
    "undefined",
    "true",
    "false",
    "interface",
    "type",
    "enum",
    "namespace",
    "abstract",
    "implements",
    "public",
    "private",
    "protected",
    "readonly",
  ]),
  python: new Set([
    "def",
    "return",
    "if",
    "elif",
    "else",
    "for",
    "while",
    "break",
    "continue",
    "class",
    "import",
    "from",
    "as",
    "try",
    "except",
    "finally",
    "raise",
    "with",
    "async",
    "await",
    "lambda",
    "yield",
    "global",
    "nonlocal",
    "pass",
    "None",
    "True",
    "False",
    "and",
    "or",
    "not",
    "in",
    "is",
  ]),
};

/**
 * Tokenize source code
 * @agent @PHANTOM - Code analysis
 */
export function tokenize(
  source: string,
  language: string = "javascript",
  options: Partial<SimilarityOptions> = {}
): Token[] {
  const opts = { ...DEFAULT_SIMILARITY_OPTIONS, ...options };
  const keywords = KEYWORDS[language] ?? KEYWORDS["javascript"]!;
  const tokens: Token[] = [];

  // Preprocess
  let processed = source;
  if (opts.ignoreComments) {
    processed = removeComments(processed);
  }
  if (!opts.caseSensitive) {
    processed = processed.toLowerCase();
  }

  // Tokenize using regex
  const tokenRegex =
    /([a-zA-Z_][a-zA-Z0-9_]*)|(\d+\.?\d*)|([+\-*/%=<>!&|^~]+)|([(){}\[\];,.])|(\s+)/g;

  let match;
  let line = 1;
  let column = 1;
  let index = 0;

  while ((match = tokenRegex.exec(processed)) !== null) {
    const value = match[0]!;

    // Skip whitespace if configured
    if (opts.ignoreWhitespace && /^\s+$/.test(value)) {
      // Update line/column tracking
      const newlines = (value.match(/\n/g) || []).length;
      if (newlines > 0) {
        line += newlines;
        column = value.length - value.lastIndexOf("\n");
      } else {
        column += value.length;
      }
      continue;
    }

    // Determine token type
    let type: Token["type"] = "identifier";
    if (keywords.has(value)) {
      type = "keyword";
    } else if (/^\d/.test(value)) {
      type = "literal";
    } else if (/^[+\-*/%=<>!&|^~]+$/.test(value)) {
      type = "operator";
    } else if (/^[(){}\[\];,.]$/.test(value)) {
      type = "punctuation";
    }

    tokens.push({
      value,
      type,
      line,
      column,
      index: index++,
    });

    // Update position
    const newlines = (value.match(/\n/g) || []).length;
    if (newlines > 0) {
      line += newlines;
      column = value.length - value.lastIndexOf("\n");
    } else {
      column += value.length;
    }
  }

  return tokens;
}

/**
 * Remove comments from source code
 */
function removeComments(source: string): string {
  // Remove single-line comments
  let result = source.replace(/\/\/.*$/gm, "");
  // Remove multi-line comments
  result = result.replace(/\/\*[\s\S]*?\*\//g, "");
  // Remove Python/Shell comments
  result = result.replace(/#.*$/gm, "");
  return result;
}

// ============================================================================
// N-GRAM FINGERPRINTING (@PHANTOM)
// ============================================================================

/**
 * Generate n-grams from tokens
 */
export function generateNgrams(tokens: Token[], n: number = 5): string[] {
  const ngrams: string[] = [];

  for (let i = 0; i <= tokens.length - n; i++) {
    const ngram = tokens
      .slice(i, i + n)
      .map((t) => t.value)
      .join(" ");
    ngrams.push(ngram);
  }

  return ngrams;
}

/**
 * Hash n-grams
 */
export function hashNgrams(ngrams: string[]): string[] {
  return ngrams.map((ng) =>
    CryptoJS.MD5(ng).toString(CryptoJS.enc.Hex).substring(0, 16)
  );
}

// ============================================================================
// WINNOWING ALGORITHM (@PHANTOM)
// ============================================================================

/**
 * Winnowing algorithm for fingerprint generation
 * Robust document fingerprinting algorithm
 * @agent @PHANTOM - Fingerprint extraction
 */
export function winnow(
  tokens: Token[],
  k: number = 5, // k-gram size
  w: number = 4 // window size
): CodeFingerprint {
  const kgrams = generateNgrams(tokens, k);
  const hashes = hashNgrams(kgrams);

  const fingerprints: string[] = [];
  const positions: Array<{ hash: string; start: number; end: number }> = [];

  // Sliding window selection
  for (let i = 0; i <= hashes.length - w; i++) {
    const window = hashes.slice(i, i + w);

    // Select minimum hash in window (rightmost if tie)
    let minHash = window[0]!;
    let minIndex = 0;

    for (let j = 1; j < window.length; j++) {
      if (window[j]! <= minHash) {
        minHash = window[j]!;
        minIndex = j;
      }
    }

    // Add if not duplicate of last added
    if (
      fingerprints.length === 0 ||
      fingerprints[fingerprints.length - 1] !== minHash
    ) {
      fingerprints.push(minHash);
      positions.push({
        hash: minHash,
        start: i + minIndex,
        end: i + minIndex + k,
      });
    }
  }

  return {
    hashes: fingerprints,
    positions,
    tokens,
  };
}

// ============================================================================
// MINHASH (@TENSOR)
// ============================================================================

/**
 * MinHash signature generation
 * Locality-Sensitive Hashing for similarity
 * @agent @TENSOR - Approximate matching
 */
export class MinHashGenerator {
  private numHashes: number;
  private hashFunctions: Array<(x: string) => number>;

  constructor(numHashes: number = 100) {
    this.numHashes = numHashes;
    this.hashFunctions = this.generateHashFunctions(numHashes);
  }

  /**
   * Generate hash functions using random coefficients
   */
  private generateHashFunctions(n: number): Array<(x: string) => number> {
    const prime = 2147483647; // Large prime
    const functions: Array<(x: string) => number> = [];

    // Use deterministic "random" values based on index
    for (let i = 0; i < n; i++) {
      const a = ((i * 1103515245 + 12345) % prime) + 1;
      const b = (i * 214013 + 2531011) % prime;

      functions.push((x: string) => {
        const hash = parseInt(
          CryptoJS.MD5(x).toString(CryptoJS.enc.Hex).substring(0, 8),
          16
        );
        return (a * hash + b) % prime;
      });
    }

    return functions;
  }

  /**
   * Generate MinHash signature for a set
   */
  generateSignature(elements: string[]): number[] {
    const signature: number[] = new Array(this.numHashes).fill(Infinity);

    for (const element of elements) {
      for (let i = 0; i < this.numHashes; i++) {
        const hash = this.hashFunctions[i]!(element);
        if (hash < signature[i]!) {
          signature[i] = hash;
        }
      }
    }

    return signature;
  }

  /**
   * Estimate Jaccard similarity from signatures
   */
  estimateSimilarity(sig1: number[], sig2: number[]): number {
    if (sig1.length !== sig2.length) {
      throw new Error("Signatures must have same length");
    }

    let matches = 0;
    for (let i = 0; i < sig1.length; i++) {
      if (sig1[i] === sig2[i]) {
        matches++;
      }
    }

    return matches / sig1.length;
  }
}

// ============================================================================
// SIMILARITY ANALYZER (@PHANTOM @TENSOR)
// ============================================================================

/**
 * Code Similarity Analyzer
 * @agent @PHANTOM @TENSOR - Complete similarity analysis
 */
export class SimilarityAnalyzer {
  private options: SimilarityOptions;
  private minHashGenerator: MinHashGenerator;

  constructor(options: Partial<SimilarityOptions> = {}) {
    this.options = { ...DEFAULT_SIMILARITY_OPTIONS, ...options };
    this.minHashGenerator = new MinHashGenerator(this.options.numHashFunctions);
  }

  /**
   * Analyze similarity between two code files
   */
  analyze(
    sourceCode: string,
    targetCode: string,
    sourcePath: string = "source",
    targetPath: string = "target",
    language: string = "javascript"
  ): SimilarityResult {
    const startTime = Date.now();

    // Tokenize
    const sourceTokens = tokenize(sourceCode, language, this.options);
    const targetTokens = tokenize(targetCode, language, this.options);

    // Calculate similarity based on algorithm
    let similarity: number;
    let matchedRegions: MatchedRegion[] = [];
    let fingerprints: {
      sourceUnique: number;
      targetUnique: number;
      shared: number;
    };

    switch (this.options.algorithm) {
      case "winnowing":
        ({ similarity, matchedRegions, fingerprints } =
          this.winnowingSimilarity(sourceTokens, targetTokens));
        break;

      case "ngram":
        ({ similarity, matchedRegions, fingerprints } = this.ngramSimilarity(
          sourceTokens,
          targetTokens
        ));
        break;

      case "jaccard":
        ({ similarity, fingerprints } = this.jaccardSimilarity(
          sourceTokens,
          targetTokens
        ));
        matchedRegions = [];
        break;

      case "minhash":
        ({ similarity, fingerprints } = this.minHashSimilarity(
          sourceTokens,
          targetTokens
        ));
        matchedRegions = [];
        break;

      case "combined":
      default:
        ({ similarity, matchedRegions, fingerprints } = this.combinedSimilarity(
          sourceTokens,
          targetTokens
        ));
    }

    return {
      sourceFile: sourcePath,
      targetFile: targetPath,
      overallSimilarity: similarity,
      algorithm: this.options.algorithm,
      matchedRegions,
      fingerprints,
      analysisTime: Date.now() - startTime,
    };
  }

  /**
   * Batch analysis of multiple files
   */
  batchAnalyze(
    files: Array<{ path: string; content: string }>,
    language: string = "javascript"
  ): SimilarityResult[] {
    const results: SimilarityResult[] = [];

    // Compare all pairs
    for (let i = 0; i < files.length; i++) {
      for (let j = i + 1; j < files.length; j++) {
        const result = this.analyze(
          files[i]!.content,
          files[j]!.content,
          files[i]!.path,
          files[j]!.path,
          language
        );

        if (result.overallSimilarity >= this.options.threshold) {
          results.push(result);
        }
      }
    }

    // Sort by similarity
    results.sort((a, b) => b.overallSimilarity - a.overallSimilarity);

    return results;
  }

  // ============================================================================
  // PRIVATE METHODS
  // ============================================================================

  /**
   * Winnowing-based similarity
   */
  private winnowingSimilarity(
    sourceTokens: Token[],
    targetTokens: Token[]
  ): {
    similarity: number;
    matchedRegions: MatchedRegion[];
    fingerprints: {
      sourceUnique: number;
      targetUnique: number;
      shared: number;
    };
  } {
    const sourceFingerprint = winnow(
      sourceTokens,
      this.options.ngramSize,
      this.options.windowSize
    );
    const targetFingerprint = winnow(
      targetTokens,
      this.options.ngramSize,
      this.options.windowSize
    );

    const sourceHashSet = new Set(sourceFingerprint.hashes);
    const targetHashSet = new Set(targetFingerprint.hashes);

    // Calculate intersection
    const shared = [...sourceHashSet].filter((h) => targetHashSet.has(h));
    const union = new Set([...sourceHashSet, ...targetHashSet]);

    const similarity = shared.length / union.size;

    // Find matched regions
    const matchedRegions = this.findMatchedRegions(
      sourceFingerprint,
      targetFingerprint,
      new Set(shared)
    );

    return {
      similarity,
      matchedRegions,
      fingerprints: {
        sourceUnique: sourceHashSet.size - shared.length,
        targetUnique: targetHashSet.size - shared.length,
        shared: shared.length,
      },
    };
  }

  /**
   * N-gram based similarity
   */
  private ngramSimilarity(
    sourceTokens: Token[],
    targetTokens: Token[]
  ): {
    similarity: number;
    matchedRegions: MatchedRegion[];
    fingerprints: {
      sourceUnique: number;
      targetUnique: number;
      shared: number;
    };
  } {
    const sourceNgrams = new Set(
      generateNgrams(sourceTokens, this.options.ngramSize)
    );
    const targetNgrams = new Set(
      generateNgrams(targetTokens, this.options.ngramSize)
    );

    const shared = [...sourceNgrams].filter((ng) => targetNgrams.has(ng));
    const union = new Set([...sourceNgrams, ...targetNgrams]);

    const similarity = union.size > 0 ? shared.length / union.size : 0;

    return {
      similarity,
      matchedRegions: [], // N-gram doesn't provide regions easily
      fingerprints: {
        sourceUnique: sourceNgrams.size - shared.length,
        targetUnique: targetNgrams.size - shared.length,
        shared: shared.length,
      },
    };
  }

  /**
   * Jaccard similarity on token sets
   */
  private jaccardSimilarity(
    sourceTokens: Token[],
    targetTokens: Token[]
  ): {
    similarity: number;
    fingerprints: {
      sourceUnique: number;
      targetUnique: number;
      shared: number;
    };
  } {
    const sourceSet = new Set(sourceTokens.map((t) => t.value));
    const targetSet = new Set(targetTokens.map((t) => t.value));

    const intersection = [...sourceSet].filter((v) => targetSet.has(v));
    const union = new Set([...sourceSet, ...targetSet]);

    const similarity = union.size > 0 ? intersection.length / union.size : 0;

    return {
      similarity,
      fingerprints: {
        sourceUnique: sourceSet.size - intersection.length,
        targetUnique: targetSet.size - intersection.length,
        shared: intersection.length,
      },
    };
  }

  /**
   * MinHash approximate similarity
   */
  private minHashSimilarity(
    sourceTokens: Token[],
    targetTokens: Token[]
  ): {
    similarity: number;
    fingerprints: {
      sourceUnique: number;
      targetUnique: number;
      shared: number;
    };
  } {
    const sourceNgrams = generateNgrams(sourceTokens, this.options.ngramSize);
    const targetNgrams = generateNgrams(targetTokens, this.options.ngramSize);

    const sourceSig = this.minHashGenerator.generateSignature(sourceNgrams);
    const targetSig = this.minHashGenerator.generateSignature(targetNgrams);

    const similarity = this.minHashGenerator.estimateSimilarity(
      sourceSig,
      targetSig
    );

    // Estimate fingerprint counts from similarity
    const total = sourceNgrams.length + targetNgrams.length;
    const estimatedShared = Math.round((total * similarity) / (1 + similarity));

    return {
      similarity,
      fingerprints: {
        sourceUnique: sourceNgrams.length - estimatedShared,
        targetUnique: targetNgrams.length - estimatedShared,
        shared: estimatedShared,
      },
    };
  }

  /**
   * Combined similarity using multiple methods
   * @agent @TENSOR - Ensemble approach
   */
  private combinedSimilarity(
    sourceTokens: Token[],
    targetTokens: Token[]
  ): {
    similarity: number;
    matchedRegions: MatchedRegion[];
    fingerprints: {
      sourceUnique: number;
      targetUnique: number;
      shared: number;
    };
  } {
    const winnowing = this.winnowingSimilarity(sourceTokens, targetTokens);
    const jaccard = this.jaccardSimilarity(sourceTokens, targetTokens);
    const minhash = this.minHashSimilarity(sourceTokens, targetTokens);

    // Weighted average - winnowing has best precision for code
    const similarity =
      winnowing.similarity * 0.5 +
      jaccard.similarity * 0.2 +
      minhash.similarity * 0.3;

    return {
      similarity,
      matchedRegions: winnowing.matchedRegions,
      fingerprints: winnowing.fingerprints,
    };
  }

  /**
   * Find matched regions from fingerprints
   */
  private findMatchedRegions(
    sourceFingerprint: CodeFingerprint,
    targetFingerprint: CodeFingerprint,
    sharedHashes: Set<string>
  ): MatchedRegion[] {
    const regions: MatchedRegion[] = [];

    // Build lookup for target positions
    const targetPositions = new Map<string, { start: number; end: number }[]>();
    for (const pos of targetFingerprint.positions) {
      if (!targetPositions.has(pos.hash)) {
        targetPositions.set(pos.hash, []);
      }
      targetPositions.get(pos.hash)!.push({ start: pos.start, end: pos.end });
    }

    // Find matching regions
    for (const sourcePos of sourceFingerprint.positions) {
      if (sharedHashes.has(sourcePos.hash)) {
        const targets = targetPositions.get(sourcePos.hash) ?? [];

        for (const targetPos of targets) {
          regions.push({
            sourceStart: sourcePos.start,
            sourceEnd: sourcePos.end,
            targetStart: targetPos.start,
            targetEnd: targetPos.end,
            similarity: 1.0, // Exact fingerprint match
            type: "exact",
          });
        }
      }
    }

    // Merge adjacent regions
    return this.mergeAdjacentRegions(regions);
  }

  /**
   * Merge adjacent matched regions
   */
  private mergeAdjacentRegions(regions: MatchedRegion[]): MatchedRegion[] {
    if (regions.length === 0) return [];

    // Sort by source position
    const sorted = [...regions].sort((a, b) => a.sourceStart - b.sourceStart);
    const merged: MatchedRegion[] = [sorted[0]!];

    for (let i = 1; i < sorted.length; i++) {
      const current = sorted[i]!;
      const last = merged[merged.length - 1]!;

      // Check if adjacent (within 2 positions)
      if (
        current.sourceStart <= last.sourceEnd + 2 &&
        current.targetStart <= last.targetEnd + 2
      ) {
        // Merge
        last.sourceEnd = Math.max(last.sourceEnd, current.sourceEnd);
        last.targetEnd = Math.max(last.targetEnd, current.targetEnd);
      } else {
        merged.push(current);
      }
    }

    return merged;
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export {
  tokenize,
  generateNgrams,
  hashNgrams,
  winnow,
  MinHashGenerator,
  SimilarityAnalyzer,
  DEFAULT_SIMILARITY_OPTIONS,
};

export type {
  SimilarityOptions,
  Token,
  CodeFingerprint,
  SimilarityResult,
  MatchedRegion,
};
