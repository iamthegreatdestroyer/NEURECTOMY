/**
 * @fileoverview Semantic Code Analysis Engine
 * @module @neurectomy/legal-fortress/plagiarism/semantic
 *
 * Agent Assignment: @LINGUA (NLP/LLM) + @TENSOR (ML/Deep Learning)
 *
 * Implements deep semantic analysis for code comparison beyond surface-level
 * and structural matching. Uses embeddings, semantic similarity, and
 * intent recognition to detect conceptually similar code.
 *
 * @author NEURECTOMY Phase 5 - Legal Fortress
 * @version 1.0.0
 */

import { EventEmitter } from "events";
import CryptoJS from "crypto-js";

// ============================================================================
// Semantic Analysis Types (@LINGUA @TENSOR)
// ============================================================================

/**
 * Embedding model configuration
 */
export interface EmbeddingModelConfig {
  modelId: string;
  dimensions: number;
  maxTokens: number;
  endpoint?: string;
  apiKey?: string;
  batchSize: number;
  cacheEnabled: boolean;
}

/**
 * Code embedding result
 */
export interface CodeEmbedding {
  id: string;
  code: string;
  language: string;
  vector: number[];
  metadata: {
    functionName?: string;
    className?: string;
    complexity: number;
    lineCount: number;
    tokenCount: number;
  };
  timestamp: Date;
}

/**
 * Semantic similarity result
 */
export interface SemanticSimilarityResult {
  sourceId: string;
  targetId: string;
  cosineSimilarity: number;
  euclideanDistance: number;
  manhattanDistance: number;
  intentSimilarity: number;
  overallScore: number;
  confidence: number;
}

/**
 * Code intent classification
 */
export interface CodeIntent {
  primary: string;
  secondary: string[];
  patterns: string[];
  domain: string;
  confidence: number;
}

/**
 * Semantic match with detailed analysis
 */
export interface SemanticMatch {
  source: CodeRegion;
  target: CodeRegion;
  similarity: SemanticSimilarityResult;
  intent: {
    sourceIntent: CodeIntent;
    targetIntent: CodeIntent;
    intentOverlap: number;
  };
  explanation: string;
  verdict: "identical" | "similar" | "related" | "different";
}

/**
 * Code region for analysis
 */
export interface CodeRegion {
  file: string;
  startLine: number;
  endLine: number;
  content: string;
  language: string;
}

/**
 * Semantic analysis configuration
 */
export interface SemanticAnalysisConfig {
  embeddingModel: EmbeddingModelConfig;
  similarityThreshold: number;
  intentRecognition: boolean;
  contextWindow: number;
  crossLanguageSupport: boolean;
  parallelization: number;
  cacheDirectory?: string;
}

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_EMBEDDING_CONFIG: EmbeddingModelConfig = {
  modelId: "code-embedding-v1",
  dimensions: 768,
  maxTokens: 8192,
  batchSize: 32,
  cacheEnabled: true,
};

const DEFAULT_SEMANTIC_CONFIG: SemanticAnalysisConfig = {
  embeddingModel: DEFAULT_EMBEDDING_CONFIG,
  similarityThreshold: 0.75,
  intentRecognition: true,
  contextWindow: 512,
  crossLanguageSupport: true,
  parallelization: 4,
};

// ============================================================================
// Embedding Generator (@TENSOR)
// ============================================================================

/**
 * Generates semantic embeddings for code snippets
 * Uses transformer-based models optimized for code understanding
 */
export class EmbeddingGenerator extends EventEmitter {
  private config: EmbeddingModelConfig;
  private cache: Map<string, CodeEmbedding>;
  private tokenizer: CodeTokenizer;

  constructor(config: Partial<EmbeddingModelConfig> = {}) {
    super();
    this.config = { ...DEFAULT_EMBEDDING_CONFIG, ...config };
    this.cache = new Map();
    this.tokenizer = new CodeTokenizer();
  }

  /**
   * Generate embedding for a code snippet
   */
  async generateEmbedding(
    code: string,
    language: string
  ): Promise<CodeEmbedding> {
    const cacheKey = this.computeCacheKey(code, language);

    // Check cache first
    if (this.config.cacheEnabled && this.cache.has(cacheKey)) {
      this.emit("cache-hit", { cacheKey });
      return this.cache.get(cacheKey)!;
    }

    // Tokenize code
    const tokens = this.tokenizer.tokenize(code, language);
    const truncatedTokens = tokens.slice(0, this.config.maxTokens);

    // Generate embedding using the configured model
    const vector = await this.computeEmbedding(truncatedTokens, language);

    const embedding: CodeEmbedding = {
      id: cacheKey,
      code,
      language,
      vector,
      metadata: {
        complexity: this.computeComplexity(code),
        lineCount: code.split("\n").length,
        tokenCount: tokens.length,
      },
      timestamp: new Date(),
    };

    // Cache result
    if (this.config.cacheEnabled) {
      this.cache.set(cacheKey, embedding);
    }

    this.emit("embedding-generated", { embedding });
    return embedding;
  }

  /**
   * Generate embeddings for multiple code snippets in batch
   */
  async generateBatchEmbeddings(
    snippets: Array<{ code: string; language: string }>
  ): Promise<CodeEmbedding[]> {
    const embeddings: CodeEmbedding[] = [];
    const batches: Array<{
      code: string;
      language: string;
      index: number;
    }>[] = [];

    // Split into batches
    let currentBatch: Array<{
      code: string;
      language: string;
      index: number;
    }> = [];
    snippets.forEach((snippet, index) => {
      currentBatch.push({ ...snippet, index });
      if (currentBatch.length >= this.config.batchSize) {
        batches.push(currentBatch);
        currentBatch = [];
      }
    });
    if (currentBatch.length > 0) {
      batches.push(currentBatch);
    }

    // Process batches
    for (const batch of batches) {
      const batchEmbeddings = await Promise.all(
        batch.map((item) => this.generateEmbedding(item.code, item.language))
      );
      embeddings.push(...batchEmbeddings);
      this.emit("batch-processed", { batchSize: batch.length });
    }

    return embeddings;
  }

  /**
   * Compute embedding vector using neural network
   */
  private async computeEmbedding(
    tokens: string[],
    _language: string
  ): Promise<number[]> {
    // Simulated embedding computation
    // In production, this would call an actual embedding model API
    const vector: number[] = new Array(this.config.dimensions).fill(0);

    // Token-based feature extraction
    for (let i = 0; i < tokens.length; i++) {
      const tokenHash = this.hashToken(tokens[i]);
      for (let d = 0; d < this.config.dimensions; d++) {
        // Distribute token influence across dimensions
        vector[d] += Math.sin(tokenHash * (d + 1)) / Math.sqrt(tokens.length);
      }
    }

    // Normalize to unit vector
    const magnitude = Math.sqrt(
      vector.reduce((sum, val) => sum + val * val, 0)
    );
    return vector.map((v) => v / (magnitude || 1));
  }

  /**
   * Hash a token to a numeric value
   */
  private hashToken(token: string): number {
    const hash = CryptoJS.SHA256(token).toString();
    return parseInt(hash.slice(0, 8), 16) / 0xffffffff;
  }

  /**
   * Compute code complexity score
   */
  private computeComplexity(code: string): number {
    let complexity = 0;

    // Control flow keywords
    const controlFlowKeywords = [
      "if",
      "else",
      "for",
      "while",
      "switch",
      "case",
      "try",
      "catch",
      "finally",
      "do",
    ];
    for (const keyword of controlFlowKeywords) {
      const regex = new RegExp(`\\b${keyword}\\b`, "g");
      complexity += (code.match(regex) || []).length;
    }

    // Nesting depth estimation
    const braceMatches = code.match(/[{}]/g) || [];
    let depth = 0;
    let maxDepth = 0;
    for (const brace of braceMatches) {
      if (brace === "{") {
        depth++;
        maxDepth = Math.max(maxDepth, depth);
      } else {
        depth--;
      }
    }
    complexity += maxDepth * 2;

    return complexity;
  }

  /**
   * Compute cache key for code+language combination
   */
  private computeCacheKey(code: string, language: string): string {
    return CryptoJS.SHA256(`${language}:${code}`).toString().slice(0, 16);
  }

  /**
   * Clear embedding cache
   */
  clearCache(): void {
    this.cache.clear();
    this.emit("cache-cleared");
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { size: number; hits: number; misses: number } {
    return {
      size: this.cache.size,
      hits: 0, // Would track in production
      misses: 0,
    };
  }
}

// ============================================================================
// Code Tokenizer (@LINGUA)
// ============================================================================

/**
 * Language-aware code tokenizer for semantic analysis
 */
export class CodeTokenizer {
  private languagePatterns: Map<string, RegExp[]>;

  constructor() {
    this.languagePatterns = new Map();
    this.initializePatterns();
  }

  /**
   * Tokenize code based on language
   */
  tokenize(code: string, language: string): string[] {
    const patterns =
      this.languagePatterns.get(language) || this.getDefaultPatterns();
    const tokens: string[] = [];
    let remaining = code;

    while (remaining.length > 0) {
      let matched = false;

      for (const pattern of patterns) {
        const match = remaining.match(pattern);
        if (match && match.index === 0) {
          const token = match[0].trim();
          if (token.length > 0) {
            tokens.push(token);
          }
          remaining = remaining.slice(match[0].length);
          matched = true;
          break;
        }
      }

      if (!matched) {
        // Skip single character
        remaining = remaining.slice(1);
      }
    }

    return tokens;
  }

  /**
   * Initialize language-specific patterns
   */
  private initializePatterns(): void {
    // TypeScript/JavaScript patterns
    const jsPatterns: RegExp[] = [
      /^\s+/, // Whitespace
      /^\/\/[^\n]*/, // Single-line comment
      /^\/\*[\s\S]*?\*\//, // Multi-line comment
      /^"(?:[^"\\]|\\.)*"/, // Double-quoted string
      /^'(?:[^'\\]|\\.)*'/, // Single-quoted string
      /^`(?:[^`\\]|\\.)*`/, // Template literal
      /^0x[0-9a-fA-F]+/, // Hex number
      /^\d+\.?\d*(?:[eE][+-]?\d+)?/, // Number
      /^(?:async|await|break|case|catch|class|const|continue|debugger|default|delete|do|else|enum|export|extends|finally|for|function|if|import|in|instanceof|interface|let|new|return|static|super|switch|this|throw|try|typeof|var|void|while|with|yield)\b/, // Keywords
      /^[a-zA-Z_$][a-zA-Z0-9_$]*/, // Identifier
      /^(?:===|!==|==|!=|<=|>=|=>|&&|\|\||<<|>>|>>>|\+\+|--|[+\-*/%&|^~!<>=]=?)/, // Operators
      /^[{}()\[\];,.:?]/, // Punctuation
    ];
    this.languagePatterns.set("typescript", jsPatterns);
    this.languagePatterns.set("javascript", jsPatterns);

    // Python patterns
    const pythonPatterns: RegExp[] = [
      /^\s+/,
      /^#[^\n]*/,
      /^"""[\s\S]*?"""/,
      /^'''[\s\S]*?'''/,
      /^"(?:[^"\\]|\\.)*"/,
      /^'(?:[^'\\]|\\.)*'/,
      /^0x[0-9a-fA-F]+/,
      /^\d+\.?\d*(?:[eE][+-]?\d+)?/,
      /^(?:and|as|assert|async|await|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|None|nonlocal|not|or|pass|raise|return|try|while|with|yield|True|False)\b/,
      /^[a-zA-Z_][a-zA-Z0-9_]*/,
      /^(?:==|!=|<=|>=|<<|>>|\*\*|\/\/|[+\-*/%&|^~!<>=@]=?)/,
      /^[{}()\[\];,:]/,
    ];
    this.languagePatterns.set("python", pythonPatterns);

    // Java patterns
    const javaPatterns: RegExp[] = [
      /^\s+/,
      /^\/\/[^\n]*/,
      /^\/\*[\s\S]*?\*\//,
      /^"(?:[^"\\]|\\.)*"/,
      /^'(?:[^'\\]|\\.)*'/,
      /^0x[0-9a-fA-F]+/,
      /^\d+\.?\d*(?:[eE][+-]?\d+)?[fFdDlL]?/,
      /^(?:abstract|assert|boolean|break|byte|case|catch|char|class|const|continue|default|do|double|else|enum|extends|final|finally|float|for|goto|if|implements|import|instanceof|int|interface|long|native|new|package|private|protected|public|return|short|static|strictfp|super|switch|synchronized|this|throw|throws|transient|try|void|volatile|while|true|false|null)\b/,
      /^[a-zA-Z_$][a-zA-Z0-9_$]*/,
      /^(?:===|!==|==|!=|<=|>=|&&|\|\||\+\+|--|[+\-*/%&|^~!<>=]=?)/,
      /^[{}()\[\];,.:?@]/,
    ];
    this.languagePatterns.set("java", javaPatterns);
  }

  /**
   * Get default tokenization patterns
   */
  private getDefaultPatterns(): RegExp[] {
    return [
      /^\s+/,
      /^\/\/[^\n]*/,
      /^\/\*[\s\S]*?\*\//,
      /^"(?:[^"\\]|\\.)*"/,
      /^'(?:[^'\\]|\\.)*'/,
      /^\d+\.?\d*/,
      /^[a-zA-Z_][a-zA-Z0-9_]*/,
      /^[+\-*/%=<>!&|^~]+/,
      /^[{}()\[\];,.:]/,
    ];
  }
}

// ============================================================================
// Intent Recognizer (@LINGUA)
// ============================================================================

/**
 * Recognizes code intent and purpose for semantic comparison
 */
export class IntentRecognizer extends EventEmitter {
  private intentPatterns: Map<string, IntentPattern[]>;

  constructor() {
    super();
    this.intentPatterns = new Map();
    this.initializeIntentPatterns();
  }

  /**
   * Recognize intent from code snippet
   */
  async recognizeIntent(code: string, language: string): Promise<CodeIntent> {
    const patterns =
      this.intentPatterns.get(language) || this.intentPatterns.get("default")!;
    const matchedIntents: Array<{ intent: string; score: number }> = [];

    for (const pattern of patterns) {
      const score = this.matchPattern(code, pattern);
      if (score > 0) {
        matchedIntents.push({ intent: pattern.intent, score });
      }
    }

    // Sort by score
    matchedIntents.sort((a, b) => b.score - a.score);

    // Determine primary and secondary intents
    const primary = matchedIntents[0]?.intent || "general";
    const secondary = matchedIntents.slice(1, 4).map((m) => m.intent);

    const detectedPatterns = matchedIntents.map((m) => m.intent);
    const domain = this.inferDomain(detectedPatterns);
    const confidence = matchedIntents[0]?.score || 0;

    return {
      primary,
      secondary,
      patterns: detectedPatterns,
      domain,
      confidence,
    };
  }

  /**
   * Match code against intent pattern
   */
  private matchPattern(code: string, pattern: IntentPattern): number {
    let score = 0;

    // Check keywords
    for (const keyword of pattern.keywords) {
      const regex = new RegExp(`\\b${keyword}\\b`, "gi");
      const matches = code.match(regex);
      if (matches) {
        score += matches.length * pattern.weight;
      }
    }

    // Check structural patterns
    for (const struct of pattern.structures) {
      if (struct.test(code)) {
        score += pattern.weight * 2;
      }
    }

    return Math.min(score / 100, 1); // Normalize to [0, 1]
  }

  /**
   * Infer domain from detected patterns
   */
  private inferDomain(patterns: string[]): string {
    const domainMapping: Record<string, string[]> = {
      "data-processing": ["sorting", "filtering", "mapping", "aggregation"],
      "api-integration": ["http-request", "rest-api", "authentication"],
      "file-io": ["file-read", "file-write", "stream"],
      database: ["query", "crud", "transaction"],
      "ui-rendering": ["component", "render", "event-handler"],
      algorithm: ["sorting", "searching", "graph", "dynamic-programming"],
      validation: ["input-validation", "schema", "sanitization"],
      "error-handling": ["try-catch", "error-boundary", "logging"],
    };

    for (const [domain, domainPatterns] of Object.entries(domainMapping)) {
      const overlap = patterns.filter((p) => domainPatterns.includes(p)).length;
      if (overlap >= 2) {
        return domain;
      }
    }

    return "general";
  }

  /**
   * Initialize intent patterns for different languages
   */
  private initializeIntentPatterns(): void {
    const defaultPatterns: IntentPattern[] = [
      {
        intent: "sorting",
        keywords: ["sort", "compare", "swap", "partition", "pivot", "merge"],
        structures: [/\.sort\s*\(/, /for\s*\([^)]*\)\s*{\s*for/],
        weight: 10,
      },
      {
        intent: "searching",
        keywords: ["find", "search", "lookup", "index", "binary", "linear"],
        structures: [
          /\.find\s*\(/,
          /\.indexOf\s*\(/,
          /while\s*\([^)]*low[^)]*high/,
        ],
        weight: 10,
      },
      {
        intent: "filtering",
        keywords: ["filter", "where", "select", "predicate", "condition"],
        structures: [/\.filter\s*\(/, /\.where\s*\(/],
        weight: 10,
      },
      {
        intent: "mapping",
        keywords: ["map", "transform", "convert", "project"],
        structures: [/\.map\s*\(/, /\.select\s*\(/],
        weight: 10,
      },
      {
        intent: "aggregation",
        keywords: ["reduce", "aggregate", "sum", "count", "average", "group"],
        structures: [/\.reduce\s*\(/, /\.groupBy\s*\(/],
        weight: 10,
      },
      {
        intent: "http-request",
        keywords: ["fetch", "axios", "http", "request", "response", "api"],
        structures: [/fetch\s*\(/, /axios\s*\./, /\.get\s*\(|\.post\s*\(/],
        weight: 10,
      },
      {
        intent: "authentication",
        keywords: ["auth", "login", "token", "jwt", "session", "password"],
        structures: [/authorize/, /authenticate/, /Bearer\s+/],
        weight: 10,
      },
      {
        intent: "file-read",
        keywords: ["read", "readFile", "readFileSync", "fs", "open"],
        structures: [/fs\.read/, /readFile\s*\(/],
        weight: 10,
      },
      {
        intent: "file-write",
        keywords: ["write", "writeFile", "writeFileSync", "save", "output"],
        structures: [/fs\.write/, /writeFile\s*\(/],
        weight: 10,
      },
      {
        intent: "query",
        keywords: ["select", "from", "where", "join", "query", "execute"],
        structures: [/SELECT\s+.*\s+FROM/, /\.query\s*\(/i],
        weight: 10,
      },
      {
        intent: "crud",
        keywords: ["create", "read", "update", "delete", "insert", "upsert"],
        structures: [/INSERT\s+INTO/, /UPDATE\s+.*\s+SET/, /DELETE\s+FROM/i],
        weight: 10,
      },
      {
        intent: "validation",
        keywords: ["validate", "check", "verify", "assert", "schema"],
        structures: [/\.validate\s*\(/, /if\s*\(\s*!/, /throw\s+new\s+Error/],
        weight: 10,
      },
      {
        intent: "error-handling",
        keywords: ["try", "catch", "finally", "error", "exception", "throw"],
        structures: [/try\s*{/, /catch\s*\(/, /finally\s*{/],
        weight: 10,
      },
      {
        intent: "event-handler",
        keywords: [
          "onClick",
          "onChange",
          "onSubmit",
          "addEventListener",
          "handler",
        ],
        structures: [/on[A-Z][a-z]+\s*=/, /addEventListener\s*\(/],
        weight: 10,
      },
      {
        intent: "component",
        keywords: [
          "render",
          "component",
          "props",
          "state",
          "useEffect",
          "useState",
        ],
        structures: [/function\s+[A-Z]/, /const\s+[A-Z]\w+\s*=\s*\(/],
        weight: 10,
      },
    ];

    this.intentPatterns.set("default", defaultPatterns);
    this.intentPatterns.set("typescript", defaultPatterns);
    this.intentPatterns.set("javascript", defaultPatterns);
    this.intentPatterns.set("python", defaultPatterns);
  }
}

/**
 * Intent pattern definition
 */
interface IntentPattern {
  intent: string;
  keywords: string[];
  structures: RegExp[];
  weight: number;
}

// ============================================================================
// Semantic Comparator (@LINGUA @TENSOR)
// ============================================================================

/**
 * Main semantic comparison engine combining embeddings and intent analysis
 */
export class SemanticComparator extends EventEmitter {
  private config: SemanticAnalysisConfig;
  private embeddingGenerator: EmbeddingGenerator;
  private intentRecognizer: IntentRecognizer;
  private comparisonCache: Map<string, SemanticSimilarityResult>;

  constructor(config: Partial<SemanticAnalysisConfig> = {}) {
    super();
    this.config = { ...DEFAULT_SEMANTIC_CONFIG, ...config };
    this.embeddingGenerator = new EmbeddingGenerator(
      this.config.embeddingModel
    );
    this.intentRecognizer = new IntentRecognizer();
    this.comparisonCache = new Map();
  }

  /**
   * Compare two code snippets semantically
   */
  async compare(
    source: CodeRegion,
    target: CodeRegion
  ): Promise<SemanticMatch> {
    const startTime = Date.now();

    // Generate embeddings
    const [sourceEmbedding, targetEmbedding] = await Promise.all([
      this.embeddingGenerator.generateEmbedding(
        source.content,
        source.language
      ),
      this.embeddingGenerator.generateEmbedding(
        target.content,
        target.language
      ),
    ]);

    // Compute vector similarities
    const similarity = this.computeSimilarity(
      sourceEmbedding.vector,
      targetEmbedding.vector
    );

    // Recognize intents
    let intentAnalysis = {
      sourceIntent: {
        primary: "general",
        secondary: [],
        patterns: [],
        domain: "general",
        confidence: 0,
      } as CodeIntent,
      targetIntent: {
        primary: "general",
        secondary: [],
        patterns: [],
        domain: "general",
        confidence: 0,
      } as CodeIntent,
      intentOverlap: 0,
    };

    if (this.config.intentRecognition) {
      const [sourceIntent, targetIntent] = await Promise.all([
        this.intentRecognizer.recognizeIntent(source.content, source.language),
        this.intentRecognizer.recognizeIntent(target.content, target.language),
      ]);

      const intentOverlap = this.computeIntentOverlap(
        sourceIntent,
        targetIntent
      );

      intentAnalysis = {
        sourceIntent,
        targetIntent,
        intentOverlap,
      };
    }

    // Compute overall score
    const overallScore = this.computeOverallScore(
      similarity,
      intentAnalysis.intentOverlap
    );

    // Determine verdict
    const verdict = this.determineVerdict(overallScore);

    // Generate explanation
    const explanation = this.generateExplanation(
      similarity,
      intentAnalysis,
      verdict
    );

    const result: SemanticMatch = {
      source,
      target,
      similarity: {
        sourceId: sourceEmbedding.id,
        targetId: targetEmbedding.id,
        ...similarity,
        intentSimilarity: intentAnalysis.intentOverlap,
        overallScore,
        confidence: Math.min(
          sourceEmbedding.metadata.complexity / 10,
          intentAnalysis.sourceIntent.confidence,
          1
        ),
      },
      intent: intentAnalysis,
      explanation,
      verdict,
    };

    this.emit("comparison-complete", {
      result,
      duration: Date.now() - startTime,
    });

    return result;
  }

  /**
   * Batch compare multiple code snippets
   */
  async batchCompare(
    sources: CodeRegion[],
    targets: CodeRegion[]
  ): Promise<SemanticMatch[]> {
    const results: SemanticMatch[] = [];

    // Generate all embeddings in batch
    const allSnippets = [
      ...sources.map((s) => ({ code: s.content, language: s.language })),
      ...targets.map((t) => ({ code: t.content, language: t.language })),
    ];

    await this.embeddingGenerator.generateBatchEmbeddings(allSnippets);

    // Compare each pair
    for (const source of sources) {
      for (const target of targets) {
        const result = await this.compare(source, target);
        if (result.similarity.overallScore >= this.config.similarityThreshold) {
          results.push(result);
        }
      }
    }

    // Sort by similarity
    results.sort(
      (a, b) => b.similarity.overallScore - a.similarity.overallScore
    );

    return results;
  }

  /**
   * Compute similarity metrics between two vectors
   */
  private computeSimilarity(
    vecA: number[],
    vecB: number[]
  ): Omit<
    SemanticSimilarityResult,
    "sourceId" | "targetId" | "intentSimilarity" | "overallScore" | "confidence"
  > {
    // Cosine similarity
    let dotProduct = 0;
    let magA = 0;
    let magB = 0;

    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      magA += vecA[i] * vecA[i];
      magB += vecB[i] * vecB[i];
    }

    const cosineSimilarity =
      dotProduct / (Math.sqrt(magA) * Math.sqrt(magB) || 1);

    // Euclidean distance
    let sumSquaredDiff = 0;
    for (let i = 0; i < vecA.length; i++) {
      sumSquaredDiff += Math.pow(vecA[i] - vecB[i], 2);
    }
    const euclideanDistance = Math.sqrt(sumSquaredDiff);

    // Manhattan distance
    let sumAbsDiff = 0;
    for (let i = 0; i < vecA.length; i++) {
      sumAbsDiff += Math.abs(vecA[i] - vecB[i]);
    }
    const manhattanDistance = sumAbsDiff;

    return {
      cosineSimilarity,
      euclideanDistance,
      manhattanDistance,
    };
  }

  /**
   * Compute overlap between two code intents
   */
  private computeIntentOverlap(
    intentA: CodeIntent,
    intentB: CodeIntent
  ): number {
    // Primary intent match
    let overlap = intentA.primary === intentB.primary ? 0.5 : 0;

    // Domain match
    if (intentA.domain === intentB.domain) {
      overlap += 0.2;
    }

    // Pattern overlap (Jaccard similarity)
    const patternsA = new Set(intentA.patterns);
    const patternsB = new Set(intentB.patterns);
    const intersection = new Set(
      [...patternsA].filter((x) => patternsB.has(x))
    );
    const union = new Set([...patternsA, ...patternsB]);

    if (union.size > 0) {
      overlap += (intersection.size / union.size) * 0.3;
    }

    return overlap;
  }

  /**
   * Compute overall similarity score
   */
  private computeOverallScore(
    similarity: Omit<
      SemanticSimilarityResult,
      | "sourceId"
      | "targetId"
      | "intentSimilarity"
      | "overallScore"
      | "confidence"
    >,
    intentOverlap: number
  ): number {
    // Weighted combination
    const weights = {
      cosine: 0.5,
      euclidean: 0.2,
      intent: 0.3,
    };

    // Normalize euclidean distance to similarity
    const euclideanSimilarity = 1 / (1 + similarity.euclideanDistance);

    return (
      weights.cosine * similarity.cosineSimilarity +
      weights.euclidean * euclideanSimilarity +
      weights.intent * intentOverlap
    );
  }

  /**
   * Determine verdict based on overall score
   */
  private determineVerdict(
    score: number
  ): "identical" | "similar" | "related" | "different" {
    if (score >= 0.95) return "identical";
    if (score >= 0.75) return "similar";
    if (score >= 0.5) return "related";
    return "different";
  }

  /**
   * Generate human-readable explanation
   */
  private generateExplanation(
    similarity: Omit<
      SemanticSimilarityResult,
      | "sourceId"
      | "targetId"
      | "intentSimilarity"
      | "overallScore"
      | "confidence"
    >,
    intent: {
      sourceIntent: CodeIntent;
      targetIntent: CodeIntent;
      intentOverlap: number;
    },
    verdict: string
  ): string {
    const parts: string[] = [];

    parts.push(
      `Semantic analysis indicates ${verdict} code with ${(similarity.cosineSimilarity * 100).toFixed(1)}% vector similarity.`
    );

    if (intent.sourceIntent.primary === intent.targetIntent.primary) {
      parts.push(
        `Both code snippets implement ${intent.sourceIntent.primary} functionality.`
      );
    } else {
      parts.push(
        `Source implements ${intent.sourceIntent.primary} while target implements ${intent.targetIntent.primary}.`
      );
    }

    if (intent.sourceIntent.domain === intent.targetIntent.domain) {
      parts.push(`Both operate in the ${intent.sourceIntent.domain} domain.`);
    }

    return parts.join(" ");
  }

  /**
   * Clear all caches
   */
  clearCaches(): void {
    this.embeddingGenerator.clearCache();
    this.comparisonCache.clear();
    this.emit("caches-cleared");
  }

  /**
   * Get analysis statistics
   */
  getStatistics(): {
    embeddingCache: { size: number; hits: number; misses: number };
    comparisonCache: number;
  } {
    return {
      embeddingCache: this.embeddingGenerator.getCacheStats(),
      comparisonCache: this.comparisonCache.size,
    };
  }
}

// ============================================================================
// Cross-Language Semantic Analyzer (@LINGUA @TENSOR)
// ============================================================================

/**
 * Analyzes semantic similarity across different programming languages
 */
export class CrossLanguageAnalyzer extends EventEmitter {
  private semanticComparator: SemanticComparator;
  private languageNormalizers: Map<string, (code: string) => string>;

  constructor(config: Partial<SemanticAnalysisConfig> = {}) {
    super();
    this.semanticComparator = new SemanticComparator({
      ...config,
      crossLanguageSupport: true,
    });
    this.languageNormalizers = new Map();
    this.initializeNormalizers();
  }

  /**
   * Compare code across different languages
   */
  async compareAcrossLanguages(
    source: CodeRegion,
    target: CodeRegion
  ): Promise<SemanticMatch> {
    // Normalize both code snippets to canonical form
    const normalizedSource: CodeRegion = {
      ...source,
      content: this.normalize(source.content, source.language),
    };

    const normalizedTarget: CodeRegion = {
      ...target,
      content: this.normalize(target.content, target.language),
    };

    // Perform semantic comparison on normalized code
    return this.semanticComparator.compare(normalizedSource, normalizedTarget);
  }

  /**
   * Normalize code to language-agnostic form
   */
  private normalize(code: string, language: string): string {
    const normalizer = this.languageNormalizers.get(language);
    if (normalizer) {
      return normalizer(code);
    }
    return this.defaultNormalize(code);
  }

  /**
   * Default normalization for unsupported languages
   */
  private defaultNormalize(code: string): string {
    return code
      .replace(/\/\*[\s\S]*?\*\//g, "") // Remove block comments
      .replace(/\/\/[^\n]*/g, "") // Remove line comments
      .replace(/#[^\n]*/g, "") // Remove Python comments
      .replace(/\s+/g, " ") // Normalize whitespace
      .trim();
  }

  /**
   * Initialize language-specific normalizers
   */
  private initializeNormalizers(): void {
    // TypeScript/JavaScript normalizer
    const jsNormalizer = (code: string): string => {
      return code
        .replace(/\/\*[\s\S]*?\*\//g, "")
        .replace(/\/\/[^\n]*/g, "")
        .replace(/\bconst\b|\blet\b|\bvar\b/g, "VAR")
        .replace(/\bfunction\b|\b=>\b/g, "FUNC")
        .replace(/\basync\b\s*/g, "")
        .replace(/\bawait\b\s*/g, "")
        .replace(/\s+/g, " ")
        .trim();
    };

    this.languageNormalizers.set("typescript", jsNormalizer);
    this.languageNormalizers.set("javascript", jsNormalizer);

    // Python normalizer
    const pythonNormalizer = (code: string): string => {
      return code
        .replace(/#[^\n]*/g, "")
        .replace(/"""[\s\S]*?"""/g, "")
        .replace(/'''[\s\S]*?'''/g, "")
        .replace(/\bdef\b/g, "FUNC")
        .replace(/\basync\s+def\b/g, "FUNC")
        .replace(/\s+/g, " ")
        .trim();
    };

    this.languageNormalizers.set("python", pythonNormalizer);

    // Java normalizer
    const javaNormalizer = (code: string): string => {
      return code
        .replace(/\/\*[\s\S]*?\*\//g, "")
        .replace(/\/\/[^\n]*/g, "")
        .replace(/\bpublic\b|\bprivate\b|\bprotected\b/g, "")
        .replace(/\bstatic\b/g, "")
        .replace(/\bfinal\b/g, "CONST")
        .replace(/\s+/g, " ")
        .trim();
    };

    this.languageNormalizers.set("java", javaNormalizer);
  }
}

// ============================================================================
// Module Exports
// ============================================================================

export { DEFAULT_EMBEDDING_CONFIG, DEFAULT_SEMANTIC_CONFIG };

export default SemanticComparator;
