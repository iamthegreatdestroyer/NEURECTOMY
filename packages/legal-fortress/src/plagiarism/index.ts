/**
 * @fileoverview Plagiarism Detection Module Exports
 * @module @neurectomy/legal-fortress/plagiarism
 *
 * Agent Assignment: @PHANTOM @TENSOR @LINGUA @CORE
 *
 * Comprehensive plagiarism detection combining:
 * - Token-based similarity (winnowing, n-gram, MinHash)
 * - AST structural comparison
 * - Semantic embedding analysis
 * - Cross-language detection
 *
 * @author NEURECTOMY Phase 5 - Legal Fortress
 * @version 1.0.0
 */

// Similarity Analysis (@PHANTOM @TENSOR)
export {
  SimilarityAnalyzer,
  MinHashGenerator,
  tokenize,
  generateNgrams,
  hashNgrams,
  winnow,
} from "./similarity";

export type {
  SimilarityOptions,
  Token,
  CodeFingerprint,
  SimilarityResult,
  MatchedRegion,
} from "./similarity";

// AST Comparison (@PHANTOM @CORE)
export {
  ASTComparator,
  NormalizedNodeType,
  MatchType,
  TransformationType,
  createASTComparator,
} from "./ast-comparator";

export type {
  SupportedLanguage,
  NormalizedASTNode,
  NodeMetadata,
  SourceLocation,
  ASTComparisonResult,
  SubtreeMatch,
  StructuralDifference,
  ComparisonStatistics,
  DetailedAnalysis,
  ASTComparatorConfig,
} from "./ast-comparator";

// Semantic Analysis (@LINGUA @TENSOR)
export {
  SemanticComparator,
  EmbeddingGenerator,
  CodeTokenizer,
  IntentRecognizer,
  CrossLanguageAnalyzer,
} from "./semantic";

export type {
  EmbeddingModelConfig,
  CodeEmbedding,
  SemanticSimilarityResult,
  CodeIntent,
  SemanticMatch,
  CodeRegion,
  SemanticAnalysisConfig,
} from "./semantic";

// ============================================================================
// Unified Plagiarism Detector
// ============================================================================

import { EventEmitter } from "events";
import { SimilarityAnalyzer, SimilarityResult } from "./similarity";
import { ASTComparator, ASTComparisonResult } from "./ast-comparator";
import {
  SemanticComparator,
  SemanticMatch,
  CodeRegion,
  CrossLanguageAnalyzer,
} from "./semantic";

/**
 * Plagiarism detection configuration
 */
export interface PlagiarismDetectorConfig {
  /** Enable token-based similarity analysis */
  enableSimilarity: boolean;
  /** Enable AST structural comparison */
  enableAST: boolean;
  /** Enable semantic embedding analysis */
  enableSemantic: boolean;
  /** Enable cross-language detection */
  enableCrossLanguage: boolean;
  /** Minimum similarity threshold to report */
  threshold: number;
  /** Weight for similarity score */
  similarityWeight: number;
  /** Weight for AST score */
  astWeight: number;
  /** Weight for semantic score */
  semanticWeight: number;
  /** Maximum file size to analyze (bytes) */
  maxFileSize: number;
  /** Parallel analysis limit */
  parallelLimit: number;
}

/**
 * Unified plagiarism detection result
 */
export interface PlagiarismResult {
  /** Unique result ID */
  id: string;
  /** Source file path */
  sourceFile: string;
  /** Target file path */
  targetFile: string;
  /** Overall plagiarism score (0-1) */
  overallScore: number;
  /** Confidence level */
  confidence: number;
  /** Verdict */
  verdict: "clean" | "suspicious" | "likely_plagiarism" | "definite_plagiarism";
  /** Component scores */
  scores: {
    similarity?: number;
    ast?: number;
    semantic?: number;
  };
  /** Matched regions */
  matches: PlagiarismMatch[];
  /** Analysis metadata */
  metadata: {
    analysisTime: number;
    sourceLines: number;
    targetLines: number;
    languagesAnalyzed: string[];
  };
  /** Timestamp */
  timestamp: Date;
}

/**
 * Individual plagiarism match
 */
export interface PlagiarismMatch {
  /** Match type */
  type: "exact" | "near" | "structural" | "semantic";
  /** Source region */
  source: {
    startLine: number;
    endLine: number;
    content: string;
  };
  /** Target region */
  target: {
    startLine: number;
    endLine: number;
    content: string;
  };
  /** Match similarity */
  similarity: number;
  /** Detection method */
  detectedBy: "similarity" | "ast" | "semantic" | "cross-language";
}

/**
 * Default configuration
 */
const DEFAULT_PLAGIARISM_CONFIG: PlagiarismDetectorConfig = {
  enableSimilarity: true,
  enableAST: true,
  enableSemantic: true,
  enableCrossLanguage: false,
  threshold: 0.3,
  similarityWeight: 0.3,
  astWeight: 0.4,
  semanticWeight: 0.3,
  maxFileSize: 1024 * 1024, // 1MB
  parallelLimit: 4,
};

/**
 * Unified Plagiarism Detection Engine
 *
 * Combines multiple detection methods for comprehensive analysis:
 * - Token-based similarity for surface-level copying
 * - AST comparison for structural plagiarism
 * - Semantic analysis for conceptual copying
 * - Cross-language detection for translated code
 */
export class PlagiarismDetector extends EventEmitter {
  private config: PlagiarismDetectorConfig;
  private similarityAnalyzer: SimilarityAnalyzer;
  private astComparator: ASTComparator;
  private semanticComparator: SemanticComparator;
  private crossLanguageAnalyzer: CrossLanguageAnalyzer;
  private resultCache: Map<string, PlagiarismResult>;

  constructor(config: Partial<PlagiarismDetectorConfig> = {}) {
    super();
    this.config = { ...DEFAULT_PLAGIARISM_CONFIG, ...config };
    this.similarityAnalyzer = new SimilarityAnalyzer();
    this.astComparator = new ASTComparator();
    this.semanticComparator = new SemanticComparator();
    this.crossLanguageAnalyzer = new CrossLanguageAnalyzer();
    this.resultCache = new Map();
  }

  /**
   * Detect plagiarism between two code files
   */
  async detect(
    sourceFile: string,
    sourceCode: string,
    sourceLanguage: string,
    targetFile: string,
    targetCode: string,
    targetLanguage: string
  ): Promise<PlagiarismResult> {
    const startTime = Date.now();
    const resultId = this.generateResultId(sourceFile, targetFile);

    // Check cache
    if (this.resultCache.has(resultId)) {
      this.emit("cache-hit", { resultId });
      return this.resultCache.get(resultId)!;
    }

    // Validate input
    if (
      sourceCode.length > this.config.maxFileSize ||
      targetCode.length > this.config.maxFileSize
    ) {
      throw new Error(
        `File size exceeds maximum allowed (${this.config.maxFileSize} bytes)`
      );
    }

    this.emit("analysis-started", { sourceFile, targetFile });

    // Run enabled analyses
    let similarityResult: SimilarityResult | null = null;
    let astResult: ASTComparisonResult | null = null;
    let semanticResult: SemanticMatch | null = null;

    // Similarity analysis is synchronous
    if (this.config.enableSimilarity) {
      similarityResult = this.similarityAnalyzer.analyze(
        sourceCode,
        targetCode,
        sourceFile,
        targetFile
      );
    }

    // AST comparison is async (takes 3 args: sourceCode, targetCode, language)
    // Use sourceLanguage as the common language for comparison
    if (this.config.enableAST) {
      astResult = await this.astComparator.compare(
        sourceCode,
        targetCode,
        sourceLanguage as import("./ast-comparator").SupportedLanguage
      );
    }

    // Semantic comparison is async
    if (this.config.enableSemantic) {
      const sourceRegion: CodeRegion = {
        file: sourceFile,
        startLine: 1,
        endLine: sourceCode.split("\n").length,
        content: sourceCode,
        language: sourceLanguage,
      };
      const targetRegion: CodeRegion = {
        file: targetFile,
        startLine: 1,
        endLine: targetCode.split("\n").length,
        content: targetCode,
        language: targetLanguage,
      };

      semanticResult =
        this.config.enableCrossLanguage && sourceLanguage !== targetLanguage
          ? await this.crossLanguageAnalyzer.compareAcrossLanguages(
              sourceRegion,
              targetRegion
            )
          : await this.semanticComparator.compare(sourceRegion, targetRegion);
    }

    // Combine scores
    const scores: PlagiarismResult["scores"] = {};
    let totalWeight = 0;
    let weightedScore = 0;

    if (similarityResult) {
      scores.similarity = similarityResult.overallSimilarity;
      weightedScore += scores.similarity * this.config.similarityWeight;
      totalWeight += this.config.similarityWeight;
    }

    if (astResult) {
      scores.ast = astResult.similarity ?? 0;
      weightedScore += scores.ast * this.config.astWeight;
      totalWeight += this.config.astWeight;
    }

    if (semanticResult) {
      scores.semantic = semanticResult.similarity?.overallScore ?? 0;
      weightedScore += scores.semantic * this.config.semanticWeight;
      totalWeight += this.config.semanticWeight;
    }

    const overallScore = totalWeight > 0 ? weightedScore / totalWeight : 0;

    // Collect matches
    const matches = this.collectMatches(
      similarityResult,
      astResult,
      semanticResult
    );

    // Calculate confidence
    const confidence = this.calculateConfidence(
      similarityResult,
      astResult,
      semanticResult
    );

    // Determine verdict
    const verdict = this.determineVerdict(overallScore, confidence);

    const result: PlagiarismResult = {
      id: resultId,
      sourceFile,
      targetFile,
      overallScore,
      confidence,
      verdict,
      scores,
      matches,
      metadata: {
        analysisTime: Date.now() - startTime,
        sourceLines: sourceCode.split("\n").length,
        targetLines: targetCode.split("\n").length,
        languagesAnalyzed: [...new Set([sourceLanguage, targetLanguage])],
      },
      timestamp: new Date(),
    };

    // Cache result
    this.resultCache.set(resultId, result);

    this.emit("analysis-complete", { result });
    return result;
  }

  /**
   * Batch detect plagiarism across multiple files
   */
  async batchDetect(
    files: Array<{
      path: string;
      content: string;
      language: string;
    }>
  ): Promise<PlagiarismResult[]> {
    const results: PlagiarismResult[] = [];
    const pairs: Array<[number, number]> = [];

    // Generate all unique pairs
    for (let i = 0; i < files.length; i++) {
      for (let j = i + 1; j < files.length; j++) {
        pairs.push([i, j]);
      }
    }

    this.emit("batch-started", { totalPairs: pairs.length });

    // Process pairs with parallelism limit
    const chunks = this.chunkArray(pairs, this.config.parallelLimit);

    for (const chunk of chunks) {
      const chunkResults = await Promise.all(
        chunk.map(([i, j]) => {
          const fileI = files[i]!;
          const fileJ = files[j]!;
          return this.detect(
            fileI.path,
            fileI.content,
            fileI.language,
            fileJ.path,
            fileJ.content,
            fileJ.language
          );
        })
      );

      // Filter by threshold
      for (const result of chunkResults) {
        if (result.overallScore >= this.config.threshold) {
          results.push(result);
        }
      }

      this.emit("batch-progress", {
        processed: results.length,
        total: pairs.length,
      });
    }

    // Sort by score descending
    results.sort((a, b) => b.overallScore - a.overallScore);

    this.emit("batch-complete", { results: results.length });
    return results;
  }

  /**
   * Generate unique result ID
   */
  private generateResultId(sourceFile: string, targetFile: string): string {
    const sorted = [sourceFile, targetFile].sort();
    return `${sorted[0]}::${sorted[1]}`;
  }

  /**
   * Collect all matches from different analyzers
   */
  private collectMatches(
    similarityResult: SimilarityResult | null,
    astResult: ASTComparisonResult | null,
    semanticResult: SemanticMatch | null
  ): PlagiarismMatch[] {
    const matches: PlagiarismMatch[] = [];

    // Add similarity matches
    if (similarityResult?.matchedRegions) {
      for (const region of similarityResult.matchedRegions) {
        matches.push({
          type: region.type,
          source: {
            startLine: region.sourceStart,
            endLine: region.sourceEnd,
            content: "",
          },
          target: {
            startLine: region.targetStart,
            endLine: region.targetEnd,
            content: "",
          },
          similarity: region.similarity,
          detectedBy: "similarity",
        });
      }
    }

    // Add AST matches
    if (astResult?.matchedSubtrees) {
      for (const subtree of astResult.matchedSubtrees) {
        matches.push({
          type: "structural",
          source: {
            startLine: subtree.sourceNode.location?.startLine ?? 0,
            endLine: subtree.sourceNode.location?.endLine ?? 0,
            content: "",
          },
          target: {
            startLine: subtree.targetNode.location?.startLine ?? 0,
            endLine: subtree.targetNode.location?.endLine ?? 0,
            content: "",
          },
          similarity: subtree.similarity,
          detectedBy: "ast",
        });
      }
    }

    // Add semantic matches
    if (semanticResult && semanticResult.verdict !== "different") {
      matches.push({
        type: "semantic",
        source: {
          startLine: semanticResult.source.startLine,
          endLine: semanticResult.source.endLine,
          content: "",
        },
        target: {
          startLine: semanticResult.target.startLine,
          endLine: semanticResult.target.endLine,
          content: "",
        },
        similarity: semanticResult.similarity.overallScore,
        detectedBy: "semantic",
      });
    }

    return matches;
  }

  /**
   * Calculate confidence based on agreement between methods
   */
  private calculateConfidence(
    similarityResult: SimilarityResult | null,
    astResult: ASTComparisonResult | null,
    semanticResult: SemanticMatch | null
  ): number {
    const scores: number[] = [];

    if (similarityResult) {
      scores.push(similarityResult.overallSimilarity);
    }
    if (astResult) {
      scores.push(astResult.similarity);
    }
    if (semanticResult) {
      scores.push(semanticResult.similarity.overallScore);
    }

    if (scores.length < 2) {
      return 0.5; // Low confidence with single method
    }

    // Calculate variance (lower variance = higher confidence)
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const variance =
      scores.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / scores.length;

    // Convert variance to confidence (inverse relationship)
    // Max variance is 0.25 (when scores are 0 and 1)
    return Math.max(0, 1 - variance * 4);
  }

  /**
   * Determine verdict based on score and confidence
   */
  private determineVerdict(
    score: number,
    confidence: number
  ): PlagiarismResult["verdict"] {
    const adjustedScore = score * confidence;

    if (adjustedScore >= 0.9) return "definite_plagiarism";
    if (adjustedScore >= 0.7) return "likely_plagiarism";
    if (adjustedScore >= 0.4) return "suspicious";
    return "clean";
  }

  /**
   * Split array into chunks
   */
  private chunkArray<T>(array: T[], size: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    return chunks;
  }

  /**
   * Clear result cache
   */
  clearCache(): void {
    this.resultCache.clear();
    // SimilarityAnalyzer doesn't have a cache to clear
    this.semanticComparator.clearCaches();
    this.emit("caches-cleared");
  }

  /**
   * Get detector statistics
   */
  getStatistics(): {
    cacheSize: number;
    analysisCount: number;
  } {
    return {
      cacheSize: this.resultCache.size,
      analysisCount: this.resultCache.size,
    };
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<PlagiarismDetectorConfig>): void {
    this.config = { ...this.config, ...config };
    this.emit("config-updated", { config: this.config });
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a plagiarism detector with default configuration
 */
export function createPlagiarismDetector(
  config?: Partial<PlagiarismDetectorConfig>
): PlagiarismDetector {
  return new PlagiarismDetector(config);
}

// NOTE: createSimilarityAnalyzer, createASTComparator, and createSemanticComparator
// are re-exported from their respective modules above

// ============================================================================
// Default Export
// ============================================================================

export default PlagiarismDetector;
