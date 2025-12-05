/**
 * @fileoverview Legal Fortress - IP Protection, License Compliance & Plagiarism Detection
 * @module @neurectomy/legal-fortress
 *
 * Agent Collective: @CRYPTO @CIPHER @AEGIS @LINGUA @PHANTOM @TENSOR @CORE @FORTRESS
 *
 * Comprehensive legal protection system for software development:
 * - Blockchain timestamping for IP protection
 * - Digital signatures and evidence vaulting
 * - License detection and compliance
 * - SBOM (Software Bill of Materials) generation
 * - Plagiarism detection across multiple methods
 *
 * @author NEURECTOMY Phase 5 - Excellence & Polish
 * @version 1.0.0
 */

// ============================================================================
// Type Definitions
// ============================================================================

export {
  // Blockchain Types
  BlockchainNetwork,
  BlockchainNetworkSchema,
  HashAlgorithm,
  HashAlgorithmSchema,
  ContentFingerprint,
  ContentFingerprintSchema,
  MerkleNode,
  MerkleNodeSchema,
  MerkleProof,
  MerkleProofSchema,
  TimestampAnchor,
  TimestampAnchorSchema,
  TimestampedContent,
  TimestampedContentSchema,

  // Signature Types
  SignatureAlgorithm,
  SignatureAlgorithmSchema,
  KeyPair,
  KeyPairSchema,
  DigitalSignature,
  DigitalSignatureSchema,
  SignedContent,
  SignedContentSchema,

  // Provenance Types
  ProvenanceEvent,
  ProvenanceEventSchema,
  ProvenanceChain,
  ProvenanceChainSchema,

  // Evidence Types
  EvidenceItem,
  EvidenceItemSchema,
  EvidenceVault,
  EvidenceVaultSchema,

  // License Types
  LicenseType,
  LicenseTypeSchema,
  LicenseInfo,
  LicenseInfoSchema,
  LicenseCompatibility,
  LicenseCompatibilitySchema,

  // SBOM Types
  SBOMFormat,
  SBOMFormatSchema,
  Dependency,
  DependencySchema,
  SBOM,
  SBOMSchema,

  // Similarity Types
  SimilarityMatch,
  SimilarityMatchSchema,
  SimilarityScore,
  SimilarityScoreSchema,
  CodeRegion,
  CodeRegionSchema,
} from "./types";

// ============================================================================
// Blockchain Module (@CRYPTO @CIPHER)
// ============================================================================

export {
  // Timestamping
  BlockchainTimestamper,
  createBlockchainTimestamper,
  type TimestamperConfig,

  // Fingerprinting
  CodeFingerprinter,
  createCodeFingerprinter,
  type FingerprintConfig,

  // Digital Signatures
  SignatureManager,
  createSignatureManager,
  type SignatureConfig,

  // Provenance
  ProvenanceTracker,
  createProvenanceTracker,
  type ProvenanceConfig,

  // Evidence Vault
  EvidenceVaultManager,
  createEvidenceVault,
  type EvidenceVaultConfig,
} from "./blockchain";

// ============================================================================
// License Module (@AEGIS @LINGUA)
// ============================================================================

export {
  // License Detection
  LicenseDetector,
  createLicenseDetector,
  type LicenseDetectorConfig,

  // SBOM Generation
  SBOMGenerator,
  createSBOMGenerator,
  type SBOMConfig,

  // Compatibility Checking
  CompatibilityChecker,
  createCompatibilityChecker,
  type CompatibilityConfig,
  COMPATIBILITY_MATRIX,
} from "./license";

// ============================================================================
// Plagiarism Module (@PHANTOM @TENSOR @LINGUA @CORE)
// ============================================================================

export {
  // Unified Detector
  PlagiarismDetector,
  createPlagiarismDetector,
  type PlagiarismDetectorConfig,
  type PlagiarismResult,
  type PlagiarismMatch,

  // Similarity Analysis
  SimilarityAnalyzer,
  WinnowingAnalyzer,
  NgramAnalyzer,
  MinHashAnalyzer,
  JaccardAnalyzer,
  type SimilarityOptions,
  type Token,
  type CodeFingerprint as SimilarityFingerprint,
  type SimilarityResult,
  type MatchedRegion,
  DEFAULT_SIMILARITY_OPTIONS,

  // AST Comparison
  ASTComparator,
  ASTNormalizer,
  TreeEditDistance,
  SubtreeMatcher,
  CloneDetector,
  NormalizedNodeType,
  type SupportedLanguage,
  type NormalizedASTNode,
  type ASTComparisonResult,
  type CloneInstance,
  type CloneClass,
  DEFAULT_AST_OPTIONS,

  // Semantic Analysis
  SemanticComparator,
  EmbeddingGenerator,
  CodeTokenizer,
  IntentRecognizer,
  CrossLanguageAnalyzer,
  type EmbeddingModelConfig,
  type CodeEmbedding,
  type SemanticSimilarityResult,
  type CodeIntent,
  type SemanticMatch,
  type CodeRegion as SemanticCodeRegion,
  type SemanticAnalysisConfig,
  DEFAULT_EMBEDDING_CONFIG,
  DEFAULT_SEMANTIC_CONFIG,
} from "./plagiarism";

// ============================================================================
// Unified Legal Fortress API
// ============================================================================

import { EventEmitter } from "events";
import { BlockchainTimestamper } from "./blockchain";
import {
  LicenseDetector,
  SBOMGenerator,
  CompatibilityChecker,
} from "./license";
import { PlagiarismDetector } from "./plagiarism";

/**
 * Legal Fortress unified configuration
 */
export interface LegalFortressConfig {
  /** Enable blockchain timestamping */
  enableTimestamping: boolean;
  /** Enable license detection */
  enableLicenseDetection: boolean;
  /** Enable plagiarism detection */
  enablePlagiarismDetection: boolean;
  /** Blockchain network for timestamping */
  blockchainNetwork?: string;
  /** Custom similarity threshold */
  similarityThreshold?: number;
}

/**
 * Legal Fortress unified analysis result
 */
export interface LegalAnalysisResult {
  /** Analysis ID */
  id: string;
  /** Analyzed path */
  path: string;
  /** Timestamp if created */
  timestamp?: {
    merkleRoot: string;
    transactionHash?: string;
    status: string;
  };
  /** Detected licenses */
  licenses?: Array<{
    file: string;
    license: string;
    confidence: number;
  }>;
  /** Plagiarism findings */
  plagiarism?: Array<{
    file: string;
    matches: number;
    maxSimilarity: number;
    verdict: string;
  }>;
  /** Overall compliance status */
  compliance: {
    status: "compliant" | "warning" | "violation";
    issues: string[];
  };
  /** Analysis metadata */
  metadata: {
    analysisTime: number;
    filesAnalyzed: number;
    timestamp: Date;
  };
}

/**
 * Default configuration
 */
const DEFAULT_LEGAL_FORTRESS_CONFIG: LegalFortressConfig = {
  enableTimestamping: true,
  enableLicenseDetection: true,
  enablePlagiarismDetection: true,
  similarityThreshold: 0.3,
};

/**
 * Legal Fortress - Unified Legal Protection System
 *
 * Provides comprehensive legal protection for software projects:
 * - IP protection via blockchain timestamping
 * - License compliance via detection and SBOM
 * - Plagiarism detection via multi-method analysis
 */
export class LegalFortress extends EventEmitter {
  private config: LegalFortressConfig;
  private timestamper: BlockchainTimestamper;
  private licenseDetector: LicenseDetector;
  private sbomGenerator: SBOMGenerator;
  private compatibilityChecker: CompatibilityChecker;
  private plagiarismDetector: PlagiarismDetector;

  constructor(config: Partial<LegalFortressConfig> = {}) {
    super();
    this.config = { ...DEFAULT_LEGAL_FORTRESS_CONFIG, ...config };
    this.timestamper = new BlockchainTimestamper();
    this.licenseDetector = new LicenseDetector();
    this.sbomGenerator = new SBOMGenerator();
    this.compatibilityChecker = new CompatibilityChecker();
    this.plagiarismDetector = new PlagiarismDetector({
      threshold: this.config.similarityThreshold,
    });
  }

  /**
   * Analyze a project for legal compliance
   */
  async analyzeProject(
    projectPath: string,
    files: Array<{
      path: string;
      content: string;
      language: string;
    }>
  ): Promise<LegalAnalysisResult> {
    const startTime = Date.now();
    const resultId = this.generateId();

    this.emit("analysis-started", { projectPath, fileCount: files.length });

    const issues: string[] = [];
    let timestampResult: LegalAnalysisResult["timestamp"] | undefined;
    let licenseResults: LegalAnalysisResult["licenses"] | undefined;
    let plagiarismResults: LegalAnalysisResult["plagiarism"] | undefined;

    // Timestamp project
    if (this.config.enableTimestamping) {
      try {
        const allContent = files.map((f) => f.content).join("\n");
        const timestamp = await this.timestamper.timestamp(allContent);
        timestampResult = {
          merkleRoot: timestamp.merkleRoot,
          transactionHash: timestamp.transactionHash,
          status: timestamp.status,
        };
        this.emit("timestamp-complete", { timestamp: timestampResult });
      } catch (error) {
        issues.push(`Timestamping failed: ${(error as Error).message}`);
      }
    }

    // Detect licenses
    if (this.config.enableLicenseDetection) {
      try {
        licenseResults = [];
        for (const file of files) {
          const detection = await this.licenseDetector.detect(file.content);
          if (detection.license !== "unknown") {
            licenseResults.push({
              file: file.path,
              license: detection.license,
              confidence: detection.confidence,
            });
          }
        }

        // Check compatibility
        const licenses = licenseResults.map((l) => l.license);
        const compatibility =
          await this.compatibilityChecker.checkAll(licenses);
        if (!compatibility.compatible) {
          issues.push(
            `License incompatibility detected: ${compatibility.conflicts.join(", ")}`
          );
        }

        this.emit("license-detection-complete", {
          licenses: licenseResults.length,
        });
      } catch (error) {
        issues.push(`License detection failed: ${(error as Error).message}`);
      }
    }

    // Detect plagiarism
    if (this.config.enablePlagiarismDetection) {
      try {
        const plagiarismFindings =
          await this.plagiarismDetector.batchDetect(files);

        plagiarismResults = plagiarismFindings.map((finding) => ({
          file: finding.sourceFile,
          matches: finding.matches.length,
          maxSimilarity: finding.overallScore,
          verdict: finding.verdict,
        }));

        // Add issues for suspicious findings
        for (const finding of plagiarismFindings) {
          if (
            finding.verdict === "likely_plagiarism" ||
            finding.verdict === "definite_plagiarism"
          ) {
            issues.push(
              `Potential plagiarism: ${finding.sourceFile} ↔ ${finding.targetFile} (${(finding.overallScore * 100).toFixed(1)}%)`
            );
          }
        }

        this.emit("plagiarism-detection-complete", {
          findings: plagiarismResults.length,
        });
      } catch (error) {
        issues.push(`Plagiarism detection failed: ${(error as Error).message}`);
      }
    }

    // Determine compliance status
    const status = this.determineComplianceStatus(issues);

    const result: LegalAnalysisResult = {
      id: resultId,
      path: projectPath,
      timestamp: timestampResult,
      licenses: licenseResults,
      plagiarism: plagiarismResults,
      compliance: {
        status,
        issues,
      },
      metadata: {
        analysisTime: Date.now() - startTime,
        filesAnalyzed: files.length,
        timestamp: new Date(),
      },
    };

    this.emit("analysis-complete", { result });
    return result;
  }

  /**
   * Create timestamp for single content
   */
  async timestampContent(content: string): Promise<{
    merkleRoot: string;
    fingerprint: string;
    status: string;
  }> {
    return this.timestamper.timestamp(content);
  }

  /**
   * Detect license in content
   */
  async detectLicense(content: string): Promise<{
    license: string;
    confidence: number;
    spdxId?: string;
  }> {
    return this.licenseDetector.detect(content);
  }

  /**
   * Generate SBOM for project
   */
  async generateSBOM(
    projectPath: string,
    format: "spdx" | "cyclonedx" = "spdx"
  ): Promise<string> {
    return this.sbomGenerator.generate(projectPath, format);
  }

  /**
   * Check code for plagiarism
   */
  async checkPlagiarism(
    sourceCode: string,
    sourceLanguage: string,
    targetCode: string,
    targetLanguage: string
  ): Promise<{
    score: number;
    verdict: string;
    matches: number;
  }> {
    const result = await this.plagiarismDetector.detect(
      "source",
      sourceCode,
      sourceLanguage,
      "target",
      targetCode,
      targetLanguage
    );

    return {
      score: result.overallScore,
      verdict: result.verdict,
      matches: result.matches.length,
    };
  }

  /**
   * Generate unique ID
   */
  private generateId(): string {
    return `lf_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
  }

  /**
   * Determine compliance status based on issues
   */
  private determineComplianceStatus(
    issues: string[]
  ): "compliant" | "warning" | "violation" {
    const hasViolation = issues.some(
      (i) =>
        i.includes("plagiarism") ||
        i.includes("incompatibility") ||
        i.includes("definite")
    );
    const hasWarning = issues.length > 0;

    if (hasViolation) return "violation";
    if (hasWarning) return "warning";
    return "compliant";
  }

  /**
   * Get component instances for advanced usage
   */
  getComponents(): {
    timestamper: BlockchainTimestamper;
    licenseDetector: LicenseDetector;
    sbomGenerator: SBOMGenerator;
    compatibilityChecker: CompatibilityChecker;
    plagiarismDetector: PlagiarismDetector;
  } {
    return {
      timestamper: this.timestamper,
      licenseDetector: this.licenseDetector,
      sbomGenerator: this.sbomGenerator,
      compatibilityChecker: this.compatibilityChecker,
      plagiarismDetector: this.plagiarismDetector,
    };
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a Legal Fortress instance with default configuration
 */
export function createLegalFortress(
  config?: Partial<LegalFortressConfig>
): LegalFortress {
  return new LegalFortress(config);
}

// ============================================================================
// Default Export
// ============================================================================

export default LegalFortress;
