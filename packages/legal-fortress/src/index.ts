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
// Type Definitions (re-export everything from types.ts)
// ============================================================================

export * from "./types";

// ============================================================================
// Blockchain Module (@CRYPTO @CIPHER)
// ============================================================================

// Timestamping
export {
  computeHash,
  createFingerprint,
  MerkleTreeBuilder,
  BlockchainAnchorService,
  TimestampingService,
  DEFAULT_NETWORK_CONFIGS,
} from "./blockchain";

export type {
  NetworkConfig,
  TimestampingConfig,
  TimestampingEvents,
} from "./blockchain";

// Fingerprinting
export {
  DEFAULT_FINGERPRINTING_CONFIG,
  detectLanguage,
  generateCodeFingerprint,
  compareFingerprints,
  batchFingerprint,
} from "./blockchain";

export type {
  SupportedLanguage as FingerprintingSupportedLanguage,
  FingerprintingConfig,
} from "./blockchain";

// Digital Signatures
export {
  KeyPairGenerator,
  InMemoryKeyStore,
  DigitalSignatureService,
  MultiSignatureCoordinator,
  DEFAULT_KEY_DERIVATION,
  generateSalt,
  deriveKey,
  encryptWithKey,
  decryptWithKey,
} from "./blockchain";

export type {
  KeyDerivationParams,
  KeyStore,
  SignatureEvents,
  SignatureServiceConfig,
  MultiSigRequirement,
  MultiSignature,
} from "./blockchain";

// Provenance
export {
  createProvenanceEvent,
  computeEventHash,
  ProvenanceChainManager,
  ProvenanceStreamProcessor,
  DEFAULT_PROVENANCE_CONFIG,
} from "./blockchain";

export type {
  ProvenanceActor,
  ProvenanceConfig,
  ProvenanceEvents,
} from "./blockchain";

// Evidence Vault
export {
  EvidenceEncryptionService,
  InMemoryStorageBackend,
  MultiRegionStorageBackend,
  EvidenceVault,
} from "./blockchain";

export type {
  EvidenceMetadata,
  EvidenceStorageOptions,
  StoredEvidence,
  EvidenceAccessEntry,
  VaultStatistics,
  IStorageBackend,
} from "./blockchain";

// ============================================================================
// License Module (@AEGIS @LINGUA)
// ============================================================================

// License Detection
export {
  LICENSE_DATABASE,
  LicenseDetectionEngine,
  BatchLicenseDetector,
} from "./license";

export type {
  LicenseDefinition,
  DetectionResult,
  DetectionOptions,
} from "./license";

// SBOM Generation
export { SBOMGenerator, SBOMValidator, DEFAULT_SBOM_OPTIONS } from "./license";

export type {
  PackageInfo,
  SBOMOptions,
  SPDXDocument,
  SPDXPackage,
  SPDXRelationship,
  CycloneDXDocument,
  CycloneDXComponent,
  ValidationResult,
} from "./license";

// Compatibility Checking
export {
  COMPATIBILITY_MATRIX,
  parseSPDXExpression,
  evaluateSPDXExpression,
  LicenseCompatibilityChecker,
} from "./license";

export type {
  CompatibilityLevel,
  CompatibilityResult,
  ProjectLicenseAnalysis,
  LicenseIssue,
  SPDXExpressionNode,
} from "./license";

// ============================================================================
// Plagiarism Module (@PHANTOM @TENSOR @LINGUA @CORE)
// ============================================================================

// Unified Detector
export {
  PlagiarismDetector,
  createPlagiarismDetector,
  createASTComparator,
} from "./plagiarism";

export type {
  PlagiarismDetectorConfig,
  PlagiarismResult,
  PlagiarismMatch,
} from "./plagiarism";

// Similarity Analysis
export {
  SimilarityAnalyzer,
  MinHashGenerator,
  tokenize,
  generateNgrams,
  hashNgrams,
  winnow,
} from "./plagiarism";

export type {
  SimilarityOptions,
  Token,
  CodeFingerprint as SimilarityFingerprint,
  SimilarityResult,
  MatchedRegion,
} from "./plagiarism";

// AST Comparison
export {
  ASTComparator,
  NormalizedNodeType,
  MatchType,
  TransformationType,
} from "./plagiarism";

export type {
  SupportedLanguage as ASTSupportedLanguage,
  NormalizedASTNode,
  NodeMetadata,
  SourceLocation,
  ASTComparisonResult,
  SubtreeMatch,
  StructuralDifference,
  ComparisonStatistics,
  DetailedAnalysis,
  ASTComparatorConfig,
} from "./plagiarism";

// Semantic Analysis
export {
  SemanticComparator,
  EmbeddingGenerator,
  CodeTokenizer,
  IntentRecognizer,
  CrossLanguageAnalyzer,
} from "./plagiarism";

export type {
  EmbeddingModelConfig,
  CodeEmbedding,
  SemanticSimilarityResult,
  CodeIntent,
  SemanticMatch,
  CodeRegion as SemanticCodeRegion,
  SemanticAnalysisConfig,
} from "./plagiarism";

// ============================================================================
// Default Export - Plagiarism Detector
// ============================================================================

export { default } from "./plagiarism";
