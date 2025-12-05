/**
 * @fileoverview Blockchain Module - IP Protection Core
 * @module @neurectomy/legal-fortress/blockchain
 *
 * Comprehensive blockchain-based intellectual property protection:
 * - Timestamping with Merkle tree aggregation
 * - Code fingerprinting and comparison
 * - Digital signatures and key management
 * - Provenance tracking with immutable chains
 * - Evidence vault with tamper-proof storage
 */

// Timestamping (@CRYPTO @CIPHER)
export {
  computeHash,
  createFingerprint,
  MerkleTreeBuilder,
  BlockchainAnchorService,
  TimestampingService,
  DEFAULT_TIMESTAMPING_CONFIG,
} from "./timestamping";

export type {
  MerkleNode,
  AnchorConfig,
  TimestampingConfig,
} from "./timestamping";

// Fingerprinting (@CIPHER @APEX)
export {
  LANGUAGE_CONFIGS,
  removeComments,
  normalizeWhitespace,
  tokenize,
  generateStructureHash,
  calculateCyclomaticComplexity,
  calculateHalsteadMetrics,
  generateCodeFingerprint,
  compareFingerprints,
} from "./fingerprinting";

export type {
  LanguageConfig,
  HalsteadMetrics,
  FingerprintComparisonResult,
} from "./fingerprinting";

// Signatures (@CIPHER @FORTRESS)
export {
  KeyPairGenerator,
  InMemoryKeyStore,
  DigitalSignatureService,
  MultiSignatureCoordinator,
} from "./signatures";

export type {
  SignatureResult,
  SignatureVerificationResult,
  MultiSignatureSession,
} from "./signatures";

// Provenance (@CRYPTO @STREAM)
export {
  createProvenanceEvent,
  computeEventHash,
  ProvenanceChainManager,
  ProvenanceStreamProcessor,
  DEFAULT_PROVENANCE_CONFIG,
} from "./provenance";

export type {
  ProvenanceActor,
  ProvenanceConfig,
  ProvenanceEvents,
} from "./provenance";

// Evidence Vault (@CRYPTO @ATLAS)
export {
  EvidenceEncryptionService,
  InMemoryStorageBackend,
  MultiRegionStorageBackend,
  EvidenceVault,
  DEFAULT_STORAGE_OPTIONS,
} from "./evidence-vault";

export type {
  EvidenceMetadata,
  EvidenceStorageOptions,
  StoredEvidence,
  EvidenceAccessEntry,
  VaultStatistics,
  IStorageBackend,
} from "./evidence-vault";
