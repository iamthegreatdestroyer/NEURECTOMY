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
  DEFAULT_NETWORK_CONFIGS,
} from "./timestamping";

export type {
  NetworkConfig,
  TimestampingConfig,
  TimestampingEvents,
} from "./timestamping";

// Fingerprinting (@CIPHER @APEX)
export {
  DEFAULT_FINGERPRINTING_CONFIG,
  detectLanguage,
  generateCodeFingerprint,
  compareFingerprints,
  batchFingerprint,
} from "./fingerprinting";

export type { SupportedLanguage, FingerprintingConfig } from "./fingerprinting";

// Signatures (@CIPHER @FORTRESS)
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
} from "./signatures";

export type {
  KeyDerivationParams,
  KeyStore,
  SignatureEvents,
  SignatureServiceConfig,
  MultiSigRequirement,
  MultiSignature,
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
} from "./evidence-vault";

export type {
  EvidenceMetadata,
  EvidenceStorageOptions,
  StoredEvidence,
  EvidenceAccessEntry,
  VaultStatistics,
  IStorageBackend,
} from "./evidence-vault";
