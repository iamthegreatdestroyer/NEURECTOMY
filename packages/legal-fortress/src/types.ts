/**
 * @fileoverview Legal Fortress Type Definitions
 * @module @neurectomy/legal-fortress/types
 *
 * @agents @CRYPTO @CIPHER - Blockchain + Cryptography Specialists
 *
 * Comprehensive type system for IP protection, license compliance,
 * and plagiarism detection systems.
 */

import { z } from "zod";

// ============================================================================
// BLOCKCHAIN TIMESTAMPING TYPES (@CRYPTO @CIPHER)
// ============================================================================

/**
 * Supported blockchain networks for timestamping
 */
export const BlockchainNetworkSchema = z.enum([
  "ethereum_mainnet",
  "ethereum_sepolia",
  "polygon_mainnet",
  "polygon_mumbai",
  "arbitrum_mainnet",
  "optimism_mainnet",
  "base_mainnet",
]);
export type BlockchainNetwork = z.infer<typeof BlockchainNetworkSchema>;

/**
 * Hash algorithms for content fingerprinting
 */
export const HashAlgorithmSchema = z.enum([
  "sha256",
  "sha384",
  "sha512",
  "sha3-256",
  "sha3-512",
  "blake3",
  "keccak256", // Ethereum native
]);
export type HashAlgorithm = z.infer<typeof HashAlgorithmSchema>;

/**
 * Content fingerprint with cryptographic hash
 */
export const ContentFingerprintSchema = z.object({
  hash: z.string().min(64).max(128),
  algorithm: HashAlgorithmSchema,
  contentSize: z.number().positive(),
  contentType: z.string(),
  metadata: z.record(z.string()).optional(),
  createdAt: z.date(),
});
export type ContentFingerprint = z.infer<typeof ContentFingerprintSchema>;

/**
 * Merkle tree node for batch timestamping
 */
export const MerkleNodeSchema = z.object({
  hash: z.string(),
  position: z.enum(["left", "right"]),
  level: z.number().nonnegative(),
});
export type MerkleNode = z.infer<typeof MerkleNodeSchema>;

/**
 * Merkle proof for timestamp verification
 */
export const MerkleProofSchema = z.object({
  leaf: z.string(),
  root: z.string(),
  proof: z.array(MerkleNodeSchema),
  leafIndex: z.number().nonnegative(),
  totalLeaves: z.number().positive(),
});
export type MerkleProof = z.infer<typeof MerkleProofSchema>;

/**
 * Blockchain timestamp anchor
 */
export const TimestampAnchorSchema = z.object({
  id: z.string().uuid(),
  merkleRoot: z.string(),
  transactionHash: z.string(),
  blockNumber: z.number().positive(),
  blockTimestamp: z.date(),
  network: BlockchainNetworkSchema,
  contractAddress: z.string().optional(),
  gasUsed: z.number().optional(),
  confirmations: z.number().nonnegative(),
  status: z.enum(["pending", "confirmed", "failed"]),
  createdAt: z.date(),
});
export type TimestampAnchor = z.infer<typeof TimestampAnchorSchema>;

/**
 * Timestamped content record
 */
export const TimestampedContentSchema = z.object({
  id: z.string().uuid(),
  fingerprint: ContentFingerprintSchema,
  merkleProof: MerkleProofSchema.optional(),
  anchor: TimestampAnchorSchema.optional(),
  status: z.enum(["pending", "anchored", "verified", "failed"]),
  verificationCount: z.number().nonnegative().default(0),
  lastVerifiedAt: z.date().optional(),
  createdAt: z.date(),
  updatedAt: z.date(),
});
export type TimestampedContent = z.infer<typeof TimestampedContentSchema>;

// ============================================================================
// DIGITAL SIGNATURE TYPES (@CIPHER @FORTRESS)
// ============================================================================

/**
 * Signature algorithms supported
 */
export const SignatureAlgorithmSchema = z.enum([
  "ed25519",
  "ecdsa_secp256k1", // Ethereum compatible
  "ecdsa_p256",
  "rsa_pss_sha256",
  "rsa_pkcs1_sha256",
]);
export type SignatureAlgorithm = z.infer<typeof SignatureAlgorithmSchema>;

/**
 * Key pair for digital signatures
 */
export const KeyPairSchema = z.object({
  id: z.string().uuid(),
  algorithm: SignatureAlgorithmSchema,
  publicKey: z.string(),
  encryptedPrivateKey: z.string().optional(), // Encrypted at rest
  keyDerivationParams: z
    .object({
      algorithm: z.string(),
      salt: z.string(),
      iterations: z.number().positive(),
    })
    .optional(),
  createdAt: z.date(),
  expiresAt: z.date().optional(),
  revokedAt: z.date().optional(),
  metadata: z.record(z.string()).optional(),
});
export type KeyPair = z.infer<typeof KeyPairSchema>;

/**
 * Digital signature
 */
export const DigitalSignatureSchema = z.object({
  id: z.string().uuid(),
  contentHash: z.string(),
  signature: z.string(),
  algorithm: SignatureAlgorithmSchema,
  publicKeyId: z.string().uuid(),
  signedAt: z.date(),
  expiresAt: z.date().optional(),
  metadata: z.record(z.string()).optional(),
});
export type DigitalSignature = z.infer<typeof DigitalSignatureSchema>;

// ============================================================================
// CODE FINGERPRINTING TYPES (@CIPHER @APEX)
// ============================================================================

/**
 * Code fingerprint components
 */
export const CodeFingerprintSchema = z.object({
  id: z.string().uuid(),
  filePath: z.string(),
  contentHash: z.string(),
  structureHash: z.string(), // AST-based
  semanticHash: z.string().optional(), // ML-based embedding
  normalizedHash: z.string(), // Whitespace/comment normalized
  language: z.string(),
  lineCount: z.number().positive(),
  charCount: z.number().positive(),
  complexity: z
    .object({
      cyclomatic: z.number().nonnegative(),
      cognitive: z.number().nonnegative().optional(),
      halstead: z
        .object({
          vocabulary: z.number(),
          length: z.number(),
          difficulty: z.number(),
          effort: z.number(),
        })
        .optional(),
    })
    .optional(),
  tokens: z.array(z.string()).optional(),
  createdAt: z.date(),
  updatedAt: z.date(),
});
export type CodeFingerprint = z.infer<typeof CodeFingerprintSchema>;

// ============================================================================
// PROVENANCE TRACKING TYPES (@CRYPTO @STREAM)
// ============================================================================

/**
 * Provenance event types
 */
export const ProvenanceEventTypeSchema = z.enum([
  "created",
  "modified",
  "renamed",
  "moved",
  "deleted",
  "restored",
  "forked",
  "merged",
  "signed",
  "timestamped",
  "licensed",
  "transferred",
  "reviewed",
  "approved",
]);
export type ProvenanceEventType = z.infer<typeof ProvenanceEventTypeSchema>;

/**
 * Provenance event record
 */
export const ProvenanceEventSchema = z.object({
  id: z.string().uuid(),
  contentId: z.string().uuid(),
  eventType: ProvenanceEventTypeSchema,
  previousHash: z.string().optional(),
  currentHash: z.string(),
  actor: z.object({
    id: z.string(),
    type: z.enum(["user", "system", "agent"]),
    name: z.string().optional(),
  }),
  metadata: z.record(z.unknown()).optional(),
  signature: z.string().optional(),
  timestamp: z.date(),
  blockchainAnchor: z.string().optional(),
});
export type ProvenanceEvent = z.infer<typeof ProvenanceEventSchema>;

/**
 * Full provenance chain for content
 */
export const ProvenanceChainSchema = z.object({
  contentId: z.string().uuid(),
  events: z.array(ProvenanceEventSchema),
  rootHash: z.string(),
  latestHash: z.string(),
  chainIntegrity: z.enum(["valid", "broken", "unverified"]),
  verifiedAt: z.date().optional(),
});
export type ProvenanceChain = z.infer<typeof ProvenanceChainSchema>;

// ============================================================================
// LICENSE DETECTION TYPES (@AEGIS @LINGUA)
// ============================================================================

/**
 * SPDX license identifiers and categories
 */
export const LicenseCategorySchema = z.enum([
  "permissive",
  "copyleft_weak",
  "copyleft_strong",
  "proprietary",
  "public_domain",
  "unknown",
  "custom",
]);
export type LicenseCategory = z.infer<typeof LicenseCategorySchema>;

/**
 * Detected license information
 */
export const DetectedLicenseSchema = z.object({
  id: z.string().uuid(),
  spdxId: z.string().optional(),
  name: z.string(),
  category: LicenseCategorySchema,
  confidence: z.number().min(0).max(1),
  source: z.enum(["file", "package_json", "readme", "header", "inferred"]),
  filePath: z.string().optional(),
  startLine: z.number().optional(),
  endLine: z.number().optional(),
  text: z.string().optional(),
  url: z.string().url().optional(),
  permissions: z.array(z.string()),
  conditions: z.array(z.string()),
  limitations: z.array(z.string()),
  compatibleWith: z.array(z.string()),
  incompatibleWith: z.array(z.string()),
});
export type DetectedLicense = z.infer<typeof DetectedLicenseSchema>;

/**
 * License compatibility result
 */
export const LicenseCompatibilitySchema = z.object({
  license1: z.string(),
  license2: z.string(),
  compatible: z.boolean(),
  direction: z.enum(["both", "one_way", "none"]),
  notes: z.string().optional(),
  conditions: z.array(z.string()),
});
export type LicenseCompatibility = z.infer<typeof LicenseCompatibilitySchema>;

// ============================================================================
// SBOM (SOFTWARE BILL OF MATERIALS) TYPES (@AEGIS @FORGE)
// ============================================================================

/**
 * SBOM format standards
 */
export const SBOMFormatSchema = z.enum([
  "spdx_2_3",
  "spdx_3_0",
  "cyclonedx_1_4",
  "cyclonedx_1_5",
]);
export type SBOMFormat = z.infer<typeof SBOMFormatSchema>;

/**
 * Package reference in SBOM
 */
export const SBOMPackageSchema = z.object({
  name: z.string(),
  version: z.string(),
  purl: z.string().optional(), // Package URL
  cpe: z.string().optional(), // Common Platform Enumeration
  license: z.string().optional(),
  licenses: z.array(DetectedLicenseSchema).optional(),
  supplier: z.string().optional(),
  downloadUrl: z.string().url().optional(),
  checksum: z
    .object({
      algorithm: HashAlgorithmSchema,
      value: z.string(),
    })
    .optional(),
  dependencies: z.array(z.string()),
  vulnerabilities: z
    .array(
      z.object({
        id: z.string(),
        severity: z.enum(["critical", "high", "medium", "low", "info"]),
        description: z.string().optional(),
        fixedIn: z.string().optional(),
      })
    )
    .optional(),
});
export type SBOMPackage = z.infer<typeof SBOMPackageSchema>;

/**
 * Complete SBOM document
 */
export const SBOMDocumentSchema = z.object({
  id: z.string().uuid(),
  format: SBOMFormatSchema,
  specVersion: z.string(),
  name: z.string(),
  version: z.string(),
  createdAt: z.date(),
  creator: z.object({
    tool: z.string(),
    version: z.string(),
  }),
  packages: z.array(SBOMPackageSchema),
  relationships: z.array(
    z.object({
      source: z.string(),
      target: z.string(),
      type: z.enum(["depends_on", "dev_dependency", "optional", "peer"]),
    })
  ),
  licenseInfo: z.object({
    declaredLicenses: z.array(z.string()),
    concludedLicenses: z.array(z.string()),
    licenseConflicts: z.array(LicenseCompatibilitySchema),
  }),
  metadata: z.record(z.unknown()).optional(),
});
export type SBOMDocument = z.infer<typeof SBOMDocumentSchema>;

// ============================================================================
// PLAGIARISM DETECTION TYPES (@PHANTOM @TENSOR)
// ============================================================================

/**
 * Code similarity match
 */
export const SimilarityMatchSchema = z.object({
  id: z.string().uuid(),
  sourceFile: z.string(),
  sourceStartLine: z.number(),
  sourceEndLine: z.number(),
  targetFile: z.string(),
  targetStartLine: z.number(),
  targetEndLine: z.number(),
  targetRepository: z.string().optional(),
  similarityScore: z.number().min(0).max(1),
  matchType: z.enum([
    "exact", // Identical code
    "renamed", // Variables/functions renamed
    "reordered", // Statements reordered
    "structural", // Same AST structure
    "semantic", // Same functionality, different implementation
    "partial", // Partial overlap
  ]),
  confidence: z.number().min(0).max(1),
  snippet: z
    .object({
      source: z.string(),
      target: z.string(),
    })
    .optional(),
});
export type SimilarityMatch = z.infer<typeof SimilarityMatchSchema>;

/**
 * Plagiarism analysis report
 */
export const PlagiarismReportSchema = z.object({
  id: z.string().uuid(),
  analyzedFile: z.string(),
  analyzedHash: z.string(),
  totalLines: z.number(),
  originalLines: z.number(),
  matchedLines: z.number(),
  originalityScore: z.number().min(0).max(1),
  matches: z.array(SimilarityMatchSchema),
  analysisMethod: z.array(
    z.enum([
      "exact_hash",
      "normalized_hash",
      "ast_comparison",
      "token_sequence",
      "semantic_embedding",
      "winnowing",
    ])
  ),
  externalSources: z.array(z.string()),
  analyzedAt: z.date(),
  processingTime: z.number(), // milliseconds
});
export type PlagiarismReport = z.infer<typeof PlagiarismReportSchema>;

// ============================================================================
// COMPLIANCE TYPES (@AEGIS @SENTRY)
// ============================================================================

/**
 * Compliance standards supported
 */
export const ComplianceStandardSchema = z.enum([
  "soc2_type1",
  "soc2_type2",
  "iso_27001",
  "gdpr",
  "hipaa",
  "pci_dss",
  "fedramp",
  "ccpa",
]);
export type ComplianceStandard = z.infer<typeof ComplianceStandardSchema>;

/**
 * Policy violation severity
 */
export const ViolationSeveritySchema = z.enum([
  "critical",
  "high",
  "medium",
  "low",
  "info",
]);
export type ViolationSeverity = z.infer<typeof ViolationSeveritySchema>;

/**
 * Policy violation record
 */
export const PolicyViolationSchema = z.object({
  id: z.string().uuid(),
  policyId: z.string(),
  policyName: z.string(),
  standard: ComplianceStandardSchema.optional(),
  severity: ViolationSeveritySchema,
  resource: z.object({
    type: z.string(),
    id: z.string(),
    path: z.string().optional(),
  }),
  description: z.string(),
  remediation: z.string().optional(),
  detectedAt: z.date(),
  resolvedAt: z.date().optional(),
  status: z.enum(["open", "acknowledged", "resolved", "false_positive"]),
});
export type PolicyViolation = z.infer<typeof PolicyViolationSchema>;

/**
 * Compliance report
 */
export const ComplianceReportSchema = z.object({
  id: z.string().uuid(),
  standards: z.array(ComplianceStandardSchema),
  scope: z.object({
    repositories: z.array(z.string()),
    packages: z.array(z.string()),
    dateRange: z.object({
      start: z.date(),
      end: z.date(),
    }),
  }),
  summary: z.object({
    totalChecks: z.number(),
    passed: z.number(),
    failed: z.number(),
    warnings: z.number(),
    notApplicable: z.number(),
  }),
  violations: z.array(PolicyViolationSchema),
  recommendations: z.array(
    z.object({
      priority: ViolationSeveritySchema,
      title: z.string(),
      description: z.string(),
      effort: z.enum(["low", "medium", "high"]),
    })
  ),
  generatedAt: z.date(),
  validUntil: z.date().optional(),
});
export type ComplianceReport = z.infer<typeof ComplianceReportSchema>;

// ============================================================================
// EVIDENCE VAULT TYPES (@CRYPTO @ATLAS)
// ============================================================================

/**
 * Evidence item for legal protection
 */
export const EvidenceItemSchema = z.object({
  id: z.string().uuid(),
  type: z.enum([
    "timestamp_proof",
    "signature",
    "provenance_chain",
    "license_detection",
    "plagiarism_report",
    "compliance_report",
    "audit_log",
    "sbom",
  ]),
  contentHash: z.string(),
  encryptedContent: z.string().optional(),
  storageLocation: z.object({
    type: z.enum(["local", "s3", "azure_blob", "gcs", "ipfs"]),
    uri: z.string(),
  }),
  blockchainAnchors: z.array(TimestampAnchorSchema),
  retentionPolicy: z.object({
    type: z.enum(["indefinite", "years", "compliance_period"]),
    value: z.number().optional(),
    deletionDate: z.date().optional(),
  }),
  accessLog: z.array(
    z.object({
      actor: z.string(),
      action: z.enum(["read", "verify", "export"]),
      timestamp: z.date(),
    })
  ),
  createdAt: z.date(),
  lastAccessedAt: z.date().optional(),
});
export type EvidenceItem = z.infer<typeof EvidenceItemSchema>;

// ============================================================================
// EXPORT TYPES
// ============================================================================

export type {
  BlockchainNetwork,
  HashAlgorithm,
  ContentFingerprint,
  MerkleNode,
  MerkleProof,
  TimestampAnchor,
  TimestampedContent,
  SignatureAlgorithm,
  KeyPair,
  DigitalSignature,
  CodeFingerprint,
  ProvenanceEventType,
  ProvenanceEvent,
  ProvenanceChain,
  LicenseCategory,
  DetectedLicense,
  LicenseCompatibility,
  SBOMFormat,
  SBOMPackage,
  SBOMDocument,
  SimilarityMatch,
  PlagiarismReport,
  ComplianceStandard,
  ViolationSeverity,
  PolicyViolation,
  ComplianceReport,
  EvidenceItem,
};
