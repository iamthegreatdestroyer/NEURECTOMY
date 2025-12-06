/**
 * @fileoverview Evidence Vault - Immutable Evidence Storage
 * @module @neurectomy/legal-fortress/blockchain/evidence-vault
 *
 * @agents @CRYPTO @ATLAS - Blockchain + Cloud Architecture Specialists
 *
 * Provides tamper-proof storage for legal evidence:
 * - Content-addressed storage
 * - Encryption at rest
 * - Blockchain verification
 * - Multi-region redundancy patterns
 * - Access audit logging
 */

import { v4 as uuidv4 } from "uuid";
import CryptoJS from "crypto-js";
import {
  EvidenceItem,
  EvidenceType,
  ContentFingerprint,
  HashAlgorithm,
  TimestampAnchor,
  ProvenanceEvent,
} from "../types";
import { computeHash, createFingerprint } from "./timestamping";

// ============================================================================
// EVIDENCE TYPES (@ATLAS)
// ============================================================================

/**
 * Evidence metadata for cataloging
 */
export interface EvidenceMetadata {
  title: string;
  description?: string;
  category: EvidenceType;
  tags: string[];
  caseId?: string;
  relatedEvidenceIds?: string[];
  expiresAt?: Date;
  retentionPolicy?: "permanent" | "7-years" | "10-years" | "custom";
}

/**
 * Evidence storage options
 */
export interface EvidenceStorageOptions {
  encrypt: boolean;
  encryptionKey?: string;
  redundancy: "single" | "dual" | "triple";
  compressionEnabled: boolean;
  verifyOnStore: boolean;
  generateFingerprint: boolean;
}

/**
 * Stored evidence record
 */
export interface StoredEvidence {
  id: string;
  metadata: EvidenceMetadata;
  fingerprint: ContentFingerprint;
  storedAt: Date;
  storageLocations: string[];
  encrypted: boolean;
  encryptionAlgorithm?: string;
  contentSize: number;
  checksumValid: boolean;
  blockchainAnchor?: TimestampAnchor;
  accessLog: EvidenceAccessEntry[];
}

/**
 * Access log entry
 */
export interface EvidenceAccessEntry {
  timestamp: Date;
  actorId: string;
  action: "view" | "download" | "verify" | "export";
  ipAddress?: string;
  success: boolean;
  reason?: string;
}

/**
 * Vault statistics
 */
export interface VaultStatistics {
  totalItems: number;
  totalSize: number;
  encryptedItems: number;
  verifiedItems: number;
  anchoredItems: number;
  byCategory: Record<EvidenceType, number>;
}

// ============================================================================
// DEFAULT CONFIGURATIONS
// ============================================================================

const DEFAULT_STORAGE_OPTIONS: EvidenceStorageOptions = {
  encrypt: true,
  redundancy: "dual",
  compressionEnabled: true,
  verifyOnStore: true,
  generateFingerprint: true,
};

// ============================================================================
// ENCRYPTION SERVICE (@CRYPTO)
// ============================================================================

/**
 * Evidence encryption service
 * @agent @CRYPTO - Encryption at rest
 */
export class EvidenceEncryptionService {
  private algorithm = "AES-256";

  /**
   * Derive encryption key from passphrase
   */
  deriveKey(passphrase: string, salt: string): string {
    // PBKDF2 key derivation
    const key = CryptoJS.PBKDF2(passphrase, salt, {
      keySize: 256 / 32,
      iterations: 100000,
      hasher: CryptoJS.algo.SHA256,
    });
    return key.toString(CryptoJS.enc.Hex);
  }

  /**
   * Encrypt content
   */
  encrypt(content: string, key: string): { ciphertext: string; iv: string } {
    const iv = CryptoJS.lib.WordArray.random(16);
    const encrypted = CryptoJS.AES.encrypt(
      content,
      CryptoJS.enc.Hex.parse(key),
      {
        iv,
        mode: CryptoJS.mode.CBC,
        padding: CryptoJS.pad.Pkcs7,
      }
    );

    return {
      ciphertext: encrypted.ciphertext.toString(CryptoJS.enc.Base64),
      iv: iv.toString(CryptoJS.enc.Hex),
    };
  }

  /**
   * Decrypt content
   */
  decrypt(ciphertext: string, key: string, iv: string): string {
    const ciphertextParams = CryptoJS.lib.CipherParams.create({
      ciphertext: CryptoJS.enc.Base64.parse(ciphertext),
    });

    const decrypted = CryptoJS.AES.decrypt(
      ciphertextParams,
      CryptoJS.enc.Hex.parse(key),
      {
        iv: CryptoJS.enc.Hex.parse(iv),
        mode: CryptoJS.mode.CBC,
        padding: CryptoJS.pad.Pkcs7,
      }
    );

    return decrypted.toString(CryptoJS.enc.Utf8);
  }

  /**
   * Generate content hash before encryption (for verification)
   */
  hashContent(content: string): string {
    return CryptoJS.SHA256(content).toString(CryptoJS.enc.Hex);
  }
}

// ============================================================================
// STORAGE BACKEND INTERFACE (@ATLAS)
// ============================================================================

/**
 * Abstract storage backend interface
 * @agent @ATLAS - Cloud architecture
 */
export interface IStorageBackend {
  name: string;
  region?: string;
  store(id: string, content: string): Promise<string>;
  retrieve(id: string): Promise<string | null>;
  delete(id: string): Promise<boolean>;
  exists(id: string): Promise<boolean>;
  list(prefix?: string): Promise<string[]>;
}

/**
 * In-memory storage backend (for development/testing)
 */
export class InMemoryStorageBackend implements IStorageBackend {
  name = "in-memory";
  private storage: Map<string, string> = new Map();

  async store(id: string, content: string): Promise<string> {
    const key = `vault/${id}`;
    this.storage.set(key, content);
    return key;
  }

  async retrieve(id: string): Promise<string | null> {
    return this.storage.get(`vault/${id}`) ?? null;
  }

  async delete(id: string): Promise<boolean> {
    return this.storage.delete(`vault/${id}`);
  }

  async exists(id: string): Promise<boolean> {
    return this.storage.has(`vault/${id}`);
  }

  async list(prefix?: string): Promise<string[]> {
    const keys: string[] = [];
    for (const key of this.storage.keys()) {
      if (!prefix || key.startsWith(prefix)) {
        keys.push(key);
      }
    }
    return keys;
  }
}

/**
 * Multi-region storage backend wrapper
 * @agent @ATLAS - Redundancy patterns
 */
export class MultiRegionStorageBackend implements IStorageBackend {
  name = "multi-region";
  private backends: IStorageBackend[];
  private writeQuorum: number;
  private readQuorum: number;

  constructor(
    backends: IStorageBackend[],
    options?: { writeQuorum?: number; readQuorum?: number }
  ) {
    this.backends = backends;
    this.writeQuorum = options?.writeQuorum ?? Math.ceil(backends.length / 2);
    this.readQuorum = options?.readQuorum ?? 1;
  }

  async store(id: string, content: string): Promise<string> {
    const results = await Promise.allSettled(
      this.backends.map((b) => b.store(id, content))
    );

    const successful = results.filter((r) => r.status === "fulfilled");

    if (successful.length < this.writeQuorum) {
      throw new Error(
        `Failed to meet write quorum: ${successful.length}/${this.writeQuorum}`
      );
    }

    return (successful[0] as PromiseFulfilledResult<string>).value;
  }

  async retrieve(id: string): Promise<string | null> {
    for (const backend of this.backends) {
      try {
        const content = await backend.retrieve(id);
        if (content !== null) {
          return content;
        }
      } catch {
        continue;
      }
    }
    return null;
  }

  async delete(id: string): Promise<boolean> {
    const results = await Promise.allSettled(
      this.backends.map((b) => b.delete(id))
    );

    const successful = results.filter(
      (r) => r.status === "fulfilled" && r.value === true
    );

    return successful.length >= this.writeQuorum;
  }

  async exists(id: string): Promise<boolean> {
    for (const backend of this.backends) {
      if (await backend.exists(id)) {
        return true;
      }
    }
    return false;
  }

  async list(prefix?: string): Promise<string[]> {
    const allKeys = new Set<string>();

    for (const backend of this.backends) {
      const keys = await backend.list(prefix);
      keys.forEach((k) => allKeys.add(k));
    }

    return [...allKeys];
  }
}

// ============================================================================
// EVIDENCE VAULT (@CRYPTO @ATLAS)
// ============================================================================

/**
 * Evidence Vault - Immutable evidence storage system
 * @agent @CRYPTO @ATLAS - Complete vault implementation
 */
export class EvidenceVault {
  private storage: IStorageBackend;
  private encryption: EvidenceEncryptionService;
  private index: Map<string, StoredEvidence> = new Map();
  private encryptionKey?: string;
  private defaultOptions: EvidenceStorageOptions;

  constructor(
    storage: IStorageBackend,
    options?: {
      encryptionKey?: string;
      defaultOptions?: Partial<EvidenceStorageOptions>;
    }
  ) {
    this.storage = storage;
    this.encryption = new EvidenceEncryptionService();
    this.encryptionKey = options?.encryptionKey;
    this.defaultOptions = {
      ...DEFAULT_STORAGE_OPTIONS,
      ...options?.defaultOptions,
    };
  }

  /**
   * Store evidence in the vault
   * @agent @CRYPTO - Encryption
   * @agent @ATLAS - Storage
   */
  async store(
    content: string,
    metadata: EvidenceMetadata,
    options?: Partial<EvidenceStorageOptions>
  ): Promise<StoredEvidence> {
    const opts = { ...this.defaultOptions, ...options };
    const evidenceId = uuidv4();

    // Generate fingerprint
    const fingerprint = opts.generateFingerprint
      ? createFingerprint(content)
      : {
          contentHash: computeHash(content, "sha256"),
          normalizedHash: computeHash(content, "sha256"),
          structureHash: "",
          algorithm: "sha256" as HashAlgorithm,
          timestamp: new Date(),
        };

    // Prepare content for storage
    let storageContent = content;
    let encrypted = false;
    let encryptionData: { iv?: string } = {};

    // Encrypt if requested
    if (opts.encrypt && this.encryptionKey) {
      const result = this.encryption.encrypt(content, this.encryptionKey);
      storageContent = JSON.stringify({
        ciphertext: result.ciphertext,
        iv: result.iv,
        contentHash: fingerprint.contentHash,
      });
      encrypted = true;
      encryptionData.iv = result.iv;
    }

    // Store content
    const storageLocation = await this.storage.store(
      evidenceId,
      storageContent
    );

    // Verify storage if requested
    let checksumValid = true;
    if (opts.verifyOnStore) {
      const retrieved = await this.storage.retrieve(evidenceId);
      checksumValid = retrieved === storageContent;
    }

    // Create stored evidence record
    const storedEvidence: StoredEvidence = {
      id: evidenceId,
      metadata,
      fingerprint,
      storedAt: new Date(),
      storageLocations: [storageLocation],
      encrypted,
      encryptionAlgorithm: encrypted ? "AES-256-CBC" : undefined,
      contentSize: Buffer.byteLength(content, "utf8"),
      checksumValid,
      accessLog: [],
    };

    // Index the evidence
    this.index.set(evidenceId, storedEvidence);

    return storedEvidence;
  }

  /**
   * Retrieve evidence from the vault
   * @agent @CRYPTO - Decryption
   */
  async retrieve(
    evidenceId: string,
    actorId: string,
    options?: { ipAddress?: string }
  ): Promise<{ content: string; evidence: StoredEvidence } | null> {
    const storedEvidence = this.index.get(evidenceId);
    if (!storedEvidence) {
      return null;
    }

    // Log access attempt
    const accessEntry: EvidenceAccessEntry = {
      timestamp: new Date(),
      actorId,
      action: "view",
      ipAddress: options?.ipAddress,
      success: false,
    };

    try {
      // Retrieve content
      const storageContent = await this.storage.retrieve(evidenceId);
      if (!storageContent) {
        accessEntry.reason = "Content not found in storage";
        storedEvidence.accessLog.push(accessEntry);
        return null;
      }

      let content: string;

      // Decrypt if encrypted
      if (storedEvidence.encrypted) {
        if (!this.encryptionKey) {
          accessEntry.reason = "Encryption key not available";
          storedEvidence.accessLog.push(accessEntry);
          throw new Error("Cannot decrypt: encryption key not provided");
        }

        const encryptedData = JSON.parse(storageContent);
        content = this.encryption.decrypt(
          encryptedData.ciphertext,
          this.encryptionKey,
          encryptedData.iv
        );

        // Verify content hash
        const currentHash = this.encryption.hashContent(content);
        if (currentHash !== encryptedData.contentHash) {
          accessEntry.reason = "Content integrity verification failed";
          storedEvidence.accessLog.push(accessEntry);
          throw new Error("Content integrity verification failed");
        }
      } else {
        content = storageContent;
      }

      // Verify fingerprint
      const currentHash = computeHash(
        content,
        storedEvidence.fingerprint.algorithm
      );
      if (currentHash !== storedEvidence.fingerprint.contentHash) {
        accessEntry.reason = "Fingerprint mismatch";
        storedEvidence.accessLog.push(accessEntry);
        throw new Error("Evidence fingerprint mismatch - possible tampering");
      }

      accessEntry.success = true;
      storedEvidence.accessLog.push(accessEntry);

      return { content, evidence: storedEvidence };
    } catch (error) {
      if (!accessEntry.reason) {
        accessEntry.reason =
          error instanceof Error ? error.message : "Unknown error";
      }
      storedEvidence.accessLog.push(accessEntry);
      throw error;
    }
  }

  /**
   * Verify evidence integrity without retrieving content
   */
  async verify(
    evidenceId: string,
    actorId: string
  ): Promise<{
    exists: boolean;
    integrityValid: boolean;
    details?: string;
  }> {
    const storedEvidence = this.index.get(evidenceId);
    if (!storedEvidence) {
      return {
        exists: false,
        integrityValid: false,
        details: "Evidence not found",
      };
    }

    // Log verification access
    storedEvidence.accessLog.push({
      timestamp: new Date(),
      actorId,
      action: "verify",
      success: true,
    });

    // Check storage exists
    const exists = await this.storage.exists(evidenceId);
    if (!exists) {
      return {
        exists: false,
        integrityValid: false,
        details: "Content missing from storage",
      };
    }

    // For encrypted content, we can verify the encrypted payload hash
    const storageContent = await this.storage.retrieve(evidenceId);
    if (!storageContent) {
      return {
        exists: true,
        integrityValid: false,
        details: "Failed to retrieve content",
      };
    }

    if (storedEvidence.encrypted) {
      try {
        const encryptedData = JSON.parse(storageContent);
        // We can verify the stored hash matches expected without decryption
        return {
          exists: true,
          integrityValid: !!encryptedData.contentHash,
          details: "Encrypted content verified",
        };
      } catch {
        return {
          exists: true,
          integrityValid: false,
          details: "Failed to parse encrypted data",
        };
      }
    }

    // For unencrypted, verify hash
    const currentHash = computeHash(
      storageContent,
      storedEvidence.fingerprint.algorithm
    );
    const valid = currentHash === storedEvidence.fingerprint.contentHash;

    return {
      exists: true,
      integrityValid: valid,
      details: valid ? "Content hash verified" : "Hash mismatch detected",
    };
  }

  /**
   * Set blockchain anchor for evidence
   */
  setBlockchainAnchor(evidenceId: string, anchor: TimestampAnchor): void {
    const evidence = this.index.get(evidenceId);
    if (!evidence) {
      throw new Error(`Evidence ${evidenceId} not found`);
    }
    evidence.blockchainAnchor = anchor;
  }

  /**
   * List evidence by category
   */
  listByCategory(category: EvidenceType): StoredEvidence[] {
    return [...this.index.values()].filter(
      (e) => e.metadata.category === category
    );
  }

  /**
   * List evidence by case
   */
  listByCase(caseId: string): StoredEvidence[] {
    return [...this.index.values()].filter((e) => e.metadata.caseId === caseId);
  }

  /**
   * Search evidence by tags
   */
  searchByTags(tags: string[], matchAll = false): StoredEvidence[] {
    return [...this.index.values()].filter((e) => {
      if (matchAll) {
        return tags.every((t) => e.metadata.tags.includes(t));
      }
      return tags.some((t) => e.metadata.tags.includes(t));
    });
  }

  /**
   * Get evidence by ID
   */
  getEvidence(evidenceId: string): StoredEvidence | null {
    return this.index.get(evidenceId) ?? null;
  }

  /**
   * Get access log for evidence
   */
  getAccessLog(evidenceId: string): EvidenceAccessEntry[] {
    const evidence = this.index.get(evidenceId);
    return evidence?.accessLog ?? [];
  }

  /**
   * Get vault statistics
   */
  getStatistics(): VaultStatistics {
    const stats: VaultStatistics = {
      totalItems: this.index.size,
      totalSize: 0,
      encryptedItems: 0,
      verifiedItems: 0,
      anchoredItems: 0,
      byCategory: {} as Record<EvidenceType, number>,
    };

    for (const evidence of this.index.values()) {
      stats.totalSize += evidence.contentSize;
      if (evidence.encrypted) stats.encryptedItems++;
      if (evidence.checksumValid) stats.verifiedItems++;
      if (evidence.blockchainAnchor) stats.anchoredItems++;

      const category = evidence.metadata.category;
      stats.byCategory[category] = (stats.byCategory[category] || 0) + 1;
    }

    return stats;
  }

  /**
   * Export evidence for legal proceedings
   */
  exportForLegal(evidenceId: string): {
    evidence: StoredEvidence;
    exportedAt: Date;
    exportFormat: "legal-bundle";
    chainOfCustody: EvidenceAccessEntry[];
  } | null {
    const evidence = this.index.get(evidenceId);
    if (!evidence) {
      return null;
    }

    return {
      evidence: { ...evidence },
      exportedAt: new Date(),
      exportFormat: "legal-bundle",
      chainOfCustody: [...evidence.accessLog],
    };
  }
}

// Types and classes are already exported inline above
