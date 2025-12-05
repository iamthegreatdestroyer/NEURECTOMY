/**
 * @fileoverview Digital Signature Infrastructure
 * @module @neurectomy/legal-fortress/blockchain/signatures
 *
 * @agents @CIPHER @FORTRESS - Cryptography + Security Specialists
 *
 * Provides digital signature capabilities for:
 * - Code signing and verification
 * - Document authentication
 * - Key management and rotation
 * - Multi-signature support
 */

import CryptoJS from "crypto-js";
import { ethers } from "ethers";
import { v4 as uuidv4 } from "uuid";
import { EventEmitter } from "eventemitter3";
import {
  SignatureAlgorithm,
  KeyPair,
  DigitalSignature,
  HashAlgorithm,
} from "../types";
import { computeHash } from "./timestamping";

// ============================================================================
// KEY DERIVATION (@CIPHER)
// ============================================================================

/**
 * Key derivation parameters
 */
export interface KeyDerivationParams {
  algorithm: "pbkdf2" | "scrypt" | "argon2id";
  salt: string;
  iterations: number;
  keyLength?: number;
}

/**
 * Default key derivation parameters
 * @agent @CIPHER - Secure defaults
 */
export const DEFAULT_KEY_DERIVATION: KeyDerivationParams = {
  algorithm: "pbkdf2",
  salt: "", // Generated per key
  iterations: 310000, // OWASP recommendation for PBKDF2-SHA256
  keyLength: 32,
};

/**
 * Generate cryptographic salt
 */
export function generateSalt(length: number = 32): string {
  const array = new Uint8Array(length);
  crypto.getRandomValues(array);
  return Array.from(array, (byte) => byte.toString(16).padStart(2, "0")).join(
    ""
  );
}

/**
 * Derive key from password using PBKDF2
 * @agent @CIPHER - Key derivation
 */
export function deriveKey(
  password: string,
  params: KeyDerivationParams
): string {
  if (params.algorithm !== "pbkdf2") {
    throw new Error(
      `Key derivation algorithm ${params.algorithm} not yet implemented`
    );
  }

  const key = CryptoJS.PBKDF2(password, params.salt, {
    keySize: (params.keyLength ?? 32) / 4, // Convert bytes to words
    iterations: params.iterations,
    hasher: CryptoJS.algo.SHA256,
  });

  return key.toString(CryptoJS.enc.Hex);
}

/**
 * Encrypt data with derived key
 */
export function encryptWithKey(data: string, key: string): string {
  const iv = CryptoJS.lib.WordArray.random(16);
  const keyWordArray = CryptoJS.enc.Hex.parse(key);

  const encrypted = CryptoJS.AES.encrypt(data, keyWordArray, {
    iv,
    mode: CryptoJS.mode.GCM,
    padding: CryptoJS.pad.Pkcs7,
  });

  // Return IV + ciphertext
  return iv.toString(CryptoJS.enc.Hex) + ":" + encrypted.toString();
}

/**
 * Decrypt data with derived key
 */
export function decryptWithKey(encryptedData: string, key: string): string {
  const [ivHex, ciphertext] = encryptedData.split(":");
  if (!ivHex || !ciphertext) {
    throw new Error("Invalid encrypted data format");
  }

  const iv = CryptoJS.enc.Hex.parse(ivHex);
  const keyWordArray = CryptoJS.enc.Hex.parse(key);

  const decrypted = CryptoJS.AES.decrypt(ciphertext, keyWordArray, {
    iv,
    mode: CryptoJS.mode.GCM,
    padding: CryptoJS.pad.Pkcs7,
  });

  return decrypted.toString(CryptoJS.enc.Utf8);
}

// ============================================================================
// KEY PAIR MANAGEMENT (@CIPHER @FORTRESS)
// ============================================================================

/**
 * Key pair generator for different algorithms
 * @agent @CIPHER @FORTRESS - Key generation
 */
export class KeyPairGenerator {
  /**
   * Generate ECDSA secp256k1 key pair (Ethereum compatible)
   */
  static generateSecp256k1(): { publicKey: string; privateKey: string } {
    const wallet = ethers.Wallet.createRandom();
    return {
      publicKey: wallet.publicKey,
      privateKey: wallet.privateKey,
    };
  }

  /**
   * Generate key pair for specified algorithm
   */
  static generate(algorithm: SignatureAlgorithm): {
    publicKey: string;
    privateKey: string;
  } {
    switch (algorithm) {
      case "ecdsa_secp256k1":
        return this.generateSecp256k1();

      case "ed25519":
      case "ecdsa_p256":
      case "rsa_pss_sha256":
      case "rsa_pkcs1_sha256":
        // These would require additional libraries (tweetnacl, etc.)
        throw new Error(
          `Algorithm ${algorithm} not yet implemented. Using ecdsa_secp256k1 is recommended.`
        );

      default:
        throw new Error(`Unsupported signature algorithm: ${algorithm}`);
    }
  }
}

/**
 * Key pair store interface
 */
export interface KeyStore {
  save(keyPair: KeyPair): Promise<void>;
  get(id: string): Promise<KeyPair | null>;
  list(): Promise<KeyPair[]>;
  delete(id: string): Promise<void>;
  rotate(id: string): Promise<KeyPair>;
}

/**
 * In-memory key store (for development/testing)
 * @agent @FORTRESS - Secure storage
 */
export class InMemoryKeyStore implements KeyStore {
  private keys: Map<string, KeyPair> = new Map();

  async save(keyPair: KeyPair): Promise<void> {
    this.keys.set(keyPair.id, keyPair);
  }

  async get(id: string): Promise<KeyPair | null> {
    return this.keys.get(id) ?? null;
  }

  async list(): Promise<KeyPair[]> {
    return Array.from(this.keys.values()).filter((k) => !k.revokedAt);
  }

  async delete(id: string): Promise<void> {
    this.keys.delete(id);
  }

  async rotate(id: string): Promise<KeyPair> {
    const existing = await this.get(id);
    if (!existing) {
      throw new Error(`Key pair ${id} not found`);
    }

    // Revoke old key
    existing.revokedAt = new Date();

    // Generate new key with same algorithm
    const newKeys = KeyPairGenerator.generate(existing.algorithm);
    const newKeyPair: KeyPair = {
      id: uuidv4(),
      algorithm: existing.algorithm,
      publicKey: newKeys.publicKey,
      encryptedPrivateKey: existing.encryptedPrivateKey
        ? encryptWithKey(newKeys.privateKey, "rotation-temp-key") // Would use proper key in production
        : undefined,
      keyDerivationParams: existing.keyDerivationParams,
      createdAt: new Date(),
      metadata: {
        ...existing.metadata,
        previousKeyId: id,
        rotatedFrom: existing.createdAt.toISOString(),
      },
    };

    await this.save(newKeyPair);
    return newKeyPair;
  }
}

// ============================================================================
// DIGITAL SIGNATURE SERVICE (@CIPHER @FORTRESS)
// ============================================================================

/**
 * Events emitted by SignatureService
 */
export interface SignatureEvents {
  "key:created": (keyPair: KeyPair) => void;
  "key:rotated": (oldId: string, newKeyPair: KeyPair) => void;
  "key:revoked": (keyPairId: string) => void;
  "signature:created": (signature: DigitalSignature) => void;
  "signature:verified": (signatureId: string, valid: boolean) => void;
}

/**
 * Digital signature service configuration
 */
export interface SignatureServiceConfig {
  defaultAlgorithm: SignatureAlgorithm;
  hashAlgorithm: HashAlgorithm;
  keyStore: KeyStore;
  autoRotateDays?: number;
}

/**
 * Digital Signature Service
 * @agent @CIPHER @FORTRESS - Complete signature implementation
 */
export class DigitalSignatureService extends EventEmitter<SignatureEvents> {
  private config: SignatureServiceConfig;
  private privateKeys: Map<string, string> = new Map(); // In-memory for session

  constructor(config: SignatureServiceConfig) {
    super();
    this.config = config;
  }

  /**
   * Create a new key pair
   */
  async createKeyPair(
    password?: string,
    algorithm?: SignatureAlgorithm,
    metadata?: Record<string, string>
  ): Promise<KeyPair> {
    const algo = algorithm ?? this.config.defaultAlgorithm;
    const { publicKey, privateKey } = KeyPairGenerator.generate(algo);

    let encryptedPrivateKey: string | undefined;
    let keyDerivationParams: KeyDerivationParams | undefined;

    if (password) {
      const salt = generateSalt();
      keyDerivationParams = {
        ...DEFAULT_KEY_DERIVATION,
        salt,
      };
      const derivedKey = deriveKey(password, keyDerivationParams);
      encryptedPrivateKey = encryptWithKey(privateKey, derivedKey);
    }

    const keyPair: KeyPair = {
      id: uuidv4(),
      algorithm: algo,
      publicKey,
      encryptedPrivateKey,
      keyDerivationParams,
      createdAt: new Date(),
      metadata,
    };

    // Store key pair
    await this.config.keyStore.save(keyPair);

    // Keep private key in memory for this session
    this.privateKeys.set(keyPair.id, privateKey);

    this.emit("key:created", keyPair);

    return keyPair;
  }

  /**
   * Unlock a key pair with password
   */
  async unlockKeyPair(keyPairId: string, password: string): Promise<void> {
    const keyPair = await this.config.keyStore.get(keyPairId);
    if (!keyPair) {
      throw new Error(`Key pair ${keyPairId} not found`);
    }

    if (!keyPair.encryptedPrivateKey || !keyPair.keyDerivationParams) {
      throw new Error("Key pair is not password-protected");
    }

    const derivedKey = deriveKey(password, keyPair.keyDerivationParams);
    const privateKey = decryptWithKey(keyPair.encryptedPrivateKey, derivedKey);

    this.privateKeys.set(keyPairId, privateKey);
  }

  /**
   * Sign content
   */
  async sign(
    content: string | Buffer,
    keyPairId: string,
    metadata?: Record<string, string>
  ): Promise<DigitalSignature> {
    const keyPair = await this.config.keyStore.get(keyPairId);
    if (!keyPair) {
      throw new Error(`Key pair ${keyPairId} not found`);
    }

    if (keyPair.revokedAt) {
      throw new Error("Cannot sign with revoked key");
    }

    const privateKey = this.privateKeys.get(keyPairId);
    if (!privateKey) {
      throw new Error("Key pair not unlocked. Call unlockKeyPair() first.");
    }

    // Compute content hash
    const contentStr =
      typeof content === "string" ? content : content.toString("utf8");
    const contentHash = computeHash(contentStr, this.config.hashAlgorithm);

    // Sign based on algorithm
    let signature: string;
    switch (keyPair.algorithm) {
      case "ecdsa_secp256k1": {
        const wallet = new ethers.Wallet(privateKey);
        const messageHash = ethers.hashMessage(contentHash);
        signature = await wallet.signMessage(contentHash);
        break;
      }
      default:
        throw new Error(`Signing with ${keyPair.algorithm} not implemented`);
    }

    const digitalSignature: DigitalSignature = {
      id: uuidv4(),
      contentHash,
      signature,
      algorithm: keyPair.algorithm,
      publicKeyId: keyPairId,
      signedAt: new Date(),
      metadata,
    };

    this.emit("signature:created", digitalSignature);

    return digitalSignature;
  }

  /**
   * Verify a signature
   */
  async verify(
    content: string | Buffer,
    signature: DigitalSignature
  ): Promise<boolean> {
    const keyPair = await this.config.keyStore.get(signature.publicKeyId);
    if (!keyPair) {
      this.emit("signature:verified", signature.id, false);
      return false;
    }

    // Compute content hash
    const contentStr =
      typeof content === "string" ? content : content.toString("utf8");
    const contentHash = computeHash(contentStr, this.config.hashAlgorithm);

    // Verify hash matches
    if (contentHash !== signature.contentHash) {
      this.emit("signature:verified", signature.id, false);
      return false;
    }

    // Verify signature based on algorithm
    let valid: boolean;
    switch (signature.algorithm) {
      case "ecdsa_secp256k1": {
        try {
          const recoveredAddress = ethers.verifyMessage(
            contentHash,
            signature.signature
          );
          const expectedAddress = ethers.computeAddress(keyPair.publicKey);
          valid =
            recoveredAddress.toLowerCase() === expectedAddress.toLowerCase();
        } catch {
          valid = false;
        }
        break;
      }
      default:
        throw new Error(
          `Verification with ${signature.algorithm} not implemented`
        );
    }

    this.emit("signature:verified", signature.id, valid);
    return valid;
  }

  /**
   * Rotate a key pair
   */
  async rotateKeyPair(keyPairId: string): Promise<KeyPair> {
    const newKeyPair = await this.config.keyStore.rotate(keyPairId);
    this.emit("key:rotated", keyPairId, newKeyPair);
    return newKeyPair;
  }

  /**
   * Revoke a key pair
   */
  async revokeKeyPair(keyPairId: string): Promise<void> {
    const keyPair = await this.config.keyStore.get(keyPairId);
    if (!keyPair) {
      throw new Error(`Key pair ${keyPairId} not found`);
    }

    keyPair.revokedAt = new Date();
    await this.config.keyStore.save(keyPair);
    this.privateKeys.delete(keyPairId);

    this.emit("key:revoked", keyPairId);
  }

  /**
   * List all active key pairs
   */
  async listKeyPairs(): Promise<KeyPair[]> {
    return this.config.keyStore.list();
  }

  /**
   * Clear session (remove private keys from memory)
   */
  clearSession(): void {
    this.privateKeys.clear();
  }
}

// ============================================================================
// MULTI-SIGNATURE SUPPORT (@FORTRESS)
// ============================================================================

/**
 * Multi-signature requirement
 */
export interface MultiSigRequirement {
  threshold: number; // Minimum signatures required
  publicKeyIds: string[]; // Authorized signers
}

/**
 * Multi-signature collection
 */
export interface MultiSignature {
  id: string;
  contentHash: string;
  requirement: MultiSigRequirement;
  signatures: DigitalSignature[];
  complete: boolean;
  createdAt: Date;
  completedAt?: Date;
}

/**
 * Multi-signature coordinator
 * @agent @FORTRESS - Multi-sig implementation
 */
export class MultiSignatureCoordinator {
  private pendingMultiSigs: Map<string, MultiSignature> = new Map();

  constructor(private signatureService: DigitalSignatureService) {}

  /**
   * Create a new multi-signature request
   */
  createMultiSig(
    contentHash: string,
    requirement: MultiSigRequirement
  ): MultiSignature {
    if (requirement.threshold > requirement.publicKeyIds.length) {
      throw new Error("Threshold cannot exceed number of signers");
    }

    const multiSig: MultiSignature = {
      id: uuidv4(),
      contentHash,
      requirement,
      signatures: [],
      complete: false,
      createdAt: new Date(),
    };

    this.pendingMultiSigs.set(multiSig.id, multiSig);
    return multiSig;
  }

  /**
   * Add a signature to a multi-sig request
   */
  async addSignature(
    multiSigId: string,
    signature: DigitalSignature
  ): Promise<MultiSignature> {
    const multiSig = this.pendingMultiSigs.get(multiSigId);
    if (!multiSig) {
      throw new Error(`Multi-signature request ${multiSigId} not found`);
    }

    if (multiSig.complete) {
      throw new Error("Multi-signature already complete");
    }

    // Verify signer is authorized
    if (!multiSig.requirement.publicKeyIds.includes(signature.publicKeyId)) {
      throw new Error("Signer not authorized for this multi-signature");
    }

    // Check for duplicate
    if (
      multiSig.signatures.some((s) => s.publicKeyId === signature.publicKeyId)
    ) {
      throw new Error("This key has already signed");
    }

    // Verify content hash matches
    if (signature.contentHash !== multiSig.contentHash) {
      throw new Error("Signature content hash does not match");
    }

    multiSig.signatures.push(signature);

    // Check if threshold reached
    if (multiSig.signatures.length >= multiSig.requirement.threshold) {
      multiSig.complete = true;
      multiSig.completedAt = new Date();
    }

    return multiSig;
  }

  /**
   * Verify a complete multi-signature
   */
  async verifyMultiSig(
    multiSig: MultiSignature,
    content: string | Buffer
  ): Promise<boolean> {
    if (!multiSig.complete) {
      return false;
    }

    // Verify each signature
    for (const signature of multiSig.signatures) {
      const valid = await this.signatureService.verify(content, signature);
      if (!valid) {
        return false;
      }
    }

    return true;
  }

  /**
   * Get pending multi-signature status
   */
  getMultiSig(multiSigId: string): MultiSignature | null {
    return this.pendingMultiSigs.get(multiSigId) ?? null;
  }

  /**
   * Cancel a pending multi-signature
   */
  cancelMultiSig(multiSigId: string): void {
    this.pendingMultiSigs.delete(multiSigId);
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export {
  generateSalt,
  deriveKey,
  encryptWithKey,
  decryptWithKey,
  KeyPairGenerator,
  InMemoryKeyStore,
  DigitalSignatureService,
  MultiSignatureCoordinator,
};

export type {
  KeyDerivationParams,
  KeyStore,
  SignatureEvents,
  SignatureServiceConfig,
  MultiSigRequirement,
  MultiSignature,
};
