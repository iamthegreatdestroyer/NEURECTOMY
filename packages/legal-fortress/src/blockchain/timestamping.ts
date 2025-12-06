/**
 * @fileoverview Blockchain Timestamping System
 * @module @neurectomy/legal-fortress/blockchain
 *
 * @agents @CRYPTO @CIPHER - Blockchain + Cryptography Specialists
 *
 * Implements blockchain-anchored timestamping using Merkle tree aggregation
 * for efficient batch processing. Supports multiple EVM-compatible networks.
 *
 * Key Features:
 * - Merkle tree aggregation for batch timestamping
 * - Multi-chain support (Ethereum, Polygon, Arbitrum, etc.)
 * - Cryptographic proof generation and verification
 * - Gas-optimized anchoring strategies
 */

import { MerkleTree } from "merkletreejs";
import CryptoJS from "crypto-js";
import { ethers } from "ethers";
import { v4 as uuidv4 } from "uuid";
import { EventEmitter } from "eventemitter3";
import {
  BlockchainNetwork,
  HashAlgorithm,
  ContentFingerprint,
  MerkleProof,
  MerkleNode,
  TimestampAnchor,
  TimestampedContent,
} from "../types";

// ============================================================================
// CONFIGURATION
// ============================================================================

/**
 * Network configuration for blockchain connections
 */
export interface NetworkConfig {
  network: BlockchainNetwork;
  rpcUrl: string;
  chainId: number;
  contractAddress?: string;
  explorerUrl?: string;
  gasLimit?: number;
  maxGasPrice?: bigint;
}

/**
 * Default network configurations
 */
export const DEFAULT_NETWORK_CONFIGS: Record<
  BlockchainNetwork,
  Partial<NetworkConfig>
> = {
  ethereum_mainnet: {
    chainId: 1,
    explorerUrl: "https://etherscan.io",
  },
  ethereum_sepolia: {
    chainId: 11155111,
    explorerUrl: "https://sepolia.etherscan.io",
  },
  polygon_mainnet: {
    chainId: 137,
    explorerUrl: "https://polygonscan.com",
  },
  polygon_mumbai: {
    chainId: 80001,
    explorerUrl: "https://mumbai.polygonscan.com",
  },
  arbitrum_mainnet: {
    chainId: 42161,
    explorerUrl: "https://arbiscan.io",
  },
  optimism_mainnet: {
    chainId: 10,
    explorerUrl: "https://optimistic.etherscan.io",
  },
  base_mainnet: {
    chainId: 8453,
    explorerUrl: "https://basescan.org",
  },
};

/**
 * Timestamping service configuration
 */
export interface TimestampingConfig {
  network: NetworkConfig;
  batchSize: number; // Max items per Merkle tree
  batchInterval: number; // Milliseconds between batch anchors
  hashAlgorithm: HashAlgorithm;
  confirmationBlocks: number;
  enableAutoAnchor: boolean;
}

// ============================================================================
// HASH UTILITIES (@CIPHER)
// ============================================================================

/**
 * Compute hash using specified algorithm
 * @agent @CIPHER - Cryptographic hash implementation
 */
export function computeHash(
  content: string | Buffer,
  algorithm: HashAlgorithm
): string {
  const data = typeof content === "string" ? content : content.toString("hex");

  switch (algorithm) {
    case "sha256":
      return CryptoJS.SHA256(data).toString(CryptoJS.enc.Hex);
    case "sha384":
      return CryptoJS.SHA384(data).toString(CryptoJS.enc.Hex);
    case "sha512":
      return CryptoJS.SHA512(data).toString(CryptoJS.enc.Hex);
    case "sha3-256":
      return CryptoJS.SHA3(data, { outputLength: 256 }).toString(
        CryptoJS.enc.Hex
      );
    case "sha3-512":
      return CryptoJS.SHA3(data, { outputLength: 512 }).toString(
        CryptoJS.enc.Hex
      );
    case "keccak256":
      return ethers.keccak256(ethers.toUtf8Bytes(data)).slice(2); // Remove 0x prefix
    case "blake3":
      // BLAKE3 would require additional library - fallback to SHA3-256
      console.warn("BLAKE3 not available, using SHA3-256");
      return CryptoJS.SHA3(data, { outputLength: 256 }).toString(
        CryptoJS.enc.Hex
      );
    default:
      throw new Error(`Unsupported hash algorithm: ${algorithm}`);
  }
}

/**
 * Create content fingerprint
 * @agent @CIPHER - Content fingerprinting
 */
export function createFingerprint(
  content: string | Buffer,
  contentType: string,
  algorithm: HashAlgorithm = "sha256",
  metadata?: Record<string, string>
): ContentFingerprint {
  const contentStr =
    typeof content === "string" ? content : content.toString("utf8");
  const hash = computeHash(content, algorithm);

  return {
    hash,
    algorithm,
    contentSize: contentStr.length,
    contentType,
    metadata,
    createdAt: new Date(),
  };
}

// ============================================================================
// MERKLE TREE IMPLEMENTATION (@CRYPTO)
// ============================================================================

/**
 * Merkle tree builder for batch timestamping
 * @agent @CRYPTO - Merkle tree aggregation
 */
export class MerkleTreeBuilder {
  private leaves: string[] = [];
  private tree: MerkleTree | null = null;
  private hashFunction: (data: string) => string;

  constructor(private algorithm: HashAlgorithm = "sha256") {
    this.hashFunction = (data: string) => computeHash(data, algorithm);
  }

  /**
   * Add a leaf to the tree
   */
  addLeaf(hash: string): number {
    const index = this.leaves.length;
    this.leaves.push(hash);
    this.tree = null; // Invalidate cached tree
    return index;
  }

  /**
   * Add multiple leaves
   */
  addLeaves(hashes: string[]): number[] {
    return hashes.map((hash) => this.addLeaf(hash));
  }

  /**
   * Build the Merkle tree
   */
  build(): MerkleTree {
    if (this.leaves.length === 0) {
      throw new Error("Cannot build tree with no leaves");
    }

    // Use keccak256 for Ethereum compatibility
    this.tree = new MerkleTree(
      this.leaves,
      (data) => {
        return ethers.keccak256(data);
      },
      {
        sortPairs: true,
        hashLeaves: false, // Leaves are already hashed
      }
    );

    return this.tree;
  }

  /**
   * Get the Merkle root
   */
  getRoot(): string {
    if (!this.tree) {
      this.build();
    }
    return this.tree!.getHexRoot();
  }

  /**
   * Generate proof for a specific leaf
   */
  getProof(leafHash: string): MerkleProof {
    if (!this.tree) {
      this.build();
    }

    const leafIndex = this.leaves.indexOf(leafHash);
    if (leafIndex === -1) {
      throw new Error("Leaf not found in tree");
    }

    const proof = this.tree!.getProof(leafHash);

    const proofNodes: MerkleNode[] = proof.map((node, index) => ({
      hash: "0x" + node.data.toString("hex"),
      position: node.position as "left" | "right",
      level: index,
    }));

    return {
      leaf: leafHash,
      root: this.getRoot(),
      proof: proofNodes,
      leafIndex,
      totalLeaves: this.leaves.length,
    };
  }

  /**
   * Verify a Merkle proof
   */
  static verifyProof(proof: MerkleProof): boolean {
    const proofData = proof.proof.map((node) => ({
      data: Buffer.from(node.hash.replace("0x", ""), "hex"),
      position: node.position,
    }));

    return MerkleTree.verify(
      proofData,
      proof.leaf,
      proof.root,
      (data) => ethers.keccak256(data),
      { sortPairs: true }
    );
  }

  /**
   * Get tree statistics
   */
  getStats(): { leafCount: number; treeDepth: number; root: string } {
    if (!this.tree) {
      this.build();
    }

    return {
      leafCount: this.leaves.length,
      treeDepth: this.tree!.getDepth(),
      root: this.getRoot(),
    };
  }

  /**
   * Reset the tree
   */
  reset(): void {
    this.leaves = [];
    this.tree = null;
  }
}

// ============================================================================
// BLOCKCHAIN ANCHORING (@CRYPTO @CIPHER)
// ============================================================================

/**
 * Timestamp anchoring contract ABI (minimal interface)
 */
const TIMESTAMP_CONTRACT_ABI = [
  "function anchor(bytes32 merkleRoot) external returns (uint256)",
  "function getTimestamp(bytes32 merkleRoot) external view returns (uint256)",
  "event Anchored(bytes32 indexed merkleRoot, uint256 indexed timestamp, address indexed sender)",
];

/**
 * Blockchain anchor service
 * @agent @CRYPTO - Blockchain interaction
 */
export class BlockchainAnchorService extends EventEmitter {
  private provider: ethers.JsonRpcProvider;
  private wallet: ethers.Wallet | null = null;
  private contract: ethers.Contract | null = null;

  constructor(private config: NetworkConfig) {
    super();
    this.provider = new ethers.JsonRpcProvider(config.rpcUrl);

    if (config.contractAddress) {
      this.contract = new ethers.Contract(
        config.contractAddress,
        TIMESTAMP_CONTRACT_ABI,
        this.provider
      );
    }
  }

  /**
   * Set wallet for transactions
   */
  setWallet(privateKey: string): void {
    this.wallet = new ethers.Wallet(privateKey, this.provider);

    if (this.config.contractAddress) {
      this.contract = new ethers.Contract(
        this.config.contractAddress,
        TIMESTAMP_CONTRACT_ABI,
        this.wallet
      );
    }
  }

  /**
   * Anchor Merkle root to blockchain
   * Uses data field of transaction if no contract
   */
  async anchorMerkleRoot(merkleRoot: string): Promise<TimestampAnchor> {
    if (!this.wallet) {
      throw new Error("Wallet not configured");
    }

    const anchorId = uuidv4();
    let tx: ethers.TransactionResponse;

    this.emit("anchoring:start", { anchorId, merkleRoot });

    try {
      if (this.contract) {
        // Use timestamp contract
        tx = await this.contract.anchor(merkleRoot);
      } else {
        // Store in transaction data field (zero-value tx)
        tx = await this.wallet.sendTransaction({
          to: this.wallet.address, // Self-send
          value: 0n,
          data: merkleRoot,
          gasLimit: this.config.gasLimit ?? 50000,
        });
      }

      this.emit("anchoring:submitted", { anchorId, txHash: tx.hash });

      // Wait for confirmation
      const receipt = await tx.wait(this.config.gasLimit ? 1 : 3);

      if (!receipt) {
        throw new Error("Transaction failed");
      }

      const block = await this.provider.getBlock(receipt.blockNumber);

      const anchor: TimestampAnchor = {
        id: anchorId,
        merkleRoot,
        transactionHash: receipt.hash,
        blockNumber: receipt.blockNumber,
        blockTimestamp: new Date(block!.timestamp * 1000),
        network: this.config.network,
        contractAddress: this.config.contractAddress,
        gasUsed: Number(receipt.gasUsed),
        confirmations: 1,
        status: "confirmed",
        createdAt: new Date(),
      };

      this.emit("anchoring:confirmed", anchor);

      return anchor;
    } catch (error) {
      this.emit("anchoring:failed", { anchorId, error });
      throw error;
    }
  }

  /**
   * Verify a timestamp anchor
   */
  async verifyAnchor(anchor: TimestampAnchor): Promise<boolean> {
    try {
      const tx = await this.provider.getTransaction(anchor.transactionHash);

      if (!tx) {
        return false;
      }

      // Check block number matches
      if (tx.blockNumber !== anchor.blockNumber) {
        return false;
      }

      // Verify merkle root in transaction data
      if (this.contract) {
        const timestamp = await this.contract.getTimestamp(anchor.merkleRoot);
        return timestamp > 0;
      } else {
        // Check data field
        return tx.data === anchor.merkleRoot;
      }
    } catch {
      return false;
    }
  }

  /**
   * Get current confirmations for an anchor
   */
  async getConfirmations(anchor: TimestampAnchor): Promise<number> {
    const currentBlock = await this.provider.getBlockNumber();
    return currentBlock - anchor.blockNumber + 1;
  }

  /**
   * Get explorer URL for transaction
   */
  getExplorerUrl(txHash: string): string | null {
    const baseUrl = DEFAULT_NETWORK_CONFIGS[this.config.network]?.explorerUrl;
    return baseUrl ? `${baseUrl}/tx/${txHash}` : null;
  }
}

// ============================================================================
// TIMESTAMPING SERVICE (@CRYPTO @CIPHER)
// ============================================================================

/**
 * Events emitted by TimestampingService
 */
export interface TimestampingEvents {
  "content:added": (content: TimestampedContent) => void;
  "batch:building": (batchId: string, count: number) => void;
  "batch:anchoring": (batchId: string, merkleRoot: string) => void;
  "batch:anchored": (batchId: string, anchor: TimestampAnchor) => void;
  "batch:failed": (batchId: string, error: Error) => void;
  "content:verified": (contentId: string, valid: boolean) => void;
}

/**
 * Complete timestamping service
 * @agent @CRYPTO @CIPHER - Full timestamping implementation
 */
export class TimestampingService extends EventEmitter<TimestampingEvents> {
  private merkleBuilder: MerkleTreeBuilder;
  private anchorService: BlockchainAnchorService;
  private pendingContent: Map<string, TimestampedContent> = new Map();
  private batchTimer: NodeJS.Timeout | null = null;
  private currentBatchId: string = uuidv4();

  constructor(private config: TimestampingConfig) {
    super();
    this.merkleBuilder = new MerkleTreeBuilder(config.hashAlgorithm);
    this.anchorService = new BlockchainAnchorService(config.network);

    if (config.enableAutoAnchor) {
      this.startAutoBatch();
    }

    // Forward anchor service events
    this.anchorService.on("anchoring:start", (data) =>
      this.emit("batch:anchoring", data.anchorId, data.merkleRoot)
    );
    this.anchorService.on("anchoring:confirmed", (anchor) =>
      this.emit("batch:anchored", this.currentBatchId, anchor)
    );
    this.anchorService.on("anchoring:failed", (data) =>
      this.emit("batch:failed", this.currentBatchId, data.error)
    );
  }

  /**
   * Configure wallet for blockchain transactions
   */
  setWallet(privateKey: string): void {
    this.anchorService.setWallet(privateKey);
  }

  /**
   * Add content for timestamping
   */
  async addContent(
    content: string | Buffer,
    contentType: string,
    metadata?: Record<string, string>
  ): Promise<TimestampedContent> {
    const fingerprint = createFingerprint(
      content,
      contentType,
      this.config.hashAlgorithm,
      metadata
    );

    const contentId = uuidv4();
    const leafIndex = this.merkleBuilder.addLeaf(fingerprint.hash);

    const timestampedContent: TimestampedContent = {
      id: contentId,
      fingerprint,
      status: "pending",
      verificationCount: 0,
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    this.pendingContent.set(contentId, timestampedContent);
    this.emit("content:added", timestampedContent);

    // Auto-anchor if batch size reached
    if (this.pendingContent.size >= this.config.batchSize) {
      await this.anchorBatch();
    }

    return timestampedContent;
  }

  /**
   * Anchor current batch to blockchain
   */
  async anchorBatch(): Promise<TimestampAnchor | null> {
    if (this.pendingContent.size === 0) {
      return null;
    }

    const batchId = this.currentBatchId;
    this.emit("batch:building", batchId, this.pendingContent.size);

    try {
      // Build Merkle tree
      const merkleRoot = this.merkleBuilder.getRoot();

      // Anchor to blockchain
      const anchor = await this.anchorService.anchorMerkleRoot(merkleRoot);

      // Update all pending content with proofs
      for (const [contentId, content] of this.pendingContent) {
        const proof = this.merkleBuilder.getProof(content.fingerprint.hash);
        content.merkleProof = proof;
        content.anchor = anchor;
        content.status = "anchored";
        content.updatedAt = new Date();
      }

      // Clear batch
      this.pendingContent.clear();
      this.merkleBuilder.reset();
      this.currentBatchId = uuidv4();

      return anchor;
    } catch (error) {
      this.emit("batch:failed", batchId, error as Error);
      throw error;
    }
  }

  /**
   * Verify timestamped content
   */
  async verifyContent(content: TimestampedContent): Promise<boolean> {
    if (!content.merkleProof || !content.anchor) {
      this.emit("content:verified", content.id, false);
      return false;
    }

    // Verify Merkle proof
    const proofValid = MerkleTreeBuilder.verifyProof(content.merkleProof);
    if (!proofValid) {
      this.emit("content:verified", content.id, false);
      return false;
    }

    // Verify blockchain anchor
    const anchorValid = await this.anchorService.verifyAnchor(content.anchor);

    const valid = proofValid && anchorValid;
    this.emit("content:verified", content.id, valid);

    if (valid) {
      content.verificationCount++;
      content.lastVerifiedAt = new Date();
      content.status = "verified";
    }

    return valid;
  }

  /**
   * Generate verification certificate
   */
  generateCertificate(content: TimestampedContent): string {
    if (!content.anchor) {
      throw new Error("Content not yet anchored");
    }

    const explorerUrl = this.anchorService.getExplorerUrl(
      content.anchor.transactionHash
    );

    return JSON.stringify(
      {
        certificateVersion: "1.0",
        contentHash: content.fingerprint.hash,
        hashAlgorithm: content.fingerprint.algorithm,
        contentType: content.fingerprint.contentType,
        contentSize: content.fingerprint.contentSize,
        timestamp: {
          blockTimestamp: content.anchor.blockTimestamp.toISOString(),
          blockNumber: content.anchor.blockNumber,
          transactionHash: content.anchor.transactionHash,
          network: content.anchor.network,
          explorerUrl,
        },
        merkleProof: content.merkleProof,
        metadata: content.fingerprint.metadata,
        generatedAt: new Date().toISOString(),
      },
      null,
      2
    );
  }

  /**
   * Start automatic batch anchoring
   */
  private startAutoBatch(): void {
    this.batchTimer = setInterval(async () => {
      if (this.pendingContent.size > 0) {
        try {
          await this.anchorBatch();
        } catch (error) {
          console.error("Auto-batch failed:", error);
        }
      }
    }, this.config.batchInterval);
  }

  /**
   * Stop automatic batch anchoring
   */
  stopAutoBatch(): void {
    if (this.batchTimer) {
      clearInterval(this.batchTimer);
      this.batchTimer = null;
    }
  }

  /**
   * Get pending content count
   */
  getPendingCount(): number {
    return this.pendingContent.size;
  }

  /**
   * Get batch statistics
   */
  getBatchStats(): { pending: number; batchId: string; treeDepth: number } {
    const stats =
      this.pendingContent.size > 0
        ? this.merkleBuilder.getStats()
        : { treeDepth: 0 };

    return {
      pending: this.pendingContent.size,
      batchId: this.currentBatchId,
      treeDepth: stats.treeDepth,
    };
  }

  /**
   * Cleanup resources
   */
  destroy(): void {
    this.stopAutoBatch();
    this.removeAllListeners();
  }
}

// Types and functions are already exported inline above
