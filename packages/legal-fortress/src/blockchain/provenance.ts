/**
 * @fileoverview Provenance Tracking System
 * @module @neurectomy/legal-fortress/blockchain/provenance
 *
 * @agents @CRYPTO @STREAM - Blockchain + Event Tracking Specialists
 *
 * Tracks the complete history and lineage of code artifacts:
 * - Immutable event chain
 * - Cryptographic linking
 * - Blockchain anchoring for tamper evidence
 * - Real-time event streaming
 */

import { v4 as uuidv4 } from "uuid";
import { EventEmitter } from "eventemitter3";
import {
  ProvenanceEvent,
  ProvenanceEventType,
  ProvenanceChain,
  HashAlgorithm,
  TimestampAnchor,
} from "../types";
import { computeHash } from "./timestamping";

// ============================================================================
// PROVENANCE EVENT CREATION (@STREAM)
// ============================================================================

/**
 * Actor information for provenance events
 */
export interface ProvenanceActor {
  id: string;
  type: "user" | "system" | "agent";
  name?: string;
}

/**
 * Configuration for provenance tracking
 */
export interface ProvenanceConfig {
  hashAlgorithm: HashAlgorithm;
  signEvents: boolean;
  anchorToBlockchain: boolean;
  maxChainLength?: number;
}

/**
 * Default provenance configuration
 */
export const DEFAULT_PROVENANCE_CONFIG: ProvenanceConfig = {
  hashAlgorithm: "sha256",
  signEvents: false,
  anchorToBlockchain: false,
  maxChainLength: 10000,
};

/**
 * Create a provenance event
 * @agent @STREAM - Event creation
 */
export function createProvenanceEvent(
  contentId: string,
  eventType: ProvenanceEventType,
  currentHash: string,
  actor: ProvenanceActor,
  previousHash?: string,
  metadata?: Record<string, unknown>
): ProvenanceEvent {
  return {
    id: uuidv4(),
    contentId,
    eventType,
    previousHash,
    currentHash,
    actor,
    metadata,
    timestamp: new Date(),
  };
}

/**
 * Compute event hash for chain linking
 * @agent @CRYPTO - Cryptographic linking
 */
export function computeEventHash(
  event: ProvenanceEvent,
  algorithm: HashAlgorithm = "sha256"
): string {
  const eventData = JSON.stringify({
    id: event.id,
    contentId: event.contentId,
    eventType: event.eventType,
    previousHash: event.previousHash,
    currentHash: event.currentHash,
    actor: event.actor,
    timestamp: event.timestamp.toISOString(),
  });

  return computeHash(eventData, algorithm);
}

// ============================================================================
// PROVENANCE CHAIN MANAGER (@CRYPTO @STREAM)
// ============================================================================

/**
 * Events emitted by ProvenanceChainManager
 */
export interface ProvenanceEvents {
  "event:added": (event: ProvenanceEvent) => void;
  "chain:created": (contentId: string) => void;
  "chain:verified": (contentId: string, valid: boolean) => void;
  "chain:anchored": (contentId: string, anchor: TimestampAnchor) => void;
  "chain:corrupted": (contentId: string, brokenAt: number) => void;
}

/**
 * Provenance Chain Manager
 * @agent @CRYPTO @STREAM - Complete provenance implementation
 */
export class ProvenanceChainManager extends EventEmitter<ProvenanceEvents> {
  private chains: Map<string, ProvenanceChain> = new Map();
  private config: ProvenanceConfig;

  constructor(config: ProvenanceConfig = DEFAULT_PROVENANCE_CONFIG) {
    super();
    this.config = config;
  }

  /**
   * Start tracking a new piece of content
   */
  startTracking(
    contentId: string,
    contentHash: string,
    actor: ProvenanceActor,
    metadata?: Record<string, unknown>
  ): ProvenanceChain {
    if (this.chains.has(contentId)) {
      throw new Error(`Content ${contentId} is already being tracked`);
    }

    // Create genesis event
    const genesisEvent = createProvenanceEvent(
      contentId,
      "created",
      contentHash,
      actor,
      undefined, // No previous hash for genesis
      metadata
    );

    const eventHash = computeEventHash(genesisEvent, this.config.hashAlgorithm);

    const chain: ProvenanceChain = {
      contentId,
      events: [genesisEvent],
      rootHash: eventHash,
      latestHash: eventHash,
      chainIntegrity: "valid",
    };

    this.chains.set(contentId, chain);
    this.emit("chain:created", contentId);
    this.emit("event:added", genesisEvent);

    return chain;
  }

  /**
   * Record an event in the provenance chain
   */
  recordEvent(
    contentId: string,
    eventType: ProvenanceEventType,
    currentHash: string,
    actor: ProvenanceActor,
    metadata?: Record<string, unknown>
  ): ProvenanceEvent {
    const chain = this.chains.get(contentId);
    if (!chain) {
      throw new Error(`No provenance chain found for content ${contentId}`);
    }

    if (chain.chainIntegrity === "broken") {
      throw new Error("Cannot add events to a broken chain");
    }

    // Check max chain length
    if (
      this.config.maxChainLength &&
      chain.events.length >= this.config.maxChainLength
    ) {
      throw new Error(
        `Chain length exceeds maximum of ${this.config.maxChainLength}`
      );
    }

    // Create event with link to previous
    const event = createProvenanceEvent(
      contentId,
      eventType,
      currentHash,
      actor,
      chain.latestHash,
      metadata
    );

    // Compute new chain hash
    const eventHash = computeEventHash(event, this.config.hashAlgorithm);

    // Update chain
    chain.events.push(event);
    chain.latestHash = eventHash;

    this.emit("event:added", event);

    return event;
  }

  /**
   * Verify chain integrity
   * @agent @CRYPTO - Chain verification
   */
  verifyChain(contentId: string): {
    valid: boolean;
    brokenAt?: number;
    details?: string;
  } {
    const chain = this.chains.get(contentId);
    if (!chain) {
      return { valid: false, details: "Chain not found" };
    }

    if (chain.events.length === 0) {
      return { valid: false, details: "Chain is empty" };
    }

    // Verify genesis event has no previous hash
    const genesis = chain.events[0]!;
    if (genesis.previousHash) {
      chain.chainIntegrity = "broken";
      this.emit("chain:corrupted", contentId, 0);
      return {
        valid: false,
        brokenAt: 0,
        details: "Genesis event has previous hash",
      };
    }

    // Verify each link in the chain
    let previousEventHash = computeEventHash(
      genesis,
      this.config.hashAlgorithm
    );

    for (let i = 1; i < chain.events.length; i++) {
      const event = chain.events[i]!;

      // Verify link
      if (event.previousHash !== previousEventHash) {
        chain.chainIntegrity = "broken";
        this.emit("chain:corrupted", contentId, i);
        return {
          valid: false,
          brokenAt: i,
          details: `Chain broken at event ${i}: expected ${previousEventHash}, got ${event.previousHash}`,
        };
      }

      previousEventHash = computeEventHash(event, this.config.hashAlgorithm);
    }

    // Verify latest hash matches
    if (previousEventHash !== chain.latestHash) {
      chain.chainIntegrity = "broken";
      this.emit("chain:corrupted", contentId, chain.events.length - 1);
      return {
        valid: false,
        brokenAt: chain.events.length - 1,
        details: "Latest hash mismatch",
      };
    }

    chain.chainIntegrity = "valid";
    chain.verifiedAt = new Date();
    this.emit("chain:verified", contentId, true);

    return { valid: true };
  }

  /**
   * Get the full provenance chain
   */
  getChain(contentId: string): ProvenanceChain | null {
    return this.chains.get(contentId) ?? null;
  }

  /**
   * Get events for a specific content
   */
  getEvents(
    contentId: string,
    options?: {
      eventTypes?: ProvenanceEventType[];
      startDate?: Date;
      endDate?: Date;
      limit?: number;
    }
  ): ProvenanceEvent[] {
    const chain = this.chains.get(contentId);
    if (!chain) {
      return [];
    }

    let events = [...chain.events];

    if (options?.eventTypes) {
      events = events.filter((e) => options.eventTypes!.includes(e.eventType));
    }

    if (options?.startDate) {
      events = events.filter((e) => e.timestamp >= options.startDate!);
    }

    if (options?.endDate) {
      events = events.filter((e) => e.timestamp <= options.endDate!);
    }

    if (options?.limit) {
      events = events.slice(-options.limit);
    }

    return events;
  }

  /**
   * Get latest event for content
   */
  getLatestEvent(contentId: string): ProvenanceEvent | null {
    const chain = this.chains.get(contentId);
    if (!chain || chain.events.length === 0) {
      return null;
    }
    return chain.events[chain.events.length - 1] ?? null;
  }

  /**
   * Set blockchain anchor for chain
   */
  setBlockchainAnchor(contentId: string, anchor: TimestampAnchor): void {
    const chain = this.chains.get(contentId);
    if (!chain) {
      throw new Error(`No provenance chain found for content ${contentId}`);
    }

    // Add anchor to latest event
    const latestEvent = chain.events[chain.events.length - 1];
    if (latestEvent) {
      latestEvent.blockchainAnchor = anchor.transactionHash;
    }

    this.emit("chain:anchored", contentId, anchor);
  }

  /**
   * Export chain for external storage/verification
   */
  exportChain(contentId: string): string {
    const chain = this.chains.get(contentId);
    if (!chain) {
      throw new Error(`No provenance chain found for content ${contentId}`);
    }

    return JSON.stringify(
      {
        version: "1.0",
        exportedAt: new Date().toISOString(),
        chain: {
          ...chain,
          events: chain.events.map((e) => ({
            ...e,
            timestamp: e.timestamp.toISOString(),
          })),
        },
      },
      null,
      2
    );
  }

  /**
   * Import chain from external source
   */
  importChain(chainData: string): ProvenanceChain {
    const parsed = JSON.parse(chainData);
    const chainInput = parsed.chain;

    const chain: ProvenanceChain = {
      contentId: chainInput.contentId,
      events: chainInput.events.map((e: Record<string, unknown>) => ({
        ...e,
        timestamp: new Date(e["timestamp"] as string),
      })),
      rootHash: chainInput.rootHash,
      latestHash: chainInput.latestHash,
      chainIntegrity: "unverified",
    };

    // Verify imported chain
    this.chains.set(chain.contentId, chain);
    const verification = this.verifyChain(chain.contentId);

    if (!verification.valid) {
      this.chains.delete(chain.contentId);
      throw new Error(`Imported chain is invalid: ${verification.details}`);
    }

    return chain;
  }

  /**
   * Get statistics for all chains
   */
  getStatistics(): {
    totalChains: number;
    totalEvents: number;
    validChains: number;
    brokenChains: number;
  } {
    let totalEvents = 0;
    let validChains = 0;
    let brokenChains = 0;

    for (const chain of this.chains.values()) {
      totalEvents += chain.events.length;
      if (chain.chainIntegrity === "valid") {
        validChains++;
      } else if (chain.chainIntegrity === "broken") {
        brokenChains++;
      }
    }

    return {
      totalChains: this.chains.size,
      totalEvents,
      validChains,
      brokenChains,
    };
  }
}

// ============================================================================
// PROVENANCE STREAM PROCESSOR (@STREAM)
// ============================================================================

/**
 * Real-time provenance event processor
 * @agent @STREAM - Event streaming
 */
export class ProvenanceStreamProcessor extends EventEmitter {
  private chainManager: ProvenanceChainManager;
  private eventQueue: ProvenanceEvent[] = [];
  private processing = false;
  private batchSize: number;
  private batchInterval: number;
  private batchTimer: NodeJS.Timeout | null = null;

  constructor(
    chainManager: ProvenanceChainManager,
    options?: {
      batchSize?: number;
      batchInterval?: number;
    }
  ) {
    super();
    this.chainManager = chainManager;
    this.batchSize = options?.batchSize ?? 100;
    this.batchInterval = options?.batchInterval ?? 5000;
  }

  /**
   * Start the stream processor
   */
  start(): void {
    this.batchTimer = setInterval(
      () => this.processBatch(),
      this.batchInterval
    );
  }

  /**
   * Stop the stream processor
   */
  stop(): void {
    if (this.batchTimer) {
      clearInterval(this.batchTimer);
      this.batchTimer = null;
    }
  }

  /**
   * Queue an event for processing
   */
  queueEvent(
    contentId: string,
    eventType: ProvenanceEventType,
    currentHash: string,
    actor: ProvenanceActor,
    metadata?: Record<string, unknown>
  ): void {
    // Check if chain exists, create if not
    let chain = this.chainManager.getChain(contentId);
    if (!chain && eventType === "created") {
      chain = this.chainManager.startTracking(
        contentId,
        currentHash,
        actor,
        metadata
      );
    } else if (!chain) {
      throw new Error(
        `No chain for content ${contentId}. Create with 'created' event first.`
      );
    }

    // Create and queue event
    const event = createProvenanceEvent(
      contentId,
      eventType,
      currentHash,
      actor,
      chain.latestHash,
      metadata
    );

    this.eventQueue.push(event);
    this.emit("event:queued", event);

    // Process immediately if batch size reached
    if (this.eventQueue.length >= this.batchSize) {
      this.processBatch();
    }
  }

  /**
   * Process queued events
   */
  private async processBatch(): Promise<void> {
    if (this.processing || this.eventQueue.length === 0) {
      return;
    }

    this.processing = true;
    const batch = this.eventQueue.splice(0, this.batchSize);

    this.emit("batch:processing", batch.length);

    for (const event of batch) {
      try {
        this.chainManager.recordEvent(
          event.contentId,
          event.eventType,
          event.currentHash,
          event.actor,
          event.metadata
        );
        this.emit("event:processed", event);
      } catch (error) {
        this.emit("event:error", event, error);
      }
    }

    this.emit("batch:complete", batch.length);
    this.processing = false;
  }

  /**
   * Get queue statistics
   */
  getQueueStats(): { pending: number; processing: boolean } {
    return {
      pending: this.eventQueue.length,
      processing: this.processing,
    };
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export {
  createProvenanceEvent,
  computeEventHash,
  ProvenanceChainManager,
  ProvenanceStreamProcessor,
};

export type { ProvenanceActor, ProvenanceConfig, ProvenanceEvents };
