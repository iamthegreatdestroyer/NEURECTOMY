/**
 * @fileoverview Immutable Audit Trail Service
 * @module @neurectomy/legal-fortress/audit
 *
 * @agents @AEGIS @CRYPTO @STREAM - Compliance + Blockchain + Event Streaming
 *
 * Enterprise-grade audit logging with:
 * - Tamper-evident append-only storage
 * - Cryptographic hash chaining
 * - Regulatory retention enforcement
 * - Legal hold management
 * - SIEM integration
 * - Multi-format export for auditors
 *
 * Compliance: SOC2 CC6.1, HIPAA 164.312(b), ISO 27001 A.12.4.2, GDPR Art. 30
 */

import { EventEmitter } from "eventemitter3";
import CryptoJS from "crypto-js";
import { v4 as uuidv4 } from "uuid";
import { z } from "zod";

// ============================================================================
// AUDIT EVENT TYPES (@AEGIS)
// ============================================================================

/**
 * Audit event categories aligned with regulatory requirements
 */
export const AuditEventCategorySchema = z.enum([
  // Access & Authentication
  "authentication",
  "authorization",
  "access_control",

  // Data Operations
  "data_create",
  "data_read",
  "data_update",
  "data_delete",
  "data_export",

  // System Operations
  "system_config",
  "system_startup",
  "system_shutdown",
  "system_error",

  // Security Events
  "security_violation",
  "security_alert",
  "intrusion_attempt",

  // Compliance Events
  "compliance_check",
  "policy_violation",
  "evidence_collection",

  // IP Protection Events
  "timestamp_anchor",
  "signature_create",
  "signature_verify",
  "plagiarism_scan",
  "license_check",

  // Administrative
  "admin_action",
  "user_management",
  "role_change",
]);
export type AuditEventCategory = z.infer<typeof AuditEventCategorySchema>;

/**
 * Actor performing the audited action
 */
export const AuditActorSchema = z.object({
  id: z.string(),
  type: z.enum(["user", "system", "agent", "service", "external"]),
  name: z.string().optional(),
  email: z.string().email().optional(),
  ipAddress: z.string().optional(),
  userAgent: z.string().optional(),
  sessionId: z.string().optional(),
  tenantId: z.string().optional(),
});
export type AuditActor = z.infer<typeof AuditActorSchema>;

/**
 * Resource affected by the action
 */
export const AuditResourceSchema = z.object({
  type: z.string(),
  id: z.string(),
  name: z.string().optional(),
  path: z.string().optional(),
  attributes: z.record(z.unknown()).optional(),
});
export type AuditResource = z.infer<typeof AuditResourceSchema>;

/**
 * Complete audit event record
 */
export const AuditEventSchema = z.object({
  // Identification
  id: z.string().uuid(),
  sequenceNumber: z.number().int().positive(),

  // Timing
  timestamp: z.date(),
  receivedAt: z.date().optional(),

  // Chain integrity
  previousHash: z.string(),
  eventHash: z.string(),

  // Event classification
  category: AuditEventCategorySchema,
  action: z.string(),
  outcome: z.enum(["success", "failure", "partial", "unknown"]),

  // Actors and resources
  actor: AuditActorSchema,
  resource: AuditResourceSchema.optional(),
  target: AuditResourceSchema.optional(),

  // Details
  description: z.string().optional(),
  reason: z.string().optional(),
  metadata: z.record(z.unknown()).optional(),

  // Compliance tracking
  complianceFrameworks: z.array(z.string()).optional(),
  controlIds: z.array(z.string()).optional(),

  // Evidence linking
  blockchainAnchor: z.string().optional(),
  evidenceIds: z.array(z.string()).optional(),

  // Retention
  retentionPolicy: z.string().optional(),
  legalHoldIds: z.array(z.string()).optional(),
  expiresAt: z.date().optional(),
});
export type AuditEvent = z.infer<typeof AuditEventSchema>;

/**
 * Retention policy configuration
 */
export const RetentionPolicySchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string().optional(),
  retentionPeriod: z.object({
    value: z.number().positive(),
    unit: z.enum(["days", "months", "years"]),
  }),
  appliesTo: z.object({
    categories: z.array(AuditEventCategorySchema).optional(),
    tenantIds: z.array(z.string()).optional(),
    resourceTypes: z.array(z.string()).optional(),
  }),
  deletionStrategy: z.enum(["archive", "purge", "anonymize"]),
  complianceFramework: z.string().optional(),
  createdAt: z.date(),
  updatedAt: z.date(),
});
export type RetentionPolicy = z.infer<typeof RetentionPolicySchema>;

/**
 * Legal hold configuration
 */
export const LegalHoldSchema = z.object({
  id: z.string().uuid(),
  name: z.string(),
  description: z.string(),
  reason: z.enum(["litigation", "investigation", "regulatory", "preservation"]),
  scope: z.object({
    dateRange: z
      .object({
        start: z.date(),
        end: z.date().optional(),
      })
      .optional(),
    actorIds: z.array(z.string()).optional(),
    resourceTypes: z.array(z.string()).optional(),
    categories: z.array(AuditEventCategorySchema).optional(),
    searchQuery: z.string().optional(),
  }),
  status: z.enum(["active", "released", "expired"]),
  createdBy: z.string(),
  createdAt: z.date(),
  releasedAt: z.date().optional(),
  releasedBy: z.string().optional(),
  expiresAt: z.date().optional(),
});
export type LegalHold = z.infer<typeof LegalHoldSchema>;

/**
 * Integrity verification report
 */
export interface IntegrityReport {
  verified: boolean;
  totalEvents: number;
  verifiedEvents: number;
  brokenChainAt?: number;
  firstEvent: Date;
  lastEvent: Date;
  verificationTime: number;
  issues: IntegrityIssue[];
}

export interface IntegrityIssue {
  sequenceNumber: number;
  eventId: string;
  issue:
    | "missing_event"
    | "hash_mismatch"
    | "sequence_gap"
    | "timestamp_anomaly";
  expected?: string;
  actual?: string;
}

/**
 * Export format options
 */
export type ExportFormat = "json" | "csv" | "pdf" | "xml";

/**
 * SIEM connector configuration
 */
export interface SIEMConfig {
  type: "splunk" | "elastic" | "sentinel" | "datadog" | "sumo" | "generic";
  endpoint: string;
  apiKey?: string;
  index?: string;
  batchSize: number;
  flushInterval: number;
  format: "cef" | "leef" | "json" | "syslog";
  tls: boolean;
}

// ============================================================================
// SERVICE CONFIGURATION
// ============================================================================

export interface AuditTrailConfig {
  /** Storage backend */
  storage: IAuditStorage;
  /** Hash algorithm for chain integrity */
  hashAlgorithm: "sha256" | "sha384" | "sha512";
  /** Enable blockchain anchoring */
  enableBlockchainAnchoring: boolean;
  /** Blockchain anchor interval (events) */
  anchorInterval: number;
  /** Default retention policy */
  defaultRetentionPolicy: RetentionPolicy;
  /** SIEM configuration (optional) */
  siemConfig?: SIEMConfig;
  /** Enable real-time streaming */
  enableStreaming: boolean;
  /** Buffer size for batch operations */
  batchBufferSize: number;
  /** Flush interval in milliseconds */
  flushIntervalMs: number;
}

/**
 * Default configuration
 */
export const DEFAULT_AUDIT_CONFIG: Partial<AuditTrailConfig> = {
  hashAlgorithm: "sha256",
  enableBlockchainAnchoring: false,
  anchorInterval: 1000,
  enableStreaming: true,
  batchBufferSize: 100,
  flushIntervalMs: 5000,
};

// ============================================================================
// STORAGE BACKEND INTERFACE
// ============================================================================

/**
 * Abstract storage interface for audit events
 * @agent @ATLAS - Storage architecture
 */
export interface IAuditStorage {
  /** Storage name */
  name: string;

  /** Initialize storage */
  initialize(): Promise<void>;

  /** Append event (must be atomic) */
  appendEvent(event: AuditEvent): Promise<void>;

  /** Append batch (must be atomic) */
  appendBatch(events: AuditEvent[]): Promise<void>;

  /** Get event by ID */
  getEvent(id: string): Promise<AuditEvent | null>;

  /** Get event by sequence number */
  getEventBySequence(sequence: number): Promise<AuditEvent | null>;

  /** Get events in range */
  getEvents(from: number, to: number): Promise<AuditEvent[]>;

  /** Search events */
  searchEvents(query: AuditSearchQuery): Promise<AuditSearchResult>;

  /** Get latest sequence number */
  getLatestSequence(): Promise<number>;

  /** Get latest event hash */
  getLatestHash(): Promise<string>;

  /** Count events matching criteria */
  countEvents(filters?: AuditFilters): Promise<number>;

  /** Apply retention (delete/archive expired) */
  applyRetention(policy: RetentionPolicy): Promise<RetentionResult>;

  /** Close storage */
  close(): Promise<void>;
}

export interface AuditSearchQuery {
  filters?: AuditFilters;
  dateRange?: { start: Date; end: Date };
  limit?: number;
  offset?: number;
  sortBy?: "timestamp" | "sequenceNumber";
  sortOrder?: "asc" | "desc";
}

export interface AuditFilters {
  categories?: AuditEventCategory[];
  actorIds?: string[];
  resourceTypes?: string[];
  outcomes?: ("success" | "failure" | "partial" | "unknown")[];
  tenantIds?: string[];
  searchText?: string;
}

export interface AuditSearchResult {
  events: AuditEvent[];
  total: number;
  hasMore: boolean;
}

export interface RetentionResult {
  processedCount: number;
  deletedCount: number;
  archivedCount: number;
  anonymizedCount: number;
  errors: string[];
}

// ============================================================================
// IN-MEMORY STORAGE (Development/Testing)
// ============================================================================

/**
 * In-memory audit storage for development and testing
 * NOT FOR PRODUCTION USE
 */
export class InMemoryAuditStorage implements IAuditStorage {
  name = "in-memory";
  private events: Map<string, AuditEvent> = new Map();
  private sequenceIndex: Map<number, string> = new Map();
  private latestSequence = 0;
  private genesisHash = "0".repeat(64);

  async initialize(): Promise<void> {
    // No-op for in-memory
  }

  async appendEvent(event: AuditEvent): Promise<void> {
    this.events.set(event.id, event);
    this.sequenceIndex.set(event.sequenceNumber, event.id);
    this.latestSequence = Math.max(this.latestSequence, event.sequenceNumber);
  }

  async appendBatch(events: AuditEvent[]): Promise<void> {
    for (const event of events) {
      await this.appendEvent(event);
    }
  }

  async getEvent(id: string): Promise<AuditEvent | null> {
    return this.events.get(id) || null;
  }

  async getEventBySequence(sequence: number): Promise<AuditEvent | null> {
    const id = this.sequenceIndex.get(sequence);
    return id ? this.events.get(id) || null : null;
  }

  async getEvents(from: number, to: number): Promise<AuditEvent[]> {
    const results: AuditEvent[] = [];
    for (let seq = from; seq <= to; seq++) {
      const event = await this.getEventBySequence(seq);
      if (event) results.push(event);
    }
    return results;
  }

  async searchEvents(query: AuditSearchQuery): Promise<AuditSearchResult> {
    let results = Array.from(this.events.values());

    // Apply filters
    if (query.filters) {
      const f = query.filters;
      if (f.categories?.length) {
        results = results.filter((e) => f.categories!.includes(e.category));
      }
      if (f.actorIds?.length) {
        results = results.filter((e) => f.actorIds!.includes(e.actor.id));
      }
      if (f.outcomes?.length) {
        results = results.filter((e) => f.outcomes!.includes(e.outcome));
      }
    }

    // Apply date range
    if (query.dateRange) {
      results = results.filter(
        (e) =>
          e.timestamp >= query.dateRange!.start &&
          e.timestamp <= query.dateRange!.end
      );
    }

    // Sort
    const sortOrder = query.sortOrder === "desc" ? -1 : 1;
    results.sort((a, b) => {
      const field =
        query.sortBy === "sequenceNumber" ? "sequenceNumber" : "timestamp";
      if (field === "timestamp") {
        return sortOrder * (a.timestamp.getTime() - b.timestamp.getTime());
      }
      return sortOrder * (a.sequenceNumber - b.sequenceNumber);
    });

    const total = results.length;
    const offset = query.offset || 0;
    const limit = query.limit || 100;

    return {
      events: results.slice(offset, offset + limit),
      total,
      hasMore: offset + limit < total,
    };
  }

  async getLatestSequence(): Promise<number> {
    return this.latestSequence;
  }

  async getLatestHash(): Promise<string> {
    if (this.latestSequence === 0) return this.genesisHash;
    const latest = await this.getEventBySequence(this.latestSequence);
    return latest?.eventHash || this.genesisHash;
  }

  async countEvents(filters?: AuditFilters): Promise<number> {
    if (!filters) return this.events.size;
    const result = await this.searchEvents({ filters });
    return result.total;
  }

  async applyRetention(policy: RetentionPolicy): Promise<RetentionResult> {
    const result: RetentionResult = {
      processedCount: 0,
      deletedCount: 0,
      archivedCount: 0,
      anonymizedCount: 0,
      errors: [],
    };

    const now = new Date();
    const cutoffMs = this.calculateRetentionCutoff(policy, now);
    const cutoffDate = new Date(now.getTime() - cutoffMs);

    for (const [id, event] of this.events) {
      // Skip events under legal hold
      if (event.legalHoldIds?.length) continue;

      if (event.timestamp < cutoffDate) {
        result.processedCount++;
        if (policy.deletionStrategy === "purge") {
          this.events.delete(id);
          result.deletedCount++;
        }
        // Archive and anonymize would need external storage
      }
    }

    return result;
  }

  private calculateRetentionCutoff(
    policy: RetentionPolicy,
    _now: Date
  ): number {
    const { value, unit } = policy.retentionPeriod;
    const multipliers = {
      days: 86400000,
      months: 2592000000,
      years: 31536000000,
    };
    return value * multipliers[unit];
  }

  async close(): Promise<void> {
    this.events.clear();
    this.sequenceIndex.clear();
  }
}

// ============================================================================
// AUDIT TRAIL SERVICE (@AEGIS)
// ============================================================================

/**
 * Events emitted by AuditTrailService
 */
export interface AuditTrailEvents {
  "event:logged": (event: AuditEvent) => void;
  "batch:flushed": (count: number) => void;
  "integrity:verified": (report: IntegrityReport) => void;
  "integrity:broken": (issue: IntegrityIssue) => void;
  "retention:applied": (result: RetentionResult) => void;
  "legal-hold:applied": (hold: LegalHold) => void;
  "legal-hold:released": (hold: LegalHold) => void;
  "anchor:created": (anchor: string, eventCount: number) => void;
  "siem:sent": (count: number) => void;
  error: (error: Error) => void;
}

/**
 * Immutable Audit Trail Service
 *
 * @agent @AEGIS - Enterprise compliance audit logging
 *
 * Features:
 * - Tamper-evident append-only logging
 * - Cryptographic hash chaining
 * - Legal hold management
 * - Retention policy enforcement
 * - SIEM integration
 * - Multi-format export
 */
export class AuditTrailService extends EventEmitter<AuditTrailEvents> {
  private storage: IAuditStorage;
  private config: AuditTrailConfig;
  private buffer: Omit<
    AuditEvent,
    "sequenceNumber" | "previousHash" | "eventHash"
  >[] = [];
  private flushTimer: NodeJS.Timeout | null = null;
  private legalHolds: Map<string, LegalHold> = new Map();
  private retentionPolicies: Map<string, RetentionPolicy> = new Map();
  private lastAnchorSequence = 0;
  private initialized = false;

  constructor(config: Partial<AuditTrailConfig> & { storage: IAuditStorage }) {
    super();
    this.config = { ...DEFAULT_AUDIT_CONFIG, ...config } as AuditTrailConfig;
    this.storage = config.storage;
  }

  /**
   * Initialize the audit trail service
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    await this.storage.initialize();

    // Start flush timer if streaming enabled
    if (this.config.enableStreaming && this.config.flushIntervalMs > 0) {
      this.flushTimer = setInterval(
        () => this.flush().catch((e) => this.emit("error", e)),
        this.config.flushIntervalMs
      );
    }

    // Register default retention policy
    if (this.config.defaultRetentionPolicy) {
      this.retentionPolicies.set(
        this.config.defaultRetentionPolicy.id,
        this.config.defaultRetentionPolicy
      );
    }

    this.initialized = true;
  }

  /**
   * Log an audit event
   */
  async logEvent(
    event: Omit<
      AuditEvent,
      "id" | "sequenceNumber" | "previousHash" | "eventHash" | "timestamp"
    >
  ): Promise<string> {
    const eventWithDefaults = {
      id: uuidv4(),
      timestamp: new Date(),
      ...event,
    };

    if (this.config.batchBufferSize > 1) {
      this.buffer.push(eventWithDefaults);

      if (this.buffer.length >= this.config.batchBufferSize) {
        await this.flush();
      }

      return eventWithDefaults.id;
    }

    // Immediate write
    const completeEvent = await this.prepareEvent(eventWithDefaults);
    await this.storage.appendEvent(completeEvent);
    this.emit("event:logged", completeEvent);

    // Check if blockchain anchor needed
    if (this.shouldAnchor(completeEvent.sequenceNumber)) {
      await this.createBlockchainAnchor();
    }

    return completeEvent.id;
  }

  /**
   * Log multiple events efficiently
   */
  async logEvents(
    events: Omit<
      AuditEvent,
      "id" | "sequenceNumber" | "previousHash" | "eventHash" | "timestamp"
    >[]
  ): Promise<string[]> {
    const ids: string[] = [];
    for (const event of events) {
      const id = await this.logEvent(event);
      ids.push(id);
    }
    return ids;
  }

  /**
   * Flush buffered events to storage
   */
  async flush(): Promise<number> {
    if (this.buffer.length === 0) return 0;

    const toFlush = [...this.buffer];
    this.buffer = [];

    const preparedEvents: AuditEvent[] = [];
    for (const event of toFlush) {
      const prepared = await this.prepareEvent(event);
      preparedEvents.push(prepared);
    }

    await this.storage.appendBatch(preparedEvents);

    for (const event of preparedEvents) {
      this.emit("event:logged", event);
    }

    this.emit("batch:flushed", preparedEvents.length);

    // Send to SIEM if configured
    if (this.config.siemConfig) {
      await this.sendToSIEM(preparedEvents);
    }

    return preparedEvents.length;
  }

  /**
   * Prepare event with sequence number and hash chain
   */
  private async prepareEvent(
    event: Omit<AuditEvent, "sequenceNumber" | "previousHash" | "eventHash">
  ): Promise<AuditEvent> {
    const latestSequence = await this.storage.getLatestSequence();
    const previousHash = await this.storage.getLatestHash();

    const sequenceNumber = latestSequence + 1;
    const eventHash = this.computeEventHash({
      ...event,
      sequenceNumber,
      previousHash,
    });

    return {
      ...event,
      sequenceNumber,
      previousHash,
      eventHash,
    } as AuditEvent;
  }

  /**
   * Compute hash for event integrity
   */
  private computeEventHash(
    event: Omit<AuditEvent, "eventHash"> & {
      sequenceNumber: number;
      previousHash: string;
    }
  ): string {
    const canonical = JSON.stringify({
      id: event.id,
      sequenceNumber: event.sequenceNumber,
      timestamp: event.timestamp.toISOString(),
      previousHash: event.previousHash,
      category: event.category,
      action: event.action,
      outcome: event.outcome,
      actor: event.actor,
      resource: event.resource,
    });

    switch (this.config.hashAlgorithm) {
      case "sha384":
        return CryptoJS.SHA384(canonical).toString();
      case "sha512":
        return CryptoJS.SHA512(canonical).toString();
      default:
        return CryptoJS.SHA256(canonical).toString();
    }
  }

  /**
   * Verify integrity of audit trail
   */
  async verifyIntegrity(from?: Date, to?: Date): Promise<IntegrityReport> {
    const start = Date.now();
    const issues: IntegrityIssue[] = [];

    const searchQuery: AuditSearchQuery = {
      sortBy: "sequenceNumber",
      sortOrder: "asc",
      limit: 100000,
    };

    if (from && to) {
      searchQuery.dateRange = { start: from, end: to };
    }

    const searchResult = await this.storage.searchEvents(searchQuery);

    let previousHash = "0".repeat(64);
    let verifiedCount = 0;
    let firstTimestamp: Date | undefined;
    let lastTimestamp: Date | undefined;

    for (let i = 0; i < searchResult.events.length; i++) {
      const event = searchResult.events[i];
      if (!event) continue;

      if (!firstTimestamp) firstTimestamp = event.timestamp;
      lastTimestamp = event.timestamp;

      // Check sequence continuity
      if (i > 0) {
        const prevEvent = searchResult.events[i - 1];
        if (
          prevEvent &&
          event.sequenceNumber !== prevEvent.sequenceNumber + 1
        ) {
          issues.push({
            sequenceNumber: event.sequenceNumber,
            eventId: event.id,
            issue: "sequence_gap",
            expected: String(prevEvent.sequenceNumber + 1),
            actual: String(event.sequenceNumber),
          });
        }
      }

      // Check hash chain
      if (event.previousHash !== previousHash) {
        issues.push({
          sequenceNumber: event.sequenceNumber,
          eventId: event.id,
          issue: "hash_mismatch",
          expected: previousHash,
          actual: event.previousHash,
        });
      }

      // Verify event hash - use same fields as computeEventHash
      const computedHash = this.computeEventHash({
        ...event,
      });

      if (computedHash !== event.eventHash) {
        issues.push({
          sequenceNumber: event.sequenceNumber,
          eventId: event.id,
          issue: "hash_mismatch",
          expected: computedHash,
          actual: event.eventHash,
        });
      }

      // Check timestamp monotonicity
      if (i > 0) {
        const prevEvent = searchResult.events[i - 1];
        if (prevEvent && event.timestamp < prevEvent.timestamp) {
          issues.push({
            sequenceNumber: event.sequenceNumber,
            eventId: event.id,
            issue: "timestamp_anomaly",
            expected: `>= ${prevEvent.timestamp.toISOString()}`,
            actual: event.timestamp.toISOString(),
          });
        }
      }

      previousHash = event.eventHash;
      verifiedCount++;
    }

    const firstIssue = issues[0];
    const report: IntegrityReport = {
      verified: issues.length === 0,
      totalEvents: searchResult.total,
      verifiedEvents: verifiedCount,
      firstEvent: firstTimestamp || new Date(),
      lastEvent: lastTimestamp || new Date(),
      verificationTime: Date.now() - start,
      issues,
      ...(firstIssue && { brokenChainAt: firstIssue.sequenceNumber }),
    };

    this.emit("integrity:verified", report);

    if (!report.verified && firstIssue) {
      this.emit("integrity:broken", firstIssue);
    }

    return report;
  }

  /**
   * Search audit events
   */
  async searchEvents(query: AuditSearchQuery): Promise<AuditSearchResult> {
    return this.storage.searchEvents(query);
  }

  /**
   * Get event by ID
   */
  async getEvent(id: string): Promise<AuditEvent | null> {
    return this.storage.getEvent(id);
  }

  // ============================================================================
  // LEGAL HOLD MANAGEMENT
  // ============================================================================

  /**
   * Apply legal hold to prevent deletion
   */
  async applyLegalHold(
    hold: Omit<LegalHold, "id" | "createdAt" | "status">
  ): Promise<LegalHold> {
    const legalHold: LegalHold = {
      ...hold,
      id: uuidv4(),
      status: "active",
      createdAt: new Date(),
    };

    this.legalHolds.set(legalHold.id, legalHold);

    // Mark affected events
    const affectedEvents = await this.findEventsForHold(legalHold);
    for (const event of affectedEvents) {
      event.legalHoldIds = [...(event.legalHoldIds || []), legalHold.id];
      // Note: In production, update in storage
    }

    this.emit("legal-hold:applied", legalHold);
    return legalHold;
  }

  /**
   * Release legal hold
   */
  async releaseLegalHold(holdId: string, releasedBy: string): Promise<void> {
    const hold = this.legalHolds.get(holdId);
    if (!hold) {
      throw new Error(`Legal hold not found: ${holdId}`);
    }

    hold.status = "released";
    hold.releasedAt = new Date();
    hold.releasedBy = releasedBy;

    this.emit("legal-hold:released", hold);
  }

  /**
   * Get active legal holds
   */
  getActiveLegalHolds(): LegalHold[] {
    return Array.from(this.legalHolds.values()).filter(
      (h) => h.status === "active"
    );
  }

  private async findEventsForHold(hold: LegalHold): Promise<AuditEvent[]> {
    const query: AuditSearchQuery = {};

    // Build date range if both dates are present
    if (hold.scope.dateRange?.start && hold.scope.dateRange?.end) {
      query.dateRange = {
        start: hold.scope.dateRange.start,
        end: hold.scope.dateRange.end,
      };
    }

    // Build filters only with defined values
    const filters: AuditFilters = {};
    if (hold.scope.actorIds) filters.actorIds = hold.scope.actorIds;
    if (hold.scope.resourceTypes)
      filters.resourceTypes = hold.scope.resourceTypes;
    if (hold.scope.categories) filters.categories = hold.scope.categories;
    if (hold.scope.searchQuery) filters.searchText = hold.scope.searchQuery;

    if (Object.keys(filters).length > 0) {
      query.filters = filters;
    }

    const result = await this.storage.searchEvents(query);
    return result.events;
  }

  // ============================================================================
  // RETENTION MANAGEMENT
  // ============================================================================

  /**
   * Register retention policy
   */
  registerRetentionPolicy(policy: RetentionPolicy): void {
    this.retentionPolicies.set(policy.id, policy);
  }

  /**
   * Apply retention policies
   */
  async applyRetention(): Promise<RetentionResult[]> {
    const results: RetentionResult[] = [];

    for (const policy of this.retentionPolicies.values()) {
      const result = await this.storage.applyRetention(policy);
      results.push(result);
      this.emit("retention:applied", result);
    }

    return results;
  }

  // ============================================================================
  // EXPORT CAPABILITIES
  // ============================================================================

  /**
   * Export audit log in specified format
   */
  async exportAuditLog(
    format: ExportFormat,
    query: AuditSearchQuery
  ): Promise<Buffer> {
    const result = await this.storage.searchEvents({
      ...query,
      limit: 100000, // Maximum export size
    });

    switch (format) {
      case "json":
        return Buffer.from(JSON.stringify(result.events, null, 2));

      case "csv":
        return this.exportToCSV(result.events);

      case "xml":
        return this.exportToXML(result.events);

      case "pdf":
        // PDF generation would require additional library
        throw new Error("PDF export requires additional configuration");

      default:
        throw new Error(`Unsupported export format: ${format}`);
    }
  }

  private exportToCSV(events: AuditEvent[]): Buffer {
    const headers = [
      "ID",
      "Sequence",
      "Timestamp",
      "Category",
      "Action",
      "Outcome",
      "Actor ID",
      "Actor Type",
      "Resource Type",
      "Resource ID",
      "Description",
      "Event Hash",
    ];

    const rows = events.map((e) =>
      [
        e.id,
        e.sequenceNumber,
        e.timestamp.toISOString(),
        e.category,
        e.action,
        e.outcome,
        e.actor.id,
        e.actor.type,
        e.resource?.type || "",
        e.resource?.id || "",
        e.description || "",
        e.eventHash,
      ]
        .map((v) => `"${String(v).replace(/"/g, '""')}"`)
        .join(",")
    );

    return Buffer.from([headers.join(","), ...rows].join("\n"));
  }

  private exportToXML(events: AuditEvent[]): Buffer {
    const xmlEvents = events.map(
      (e) => `
  <AuditEvent>
    <ID>${e.id}</ID>
    <Sequence>${e.sequenceNumber}</Sequence>
    <Timestamp>${e.timestamp.toISOString()}</Timestamp>
    <Category>${e.category}</Category>
    <Action>${e.action}</Action>
    <Outcome>${e.outcome}</Outcome>
    <Actor>
      <ID>${e.actor.id}</ID>
      <Type>${e.actor.type}</Type>
    </Actor>
    <EventHash>${e.eventHash}</EventHash>
  </AuditEvent>`
    );

    return Buffer.from(
      `<?xml version="1.0" encoding="UTF-8"?>
<AuditLog>${xmlEvents.join("")}
</AuditLog>`
    );
  }

  // ============================================================================
  // SIEM INTEGRATION
  // ============================================================================

  /**
   * Send events to SIEM
   */
  private async sendToSIEM(events: AuditEvent[]): Promise<void> {
    if (!this.config.siemConfig) return;

    const formatted = events.map((e) => this.formatForSIEM(e));

    // In production, implement actual SIEM connector
    // This is a placeholder for the integration point
    console.log(
      `[SIEM] Would send ${formatted.length} events to ${this.config.siemConfig.type}`
    );

    this.emit("siem:sent", events.length);
  }

  /**
   * Format event for SIEM consumption
   */
  private formatForSIEM(event: AuditEvent): string {
    const config = this.config.siemConfig!;

    switch (config.format) {
      case "cef":
        return this.formatCEF(event);
      case "leef":
        return this.formatLEEF(event);
      case "syslog":
        return this.formatSyslog(event);
      default:
        return JSON.stringify(event);
    }
  }

  private formatCEF(event: AuditEvent): string {
    // CEF: Common Event Format
    const severity = this.mapOutcomeToSeverity(event.outcome);
    return (
      `CEF:0|NEURECTOMY|LegalFortress|1.0|${event.category}|${event.action}|${severity}|` +
      `rt=${event.timestamp.getTime()} src=${event.actor.ipAddress || "unknown"} ` +
      `suser=${event.actor.id} act=${event.action} outcome=${event.outcome}`
    );
  }

  private formatLEEF(event: AuditEvent): string {
    // LEEF: Log Event Extended Format (IBM QRadar)
    return (
      `LEEF:1.0|NEURECTOMY|LegalFortress|1.0|${event.category}|` +
      `devTime=${event.timestamp.toISOString()}\tsrc=${event.actor.ipAddress || "unknown"}\t` +
      `usrName=${event.actor.id}\taction=${event.action}\toutcome=${event.outcome}`
    );
  }

  private formatSyslog(event: AuditEvent): string {
    const priority = 14; // facility=user, severity=info
    return (
      `<${priority}>${event.timestamp.toISOString()} neurectomy legal-fortress: ` +
      `[${event.category}] ${event.action} by ${event.actor.id} - ${event.outcome}`
    );
  }

  private mapOutcomeToSeverity(outcome: string): number {
    switch (outcome) {
      case "failure":
        return 7;
      case "partial":
        return 5;
      default:
        return 3;
    }
  }

  // ============================================================================
  // BLOCKCHAIN ANCHORING
  // ============================================================================

  private shouldAnchor(sequenceNumber: number): boolean {
    if (!this.config.enableBlockchainAnchoring) return false;
    return (
      sequenceNumber - this.lastAnchorSequence >= this.config.anchorInterval
    );
  }

  private async createBlockchainAnchor(): Promise<void> {
    // Integration point for blockchain timestamping service
    const latestSequence = await this.storage.getLatestSequence();
    const latestHash = await this.storage.getLatestHash();

    // In production, call TimestampingService here
    console.log(
      `[Anchor] Would anchor sequence ${latestSequence} with hash ${latestHash}`
    );

    this.lastAnchorSequence = latestSequence;
    this.emit(
      "anchor:created",
      latestHash,
      latestSequence - this.lastAnchorSequence
    );
  }

  // ============================================================================
  // LIFECYCLE
  // ============================================================================

  /**
   * Graceful shutdown
   */
  async shutdown(): Promise<void> {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
      this.flushTimer = null;
    }

    await this.flush();
    await this.storage.close();
  }
}

// ============================================================================
// FACTORY FUNCTION
// ============================================================================

/**
 * Create an audit trail service with default in-memory storage
 * For production, provide a persistent storage backend
 */
export function createAuditTrailService(
  config?: Partial<AuditTrailConfig>
): AuditTrailService {
  const storage = config?.storage ?? new InMemoryAuditStorage();

  const defaultRetentionPolicy: RetentionPolicy = {
    id: "default",
    name: "Default 7-Year Retention",
    description: "Standard regulatory retention period",
    retentionPeriod: { value: 7, unit: "years" },
    appliesTo: {},
    deletionStrategy: "archive",
    complianceFramework: "sox",
    createdAt: new Date(),
    updatedAt: new Date(),
  };

  return new AuditTrailService({
    storage,
    defaultRetentionPolicy,
    ...config,
  });
}

// ============================================================================
// EXPORTS
// ============================================================================

export default AuditTrailService;
