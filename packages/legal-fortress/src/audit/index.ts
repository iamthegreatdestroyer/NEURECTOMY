/**
 * @fileoverview Audit Trail Module Exports
 * @module @neurectomy/legal-fortress/audit
 *
 * @agents @AEGIS @CRYPTO @STREAM
 *
 * Enterprise-grade audit logging with:
 * - Tamper-evident append-only storage
 * - Cryptographic hash chaining
 * - Regulatory retention enforcement
 * - Legal hold management
 * - SIEM integration
 */

export {
  AuditTrailService,
  createAuditTrailService,
  InMemoryAuditStorage,
} from "./audit-trail-service";

export type {
  // Event types
  AuditEventCategory,
  AuditActor,
  AuditResource,
  AuditEvent,

  // Policy types
  RetentionPolicy,
  LegalHold,

  // Report types
  IntegrityReport,
  IntegrityIssue,

  // Configuration
  AuditTrailConfig,
  ExportFormat,
  SIEMConfig,

  // Storage interface
  IAuditStorage,
  AuditSearchQuery,
  AuditFilters,
  AuditSearchResult,
  RetentionResult,

  // Events
  AuditTrailEvents,
} from "./audit-trail-service";
