/**
 * @fileoverview Audit & Compliance Module
 * @module @neurectomy/enterprise/audit
 *
 * @description
 * Enterprise audit and compliance capabilities:
 * - Tamper-proof audit logging
 * - Compliance reporting (SOC2, GDPR, ISO 27001)
 * - Evidence collection
 * - Finding management
 *
 * @AEGIS Enterprise compliance infrastructure
 */

// Audit Logger
export {
  AuditLogger,
  createAuditLogger,
  type AuditBlock,
  type AuditStats,
  type AuditExport,
  type IntegrityVerificationResult,
} from "./audit-logger.js";

// Compliance Reporter
export {
  ComplianceReporter,
  createComplianceReporter,
  type ComplianceReporterConfig,
  type ControlEvaluation,
  type Evidence,
  type Finding,
  type SOC2Report,
  type GDPRReport,
  type ComplianceDashboard,
} from "./compliance-reporter.js";

// Re-export types
export type {
  AuditEntry,
  AuditConfig,
  AuditEventType,
  TamperProofConfig,
  AuditQuery,
  AuditQueryResult,
  ComplianceFramework,
  SOC2Control,
  ComplianceReport,
  ComplianceStatus,
  ControlStatus,
} from "../types.js";

/**
 * Create full audit system
 */
export interface AuditSystemConfig {
  audit: import("../types.js").AuditConfig;
  compliance?: import("./compliance-reporter.js").ComplianceReporterConfig;
}

export interface AuditSystem {
  logger: import("./audit-logger.js").AuditLogger;
  compliance: import("./compliance-reporter.js").ComplianceReporter;
}

export async function createAuditSystem(
  config: AuditSystemConfig
): Promise<AuditSystem> {
  const { AuditLogger } = await import("./audit-logger.js");
  const { ComplianceReporter } = await import("./compliance-reporter.js");

  const logger = new AuditLogger(config.audit);
  const compliance = new ComplianceReporter(config.compliance || {});

  // Wire up events
  logger.on("entry:logged", (entry) => {
    // Could trigger compliance checks
  });

  logger.on("security:alert", (entry) => {
    // High severity events trigger compliance tracking
    compliance.trackFinding({
      severity: "high",
      title: `Security Event: ${entry.action}`,
      description: `Security event detected: ${JSON.stringify(entry.details)}`,
      controlId: "CC7.3", // Incident Response
      remediation: ["Investigate security event", "Document findings"],
      status: "open",
    });
  });

  // Initialize
  await logger.initialize();
  await compliance.initialize();

  return { logger, compliance };
}
