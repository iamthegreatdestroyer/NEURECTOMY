/**
 * @fileoverview Compliance Engine Module Exports
 * @module @neurectomy/legal-fortress/compliance
 *
 * @agents @AEGIS @ARCHITECT @FORTRESS
 *
 * Enterprise compliance automation:
 * - Multi-framework support (SOC2, GDPR, HIPAA, ISO 27001, PCI-DSS, FedRAMP)
 * - Automated control assessment
 * - Gap analysis and remediation planning
 * - Continuous compliance monitoring
 * - Audit-ready reporting
 */

export {
  ComplianceEngine,
  createComplianceEngine,
  SOC2_CONTROLS,
  SOC2_CATEGORIES,
  GDPR_CONTROLS,
} from "./engine";

export type {
  // Control types
  ComplianceControl,
  EvidenceType,
  CrossReference,
  ComplianceEvidence,

  // Assessment types
  ControlAssessment,
  ComplianceFinding,
  FrameworkAssessment,

  // Remediation types
  RemediationTask,

  // Configuration
  ComplianceEngineConfig,
  EvidenceSource,

  // Events
  ComplianceEngineEvents,
} from "./engine";
