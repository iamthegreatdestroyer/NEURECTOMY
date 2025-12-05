/**
 * @fileoverview Compliance Reporting Engine
 * @module @neurectomy/enterprise/audit/compliance
 *
 * @description
 * Automated compliance reporting for regulatory requirements:
 * - SOC2 Type I & II
 * - GDPR
 * - HIPAA
 * - ISO 27001
 * - Custom frameworks
 *
 * @AEGIS Compliance-first reporting architecture
 */

import { EventEmitter } from "events";
import type {
  ComplianceFramework,
  SOC2Control,
  ComplianceReport,
  ComplianceStatus,
  ControlStatus,
  AuditEntry,
} from "../types.js";

/**
 * Control evaluation result
 */
export interface ControlEvaluation {
  /** Control ID */
  controlId: string;
  /** Control name */
  controlName: string;
  /** Status */
  status: ControlStatus;
  /** Score (0-100) */
  score: number;
  /** Evidence collected */
  evidence: Evidence[];
  /** Findings */
  findings: Finding[];
  /** Recommendations */
  recommendations: string[];
  /** Last evaluated */
  evaluatedAt: Date;
}

/**
 * Evidence item
 */
export interface Evidence {
  /** Evidence ID */
  id: string;
  /** Evidence type */
  type:
    | "audit_log"
    | "configuration"
    | "screenshot"
    | "document"
    | "api_response";
  /** Description */
  description: string;
  /** Reference */
  reference: string;
  /** Collected at */
  collectedAt: Date;
  /** Collector */
  collectedBy: string;
}

/**
 * Finding from evaluation
 */
export interface Finding {
  /** Finding ID */
  id: string;
  /** Severity */
  severity: "informational" | "low" | "medium" | "high" | "critical";
  /** Title */
  title: string;
  /** Description */
  description: string;
  /** Affected control */
  controlId: string;
  /** Remediation steps */
  remediation: string[];
  /** Due date */
  dueDate?: Date;
  /** Status */
  status: "open" | "in_progress" | "resolved" | "accepted";
}

/**
 * Compliance Reporting Engine
 *
 * Provides automated compliance assessment and
 * report generation for multiple frameworks.
 */
export class ComplianceReporter extends EventEmitter {
  private frameworks: Map<string, ComplianceFramework> = new Map();
  private evaluations: Map<string, ControlEvaluation[]> = new Map();
  private reports: ComplianceReport[] = [];
  private findings: Finding[] = [];
  private initialized: boolean = false;

  constructor(private config: ComplianceReporterConfig) {
    super();
    this.loadDefaultFrameworks();
  }

  /**
   * Initialize compliance reporter
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    this.emit("initializing");

    // Load custom frameworks if configured
    if (this.config.customFrameworks) {
      for (const framework of this.config.customFrameworks) {
        this.registerFramework(framework);
      }
    }

    this.initialized = true;
    this.emit("initialized");
  }

  /**
   * Register compliance framework
   */
  registerFramework(framework: ComplianceFramework): void {
    this.frameworks.set(framework.id, framework);
    this.emit("framework:registered", framework);
  }

  /**
   * Evaluate compliance for a framework
   */
  async evaluate(
    frameworkId: string,
    tenantId: string,
    auditEntries: AuditEntry[],
    configData: Record<string, unknown>
  ): Promise<ControlEvaluation[]> {
    const framework = this.frameworks.get(frameworkId);
    if (!framework) {
      throw new Error(`Framework not found: ${frameworkId}`);
    }

    const evaluations: ControlEvaluation[] = [];

    for (const control of framework.controls) {
      const evaluation = await this.evaluateControl(
        control,
        tenantId,
        auditEntries,
        configData
      );
      evaluations.push(evaluation);
    }

    // Store evaluations
    const key = `${tenantId}:${frameworkId}`;
    this.evaluations.set(key, evaluations);

    this.emit("evaluation:complete", { frameworkId, tenantId, evaluations });
    return evaluations;
  }

  /**
   * Generate compliance report
   */
  async generateReport(
    frameworkId: string,
    tenantId: string,
    reportType: "summary" | "detailed" | "executive",
    generatedBy: string
  ): Promise<ComplianceReport> {
    const key = `${tenantId}:${frameworkId}`;
    const evaluations = this.evaluations.get(key);

    if (!evaluations) {
      throw new Error("No evaluations found. Run evaluate() first.");
    }

    const framework = this.frameworks.get(frameworkId)!;

    // Calculate overall score
    const totalScore = evaluations.reduce((sum, e) => sum + e.score, 0);
    const overallScore = totalScore / evaluations.length;

    // Determine status
    const status = this.determineOverallStatus(evaluations);

    // Collect findings
    const allFindings = evaluations.flatMap((e) => e.findings);
    const openFindings = allFindings.filter((f) => f.status === "open");
    const criticalFindings = allFindings.filter(
      (f) => f.severity === "critical"
    );

    const report: ComplianceReport = {
      id: this.generateReportId(),
      frameworkId,
      frameworkName: framework.name,
      tenantId,
      reportType,
      status,
      overallScore,
      controlsTotal: evaluations.length,
      controlsPassed: evaluations.filter((e) => e.status === "passed").length,
      controlsFailed: evaluations.filter((e) => e.status === "failed").length,
      controlsWarning: evaluations.filter((e) => e.status === "warning").length,
      controlsNotApplicable: evaluations.filter(
        (e) => e.status === "not_applicable"
      ).length,
      findings: allFindings,
      openFindings: openFindings.length,
      criticalFindings: criticalFindings.length,
      evaluations: reportType === "detailed" ? evaluations : undefined,
      recommendations: this.generateRecommendations(evaluations),
      generatedAt: new Date(),
      generatedBy,
      validUntil: new Date(Date.now() + 90 * 24 * 60 * 60 * 1000), // 90 days
    };

    this.reports.push(report);
    this.emit("report:generated", report);

    return report;
  }

  /**
   * Get SOC2 Type II report
   */
  async generateSOC2Report(
    tenantId: string,
    type: "type1" | "type2",
    periodStart: Date,
    periodEnd: Date,
    auditEntries: AuditEntry[],
    configData: Record<string, unknown>,
    generatedBy: string
  ): Promise<SOC2Report> {
    // Run SOC2 evaluation
    const evaluations = await this.evaluate(
      "soc2",
      tenantId,
      auditEntries,
      configData
    );

    // Group by trust service criteria
    const byCriteria = this.groupByTrustServiceCriteria(evaluations);

    // Calculate scores per criteria
    const criteriaScores: Record<string, number> = {};
    for (const [criteria, evals] of Object.entries(byCriteria)) {
      const total = evals.reduce((sum, e) => sum + e.score, 0);
      criteriaScores[criteria] = total / evals.length;
    }

    const baseReport = await this.generateReport(
      "soc2",
      tenantId,
      "detailed",
      generatedBy
    );

    const soc2Report: SOC2Report = {
      ...baseReport,
      soc2Type: type,
      periodStart,
      periodEnd,
      trustServiceCriteria: {
        security: criteriaScores["security"] || 0,
        availability: criteriaScores["availability"] || 0,
        processingIntegrity: criteriaScores["processing_integrity"] || 0,
        confidentiality: criteriaScores["confidentiality"] || 0,
        privacy: criteriaScores["privacy"] || 0,
      },
      auditPeriodDays: Math.ceil(
        (periodEnd.getTime() - periodStart.getTime()) / (24 * 60 * 60 * 1000)
      ),
      auditorOpinion: this.generateAuditorOpinion(baseReport),
    };

    this.emit("soc2:report:generated", soc2Report);
    return soc2Report;
  }

  /**
   * Get GDPR compliance report
   */
  async generateGDPRReport(
    tenantId: string,
    auditEntries: AuditEntry[],
    configData: Record<string, unknown>,
    generatedBy: string
  ): Promise<GDPRReport> {
    const evaluations = await this.evaluate(
      "gdpr",
      tenantId,
      auditEntries,
      configData
    );

    const baseReport = await this.generateReport(
      "gdpr",
      tenantId,
      "detailed",
      generatedBy
    );

    // Calculate GDPR-specific metrics
    const dataSubjectRequests = auditEntries.filter(
      (e) => e.details?.gdprRequestType
    ).length;

    const consentRecords = auditEntries.filter(
      (e) => e.action === "consent_recorded"
    ).length;

    const dataBreaches = auditEntries.filter(
      (e) => e.eventType === "security" && e.details?.breach === true
    ).length;

    const gdprReport: GDPRReport = {
      ...baseReport,
      dataSubjectRequestsProcessed: dataSubjectRequests,
      consentRecordsCount: consentRecords,
      dataBreachesReported: dataBreaches,
      dataProtectionOfficer: (configData.dpo as string) || "Not Assigned",
      dataProcessingAgreements: (configData.dpas as string[]) || [],
      crossBorderTransfers: (configData.transfers as string[]) || [],
      retentionPoliciesCompliant: this.checkRetentionPolicies(evaluations),
    };

    this.emit("gdpr:report:generated", gdprReport);
    return gdprReport;
  }

  /**
   * Track finding
   */
  async trackFinding(finding: Omit<Finding, "id">): Promise<Finding> {
    const fullFinding: Finding = {
      ...finding,
      id: this.generateFindingId(),
    };

    this.findings.push(fullFinding);
    this.emit("finding:tracked", fullFinding);

    return fullFinding;
  }

  /**
   * Update finding status
   */
  async updateFindingStatus(
    findingId: string,
    status: Finding["status"],
    notes?: string
  ): Promise<Finding | null> {
    const finding = this.findings.find((f) => f.id === findingId);
    if (!finding) return null;

    finding.status = status;
    this.emit("finding:updated", { finding, notes });

    return finding;
  }

  /**
   * Get compliance dashboard data
   */
  getDashboard(tenantId: string): ComplianceDashboard {
    const tenantEvaluations: ControlEvaluation[] = [];
    const frameworkScores: Record<string, number> = {};

    for (const [key, evals] of this.evaluations) {
      if (key.startsWith(`${tenantId}:`)) {
        tenantEvaluations.push(...evals);
        const frameworkId = key.split(":")[1];
        const total = evals.reduce((sum, e) => sum + e.score, 0);
        frameworkScores[frameworkId] = total / evals.length;
      }
    }

    const tenantFindings = this.findings.filter((f) =>
      tenantEvaluations.some((e) => e.controlId === f.controlId)
    );

    const openFindings = tenantFindings.filter((f) => f.status === "open");
    const overdueFindings = tenantFindings.filter(
      (f) => f.status === "open" && f.dueDate && f.dueDate < new Date()
    );

    return {
      tenantId,
      frameworkScores,
      overallScore:
        Object.values(frameworkScores).reduce((a, b) => a + b, 0) /
          Object.values(frameworkScores).length || 0,
      totalControls: tenantEvaluations.length,
      passedControls: tenantEvaluations.filter((e) => e.status === "passed")
        .length,
      failedControls: tenantEvaluations.filter((e) => e.status === "failed")
        .length,
      openFindings: openFindings.length,
      criticalFindings: openFindings.filter((f) => f.severity === "critical")
        .length,
      overdueFindings: overdueFindings.length,
      recentReports: this.reports
        .filter((r) => r.tenantId === tenantId)
        .slice(-5),
      lastUpdated: new Date(),
    };
  }

  /**
   * Get available frameworks
   */
  getFrameworks(): ComplianceFramework[] {
    return Array.from(this.frameworks.values());
  }

  /**
   * Shutdown reporter
   */
  async shutdown(): Promise<void> {
    this.emit("shutting-down");
    this.initialized = false;
    this.emit("shutdown");
  }

  // Private methods

  private loadDefaultFrameworks(): void {
    // SOC2 Framework
    this.frameworks.set("soc2", {
      id: "soc2",
      name: "SOC 2",
      version: "2017",
      controls: this.getSOC2Controls(),
    });

    // GDPR Framework
    this.frameworks.set("gdpr", {
      id: "gdpr",
      name: "GDPR",
      version: "2018",
      controls: this.getGDPRControls(),
    });

    // ISO 27001 Framework
    this.frameworks.set("iso27001", {
      id: "iso27001",
      name: "ISO 27001",
      version: "2022",
      controls: this.getISO27001Controls(),
    });
  }

  private getSOC2Controls(): SOC2Control[] {
    return [
      // Security
      {
        id: "CC1.1",
        name: "COSO Principle 1",
        category: "security",
        description:
          "The entity demonstrates a commitment to integrity and ethical values",
      },
      {
        id: "CC1.2",
        name: "COSO Principle 2",
        category: "security",
        description:
          "The board of directors demonstrates independence from management",
      },
      {
        id: "CC2.1",
        name: "COSO Principle 13",
        category: "security",
        description:
          "The entity obtains or generates and uses relevant, quality information",
      },
      {
        id: "CC3.1",
        name: "COSO Principle 6",
        category: "security",
        description: "The entity specifies objectives with sufficient clarity",
      },
      {
        id: "CC4.1",
        name: "COSO Principle 16",
        category: "security",
        description:
          "The entity selects, develops, and performs ongoing evaluations",
      },
      {
        id: "CC5.1",
        name: "COSO Principle 10",
        category: "security",
        description: "The entity selects and develops control activities",
      },
      {
        id: "CC6.1",
        name: "Logical and Physical Access",
        category: "security",
        description: "The entity implements logical access security software",
      },
      {
        id: "CC6.2",
        name: "System Authentication",
        category: "security",
        description: "Prior to issuing system credentials",
      },
      {
        id: "CC6.3",
        name: "Access Removal",
        category: "security",
        description:
          "The entity removes access to protected information assets",
      },
      {
        id: "CC6.6",
        name: "Threat Detection",
        category: "security",
        description:
          "The entity implements controls to prevent or detect threats",
      },
      {
        id: "CC6.7",
        name: "Data Transmission",
        category: "security",
        description: "The entity restricts the transmission of data",
      },
      {
        id: "CC6.8",
        name: "Malware Prevention",
        category: "security",
        description:
          "The entity implements controls to prevent malicious software",
      },
      {
        id: "CC7.1",
        name: "Vulnerability Management",
        category: "security",
        description:
          "To meet its objectives, the entity uses detection and monitoring",
      },
      {
        id: "CC7.2",
        name: "Security Monitoring",
        category: "security",
        description: "The entity monitors system components",
      },
      {
        id: "CC7.3",
        name: "Incident Response",
        category: "security",
        description: "The entity evaluates security events",
      },
      {
        id: "CC7.4",
        name: "Incident Communication",
        category: "security",
        description: "The entity responds to identified security incidents",
      },
      {
        id: "CC8.1",
        name: "Change Management",
        category: "security",
        description: "The entity authorizes, designs, develops changes",
      },
      {
        id: "CC9.1",
        name: "Risk Mitigation",
        category: "security",
        description:
          "The entity identifies, selects, and develops risk mitigation",
      },
      // Availability
      {
        id: "A1.1",
        name: "Capacity Planning",
        category: "availability",
        description:
          "The entity maintains, monitors, and evaluates current processing capacity",
      },
      {
        id: "A1.2",
        name: "Recovery Planning",
        category: "availability",
        description: "The entity authorizes, designs, develops backup/recovery",
      },
      {
        id: "A1.3",
        name: "Recovery Testing",
        category: "availability",
        description: "The entity tests recovery plan procedures",
      },
      // Processing Integrity
      {
        id: "PI1.1",
        name: "Processing Accuracy",
        category: "processing_integrity",
        description: "The entity obtains data from external sources",
      },
      {
        id: "PI1.2",
        name: "Data Quality",
        category: "processing_integrity",
        description: "The entity implements policies for data input integrity",
      },
      // Confidentiality
      {
        id: "C1.1",
        name: "Confidential Information",
        category: "confidentiality",
        description:
          "The entity identifies and maintains confidential information",
      },
      {
        id: "C1.2",
        name: "Data Disposal",
        category: "confidentiality",
        description: "The entity disposes of confidential information",
      },
      // Privacy
      {
        id: "P1.1",
        name: "Privacy Notice",
        category: "privacy",
        description: "The entity provides notice to data subjects",
      },
      {
        id: "P2.1",
        name: "Data Choice",
        category: "privacy",
        description: "The entity communicates choices to data subjects",
      },
      {
        id: "P3.1",
        name: "Personal Information",
        category: "privacy",
        description:
          "Personal information is collected consistent with objectives",
      },
      {
        id: "P4.1",
        name: "Data Use",
        category: "privacy",
        description: "The entity limits the use of personal information",
      },
      {
        id: "P5.1",
        name: "Data Access",
        category: "privacy",
        description:
          "The entity grants data subjects access to their personal information",
      },
      {
        id: "P6.1",
        name: "Data Disclosure",
        category: "privacy",
        description:
          "The entity discloses personal information to third parties",
      },
      {
        id: "P7.1",
        name: "Data Quality",
        category: "privacy",
        description:
          "The entity collects and maintains accurate personal information",
      },
      {
        id: "P8.1",
        name: "Complaint Management",
        category: "privacy",
        description:
          "The entity implements a process for receiving privacy complaints",
      },
    ];
  }

  private getGDPRControls(): SOC2Control[] {
    return [
      {
        id: "GDPR.5",
        name: "Lawfulness of Processing",
        category: "processing",
        description:
          "Personal data shall be processed lawfully, fairly and transparently",
      },
      {
        id: "GDPR.6",
        name: "Legal Basis",
        category: "processing",
        description: "Processing is lawful only if certain conditions are met",
      },
      {
        id: "GDPR.7",
        name: "Consent",
        category: "consent",
        description: "Conditions for consent",
      },
      {
        id: "GDPR.12",
        name: "Transparent Communication",
        category: "rights",
        description: "Transparent information, communication and modalities",
      },
      {
        id: "GDPR.13",
        name: "Information Provision",
        category: "rights",
        description: "Information to be provided where personal data collected",
      },
      {
        id: "GDPR.15",
        name: "Right of Access",
        category: "rights",
        description: "Right of access by the data subject",
      },
      {
        id: "GDPR.16",
        name: "Right to Rectification",
        category: "rights",
        description: "Right to rectification",
      },
      {
        id: "GDPR.17",
        name: "Right to Erasure",
        category: "rights",
        description: "Right to erasure (right to be forgotten)",
      },
      {
        id: "GDPR.20",
        name: "Data Portability",
        category: "rights",
        description: "Right to data portability",
      },
      {
        id: "GDPR.25",
        name: "Privacy by Design",
        category: "design",
        description: "Data protection by design and by default",
      },
      {
        id: "GDPR.30",
        name: "Records of Processing",
        category: "documentation",
        description: "Records of processing activities",
      },
      {
        id: "GDPR.32",
        name: "Security of Processing",
        category: "security",
        description: "Security of processing",
      },
      {
        id: "GDPR.33",
        name: "Breach Notification",
        category: "breach",
        description: "Notification of a personal data breach",
      },
      {
        id: "GDPR.35",
        name: "Impact Assessment",
        category: "assessment",
        description: "Data protection impact assessment",
      },
      {
        id: "GDPR.37",
        name: "DPO Designation",
        category: "governance",
        description: "Designation of the data protection officer",
      },
      {
        id: "GDPR.44",
        name: "Transfer Restrictions",
        category: "transfers",
        description: "General principle for transfers",
      },
    ];
  }

  private getISO27001Controls(): SOC2Control[] {
    return [
      {
        id: "A.5",
        name: "Information Security Policies",
        category: "policy",
        description: "Management direction for information security",
      },
      {
        id: "A.6",
        name: "Organization of Information Security",
        category: "organization",
        description: "Internal organization and mobile devices",
      },
      {
        id: "A.7",
        name: "Human Resource Security",
        category: "hr",
        description: "Prior to, during, and termination of employment",
      },
      {
        id: "A.8",
        name: "Asset Management",
        category: "assets",
        description: "Responsibility for assets and information classification",
      },
      {
        id: "A.9",
        name: "Access Control",
        category: "access",
        description: "Business requirements and user access management",
      },
      {
        id: "A.10",
        name: "Cryptography",
        category: "crypto",
        description: "Cryptographic controls",
      },
      {
        id: "A.11",
        name: "Physical Security",
        category: "physical",
        description: "Secure areas and equipment",
      },
      {
        id: "A.12",
        name: "Operations Security",
        category: "operations",
        description: "Operational procedures and responsibilities",
      },
      {
        id: "A.13",
        name: "Communications Security",
        category: "communications",
        description: "Network security management",
      },
      {
        id: "A.14",
        name: "System Acquisition",
        category: "development",
        description: "Security requirements and development",
      },
      {
        id: "A.15",
        name: "Supplier Relationships",
        category: "suppliers",
        description: "Information security in supplier relationships",
      },
      {
        id: "A.16",
        name: "Incident Management",
        category: "incidents",
        description: "Management of security incidents",
      },
      {
        id: "A.17",
        name: "Business Continuity",
        category: "continuity",
        description: "Information security continuity",
      },
      {
        id: "A.18",
        name: "Compliance",
        category: "compliance",
        description: "Compliance with legal and contractual requirements",
      },
    ];
  }

  private async evaluateControl(
    control: SOC2Control,
    tenantId: string,
    auditEntries: AuditEntry[],
    configData: Record<string, unknown>
  ): Promise<ControlEvaluation> {
    const evidence: Evidence[] = [];
    const findings: Finding[] = [];
    let score = 0;

    // Collect evidence based on control type
    const relevantEntries = auditEntries.filter((e) =>
      this.isRelevantToControl(e, control)
    );

    if (relevantEntries.length > 0) {
      evidence.push({
        id: this.generateEvidenceId(),
        type: "audit_log",
        description: `${relevantEntries.length} audit entries related to ${control.name}`,
        reference: `audit:${control.id}`,
        collectedAt: new Date(),
        collectedBy: "system",
      });
      score += 40;
    }

    // Check configuration
    const configCheck = this.checkControlConfig(control, configData);
    if (configCheck.configured) {
      evidence.push({
        id: this.generateEvidenceId(),
        type: "configuration",
        description: configCheck.description,
        reference: `config:${control.id}`,
        collectedAt: new Date(),
        collectedBy: "system",
      });
      score += 40;
    } else {
      findings.push({
        id: this.generateFindingId(),
        severity: "medium",
        title: `${control.name} not properly configured`,
        description: configCheck.description,
        controlId: control.id,
        remediation: configCheck.remediation || [],
        status: "open",
      });
    }

    // Additional scoring based on evidence quality
    if (evidence.length >= 2) score += 20;

    // Determine status
    let status: ControlStatus = "not_evaluated";
    if (score >= 80) status = "passed";
    else if (score >= 50) status = "warning";
    else if (score > 0) status = "failed";

    return {
      controlId: control.id,
      controlName: control.name,
      status,
      score: Math.min(100, score),
      evidence,
      findings,
      recommendations: this.getControlRecommendations(control, status),
      evaluatedAt: new Date(),
    };
  }

  private isRelevantToControl(
    entry: AuditEntry,
    control: SOC2Control
  ): boolean {
    // Map control categories to audit event types
    const categoryMapping: Record<string, string[]> = {
      security: ["auth", "security", "config_change"],
      availability: ["system", "config_change"],
      processing_integrity: ["data_access", "system"],
      confidentiality: ["data_access", "security"],
      privacy: ["data_access", "config_change"],
    };

    const relevantTypes = categoryMapping[control.category] || [];
    return relevantTypes.includes(entry.eventType);
  }

  private checkControlConfig(
    control: SOC2Control,
    configData: Record<string, unknown>
  ): { configured: boolean; description: string; remediation?: string[] } {
    // Simplified config check - would be more detailed in production
    const configKey = control.id.toLowerCase().replace(/\./g, "_");
    const isConfigured = configData[configKey] !== undefined;

    return {
      configured: isConfigured,
      description: isConfigured
        ? `Configuration for ${control.name} is present`
        : `Configuration for ${control.name} is missing`,
      remediation: isConfigured
        ? undefined
        : [
            `Configure ${control.id} settings`,
            "Review control requirements",
            "Implement necessary controls",
          ],
    };
  }

  private getControlRecommendations(
    control: SOC2Control,
    status: ControlStatus
  ): string[] {
    if (status === "passed") return [];

    return [
      `Review ${control.name} implementation`,
      `Ensure ${control.description}`,
      "Document evidence of control operation",
      "Schedule periodic review",
    ];
  }

  private determineOverallStatus(
    evaluations: ControlEvaluation[]
  ): ComplianceStatus {
    const failed = evaluations.filter((e) => e.status === "failed").length;
    const warning = evaluations.filter((e) => e.status === "warning").length;
    const passed = evaluations.filter((e) => e.status === "passed").length;

    if (failed > 0) return "non_compliant";
    if (warning > evaluations.length * 0.2) return "partial";
    if (passed >= evaluations.length * 0.8) return "compliant";
    return "partial";
  }

  private generateRecommendations(evaluations: ControlEvaluation[]): string[] {
    const recommendations: string[] = [];

    const failed = evaluations.filter((e) => e.status === "failed");
    const warning = evaluations.filter((e) => e.status === "warning");

    if (failed.length > 0) {
      recommendations.push(
        `Address ${failed.length} failed controls immediately`
      );
    }

    if (warning.length > 0) {
      recommendations.push(`Review ${warning.length} controls with warnings`);
    }

    recommendations.push("Schedule regular compliance assessments");
    recommendations.push("Maintain evidence collection procedures");

    return recommendations;
  }

  private groupByTrustServiceCriteria(
    evaluations: ControlEvaluation[]
  ): Record<string, ControlEvaluation[]> {
    const grouped: Record<string, ControlEvaluation[]> = {};

    for (const evaluation of evaluations) {
      const control = this.findControl("soc2", evaluation.controlId);
      if (control) {
        const category = control.category;
        if (!grouped[category]) grouped[category] = [];
        grouped[category].push(evaluation);
      }
    }

    return grouped;
  }

  private findControl(
    frameworkId: string,
    controlId: string
  ): SOC2Control | undefined {
    const framework = this.frameworks.get(frameworkId);
    return framework?.controls.find((c) => c.id === controlId);
  }

  private generateAuditorOpinion(report: ComplianceReport): string {
    if (report.status === "compliant") {
      return "In our opinion, the controls were suitably designed and operating effectively.";
    } else if (report.status === "partial") {
      return "In our opinion, except for the matters described, controls were suitably designed.";
    } else {
      return "Significant control deficiencies were identified that require remediation.";
    }
  }

  private checkRetentionPolicies(evaluations: ControlEvaluation[]): boolean {
    const retentionControl = evaluations.find(
      (e) => e.controlId.includes("retention") || e.controlId === "GDPR.5"
    );
    return retentionControl?.status === "passed";
  }

  private generateReportId(): string {
    return `report_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  private generateFindingId(): string {
    return `finding_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  private generateEvidenceId(): string {
    return `evidence_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }
}

/**
 * Configuration for compliance reporter
 */
export interface ComplianceReporterConfig {
  customFrameworks?: ComplianceFramework[];
  autoEvaluate?: boolean;
  evaluationInterval?: number;
}

/**
 * SOC2 specific report
 */
export interface SOC2Report extends ComplianceReport {
  soc2Type: "type1" | "type2";
  periodStart: Date;
  periodEnd: Date;
  trustServiceCriteria: {
    security: number;
    availability: number;
    processingIntegrity: number;
    confidentiality: number;
    privacy: number;
  };
  auditPeriodDays: number;
  auditorOpinion: string;
}

/**
 * GDPR specific report
 */
export interface GDPRReport extends ComplianceReport {
  dataSubjectRequestsProcessed: number;
  consentRecordsCount: number;
  dataBreachesReported: number;
  dataProtectionOfficer: string;
  dataProcessingAgreements: string[];
  crossBorderTransfers: string[];
  retentionPoliciesCompliant: boolean;
}

/**
 * Compliance dashboard data
 */
export interface ComplianceDashboard {
  tenantId: string;
  frameworkScores: Record<string, number>;
  overallScore: number;
  totalControls: number;
  passedControls: number;
  failedControls: number;
  openFindings: number;
  criticalFindings: number;
  overdueFindings: number;
  recentReports: ComplianceReport[];
  lastUpdated: Date;
}

/**
 * Factory function
 */
export function createComplianceReporter(
  config: ComplianceReporterConfig
): ComplianceReporter {
  return new ComplianceReporter(config);
}

export default ComplianceReporter;
