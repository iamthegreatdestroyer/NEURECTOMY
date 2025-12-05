/**
 * @neurectomy/enterprise
 *
 * Enterprise-grade features for NEURECTOMY:
 * - SSO/SAML/OAuth Authentication
 * - Multi-Tenancy with Isolation
 * - SOC2/GDPR/ISO 27001 Compliance
 * - Tamper-Proof Audit Logging
 * - Role-Based Access Control (RBAC)
 * - Enterprise Deployment Management
 *
 * Agent Assignment: @AEGIS @CIPHER @FLUX @FORTRESS
 *
 * @author NEURECTOMY Phase 5 - Enterprise Excellence
 * @version 1.0.0
 */

// ============================================================================
// Types
// ============================================================================

export * from "./types";

// ============================================================================
// Authentication (SSO/SAML/OAuth)
// ============================================================================

export {
  // SSO Provider
  SSOProvider,
  createSSOProvider,
  type SSOConfig,
  type SSOProviderConfig,
  type SSOSession,
  type SSOUser,
  type SSOProviderType,

  // OAuth Handler
  OAuthHandler,
  createOAuthHandler,
  type OAuthConfig,
  type OAuthProviderConfig,
  type OAuthTokens,
  type OAuthUserInfo,
  type OAuthProvider,

  // SAML Handler
  SAMLHandler,
  createSAMLHandler,
  type SAMLConfig,
  type SAMLAssertion,
  type SAMLAttribute,
  type SAMLMetadata,
} from "./auth";

// ============================================================================
// Multi-Tenancy
// ============================================================================

export {
  // Tenant Manager
  TenantManager,
  createTenantManager,
  type TenantManagerConfig,
  type Tenant,
  type TenantStatus,
  type TenantTier,
  type TenantSettings,
  type TenantFeatures,

  // Tenant Isolation
  TenantIsolationEngine,
  createTenantIsolationEngine,
  type TenantIsolationConfig,
  type IsolationLevel,
  type IsolationBoundary,
  type AccessDecision,
  type IsolationViolation,

  // Resource Limits
  ResourceLimiter,
  createResourceLimiter,
  type ResourceLimiterConfig,
  type ResourceQuota,
  type ResourceUsage,
  type QuotaExceededError,
  type RateLimitResult,

  // Factory
  createTenancySystem,
  type TenancySystem,
  type TenancySystemConfig,
} from "./tenancy";

// ============================================================================
// Audit Logging & Compliance
// ============================================================================

export {
  // Audit Logger
  AuditLogger,
  createAuditLogger,
  type AuditLoggerConfig,
  type AuditEntry,
  type AuditEntryType,
  type AuditBlock,
  type IntegrityVerification,

  // Compliance Reporter
  ComplianceReporter,
  createComplianceReporter,
  type ComplianceReporterConfig,
  type ComplianceFramework,
  type ComplianceControl,
  type ComplianceEvidence,
  type ComplianceReport,
  type ComplianceFinding,
  type SOC2Report,
  type GDPRReport,

  // Factory
  createAuditSystem,
  type AuditSystem,
  type AuditSystemConfig,
} from "./audit";

// ============================================================================
// Role-Based Access Control (RBAC)
// ============================================================================

export {
  // Role Manager
  RoleManager,
  createRoleManager,
  type RoleManagerConfig,
  type Role,
  type Permission,
  type RoleAssignment,
  type RoleHierarchy,

  // Permission Evaluator
  PermissionEvaluator,
  createPermissionEvaluator,
  type PermissionEvaluatorConfig,
  type Policy,
  type PolicyCondition,
  type EvaluationResult,
  type AccessContext,

  // Factory
  createRBACSystem,
  checkAccess,
  type RBACSystem,
  type RBACSystemConfig,
} from "./rbac";

// ============================================================================
// Deployment Management
// ============================================================================

export {
  // Deployment Manager
  DeploymentManager,
  createDeploymentManager,
  type DeploymentManagerConfig,
  type DeploymentConfig,
  type Deployment,
  type DeploymentStatus,
  type DeploymentStrategy,
  type DeploymentStage,
  type DeploymentEvent,
  type DeploymentEventType,
  type DeploymentMetrics,
  type DeploymentGate,
  type GateType,
  type HealthCheckConfig,
  DEFAULT_DEPLOYMENT_MANAGER_CONFIG,

  // Rollback Manager
  RollbackManager,
  createRollbackManager,
  type RollbackManagerConfig,
  type RollbackConfig,
  type RollbackOperation,
  type RollbackStatus,
  type RollbackStrategy,
  type RollbackTrigger,
  type DeploymentVersion,
  type VersionMetrics,
  DEFAULT_ROLLBACK_CONFIG,
  DEFAULT_ROLLBACK_MANAGER_CONFIG,

  // Factory
  createDeploymentSystem,
  type DeploymentSystem,
  type DeploymentSystemConfig,
} from "./deployment";

// ============================================================================
// Scalability Infrastructure
// ============================================================================

export {
  // Database Sharding
  DatabaseShardingManager,
  createDatabaseShardingManager,
  createDefaultShardingConfig,

  // Failover Automation
  FailoverAutomationManager,
  createFailoverAutomationManager,
  createDefaultFailoverConfig,

  // Types
  type ShardConfig,
  type ShardingRule,
  type ShardAssignment,
  type ShardingConfig,
  type FailoverStrategy,
  type NodeConfig,
  type HealthCheckConfig as FailoverHealthCheckConfig,
  type FailoverPolicy,
  type FailoverEvent,
  type ScalingPolicy,
  type ScalingEvent,
  type InstanceConfig,
  type LoadBalancerConfig,
  type BackendTarget,
  type RateLimitRule,
  type RateLimitState,
  type CircuitBreakerConfig,
  type CircuitBreakerStatus,
  type ReplicaConfig,
  type ReplicationConfig,
  type RPOConfig,
  type RTOConfig,
  type BackupConfig,
  type DisasterRecoveryConfig,
  type HorizontalScalingConfig,
  type ScalabilityConfig,
  type ScalabilityEngineEvents,
} from "./scalability";

// ============================================================================
// Unified Enterprise Factory
// ============================================================================

import { SSOProvider, SSOConfig, createSSOProvider } from "./auth";
import {
  createTenancySystem,
  TenancySystemConfig,
  TenancySystem,
} from "./tenancy";
import { createAuditSystem, AuditSystemConfig, AuditSystem } from "./audit";
import { createRBACSystem, RBACSystemConfig, RBACSystem } from "./rbac";
import {
  createDeploymentSystem,
  DeploymentSystemConfig,
  DeploymentSystem,
} from "./deployment";

/**
 * Enterprise System Configuration
 */
export interface EnterpriseConfig {
  /** SSO/Authentication configuration */
  sso?: SSOConfig;
  /** Multi-tenancy configuration */
  tenancy?: TenancySystemConfig;
  /** Audit and compliance configuration */
  audit?: AuditSystemConfig;
  /** RBAC configuration */
  rbac?: RBACSystemConfig;
  /** Deployment configuration */
  deployment?: DeploymentSystemConfig;
}

/**
 * Complete Enterprise System
 */
export interface EnterpriseSystem {
  /** SSO Provider for authentication */
  sso: SSOProvider;
  /** Multi-tenancy management */
  tenancy: TenancySystem;
  /** Audit logging and compliance */
  audit: AuditSystem;
  /** Role-based access control */
  rbac: RBACSystem;
  /** Deployment and rollback management */
  deployment: DeploymentSystem;
}

/**
 * Create a complete enterprise system with all components integrated
 *
 * @example
 * ```typescript
 * const enterprise = createEnterpriseSystem({
 *   sso: {
 *     providers: [{ type: 'oauth', provider: 'google', ... }],
 *   },
 *   tenancy: {
 *     isolation: { defaultLevel: 'strict' },
 *   },
 *   audit: {
 *     logger: { retentionDays: 365 },
 *   },
 *   rbac: {
 *     roles: { defaultRoles: true },
 *   },
 * });
 *
 * // Authenticate user
 * const session = await enterprise.sso.authenticate(provider, code);
 *
 * // Check access
 * const canAccess = await enterprise.rbac.checkAccess(
 *   session.user.id,
 *   'read',
 *   'projects'
 * );
 *
 * // Log action
 * await enterprise.audit.log({
 *   type: 'data_access',
 *   action: 'read',
 *   resource: 'projects',
 *   userId: session.user.id,
 * });
 * ```
 */
export function createEnterpriseSystem(
  config: EnterpriseConfig = {}
): EnterpriseSystem {
  // Create individual subsystems
  const sso = createSSOProvider(config.sso);
  const tenancy = createTenancySystem(config.tenancy);
  const audit = createAuditSystem(config.audit);
  const rbac = createRBACSystem(config.rbac);
  const deployment = createDeploymentSystem(config.deployment);

  // Wire up cross-system integrations

  // SSO events -> Audit logging
  sso.on("session.created", (event) => {
    audit.auditLogger.log({
      type: "auth",
      action: "login",
      userId: event.user?.id || "unknown",
      resource: "session",
      resourceId: event.sessionId,
      metadata: {
        provider: event.provider,
        timestamp: new Date().toISOString(),
      },
    });
  });

  sso.on("session.destroyed", (event) => {
    audit.auditLogger.log({
      type: "auth",
      action: "logout",
      userId: event.userId || "unknown",
      resource: "session",
      resourceId: event.sessionId,
      metadata: {
        timestamp: new Date().toISOString(),
      },
    });
  });

  // Tenant events -> Audit logging
  tenancy.tenantManager.on("tenant.created", (event) => {
    audit.auditLogger.log({
      type: "admin",
      action: "create",
      userId: event.createdBy || "system",
      resource: "tenant",
      resourceId: event.tenantId,
      metadata: {
        name: event.name,
        tier: event.tier,
      },
    });
  });

  tenancy.tenantManager.on("tenant.suspended", (event) => {
    audit.auditLogger.log({
      type: "admin",
      action: "suspend",
      userId: event.suspendedBy || "system",
      resource: "tenant",
      resourceId: event.tenantId,
      metadata: {
        reason: event.reason,
      },
    });
  });

  // RBAC events -> Audit logging
  rbac.roleManager.on("role.assigned", (event) => {
    audit.auditLogger.log({
      type: "security",
      action: "role_assigned",
      userId: event.assignedBy || "system",
      resource: "role",
      resourceId: event.roleId,
      metadata: {
        targetUser: event.userId,
        role: event.roleName,
      },
    });
  });

  rbac.roleManager.on("role.revoked", (event) => {
    audit.auditLogger.log({
      type: "security",
      action: "role_revoked",
      userId: event.revokedBy || "system",
      resource: "role",
      resourceId: event.roleId,
      metadata: {
        targetUser: event.userId,
        role: event.roleName,
      },
    });
  });

  // Deployment events -> Audit logging
  deployment.deploymentManager.on("deployment.completed", (event) => {
    audit.auditLogger.log({
      type: "config_change",
      action: "deployment_completed",
      userId: event.deployedBy || "system",
      resource: "deployment",
      resourceId: event.deploymentId,
      metadata: {
        version: event.data?.version,
        environment: event.data?.environment,
      },
    });
  });

  deployment.rollbackManager.on("rollback.completed", (event) => {
    audit.auditLogger.log({
      type: "config_change",
      action: "rollback_completed",
      userId: event.initiatedBy || "system",
      resource: "deployment",
      resourceId: event.operationId,
      metadata: {
        fromVersion: event.data?.fromVersion,
        toVersion: event.data?.toVersion,
      },
    });
  });

  return {
    sso,
    tenancy,
    audit,
    rbac,
    deployment,
  };
}

// ============================================================================
// Re-export common utilities
// ============================================================================

export { EventEmitter } from "events";
