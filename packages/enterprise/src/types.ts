/**
 * @fileoverview Enterprise Package Type Definitions
 * @module @neurectomy/enterprise/types
 *
 * @elite-agent-collective @CIPHER @SYNAPSE @AEGIS @ARCHITECT @FORTRESS
 *
 * Comprehensive type definitions for enterprise features:
 * - SSO/SAML Authentication
 * - Multi-Tenancy Architecture
 * - SOC2 Compliance Automation
 * - Audit Logging System
 *
 * @author NEURECTOMY Phase 5 - Enterprise Excellence
 * @version 1.0.0
 */

// ============================================================================
// Authentication Types (@CIPHER @SYNAPSE)
// ============================================================================

/**
 * Supported authentication providers
 */
export type AuthProvider =
  | "saml"
  | "oauth2"
  | "oidc"
  | "ldap"
  | "local"
  | "api-key"
  | "jwt";

/**
 * OAuth2 grant types
 */
export type OAuth2GrantType =
  | "authorization_code"
  | "client_credentials"
  | "refresh_token"
  | "device_code"
  | "pkce";

/**
 * SAML binding types
 */
export type SAMLBinding = "HTTP-POST" | "HTTP-Redirect" | "Artifact";

/**
 * Session state
 */
export type SessionState =
  | "active"
  | "expired"
  | "revoked"
  | "locked"
  | "pending_mfa";

/**
 * MFA method types
 */
export type MFAMethod =
  | "totp"
  | "sms"
  | "email"
  | "hardware_key"
  | "push"
  | "biometric";

/**
 * User identity from any auth provider
 */
export interface UserIdentity {
  id: string;
  externalId?: string;
  email: string;
  emailVerified: boolean;
  displayName: string;
  firstName?: string;
  lastName?: string;
  avatar?: string;
  provider: AuthProvider;
  providerId?: string;
  tenantId: string;
  roles: string[];
  permissions: string[];
  groups: string[];
  attributes: Record<string, unknown>;
  metadata: IdentityMetadata;
}

/**
 * Identity metadata
 */
export interface IdentityMetadata {
  createdAt: Date;
  updatedAt: Date;
  lastLoginAt?: Date;
  lastActivityAt?: Date;
  loginCount: number;
  failedLoginAttempts: number;
  lockedUntil?: Date;
  mfaEnabled: boolean;
  mfaMethods: MFAMethod[];
  passwordChangedAt?: Date;
  passwordExpiresAt?: Date;
}

/**
 * Authentication session
 */
export interface AuthSession {
  id: string;
  userId: string;
  tenantId: string;
  state: SessionState;
  provider: AuthProvider;
  accessToken: string;
  refreshToken?: string;
  idToken?: string;
  tokenType: string;
  expiresAt: Date;
  refreshExpiresAt?: Date;
  scopes: string[];
  ipAddress: string;
  userAgent: string;
  deviceId?: string;
  createdAt: Date;
  lastAccessAt: Date;
  mfaVerified: boolean;
  metadata: Record<string, unknown>;
}

/**
 * SSO Provider configuration
 */
export interface SSOProviderConfig {
  id: string;
  name: string;
  type: AuthProvider;
  enabled: boolean;
  tenantIds: string[];
  config: SAMLConfig | OAuth2Config | OIDCConfig | LDAPConfig;
  attributeMapping: AttributeMapping;
  defaultRoles: string[];
  autoProvision: boolean;
  jitProvisioning: boolean;
}

/**
 * SAML configuration
 */
export interface SAMLConfig {
  entityId: string;
  ssoUrl: string;
  sloUrl?: string;
  certificate: string;
  privateKey?: string;
  binding: SAMLBinding;
  signRequests: boolean;
  signAssertions: boolean;
  encryptAssertions: boolean;
  wantAssertionsSigned: boolean;
  allowUnencrypted: boolean;
  nameIdFormat: string;
  authnContextClassRef?: string;
  forceAuthn: boolean;
  metadata?: string;
}

/**
 * OAuth2 configuration
 */
export interface OAuth2Config {
  clientId: string;
  clientSecret: string;
  authorizationUrl: string;
  tokenUrl: string;
  userInfoUrl?: string;
  redirectUri: string;
  scopes: string[];
  grantType: OAuth2GrantType;
  pkceEnabled: boolean;
  pkceMethod?: "S256" | "plain";
  audience?: string;
  responseType: string;
  responseMode?: string;
}

/**
 * OpenID Connect configuration
 */
export interface OIDCConfig extends OAuth2Config {
  issuer: string;
  jwksUri: string;
  discoveryUrl?: string;
  idTokenSignedResponseAlg: string;
  userInfoSignedResponseAlg?: string;
  requestObjectSigningAlg?: string;
  claims?: string[];
}

/**
 * LDAP configuration
 */
export interface LDAPConfig {
  url: string;
  baseDN: string;
  bindDN: string;
  bindPassword: string;
  searchFilter: string;
  searchScope: "base" | "one" | "sub";
  tlsEnabled: boolean;
  tlsOptions?: {
    rejectUnauthorized: boolean;
    ca?: string;
    cert?: string;
    key?: string;
  };
  groupSearchBase?: string;
  groupSearchFilter?: string;
  groupMemberAttribute?: string;
  userAttributes: string[];
  timeout: number;
  reconnectDelay: number;
}

/**
 * Attribute mapping from provider to internal
 */
export interface AttributeMapping {
  id: string;
  email: string;
  displayName: string;
  firstName?: string;
  lastName?: string;
  groups?: string;
  roles?: string;
  avatar?: string;
  custom: Record<string, string>;
}

// ============================================================================
// Multi-Tenancy Types (@ARCHITECT @FORTRESS)
// ============================================================================

/**
 * Tenant isolation level
 */
export type TenantIsolationLevel =
  | "shared" // Shared infrastructure, data segregation
  | "pool" // Pool of shared resources with soft limits
  | "silo" // Dedicated resources per tenant
  | "hybrid"; // Mix of shared and dedicated

/**
 * Tenant status
 */
export type TenantStatus =
  | "active"
  | "suspended"
  | "pending"
  | "trial"
  | "deactivated"
  | "migrating";

/**
 * Tenant subscription tier
 */
export type SubscriptionTier =
  | "free"
  | "starter"
  | "professional"
  | "enterprise"
  | "custom";

/**
 * Resource limit enforcement
 */
export type LimitEnforcement = "soft" | "hard" | "notify" | "throttle";

/**
 * Tenant definition
 */
export interface Tenant {
  id: string;
  name: string;
  slug: string;
  domain?: string;
  customDomains: string[];
  status: TenantStatus;
  isolationLevel: TenantIsolationLevel;
  subscription: TenantSubscription;
  settings: TenantSettings;
  features: TenantFeatures;
  resourceLimits: ResourceLimits;
  metadata: TenantMetadata;
}

/**
 * Tenant subscription details
 */
export interface TenantSubscription {
  tier: SubscriptionTier;
  startDate: Date;
  endDate?: Date;
  trialEndsAt?: Date;
  billingCycle: "monthly" | "annual" | "custom";
  status: "active" | "past_due" | "canceled" | "trialing";
  features: string[];
  addons: string[];
  customPricing?: Record<string, number>;
}

/**
 * Tenant-specific settings
 */
export interface TenantSettings {
  branding: BrandingSettings;
  security: SecuritySettings;
  notifications: NotificationSettings;
  integrations: IntegrationSettings;
  locale: LocaleSettings;
  custom: Record<string, unknown>;
}

/**
 * Branding settings
 */
export interface BrandingSettings {
  logo?: string;
  favicon?: string;
  primaryColor?: string;
  secondaryColor?: string;
  accentColor?: string;
  customCSS?: string;
  emailTemplate?: string;
}

/**
 * Security settings per tenant
 */
export interface SecuritySettings {
  mfaRequired: boolean;
  mfaMethods: MFAMethod[];
  passwordPolicy: PasswordPolicy;
  sessionTimeout: number;
  maxConcurrentSessions: number;
  ipWhitelist: string[];
  ipBlacklist: string[];
  allowedAuthProviders: AuthProvider[];
  apiRateLimits: RateLimitConfig;
  dataRetentionDays: number;
}

/**
 * Password policy
 */
export interface PasswordPolicy {
  minLength: number;
  maxLength: number;
  requireUppercase: boolean;
  requireLowercase: boolean;
  requireNumbers: boolean;
  requireSpecialChars: boolean;
  preventReuse: number;
  expirationDays: number;
  lockoutAttempts: number;
  lockoutDuration: number;
}

/**
 * Rate limit configuration
 */
export interface RateLimitConfig {
  requestsPerMinute: number;
  requestsPerHour: number;
  requestsPerDay: number;
  burstLimit: number;
  throttleDelay: number;
}

/**
 * Notification settings
 */
export interface NotificationSettings {
  emailEnabled: boolean;
  slackEnabled: boolean;
  webhookEnabled: boolean;
  webhookUrl?: string;
  alertEmails: string[];
  digestFrequency: "realtime" | "hourly" | "daily" | "weekly";
}

/**
 * Integration settings
 */
export interface IntegrationSettings {
  ssoProviders: string[];
  apiKeys: APIKeyConfig[];
  webhooks: WebhookConfig[];
  connectors: ConnectorConfig[];
}

/**
 * API key configuration
 */
export interface APIKeyConfig {
  id: string;
  name: string;
  prefix: string;
  scopes: string[];
  rateLimit: RateLimitConfig;
  expiresAt?: Date;
  lastUsedAt?: Date;
  createdBy: string;
}

/**
 * Webhook configuration
 */
export interface WebhookConfig {
  id: string;
  name: string;
  url: string;
  events: string[];
  secret: string;
  enabled: boolean;
  retryPolicy: RetryPolicy;
  headers: Record<string, string>;
}

/**
 * Retry policy
 */
export interface RetryPolicy {
  maxRetries: number;
  initialDelay: number;
  maxDelay: number;
  backoffMultiplier: number;
}

/**
 * Connector configuration
 */
export interface ConnectorConfig {
  id: string;
  type: string;
  name: string;
  config: Record<string, unknown>;
  enabled: boolean;
  status: "connected" | "disconnected" | "error";
  lastSyncAt?: Date;
}

/**
 * Locale settings
 */
export interface LocaleSettings {
  language: string;
  timezone: string;
  dateFormat: string;
  timeFormat: string;
  currency: string;
  numberFormat: string;
}

/**
 * Tenant feature flags
 */
export interface TenantFeatures {
  flags: Record<string, boolean>;
  limits: Record<string, number>;
  experiments: string[];
  customFeatures: Record<string, unknown>;
}

/**
 * Resource limits for tenant
 */
export interface ResourceLimits {
  users: ResourceLimit;
  storage: ResourceLimit;
  apiRequests: ResourceLimit;
  computeUnits: ResourceLimit;
  bandwidth: ResourceLimit;
  projects: ResourceLimit;
  customLimits: Record<string, ResourceLimit>;
}

/**
 * Individual resource limit
 */
export interface ResourceLimit {
  limit: number;
  used: number;
  unit: string;
  enforcement: LimitEnforcement;
  resetPeriod?: "hourly" | "daily" | "monthly" | "never";
  overage?: OveragePolicy;
}

/**
 * Overage policy
 */
export interface OveragePolicy {
  allowed: boolean;
  maxOverage: number;
  costPerUnit: number;
  notifyAt: number[];
}

/**
 * Tenant metadata
 */
export interface TenantMetadata {
  createdAt: Date;
  updatedAt: Date;
  createdBy: string;
  industry?: string;
  companySize?: string;
  country?: string;
  tags: string[];
  notes?: string;
}

// ============================================================================
// SOC2 Compliance Types (@AEGIS @SENTRY)
// ============================================================================

/**
 * SOC2 Trust Service Criteria categories
 */
export type SOC2Category =
  | "security"
  | "availability"
  | "processing_integrity"
  | "confidentiality"
  | "privacy";

/**
 * Control status
 */
export type ControlStatus =
  | "compliant"
  | "non_compliant"
  | "partial"
  | "not_applicable"
  | "pending_review"
  | "remediation_in_progress";

/**
 * Evidence type
 */
export type EvidenceType =
  | "policy"
  | "procedure"
  | "screenshot"
  | "log"
  | "report"
  | "configuration"
  | "attestation"
  | "audit_trail";

/**
 * Risk level
 */
export type RiskLevel = "critical" | "high" | "medium" | "low" | "info";

/**
 * SOC2 Control definition
 */
export interface SOC2Control {
  id: string;
  code: string;
  name: string;
  description: string;
  category: SOC2Category;
  criteria: string[];
  status: ControlStatus;
  owner: string;
  reviewer: string;
  implementation: ControlImplementation;
  evidence: ControlEvidence[];
  testResults: TestResult[];
  riskAssessment: RiskAssessment;
  exceptions: ControlException[];
  lastReviewDate: Date;
  nextReviewDate: Date;
  metadata: Record<string, unknown>;
}

/**
 * Control implementation details
 */
export interface ControlImplementation {
  type: "automated" | "manual" | "hybrid";
  description: string;
  procedures: string[];
  tools: string[];
  frequency:
    | "continuous"
    | "daily"
    | "weekly"
    | "monthly"
    | "quarterly"
    | "annual";
  automationScript?: string;
  integrations: string[];
}

/**
 * Control evidence
 */
export interface ControlEvidence {
  id: string;
  type: EvidenceType;
  title: string;
  description: string;
  url?: string;
  content?: string;
  hash: string;
  collectedAt: Date;
  collectedBy: string;
  validUntil: Date;
  verified: boolean;
  verifiedBy?: string;
  verifiedAt?: Date;
  tags: string[];
}

/**
 * Test result
 */
export interface TestResult {
  id: string;
  testId: string;
  testName: string;
  status: "pass" | "fail" | "error" | "skipped";
  executedAt: Date;
  executedBy: string;
  duration: number;
  findings: TestFinding[];
  evidence: string[];
  notes?: string;
}

/**
 * Test finding
 */
export interface TestFinding {
  id: string;
  title: string;
  description: string;
  severity: RiskLevel;
  recommendation: string;
  remediation?: string;
  status: "open" | "in_progress" | "resolved" | "accepted";
  dueDate?: Date;
  assignee?: string;
}

/**
 * Risk assessment
 */
export interface RiskAssessment {
  inherentRisk: RiskLevel;
  residualRisk: RiskLevel;
  likelihood: number;
  impact: number;
  riskScore: number;
  mitigations: string[];
  acceptedRisks: string[];
  lastAssessmentDate: Date;
}

/**
 * Control exception
 */
export interface ControlException {
  id: string;
  reason: string;
  approvedBy: string;
  approvedAt: Date;
  expiresAt: Date;
  compensatingControls: string[];
  riskAcceptance: string;
  reviewRequired: boolean;
}

/**
 * Compliance report
 */
export interface ComplianceReport {
  id: string;
  type: "soc2_type1" | "soc2_type2" | "gap_assessment" | "readiness";
  title: string;
  period: {
    startDate: Date;
    endDate: Date;
  };
  scope: string[];
  categories: SOC2Category[];
  summary: ComplianceSummary;
  controls: SOC2Control[];
  findings: TestFinding[];
  recommendations: string[];
  generatedAt: Date;
  generatedBy: string;
  status: "draft" | "review" | "final" | "published";
}

/**
 * Compliance summary
 */
export interface ComplianceSummary {
  totalControls: number;
  compliantControls: number;
  nonCompliantControls: number;
  partialControls: number;
  notApplicable: number;
  complianceScore: number;
  riskScore: number;
  criticalFindings: number;
  highFindings: number;
  mediumFindings: number;
  lowFindings: number;
  byCategory: Record<SOC2Category, CategorySummary>;
}

/**
 * Category summary
 */
export interface CategorySummary {
  total: number;
  compliant: number;
  nonCompliant: number;
  partial: number;
  score: number;
}

// ============================================================================
// Audit Logging Types (@AEGIS @CRYPTO)
// ============================================================================

/**
 * Audit event types
 */
export type AuditEventType =
  | "authentication"
  | "authorization"
  | "data_access"
  | "data_modification"
  | "configuration_change"
  | "admin_action"
  | "system_event"
  | "security_event"
  | "compliance_event"
  | "integration_event";

/**
 * Audit event outcome
 */
export type AuditOutcome =
  | "success"
  | "failure"
  | "error"
  | "denied"
  | "partial";

/**
 * Audit event severity
 */
export type AuditSeverity = "debug" | "info" | "warning" | "error" | "critical";

/**
 * Audit log entry
 */
export interface AuditLogEntry {
  id: string;
  timestamp: Date;
  eventType: AuditEventType;
  eventCode: string;
  action: string;
  outcome: AuditOutcome;
  severity: AuditSeverity;
  actor: AuditActor;
  target: AuditTarget;
  context: AuditContext;
  changes?: DataChanges;
  metadata: Record<string, unknown>;
  signature: string;
  previousHash: string;
  hash: string;
}

/**
 * Audit actor (who performed the action)
 */
export interface AuditActor {
  type: "user" | "service" | "system" | "anonymous";
  id: string;
  name?: string;
  email?: string;
  tenantId?: string;
  roles?: string[];
  sessionId?: string;
  ipAddress?: string;
  userAgent?: string;
  location?: GeoLocation;
}

/**
 * Geo location
 */
export interface GeoLocation {
  country?: string;
  region?: string;
  city?: string;
  latitude?: number;
  longitude?: number;
  timezone?: string;
}

/**
 * Audit target (what was affected)
 */
export interface AuditTarget {
  type: string;
  id: string;
  name?: string;
  tenantId?: string;
  path?: string;
  attributes?: Record<string, unknown>;
}

/**
 * Audit context
 */
export interface AuditContext {
  requestId?: string;
  correlationId?: string;
  spanId?: string;
  traceId?: string;
  source?: string;
  component?: string;
  version?: string;
  environment?: string;
  custom?: Record<string, unknown>;
}

/**
 * Data changes for modification events
 */
export interface DataChanges {
  before?: Record<string, unknown>;
  after?: Record<string, unknown>;
  diff?: ChangeDiff[];
  redacted?: string[];
}

/**
 * Individual change diff
 */
export interface ChangeDiff {
  field: string;
  oldValue?: unknown;
  newValue?: unknown;
  operation: "add" | "remove" | "modify";
}

/**
 * Audit log configuration
 */
export interface AuditLogConfig {
  enabled: boolean;
  retentionDays: number;
  compressionEnabled: boolean;
  encryptionEnabled: boolean;
  encryptionKey?: string;
  tamperProofEnabled: boolean;
  hashAlgorithm: "sha256" | "sha384" | "sha512" | "blake3";
  signingEnabled: boolean;
  signingKey?: string;
  batchSize: number;
  flushInterval: number;
  redactedFields: string[];
  excludedEvents: string[];
  destinations: AuditDestination[];
}

/**
 * Audit log destination
 */
export interface AuditDestination {
  type:
    | "database"
    | "file"
    | "s3"
    | "elasticsearch"
    | "splunk"
    | "siem"
    | "webhook";
  name: string;
  enabled: boolean;
  config: Record<string, unknown>;
  filter?: AuditFilter;
  transform?: AuditTransform;
}

/**
 * Audit filter
 */
export interface AuditFilter {
  eventTypes?: AuditEventType[];
  severities?: AuditSeverity[];
  outcomes?: AuditOutcome[];
  tenantIds?: string[];
  actorTypes?: string[];
  custom?: Record<string, unknown>;
}

/**
 * Audit transform
 */
export interface AuditTransform {
  format: "json" | "cef" | "leef" | "syslog";
  includeFields?: string[];
  excludeFields?: string[];
  renameFields?: Record<string, string>;
  timestampFormat?: string;
}

/**
 * Audit query options
 */
export interface AuditQueryOptions {
  startDate?: Date;
  endDate?: Date;
  eventTypes?: AuditEventType[];
  severities?: AuditSeverity[];
  outcomes?: AuditOutcome[];
  actorId?: string;
  tenantId?: string;
  targetType?: string;
  targetId?: string;
  searchText?: string;
  limit?: number;
  offset?: number;
  sortBy?: string;
  sortOrder?: "asc" | "desc";
}

/**
 * Audit verification result
 */
export interface AuditVerificationResult {
  valid: boolean;
  entriesChecked: number;
  entriesValid: number;
  entriesInvalid: number;
  firstInvalidEntry?: string;
  errors: AuditVerificationError[];
  chainIntact: boolean;
  signatures: SignatureVerification[];
}

/**
 * Audit verification error
 */
export interface AuditVerificationError {
  entryId: string;
  errorType:
    | "hash_mismatch"
    | "signature_invalid"
    | "chain_broken"
    | "missing_entry";
  message: string;
  expectedValue?: string;
  actualValue?: string;
}

/**
 * Signature verification
 */
export interface SignatureVerification {
  entryId: string;
  valid: boolean;
  signedAt?: Date;
  signedBy?: string;
  algorithm?: string;
}

// ============================================================================
// Enterprise Event Types
// ============================================================================

/**
 * Enterprise event types
 */
export type EnterpriseEventType =
  | "tenant_created"
  | "tenant_updated"
  | "tenant_suspended"
  | "tenant_deleted"
  | "user_provisioned"
  | "user_deprovisioned"
  | "sso_login"
  | "sso_logout"
  | "mfa_enabled"
  | "mfa_disabled"
  | "compliance_alert"
  | "resource_limit_warning"
  | "resource_limit_exceeded";

/**
 * Enterprise event
 */
export interface EnterpriseEvent<T = unknown> {
  id: string;
  type: EnterpriseEventType;
  tenantId: string;
  timestamp: Date;
  data: T;
  metadata: Record<string, unknown>;
}

// ============================================================================
// Configuration Types
// ============================================================================

/**
 * Enterprise module configuration
 */
export interface EnterpriseConfig {
  auth: AuthModuleConfig;
  tenancy: TenancyModuleConfig;
  compliance: ComplianceModuleConfig;
  audit: AuditLogConfig;
}

/**
 * Authentication module config
 */
export interface AuthModuleConfig {
  sessionTimeout: number;
  maxConcurrentSessions: number;
  tokenExpiration: number;
  refreshTokenExpiration: number;
  mfaGracePeriod: number;
  passwordPolicy: PasswordPolicy;
  providers: SSOProviderConfig[];
  jwtSecret: string;
  jwtAlgorithm: string;
  cookieSettings: CookieSettings;
}

/**
 * Cookie settings
 */
export interface CookieSettings {
  name: string;
  secure: boolean;
  httpOnly: boolean;
  sameSite: "strict" | "lax" | "none";
  domain?: string;
  path: string;
  maxAge: number;
}

/**
 * Tenancy module config
 */
export interface TenancyModuleConfig {
  defaultIsolationLevel: TenantIsolationLevel;
  defaultSubscriptionTier: SubscriptionTier;
  defaultResourceLimits: ResourceLimits;
  allowCustomDomains: boolean;
  domainVerificationRequired: boolean;
  autoProvisioningEnabled: boolean;
  trialDurationDays: number;
}

/**
 * Compliance module config
 */
export interface ComplianceModuleConfig {
  enabledCategories: SOC2Category[];
  automatedTestingEnabled: boolean;
  testingSchedule: string; // cron expression
  alertingEnabled: boolean;
  alertThreshold: RiskLevel;
  reportRetentionDays: number;
  evidenceStorageProvider: string;
}
