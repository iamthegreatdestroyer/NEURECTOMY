/**
 * @fileoverview Scalability Types
 * Type definitions for horizontal scaling, sharding, failover, and high availability
 * @module @neurectomy/enterprise/scalability
 */

// =============================================================================
// SHARDING TYPES
// =============================================================================

/**
 * Sharding strategy type
 */
export type ShardingStrategy =
  | "hash"
  | "range"
  | "directory"
  | "geographic"
  | "tenant"
  | "composite";

/**
 * Shard status
 */
export type ShardStatus =
  | "active"
  | "readonly"
  | "migrating"
  | "rebalancing"
  | "offline"
  | "degraded";

/**
 * Configuration for a single shard
 */
export interface ShardConfig {
  id: string;
  name: string;
  connectionString: string;
  region?: string;
  priority: number;
  status: ShardStatus;
  capacity: {
    maxConnections: number;
    maxStorage: number; // bytes
    currentUsage: number; // percentage
  };
  metadata: Record<string, unknown>;
}

/**
 * Sharding rule definition
 */
export interface ShardingRule {
  id: string;
  name: string;
  strategy: ShardingStrategy;
  keyField: string;
  hashFunction?: "murmur3" | "xxhash" | "md5" | "sha256";
  rangeStart?: string | number;
  rangeEnd?: string | number;
  geographicMapping?: Record<string, string>; // region -> shardId
  tenantMapping?: Record<string, string>; // tenantId -> shardId
  enabled: boolean;
  priority: number;
}

/**
 * Shard assignment result
 */
export interface ShardAssignment {
  shardId: string;
  rule: ShardingRule;
  key: string;
  timestamp: Date;
  confidence: number;
}

/**
 * Sharding configuration
 */
export interface ShardingConfig {
  enabled: boolean;
  strategy: ShardingStrategy;
  shards: ShardConfig[];
  rules: ShardingRule[];
  rebalancing: {
    enabled: boolean;
    threshold: number; // percentage imbalance to trigger
    maxConcurrent: number;
    scheduleExpression?: string; // cron expression
  };
  routing: {
    cacheEnabled: boolean;
    cacheTTL: number;
    fallbackShardId?: string;
  };
}

// =============================================================================
// FAILOVER TYPES
// =============================================================================

/**
 * Failover strategy type
 */
export type FailoverStrategy =
  | "active-passive"
  | "active-active"
  | "hot-standby"
  | "warm-standby"
  | "cold-standby"
  | "multi-region";

/**
 * Health check type
 */
export type HealthCheckType =
  | "http"
  | "tcp"
  | "database"
  | "custom"
  | "composite";

/**
 * Node status in cluster
 */
export type NodeStatus =
  | "healthy"
  | "unhealthy"
  | "unknown"
  | "draining"
  | "starting"
  | "stopping";

/**
 * Failover event type
 */
export type FailoverEventType =
  | "detection"
  | "initiated"
  | "in-progress"
  | "completed"
  | "rollback"
  | "failed";

/**
 * Node configuration
 */
export interface NodeConfig {
  id: string;
  name: string;
  host: string;
  port: number;
  region: string;
  zone?: string;
  role: "primary" | "secondary" | "arbiter";
  priority: number;
  status: NodeStatus;
  lastHealthCheck?: Date;
  metadata: Record<string, unknown>;
}

/**
 * Health check configuration
 */
export interface HealthCheckConfig {
  id: string;
  type: HealthCheckType;
  endpoint?: string;
  interval: number; // ms
  timeout: number; // ms
  healthyThreshold: number;
  unhealthyThreshold: number;
  successCodes?: number[];
  customCheck?: () => Promise<boolean>;
}

/**
 * Failover policy
 */
export interface FailoverPolicy {
  id: string;
  name: string;
  strategy: FailoverStrategy;
  triggerConditions: {
    consecutiveFailures: number;
    responseTimeThreshold: number; // ms
    errorRateThreshold: number; // percentage
    customCondition?: (metrics: FailoverMetrics) => boolean;
  };
  actions: {
    notifyBefore: boolean;
    notifyAfter: boolean;
    drainTimeout: number; // ms
    cooldownPeriod: number; // ms between failovers
    autoRollback: boolean;
    rollbackTimeout: number; // ms
  };
  enabled: boolean;
}

/**
 * Failover event
 */
export interface FailoverEvent {
  id: string;
  type: FailoverEventType;
  sourceNode: string;
  targetNode: string;
  policy: FailoverPolicy;
  startTime: Date;
  endTime?: Date;
  duration?: number;
  success: boolean;
  error?: string;
  metrics: FailoverMetrics;
}

/**
 * Failover metrics
 */
export interface FailoverMetrics {
  responseTime: number;
  errorRate: number;
  consecutiveFailures: number;
  uptime: number;
  lastSuccessfulRequest?: Date;
  requestsInFlight: number;
}

/**
 * Failover configuration
 */
export interface FailoverConfig {
  enabled: boolean;
  strategy: FailoverStrategy;
  nodes: NodeConfig[];
  healthChecks: HealthCheckConfig[];
  policies: FailoverPolicy[];
  global: {
    maxFailoversPerHour: number;
    notificationChannels: string[];
    metricsRetention: number; // days
  };
}

// =============================================================================
// HORIZONTAL SCALING TYPES
// =============================================================================

/**
 * Scaling direction
 */
export type ScalingDirection = "up" | "down" | "none";

/**
 * Scaling trigger type
 */
export type ScalingTriggerType =
  | "cpu"
  | "memory"
  | "requests"
  | "queue-depth"
  | "custom"
  | "schedule";

/**
 * Scaling policy
 */
export interface ScalingPolicy {
  id: string;
  name: string;
  trigger: ScalingTriggerType;
  threshold: {
    scaleUp: number;
    scaleDown: number;
  };
  adjustment: {
    type: "absolute" | "percentage" | "target";
    value: number;
  };
  cooldown: {
    scaleUp: number; // seconds
    scaleDown: number; // seconds
  };
  schedule?: {
    expression: string; // cron
    targetInstances: number;
  };
  enabled: boolean;
}

/**
 * Scaling event
 */
export interface ScalingEvent {
  id: string;
  direction: ScalingDirection;
  policy: ScalingPolicy;
  previousCount: number;
  targetCount: number;
  actualCount: number;
  triggerValue: number;
  timestamp: Date;
  duration: number;
  success: boolean;
  error?: string;
}

/**
 * Instance configuration
 */
export interface InstanceConfig {
  id: string;
  name: string;
  type: string;
  region: string;
  zone: string;
  status: "running" | "pending" | "terminating" | "stopped";
  health: "healthy" | "unhealthy" | "unknown";
  launchTime: Date;
  metrics: {
    cpu: number;
    memory: number;
    network: number;
  };
}

/**
 * Horizontal scaling configuration
 */
export interface HorizontalScalingConfig {
  enabled: boolean;
  minInstances: number;
  maxInstances: number;
  desiredInstances: number;
  policies: ScalingPolicy[];
  instanceTemplate: {
    type: string;
    region: string;
    zones: string[];
    tags: Record<string, string>;
  };
  healthCheck: HealthCheckConfig;
}

// =============================================================================
// LOAD BALANCING TYPES
// =============================================================================

/**
 * Load balancing algorithm
 */
export type LoadBalancingAlgorithm =
  | "round-robin"
  | "least-connections"
  | "weighted-round-robin"
  | "ip-hash"
  | "random"
  | "resource-based";

/**
 * Backend target
 */
export interface BackendTarget {
  id: string;
  host: string;
  port: number;
  weight: number;
  maxConnections: number;
  currentConnections: number;
  health: "healthy" | "unhealthy" | "draining";
  metadata: Record<string, unknown>;
}

/**
 * Load balancer configuration
 */
export interface LoadBalancerConfig {
  id: string;
  name: string;
  algorithm: LoadBalancingAlgorithm;
  targets: BackendTarget[];
  healthCheck: HealthCheckConfig;
  sessionAffinity: {
    enabled: boolean;
    type: "cookie" | "ip" | "header";
    ttl: number;
  };
  connectionDraining: {
    enabled: boolean;
    timeout: number;
  };
}

// =============================================================================
// RATE LIMITING TYPES
// =============================================================================

/**
 * Rate limit algorithm
 */
export type RateLimitAlgorithm =
  | "token-bucket"
  | "leaky-bucket"
  | "sliding-window"
  | "fixed-window";

/**
 * Rate limit scope
 */
export type RateLimitScope =
  | "global"
  | "per-user"
  | "per-ip"
  | "per-tenant"
  | "per-api-key";

/**
 * Rate limit rule
 */
export interface RateLimitRule {
  id: string;
  name: string;
  algorithm: RateLimitAlgorithm;
  scope: RateLimitScope;
  limit: number;
  window: number; // ms
  burst?: number;
  cost?: number;
  action: "reject" | "throttle" | "queue";
  enabled: boolean;
}

/**
 * Rate limit state
 */
export interface RateLimitState {
  key: string;
  rule: RateLimitRule;
  currentCount: number;
  windowStart: Date;
  lastRequest: Date;
  remaining: number;
  resetAt: Date;
}

/**
 * Rate limit configuration
 */
export interface RateLimitConfig {
  enabled: boolean;
  defaultRule: RateLimitRule;
  rules: RateLimitRule[];
  storage: "memory" | "redis" | "distributed";
  headers: {
    enabled: boolean;
    limitHeader: string;
    remainingHeader: string;
    resetHeader: string;
  };
}

// =============================================================================
// CIRCUIT BREAKER TYPES
// =============================================================================

/**
 * Circuit breaker state
 */
export type CircuitState = "closed" | "open" | "half-open";

/**
 * Circuit breaker configuration
 */
export interface CircuitBreakerConfig {
  id: string;
  name: string;
  failureThreshold: number;
  successThreshold: number;
  timeout: number; // ms to stay open
  halfOpenRequests: number;
  monitorInterval: number;
  enabled: boolean;
}

/**
 * Circuit breaker status
 */
export interface CircuitBreakerStatus {
  id: string;
  state: CircuitState;
  failures: number;
  successes: number;
  lastFailure?: Date;
  lastSuccess?: Date;
  lastStateChange: Date;
  nextAttempt?: Date;
}

// =============================================================================
// REPLICATION TYPES
// =============================================================================

/**
 * Replication mode
 */
export type ReplicationMode = "sync" | "async" | "semi-sync" | "multi-master";

/**
 * Replication status
 */
export type ReplicationStatus =
  | "synced"
  | "lagging"
  | "catching-up"
  | "disconnected"
  | "error";

/**
 * Replica configuration
 */
export interface ReplicaConfig {
  id: string;
  name: string;
  host: string;
  port: number;
  role: "primary" | "replica" | "relay";
  mode: ReplicationMode;
  status: ReplicationStatus;
  lag?: number; // ms behind primary
  priority: number;
}

/**
 * Replication configuration
 */
export interface ReplicationConfig {
  enabled: boolean;
  mode: ReplicationMode;
  replicas: ReplicaConfig[];
  consistency: "eventual" | "strong" | "causal";
  conflictResolution: "last-write-wins" | "first-write-wins" | "custom";
  monitoring: {
    lagThreshold: number; // ms
    alertOnLag: boolean;
  };
}

// =============================================================================
// DISASTER RECOVERY TYPES
// =============================================================================

/**
 * Recovery point objective (RPO) configuration
 */
export interface RPOConfig {
  target: number; // seconds of data loss acceptable
  backupInterval: number; // seconds between backups
  retentionPeriod: number; // days
}

/**
 * Recovery time objective (RTO) configuration
 */
export interface RTOConfig {
  target: number; // seconds to restore service
  automatedRecovery: boolean;
  manualSteps?: string[];
}

/**
 * Backup configuration
 */
export interface BackupConfig {
  id: string;
  type: "full" | "incremental" | "differential";
  schedule: string; // cron expression
  storage: {
    type: "s3" | "gcs" | "azure-blob" | "local";
    bucket: string;
    path: string;
    encryption: boolean;
  };
  retention: {
    daily: number;
    weekly: number;
    monthly: number;
    yearly: number;
  };
  verification: {
    enabled: boolean;
    schedule: string;
  };
}

/**
 * Disaster recovery configuration
 */
export interface DisasterRecoveryConfig {
  enabled: boolean;
  rpo: RPOConfig;
  rto: RTOConfig;
  backups: BackupConfig[];
  failoverRegions: string[];
  runbooks: {
    id: string;
    name: string;
    steps: string[];
    contacts: string[];
  }[];
}

// =============================================================================
// COMPREHENSIVE SCALABILITY CONFIG
// =============================================================================

/**
 * Complete scalability configuration
 */
export interface ScalabilityConfig {
  sharding: ShardingConfig;
  failover: FailoverConfig;
  horizontalScaling: HorizontalScalingConfig;
  loadBalancing: LoadBalancerConfig;
  rateLimit: RateLimitConfig;
  circuitBreaker: CircuitBreakerConfig[];
  replication: ReplicationConfig;
  disasterRecovery: DisasterRecoveryConfig;
}

/**
 * Scalability engine events
 */
export interface ScalabilityEngineEvents {
  "shard:assigned": ShardAssignment;
  "shard:rebalancing": { shardId: string; progress: number };
  "failover:started": FailoverEvent;
  "failover:completed": FailoverEvent;
  "failover:failed": FailoverEvent;
  "scaling:triggered": ScalingEvent;
  "scaling:completed": ScalingEvent;
  "circuit:opened": CircuitBreakerStatus;
  "circuit:closed": CircuitBreakerStatus;
  "ratelimit:exceeded": RateLimitState;
  "backup:completed": { backupId: string; size: number; duration: number };
  error: Error;
  warning: { code: string; message: string };
}
