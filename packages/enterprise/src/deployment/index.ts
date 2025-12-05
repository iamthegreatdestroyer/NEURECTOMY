/**
 * @fileoverview Deployment Module Exports
 * @module @neurectomy/enterprise/deployment
 *
 * Agent Assignment: @FLUX @ARCHITECT @SENTRY
 *
 * Enterprise deployment management:
 * - Deployment Manager: Blue-green, canary, rolling deployments
 * - Rollback Manager: Automatic failure detection and recovery
 *
 * @author NEURECTOMY Phase 5 - Enterprise Excellence
 * @version 1.0.0
 */

// Deployment Manager
export {
  DeploymentManager,
  createDeploymentManager,
  type DeploymentConfig,
  type DeploymentEnvironment,
  type DeploymentStrategy,
  type HealthCheckConfig,
  type RollbackConfig as DeploymentRollbackConfig,
  type DeploymentGate,
  type GateType,
  type GateConfig,
  type TimeWindow,
  type Deployment,
  type DeploymentStatus,
  type DeploymentStage,
  type StageStatus as DeploymentStageStatus,
  type DeploymentInstance,
  type InstanceStatus,
  type HealthStatus,
  type DeploymentMetrics,
  type DeploymentArtifact,
  type ArtifactType as DeploymentArtifactType,
  type DeploymentError,
  type DeploymentEvent,
  type DeploymentEventType,
  type DeploymentManagerConfig,
  DEFAULT_DEPLOYMENT_MANAGER_CONFIG,
} from "./deployment-manager";

// Rollback Manager
export {
  RollbackManager,
  createRollbackManager,
  type RollbackConfig,
  type RollbackStrategyConfig,
  type RollbackCondition,
  type ConditionType,
  type ConditionOperator,
  type RollbackStrategy,
  type DeploymentVersion,
  type VersionArtifact,
  type ArtifactType as VersionArtifactType,
  type DeploymentState,
  type InstanceState,
  type TrafficDistribution,
  type VersionMetrics,
  type RollbackOperation,
  type RollbackStatus,
  type RollbackTrigger,
  type RollbackStage,
  type StageStatus as RollbackStageStatus,
  type PreservedState,
  type ValidationResult,
  type RollbackError,
  type RollbackEvent,
  type RollbackEventType,
  type RollbackManagerConfig,
  DEFAULT_ROLLBACK_CONFIG,
  DEFAULT_ROLLBACK_MANAGER_CONFIG,
} from "./rollback-manager";

// ============================================================================
// Unified Deployment Factory
// ============================================================================

import {
  DeploymentManager,
  DeploymentManagerConfig,
} from "./deployment-manager";
import {
  RollbackManager,
  RollbackManagerConfig,
  RollbackConfig,
} from "./rollback-manager";

export interface DeploymentSystemConfig {
  deployment?: Partial<DeploymentManagerConfig>;
  rollback?: Partial<RollbackManagerConfig>;
  rollbackPolicy?: Partial<RollbackConfig>;
}

export interface DeploymentSystem {
  deploymentManager: DeploymentManager;
  rollbackManager: RollbackManager;
}

/**
 * Create a complete deployment system with integrated deployment and rollback managers
 */
export function createDeploymentSystem(
  config: DeploymentSystemConfig = {}
): DeploymentSystem {
  const deploymentManager = new DeploymentManager(config.deployment);
  const rollbackManager = new RollbackManager(
    config.rollback,
    config.rollbackPolicy
  );

  // Wire up deployment events to rollback manager
  deploymentManager.on("deployment.completed", (event) => {
    // Register deployed version for rollback eligibility
    rollbackManager.registerVersion({
      id: event.data.deploymentId || event.deploymentId,
      version: event.data.version || "unknown",
      deployedAt: new Date(),
      artifacts: [],
      configuration: {},
      state: {
        instances: [],
        trafficDistribution: {
          activeVersion: event.data.version || "unknown",
          distribution: {},
        },
        databaseVersion: "",
        configHash: "",
      },
      metrics: {
        errorRate: 0,
        latencyP99: 0,
        latencyP95: 0,
        latencyAvg: 0,
        requestsPerSecond: 0,
        cpuUsage: 0,
        memoryUsage: 0,
      },
      rollbackEligible: true,
    });
  });

  // Handle automatic rollback triggers
  deploymentManager.on("deployment.failed", async (event) => {
    // Find the current successful version and potentially rollback
    const eligibleVersions = rollbackManager.getEligibleVersions();
    if (eligibleVersions.length > 0) {
      const targetVersion = rollbackManager.findBestRollbackTarget(
        event.deploymentId
      );
      if (targetVersion) {
        console.log(
          `Deployment failed, eligible for rollback to ${targetVersion.version}`
        );
      }
    }
  });

  return {
    deploymentManager,
    rollbackManager,
  };
}
