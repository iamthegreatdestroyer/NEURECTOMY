/**
 * Deployment Strategies - Barrel Export
 * Exports all deployment strategy implementations
 */

export { BaseDeploymentStrategy } from "./base";
export type { DeploymentStrategyConfig, DeploymentProgress } from "./base";

export { RollingDeploymentStrategy, createRollingStrategy } from "./rolling";
export type { RollingDeploymentConfig } from "./rolling";

export {
  BlueGreenDeploymentStrategy,
  createBlueGreenStrategy,
} from "./bluegreen";
export type { BlueGreenDeploymentConfig } from "./bluegreen";

export { CanaryDeploymentStrategy, createCanaryStrategy } from "./canary";
export type { CanaryDeploymentConfig } from "./canary";

// Convenience aliases matching common naming conventions
export { RollingDeploymentStrategy as RollingUpdateStrategy } from "./rolling";
export { BlueGreenDeploymentStrategy as BlueGreenStrategy } from "./bluegreen";
export { CanaryDeploymentStrategy as CanaryStrategy } from "./canary";
