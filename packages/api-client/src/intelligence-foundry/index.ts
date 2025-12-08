/**
 * Intelligence Foundry API Client
 *
 * Centralized exports for MLflow, Optuna, and WebSocket clients.
 *
 * @module intelligence-foundry
 */

// MLflow exports
export * from "./mlflow";
export { MLflowClient, getMLflowClient, resetMLflowClient } from "./mlflow";

// Optuna exports
export * from "./optuna";
export { OptunaClient, getOptunaClient, resetOptunaClient } from "./optuna";

// WebSocket exports
export * from "./websocket";
export {
  IntelligenceFoundryWebSocket,
  getIntelligenceFoundryWebSocket,
  resetIntelligenceFoundryWebSocket,
  useIntelligenceFoundryWebSocket,
} from "./websocket";
