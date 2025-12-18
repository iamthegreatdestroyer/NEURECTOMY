/**
 * NEURECTOMY Services Index
 *
 * Exports all services for the Spectrum Workspace IDE.
 *
 * @module @neurectomy/services
 */

// AI/LLM Services
export {
  RyotService,
  chatCompletion,
  streamChatCompletion,
  executeAgent,
  streamAgentExecution,
  isRyotAvailable,
  startRyotServer,
  setAIConfig,
} from "./ryot-service";

export type {
  ChatMessage as RyotChatMessage,
  ChatCompletionRequest,
  ChatCompletionResponse,
  StreamDelta,
} from "./ryot-service";

// Encrypted Storage Services
export {
  SigmaVaultService,
  initializeVault,
  unlockVault,
  lockVault,
  isVaultUnlocked,
  encryptFile,
  decryptFile,
  listVaultFiles,
  deleteVaultFile,
  storeSecureNote,
  retrieveSecureNote,
} from "./sigmavault-service";

export type {
  DimensionalCoordinate,
  ScatteredFile,
  VaultConfig,
  EncryptionResult,
  DecryptionResult,
  SecureNote,
} from "./sigmavault-service";
