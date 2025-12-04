/**
 * Default timeout for operations in milliseconds.
 */
export const DEFAULT_TIMEOUT = 30_000;

/**
 * Maximum number of retry attempts.
 */
export const MAX_RETRIES = 3;

/**
 * Current API version.
 */
export const API_VERSION = "v1";

/**
 * WebSocket reconnection settings.
 */
export const WEBSOCKET_CONFIG = {
  RECONNECT_INTERVAL_MS: 1000,
  MAX_RECONNECT_ATTEMPTS: 10,
  PING_INTERVAL_MS: 30_000,
  PONG_TIMEOUT_MS: 10_000,
} as const;

/**
 * Rate limiting defaults.
 */
export const RATE_LIMIT_DEFAULTS = {
  REQUESTS_PER_SECOND: 100,
  BURST_SIZE: 200,
  WINDOW_MS: 60_000,
} as const;

/**
 * Cache TTL settings in milliseconds.
 */
export const CACHE_TTL = {
  SHORT: 60_000, // 1 minute
  MEDIUM: 300_000, // 5 minutes
  LONG: 3600_000, // 1 hour
  VERY_LONG: 86400_000, // 24 hours
} as const;

/**
 * Pagination defaults.
 */
export const PAGINATION_DEFAULTS = {
  PAGE_SIZE: 20,
  MAX_PAGE_SIZE: 100,
} as const;

/**
 * File size limits in bytes.
 */
export const FILE_SIZE_LIMITS = {
  MAX_UPLOAD_SIZE: 100 * 1024 * 1024, // 100 MB
  MAX_IMPORT_SIZE: 50 * 1024 * 1024, // 50 MB
  MAX_EXPORT_SIZE: 500 * 1024 * 1024, // 500 MB
} as const;
