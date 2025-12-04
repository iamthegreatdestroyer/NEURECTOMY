import { RestError } from "./rest";

export interface ApiError {
  code: string;
  message: string;
  statusCode?: number;
  details?: Record<string, unknown>;
}

/**
 * Handle and normalize API errors.
 */
export function handleApiError(error: unknown): ApiError {
  if (error instanceof RestError) {
    let details: Record<string, unknown> | undefined;
    try {
      details = JSON.parse(error.responseBody);
    } catch {
      details = { rawBody: error.responseBody };
    }

    return {
      code: `HTTP_${error.statusCode}`,
      message: error.message,
      statusCode: error.statusCode,
      details,
    };
  }

  if (error instanceof Error) {
    return {
      code: "UNKNOWN_ERROR",
      message: error.message,
    };
  }

  return {
    code: "UNKNOWN_ERROR",
    message: String(error),
  };
}

/**
 * Retry a request with exponential backoff.
 */
export async function retryRequest<T>(
  fn: () => Promise<T>,
  options: {
    maxRetries?: number;
    baseDelayMs?: number;
    maxDelayMs?: number;
    shouldRetry?: (error: unknown) => boolean;
  } = {}
): Promise<T> {
  const {
    maxRetries = 3,
    baseDelayMs = 1000,
    maxDelayMs = 10000,
    shouldRetry = () => true,
  } = options;

  let lastError: unknown;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (attempt === maxRetries || !shouldRetry(error)) {
        break;
      }

      // Exponential backoff with jitter
      const delay = Math.min(
        baseDelayMs * Math.pow(2, attempt) + Math.random() * 1000,
        maxDelayMs
      );
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  throw lastError;
}

/**
 * Check if an error is retryable (5xx or network error).
 */
export function isRetryableError(error: unknown): boolean {
  if (error instanceof RestError) {
    return error.statusCode >= 500 || error.statusCode === 429;
  }
  if (error instanceof TypeError && error.message.includes("fetch")) {
    return true; // Network error
  }
  return false;
}
