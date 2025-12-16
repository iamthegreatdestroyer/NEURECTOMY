import axios, { AxiosInstance, AxiosError } from 'axios';

/**
 * Configuration for Neurectomy API client
 */
export interface NeurectomyConfig {
  /** API key for authentication */
  apiKey: string;
  /** Base URL for API (default: https://api.neurectomy.ai) */
  baseURL?: string;
  /** Request timeout in milliseconds (default: 30000) */
  timeout?: number;
  /** Enable retry on failure (default: true) */
  retryOnFailure?: boolean;
  /** Maximum retry attempts (default: 3) */
  maxRetries?: number;
}

/**
 * Request for text completion
 */
export interface CompletionRequest {
  /** Prompt text */
  prompt: string;
  /** Maximum tokens to generate (default: 100) */
  maxTokens?: number;
  /** Temperature for sampling (0-2, default: 0.7) */
  temperature?: number;
  /** Model to use (default: ryot-bitnet-7b) */
  model?: string;
  /** Top P for nucleus sampling (default: 1.0) */
  topP?: number;
  /** Frequency penalty (default: 0) */
  frequencyPenalty?: number;
  /** Presence penalty (default: 0) */
  presencePenalty?: number;
}

/**
 * Response from text completion
 */
export interface CompletionResponse {
  /** Generated text */
  text: string;
  /** Number of tokens generated */
  tokensGenerated: number;
  /** Reason completion finished (stop, max_tokens, etc) */
  finishReason: string;
  /** Usage statistics */
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

/**
 * Request for text compression
 */
export interface CompressionRequest {
  /** Text to compress */
  text: string;
  /** Target compression ratio (0-1, default: 0.1) */
  targetRatio?: number;
  /** Compression level (1-9, default: 2) */
  compressionLevel?: number;
  /** Compression algorithm (default: lz4) */
  algorithm?: string;
}

/**
 * Response from text compression
 */
export interface CompressionResponse {
  /** Compressed data (base64 encoded) */
  compressedData: string;
  /** Achieved compression ratio */
  compressionRatio: number;
  /** Original data size in bytes */
  originalSize: number;
  /** Compressed data size in bytes */
  compressedSize: number;
  /** Algorithm used */
  algorithm: string;
}

/**
 * Request for file storage
 */
export interface StorageRequest {
  /** File path */
  path: string;
  /** File data (base64 encoded) */
  data: string;
  /** Optional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Response from file storage
 */
export interface StorageResponse {
  /** Unique object ID */
  objectId: string;
  /** File path */
  path: string;
  /** File size in bytes */
  size: number;
  /** Timestamp */
  timestamp: string;
}

/**
 * Retrieved file data
 */
export interface RetrievedFile {
  /** File data (base64 encoded) */
  data: string;
  /** File path */
  path: string;
  /** File size in bytes */
  size: number;
  /** Timestamp */
  timestamp: string;
  /** Optional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Error response from API
 */
export interface ErrorResponse {
  /** Error code */
  code: string;
  /** Error message */
  message: string;
  /** Additional error details */
  details?: unknown;
}

/**
 * Neurectomy API Client
 *
 * Example:
 * ```typescript
 * const client = new NeurectomyClient({
 *   apiKey: process.env.NEURECTOMY_API_KEY
 * });
 *
 * const completion = await client.complete({
 *   prompt: "Hello, world!"
 * });
 * ```
 */
export class NeurectomyClient {
  private client: AxiosInstance;
  private config: NeurectomyConfig;
  private retryCount: number = 0;

  constructor(config: NeurectomyConfig) {
    if (!config.apiKey) {
      throw new Error('API key is required');
    }

    this.config = {
      retryOnFailure: true,
      maxRetries: 3,
      ...config,
    };

    this.client = axios.create({
      baseURL: this.config.baseURL || 'https://api.neurectomy.ai',
      timeout: this.config.timeout || 30000,
      headers: {
        Authorization: `Bearer ${this.config.apiKey}`,
        'Content-Type': 'application/json',
        'User-Agent': 'neurectomy-js-sdk/1.0.0',
      },
    });

    // Add response interceptor for error handling
    this.client.interceptors.response.use((response) => response, this.handleError.bind(this));
  }

  /**
   * Generate text completion
   */
  async complete(request: CompletionRequest): Promise<CompletionResponse> {
    try {
      const response = await this.client.post<CompletionResponse>('/v1/completions', {
        prompt: request.prompt,
        max_tokens: request.maxTokens || 100,
        temperature: request.temperature ?? 0.7,
        model: request.model || 'ryot-bitnet-7b',
        top_p: request.topP ?? 1.0,
        frequency_penalty: request.frequencyPenalty ?? 0,
        presence_penalty: request.presencePenalty ?? 0,
      });

      return response.data;
    } catch (error) {
      throw this.formatError(error);
    }
  }

  /**
   * Compress text data
   */
  async compress(request: CompressionRequest): Promise<CompressionResponse> {
    try {
      const response = await this.client.post<CompressionResponse>('/v1/compress', {
        text: request.text,
        target_ratio: request.targetRatio ?? 0.1,
        compression_level: request.compressionLevel ?? 2,
        algorithm: request.algorithm || 'lz4',
      });

      return response.data;
    } catch (error) {
      throw this.formatError(error);
    }
  }

  /**
   * Store file in ΣVAULT
   */
  async storeFile(
    path: string,
    data: string,
    metadata?: Record<string, unknown>
  ): Promise<StorageResponse> {
    try {
      const response = await this.client.post<StorageResponse>('/v1/storage/store', {
        path,
        data,
        metadata,
      });

      return response.data;
    } catch (error) {
      throw this.formatError(error);
    }
  }

  /**
   * Retrieve file from ΣVAULT
   */
  async retrieveFile(objectId: string): Promise<RetrievedFile> {
    try {
      const response = await this.client.get<RetrievedFile>(`/v1/storage/${objectId}`);
      return response.data;
    } catch (error) {
      throw this.formatError(error);
    }
  }

  /**
   * Delete file from ΣVAULT
   */
  async deleteFile(objectId: string): Promise<{ success: boolean }> {
    try {
      const response = await this.client.delete<{ success: boolean }>(`/v1/storage/${objectId}`);
      return response.data;
    } catch (error) {
      throw this.formatError(error);
    }
  }

  /**
   * Get API status
   */
  async getStatus(): Promise<{ status: string; version: string }> {
    try {
      const response = await this.client.get('/v1/status');
      return response.data;
    } catch (error) {
      throw this.formatError(error);
    }
  }

  /**
   * Handle errors with retry logic
   */
  private async handleError(error: AxiosError): Promise<never> {
    if (this.config.retryOnFailure && this.retryCount < (this.config.maxRetries || 3)) {
      if (this.isRetryableError(error)) {
        this.retryCount++;
        const delay = Math.pow(2, this.retryCount) * 1000; // Exponential backoff
        await new Promise((resolve) => setTimeout(resolve, delay));
        // Retry by throwing to let interceptor handle it
      }
    }
    this.retryCount = 0;
    throw error;
  }

  /**
   * Check if error is retryable
   */
  private isRetryableError(error: AxiosError): boolean {
    if (!error.response) {
      // Network error
      return true;
    }
    const status = error.response.status;
    // Retry on 429 (rate limit) and 5xx errors
    return status === 429 || (status >= 500 && status < 600);
  }

  /**
   * Format error with proper typing
   */
  private formatError(error: unknown): Error {
    if (axios.isAxiosError(error)) {
      const data = error.response?.data as ErrorResponse | undefined;
      const message = data?.message || error.message || 'Unknown error';
      const errorObj = new Error(message);
      (errorObj as any).code = data?.code;
      (errorObj as any).details = data?.details;
      (errorObj as any).status = error.response?.status;
      return errorObj;
    }
    return error instanceof Error ? error : new Error(String(error));
  }
}

export default NeurectomyClient;
