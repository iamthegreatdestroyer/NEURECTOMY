/**
 * REST API Client
 *
 * HTTP client for non-GraphQL REST endpoints.
 * Includes request/response interceptors, retry logic, and error handling.
 */

export interface RequestConfig extends RequestInit {
  params?: Record<string, string | number | boolean>;
  timeout?: number;
  retry?: {
    attempts?: number;
    delay?: number;
    backoff?: boolean;
  };
}

export interface APIResponse<T = any> {
  data: T | null;
  error: APIError | null;
  status: number;
  headers: Headers;
}

export interface APIError {
  message: string;
  code?: string;
  status?: number;
  details?: any;
}

type RequestInterceptor = (
  config: RequestConfig
) => RequestConfig | Promise<RequestConfig>;
type ResponseInterceptor = <T>(
  response: APIResponse<T>
) => APIResponse<T> | Promise<APIResponse<T>>;

class RESTClient {
  private baseURL: string;
  private defaultHeaders: HeadersInit;
  private requestInterceptors: RequestInterceptor[] = [];
  private responseInterceptors: ResponseInterceptor[] = [];

  constructor(baseURL: string, defaultHeaders: HeadersInit = {}) {
    this.baseURL = baseURL;
    this.defaultHeaders = {
      "Content-Type": "application/json",
      ...defaultHeaders,
    };
  }

  /**
   * Add request interceptor
   */
  addRequestInterceptor(interceptor: RequestInterceptor): void {
    this.requestInterceptors.push(interceptor);
  }

  /**
   * Add response interceptor
   */
  addResponseInterceptor(interceptor: ResponseInterceptor): void {
    this.responseInterceptors.push(interceptor);
  }

  /**
   * Build full URL with query parameters
   */
  private buildURL(
    path: string,
    params?: Record<string, string | number | boolean>
  ): string {
    const url = new URL(path, this.baseURL);

    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        url.searchParams.append(key, String(value));
      });
    }

    return url.toString();
  }

  /**
   * Apply request interceptors
   */
  private async applyRequestInterceptors(
    config: RequestConfig
  ): Promise<RequestConfig> {
    let modifiedConfig = { ...config };

    for (const interceptor of this.requestInterceptors) {
      modifiedConfig = await interceptor(modifiedConfig);
    }

    return modifiedConfig;
  }

  /**
   * Apply response interceptors
   */
  private async applyResponseInterceptors<T>(
    response: APIResponse<T>
  ): Promise<APIResponse<T>> {
    let modifiedResponse = response;

    for (const interceptor of this.responseInterceptors) {
      modifiedResponse = await interceptor(modifiedResponse);
    }

    return modifiedResponse;
  }

  /**
   * Sleep utility for retry delays
   */
  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * Make HTTP request with retry logic
   */
  private async makeRequest<T>(
    method: string,
    path: string,
    config: RequestConfig = {}
  ): Promise<APIResponse<T>> {
    const {
      params,
      timeout = 30000,
      retry = { attempts: 3, delay: 1000, backoff: true },
      ...fetchConfig
    } = config;

    let lastError: APIError | null = null;
    const maxAttempts = retry.attempts || 3;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        // Apply request interceptors
        const interceptedConfig = await this.applyRequestInterceptors({
          ...fetchConfig,
          method,
        });

        // Build URL
        const url = this.buildURL(path, params);

        // Merge headers
        const headers = {
          ...this.defaultHeaders,
          ...interceptedConfig.headers,
        };

        // Create abort controller for timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        // Make request
        const response = await fetch(url, {
          ...interceptedConfig,
          headers,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        // Parse response
        let data: T | null = null;
        const contentType = response.headers.get("content-type");

        if (contentType?.includes("application/json")) {
          data = await response.json();
        } else if (contentType?.includes("text/")) {
          data = (await response.text()) as any;
        }

        // Build response object
        const apiResponse: APIResponse<T> = {
          data,
          error: null,
          status: response.status,
          headers: response.headers,
        };

        // Check for HTTP errors
        if (!response.ok) {
          apiResponse.error = {
            message: data?.message || response.statusText || "Request failed",
            status: response.status,
            code: data?.code,
            details: data,
          };
          apiResponse.data = null;
        }

        // Apply response interceptors
        const interceptedResponse =
          await this.applyResponseInterceptors(apiResponse);

        // If successful or non-retryable error, return
        if (response.ok || response.status < 500) {
          return interceptedResponse;
        }

        // Store error for retry
        lastError = interceptedResponse.error;
      } catch (error: any) {
        lastError = {
          message: error.message || "Network error",
          code: error.name,
          details: error,
        };

        // If it's an abort error (timeout), don't retry
        if (error.name === "AbortError") {
          return {
            data: null,
            error: { message: "Request timeout", code: "TIMEOUT" },
            status: 0,
            headers: new Headers(),
          };
        }
      }

      // Wait before retry (with exponential backoff)
      if (attempt < maxAttempts - 1) {
        const delay = retry.backoff
          ? retry.delay! * Math.pow(2, attempt)
          : retry.delay!;
        await this.sleep(delay);
      }
    }

    // All retries failed
    return {
      data: null,
      error: lastError || { message: "Request failed after retries" },
      status: lastError?.status || 0,
      headers: new Headers(),
    };
  }

  /**
   * GET request
   */
  async get<T = any>(
    path: string,
    config?: RequestConfig
  ): Promise<APIResponse<T>> {
    return this.makeRequest<T>("GET", path, config);
  }

  /**
   * POST request
   */
  async post<T = any>(
    path: string,
    data?: any,
    config?: RequestConfig
  ): Promise<APIResponse<T>> {
    return this.makeRequest<T>("POST", path, {
      ...config,
      body: JSON.stringify(data),
    });
  }

  /**
   * PUT request
   */
  async put<T = any>(
    path: string,
    data?: any,
    config?: RequestConfig
  ): Promise<APIResponse<T>> {
    return this.makeRequest<T>("PUT", path, {
      ...config,
      body: JSON.stringify(data),
    });
  }

  /**
   * PATCH request
   */
  async patch<T = any>(
    path: string,
    data?: any,
    config?: RequestConfig
  ): Promise<APIResponse<T>> {
    return this.makeRequest<T>("PATCH", path, {
      ...config,
      body: JSON.stringify(data),
    });
  }

  /**
   * DELETE request
   */
  async delete<T = any>(
    path: string,
    config?: RequestConfig
  ): Promise<APIResponse<T>> {
    return this.makeRequest<T>("DELETE", path, config);
  }
}

// Create default REST client instance
const REST_API_BASE_URL =
  import.meta.env.VITE_REST_API_URL || "http://localhost:16080/api";

export const restClient = new RESTClient(REST_API_BASE_URL);

// Add auth token interceptor
restClient.addRequestInterceptor(async (config) => {
  const token = localStorage.getItem("auth_token");

  if (token) {
    config.headers = {
      ...config.headers,
      Authorization: `Bearer ${token}`,
    };
  }

  return config;
});

// Add logging interceptor (development only)
if (import.meta.env.DEV) {
  restClient.addRequestInterceptor(async (config) => {
    console.log(`[REST] ${config.method} ${config.url}`, config);
    return config;
  });

  restClient.addResponseInterceptor(async (response) => {
    if (response.error) {
      console.error("[REST Error]", response.error);
    } else {
      console.log("[REST Success]", response.status, response.data);
    }
    return response;
  });
}

// Export client and types
export { RESTClient };
export type { RequestConfig, APIResponse, APIError };
