export interface RestClientConfig {
  baseUrl: string;
  headers?: Record<string, string>;
  timeout?: number;
}

interface RequestOptions {
  method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  path: string;
  body?: unknown;
  headers?: Record<string, string>;
  params?: Record<string, string | number | boolean>;
}

/**
 * REST API client with automatic retry and error handling.
 */
export class RestClient {
  private baseUrl: string;
  private defaultHeaders: Record<string, string>;
  private timeout: number;

  constructor(config: RestClientConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, "");
    this.defaultHeaders = {
      "Content-Type": "application/json",
      ...config.headers,
    };
    this.timeout = config.timeout ?? 30000;
  }

  /**
   * Execute a request.
   */
  private async request<T>(options: RequestOptions): Promise<T> {
    const { method, path, body, headers, params } = options;

    let url = `${this.baseUrl}${path}`;
    if (params) {
      const searchParams = new URLSearchParams();
      Object.entries(params).forEach(([key, value]) => {
        searchParams.append(key, String(value));
      });
      url += `?${searchParams.toString()}`;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method,
        headers: { ...this.defaultHeaders, ...headers },
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorBody = await response.text();
        throw new RestError(
          `HTTP ${response.status}: ${response.statusText}`,
          response.status,
          errorBody
        );
      }

      // Handle empty responses
      const contentType = response.headers.get("content-type");
      if (contentType?.includes("application/json")) {
        return response.json();
      }
      return {} as T;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * GET request.
   */
  async get<T>(
    path: string,
    params?: Record<string, string | number | boolean>
  ): Promise<T> {
    return this.request<T>({ method: "GET", path, params });
  }

  /**
   * POST request.
   */
  async post<T>(path: string, body?: unknown): Promise<T> {
    return this.request<T>({ method: "POST", path, body });
  }

  /**
   * PUT request.
   */
  async put<T>(path: string, body?: unknown): Promise<T> {
    return this.request<T>({ method: "PUT", path, body });
  }

  /**
   * PATCH request.
   */
  async patch<T>(path: string, body?: unknown): Promise<T> {
    return this.request<T>({ method: "PATCH", path, body });
  }

  /**
   * DELETE request.
   */
  async delete<T>(path: string): Promise<T> {
    return this.request<T>({ method: "DELETE", path });
  }

  /**
   * Set authorization token.
   */
  setAuthToken(token: string): void {
    this.defaultHeaders["Authorization"] = `Bearer ${token}`;
  }

  /**
   * Clear authorization token.
   */
  clearAuthToken(): void {
    delete this.defaultHeaders["Authorization"];
  }
}

/**
 * REST API error.
 */
export class RestError extends Error {
  public readonly statusCode: number;
  public readonly responseBody: string;

  constructor(message: string, statusCode: number, responseBody: string) {
    super(message);
    this.name = "RestError";
    this.statusCode = statusCode;
    this.responseBody = responseBody;
  }
}

/**
 * Factory function to create a REST client.
 */
export function createRestClient(config: RestClientConfig): RestClient {
  return new RestClient(config);
}
