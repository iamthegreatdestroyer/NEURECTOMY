/**
 * API Client Configuration
 * Centralized API client for all backend services
 */

// Base URLs from environment
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';
const ML_API_URL = import.meta.env.VITE_ML_API_URL || 'http://localhost:8000';

// Request configuration
interface RequestConfig extends RequestInit {
  params?: Record<string, string | number | boolean>;
  timeout?: number;
}

// API Error
export class APIError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string,
    public details?: unknown
  ) {
    super(message);
    this.name = 'APIError';
  }
}

// Build URL with query params
function buildUrl(baseUrl: string, path: string, params?: Record<string, string | number | boolean>): string {
  const url = new URL(path, baseUrl);
  
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        url.searchParams.append(key, String(value));
      }
    });
  }
  
  return url.toString();
}

// Create fetch with timeout
async function fetchWithTimeout(url: string, config: RequestConfig): Promise<Response> {
  const { timeout = 30000, ...fetchConfig } = config;
  
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      ...fetchConfig,
      signal: controller.signal,
    });
    
    return response;
  } finally {
    clearTimeout(timeoutId);
  }
}

// Generic request function
async function request<T>(
  baseUrl: string,
  path: string,
  config: RequestConfig = {}
): Promise<T> {
  const { params, ...fetchConfig } = config;
  const url = buildUrl(baseUrl, path, params);
  
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...config.headers,
  };
  
  // Add auth token if available
  const token = localStorage.getItem('auth_token');
  if (token) {
    (headers as Record<string, string>)['Authorization'] = `Bearer ${token}`;
  }
  
  const response = await fetchWithTimeout(url, {
    ...fetchConfig,
    headers,
  });
  
  // Handle non-OK responses
  if (!response.ok) {
    let errorData;
    try {
      errorData = await response.json();
    } catch {
      errorData = { message: response.statusText };
    }
    
    throw new APIError(
      errorData.message || 'Request failed',
      response.status,
      errorData.code,
      errorData.details
    );
  }
  
  // Handle empty responses
  if (response.status === 204) {
    return undefined as T;
  }
  
  return response.json();
}

// Rust Core API Client
export const coreApi = {
  // Health check
  health: () => request<{ status: string; version: string }>(API_BASE_URL, '/health'),
  
  // Agents
  agents: {
    list: (params?: { status?: string; tier?: number }) =>
      request<unknown[]>(API_BASE_URL, '/api/v1/agents', { params }),
    
    get: (id: string) =>
      request<unknown>(API_BASE_URL, `/api/v1/agents/${id}`),
    
    create: (data: unknown) =>
      request<unknown>(API_BASE_URL, '/api/v1/agents', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    
    update: (id: string, data: unknown) =>
      request<unknown>(API_BASE_URL, `/api/v1/agents/${id}`, {
        method: 'PATCH',
        body: JSON.stringify(data),
      }),
    
    delete: (id: string) =>
      request<void>(API_BASE_URL, `/api/v1/agents/${id}`, {
        method: 'DELETE',
      }),
    
    invoke: (id: string, prompt: string, context?: unknown) =>
      request<unknown>(API_BASE_URL, `/api/v1/agents/${id}/invoke`, {
        method: 'POST',
        body: JSON.stringify({ prompt, context }),
      }),
  },
  
  // Containers
  containers: {
    list: () =>
      request<unknown[]>(API_BASE_URL, '/api/v1/containers'),
    
    get: (id: string) =>
      request<unknown>(API_BASE_URL, `/api/v1/containers/${id}`),
    
    start: (id: string) =>
      request<void>(API_BASE_URL, `/api/v1/containers/${id}/start`, {
        method: 'POST',
      }),
    
    stop: (id: string) =>
      request<void>(API_BASE_URL, `/api/v1/containers/${id}/stop`, {
        method: 'POST',
      }),
    
    logs: (id: string, params?: { tail?: number; since?: string }) =>
      request<string>(API_BASE_URL, `/api/v1/containers/${id}/logs`, { params }),
    
    stats: (id: string) =>
      request<unknown>(API_BASE_URL, `/api/v1/containers/${id}/stats`),
  },
  
  // Kubernetes
  clusters: {
    list: () =>
      request<unknown[]>(API_BASE_URL, '/api/v1/clusters'),
    
    get: (id: string) =>
      request<unknown>(API_BASE_URL, `/api/v1/clusters/${id}`),
    
    pods: (clusterId: string, namespace?: string) =>
      request<unknown[]>(API_BASE_URL, `/api/v1/clusters/${clusterId}/pods`, {
        params: namespace ? { namespace } : undefined,
      }),
    
    deployments: (clusterId: string, namespace?: string) =>
      request<unknown[]>(API_BASE_URL, `/api/v1/clusters/${clusterId}/deployments`, {
        params: namespace ? { namespace } : undefined,
      }),
  },
  
  // Research
  research: {
    search: (query: string, sources?: string[]) =>
      request<unknown>(API_BASE_URL, '/api/v1/research/search', {
        method: 'POST',
        body: JSON.stringify({ query, sources }),
      }),
    
    analyze: (documentId: string) =>
      request<unknown>(API_BASE_URL, `/api/v1/research/analyze/${documentId}`),
  },
};

// ML Service API Client
export const mlApi = {
  // Health check
  health: () => request<{ status: string }>(ML_API_URL, '/health'),
  
  // Models
  models: {
    list: () =>
      request<unknown[]>(ML_API_URL, '/api/v1/models'),
    
    get: (id: string) =>
      request<unknown>(ML_API_URL, `/api/v1/models/${id}`),
    
    deploy: (id: string) =>
      request<void>(ML_API_URL, `/api/v1/models/${id}/deploy`, {
        method: 'POST',
      }),
  },
  
  // Training
  training: {
    start: (config: unknown) =>
      request<{ jobId: string }>(ML_API_URL, '/api/v1/training', {
        method: 'POST',
        body: JSON.stringify(config),
      }),
    
    status: (jobId: string) =>
      request<unknown>(ML_API_URL, `/api/v1/training/${jobId}`),
    
    cancel: (jobId: string) =>
      request<void>(ML_API_URL, `/api/v1/training/${jobId}/cancel`, {
        method: 'POST',
      }),
  },
  
  // Inference
  inference: {
    predict: (modelId: string, input: unknown) =>
      request<unknown>(ML_API_URL, `/api/v1/inference/${modelId}`, {
        method: 'POST',
        body: JSON.stringify({ input }),
      }),
    
    embed: (text: string | string[]) =>
      request<number[][]>(ML_API_URL, '/api/v1/embeddings', {
        method: 'POST',
        body: JSON.stringify({ text }),
      }),
  },
  
  // LLM
  llm: {
    chat: (messages: unknown[], options?: unknown) =>
      request<unknown>(ML_API_URL, '/api/v1/llm/chat', {
        method: 'POST',
        body: JSON.stringify({ messages, options }),
      }),
    
    complete: (prompt: string, options?: unknown) =>
      request<unknown>(ML_API_URL, '/api/v1/llm/complete', {
        method: 'POST',
        body: JSON.stringify({ prompt, options }),
      }),
  },
};

// Export combined API
export const api = {
  core: coreApi,
  ml: mlApi,
};

export default api;
