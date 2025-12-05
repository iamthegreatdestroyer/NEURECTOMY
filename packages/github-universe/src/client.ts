/**
 * @fileoverview GitHub API Client
 * @module @neurectomy/github-universe/client
 *
 * Core GitHub API client with GraphQL and REST support,
 * rate limiting, caching, and automatic retries.
 *
 * @agents @SYNAPSE @APEX
 */

import { Octokit } from "@octokit/core";
import { graphql } from "@octokit/graphql";
import { paginateRest } from "@octokit/plugin-paginate-rest";
import { retry } from "@octokit/plugin-retry";
import { throttling } from "@octokit/plugin-throttling";
import { EventEmitter } from "eventemitter3";
import {
  type GitHubUniverseConfig,
  GitHubUniverseConfigSchema,
  type RateLimit,
  type APIError,
} from "./types";

// =============================================================================
// ENHANCED OCTOKIT
// =============================================================================

/**
 * Create enhanced Octokit instance with plugins
 * @internal - not exported, used only within this module
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const EnhancedOctokit: any = Octokit.plugin(paginateRest, retry, throttling);

// =============================================================================
// CLIENT EVENTS
// =============================================================================

export interface GitHubClientEvents {
  "request:start": (url: string, method: string) => void;
  "request:complete": (
    url: string,
    method: string,
    status: number,
    duration: number
  ) => void;
  "request:error": (url: string, method: string, error: Error) => void;
  "ratelimit:warning": (remaining: number, reset: Date) => void;
  "ratelimit:exceeded": (reset: Date) => void;
  "cache:hit": (key: string) => void;
  "cache:miss": (key: string) => void;
}

// =============================================================================
// CACHE IMPLEMENTATION
// =============================================================================

interface CacheEntry<T> {
  data: T;
  expiry: number;
  etag?: string;
}

/**
 * Simple in-memory cache with TTL
 */
class RequestCache {
  private cache = new Map<string, CacheEntry<unknown>>();
  private maxSize = 1000;

  set<T>(key: string, data: T, ttl: number, etag?: string): void {
    // Evict oldest entries if at max size
    if (this.cache.size >= this.maxSize) {
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey) this.cache.delete(oldestKey);
    }

    this.cache.set(key, {
      data,
      expiry: Date.now() + ttl * 1000,
      etag,
    });
  }

  get<T>(key: string): { data: T; etag?: string } | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    if (Date.now() > entry.expiry) {
      this.cache.delete(key);
      return null;
    }

    return { data: entry.data as T, etag: entry.etag };
  }

  invalidate(pattern: string | RegExp): number {
    let count = 0;
    for (const key of this.cache.keys()) {
      if (
        typeof pattern === "string" ? key.includes(pattern) : pattern.test(key)
      ) {
        this.cache.delete(key);
        count++;
      }
    }
    return count;
  }

  clear(): void {
    this.cache.clear();
  }

  get size(): number {
    return this.cache.size;
  }
}

// =============================================================================
// GITHUB CLIENT
// =============================================================================

/**
 * GitHub API Client
 *
 * Provides unified access to GitHub's REST and GraphQL APIs with:
 * - Automatic rate limit handling
 * - Request caching with ETags
 * - Retry logic for transient failures
 * - Event-driven monitoring
 *
 * @example
 * ```typescript
 * const client = new GitHubClient({
 *   auth: { token: 'ghp_xxxx' },
 *   enableCache: true,
 *   maxRetries: 3,
 * });
 *
 * // REST API
 * const repo = await client.rest('GET /repos/{owner}/{repo}', {
 *   owner: 'neurectomy',
 *   repo: 'core',
 * });
 *
 * // GraphQL API
 * const data = await client.graphql(`
 *   query($owner: String!, $name: String!) {
 *     repository(owner: $owner, name: $name) {
 *       name
 *       stargazerCount
 *     }
 *   }
 * `, { owner: 'neurectomy', name: 'core' });
 * ```
 */
export class GitHubClient extends EventEmitter<GitHubClientEvents> {
  private octokit: InstanceType<typeof EnhancedOctokit>;
  private graphqlClient: typeof graphql;
  private config: GitHubUniverseConfig;
  private cache: RequestCache;
  private rateLimitInfo: Map<string, RateLimit> = new Map();

  constructor(
    config: Partial<GitHubUniverseConfig> & {
      auth: GitHubUniverseConfig["auth"];
    }
  ) {
    super();

    // Validate and merge config with defaults
    this.config = GitHubUniverseConfigSchema.parse({
      ...config,
      auth: config.auth,
    });

    this.cache = new RequestCache();

    // Initialize Octokit with plugins
    this.octokit = new EnhancedOctokit({
      auth: this.config.auth.token,
      baseUrl: this.config.auth.baseUrl,
      userAgent: this.config.userAgent,
      request: {
        timeout: this.config.timeout,
      },
      throttle: {
        onRateLimit: (
          retryAfter: number,
          options: object,
          _octokit: unknown,
          retryCount: number
        ) => {
          const resetDate = new Date(Date.now() + retryAfter * 1000);
          this.emit("ratelimit:exceeded", resetDate);

          if (retryCount < this.config.maxRetries) {
            this.log("warn", `Rate limit hit, retrying after ${retryAfter}s`);
            return true;
          }
          return false;
        },
        onSecondaryRateLimit: (
          retryAfter: number,
          options: object,
          _octokit: unknown,
          retryCount: number
        ) => {
          const resetDate = new Date(Date.now() + retryAfter * 1000);
          this.emit("ratelimit:exceeded", resetDate);

          if (retryCount < this.config.maxRetries) {
            this.log(
              "warn",
              `Secondary rate limit hit, retrying after ${retryAfter}s`
            );
            return true;
          }
          return false;
        },
      },
      retry: {
        doNotRetry: ["429"],
        retries: this.config.maxRetries,
      },
    });

    // Initialize GraphQL client
    this.graphqlClient = graphql.defaults({
      headers: {
        authorization: `token ${this.config.auth.token}`,
        "user-agent": this.config.userAgent,
        "X-GitHub-Api-Version": this.config.auth.apiVersion,
      },
      ...(this.config.auth.baseUrl && {
        baseUrl: `${this.config.auth.baseUrl}/api/graphql`,
      }),
    });
  }

  // ===========================================================================
  // REST API
  // ===========================================================================

  /**
   * Make a REST API request
   */
  async rest<T = unknown>(
    route: string,
    options?: Record<string, unknown>
  ): Promise<T> {
    const cacheKey = this.getCacheKey("rest", route, options);
    const startTime = Date.now();

    // Extract method and path from route (e.g., "GET /repos/{owner}/{repo}")
    const [method, path] = route.split(" ");
    const url = this.interpolatePath(path, options);

    this.emit("request:start", url, method);

    try {
      // Check cache for GET requests
      if (method === "GET" && this.config.enableCache) {
        const cached = this.cache.get<T>(cacheKey);
        if (cached) {
          this.emit("cache:hit", cacheKey);
          return cached.data;
        }
        this.emit("cache:miss", cacheKey);
      }

      // Make request
      const response = await this.octokit.request(route, options);

      // Update rate limit info
      this.updateRateLimitInfo(response.headers);

      // Cache successful GET responses
      if (method === "GET" && this.config.enableCache) {
        this.cache.set(
          cacheKey,
          response.data,
          this.config.cacheTtl,
          response.headers.etag
        );
      }

      const duration = Date.now() - startTime;
      this.emit("request:complete", url, method, response.status, duration);

      return response.data as T;
    } catch (error) {
      const duration = Date.now() - startTime;
      this.emit("request:error", url, method, error as Error);
      this.log("error", `REST request failed: ${method} ${url}`, error);
      throw this.normalizeError(error);
    }
  }

  /**
   * Make a paginated REST API request
   */
  async restPaginate<T = unknown>(
    route: string,
    options?: Record<string, unknown>
  ): Promise<T[]> {
    const items: T[] = [];

    const iterator = this.octokit.paginate.iterator(route, {
      ...options,
      per_page: 100,
    });

    for await (const response of iterator) {
      items.push(...(response.data as T[]));
    }

    return items;
  }

  // ===========================================================================
  // GRAPHQL API
  // ===========================================================================

  /**
   * Make a GraphQL query
   */
  async graphql<T = unknown>(
    query: string,
    variables?: Record<string, unknown>
  ): Promise<T> {
    const cacheKey = this.getCacheKey("graphql", query, variables);
    const startTime = Date.now();

    this.emit("request:start", "graphql", "POST");

    try {
      // Check cache
      if (this.config.enableCache) {
        const cached = this.cache.get<T>(cacheKey);
        if (cached) {
          this.emit("cache:hit", cacheKey);
          return cached.data;
        }
        this.emit("cache:miss", cacheKey);
      }

      const response = await this.graphqlClient<T>(query, variables);

      // Cache response
      if (this.config.enableCache) {
        this.cache.set(cacheKey, response, this.config.cacheTtl);
      }

      const duration = Date.now() - startTime;
      this.emit("request:complete", "graphql", "POST", 200, duration);

      return response;
    } catch (error) {
      const duration = Date.now() - startTime;
      this.emit("request:error", "graphql", "POST", error as Error);
      this.log("error", "GraphQL request failed", error);
      throw this.normalizeError(error);
    }
  }

  /**
   * Make a paginated GraphQL query
   */
  async graphqlPaginate<T = unknown>(
    query: string,
    variables: Record<string, unknown>,
    options: {
      /** Path to the connection in the response (e.g., 'repository.issues') */
      connectionPath: string;
      /** Maximum number of items to fetch */
      maxItems?: number;
    }
  ): Promise<T[]> {
    const items: T[] = [];
    let hasNextPage = true;
    let cursor: string | null = null;

    while (hasNextPage) {
      const response = await this.graphql<Record<string, unknown>>(query, {
        ...variables,
        after: cursor,
      });

      // Navigate to the connection
      const pathParts = options.connectionPath.split(".");
      let connection = response;
      for (const part of pathParts) {
        connection = connection[part] as Record<string, unknown>;
      }

      // Extract edges/nodes
      const edges = (connection.edges || []) as Array<{
        node: T;
        cursor: string;
      }>;
      const nodes = (connection.nodes || []) as T[];

      if (edges.length > 0) {
        items.push(...edges.map((e) => e.node));
        cursor = edges[edges.length - 1]?.cursor || null;
      } else {
        items.push(...nodes);
      }

      // Check pagination
      const pageInfo = connection.pageInfo as
        | { hasNextPage: boolean; endCursor: string }
        | undefined;
      hasNextPage = pageInfo?.hasNextPage ?? false;
      cursor = pageInfo?.endCursor ?? cursor;

      // Check max items
      if (options.maxItems && items.length >= options.maxItems) {
        break;
      }
    }

    return items.slice(0, options.maxItems);
  }

  // ===========================================================================
  // RATE LIMIT MANAGEMENT
  // ===========================================================================

  /**
   * Get current rate limit info
   */
  async getRateLimit(): Promise<Record<string, RateLimit>> {
    const response = await this.rest<{
      resources: Record<
        string,
        {
          limit: number;
          remaining: number;
          reset: number;
          used: number;
        }
      >;
    }>("GET /rate_limit");

    const result: Record<string, RateLimit> = {};
    for (const [resource, data] of Object.entries(response.resources)) {
      result[resource] = { ...data, resource };
    }

    return result;
  }

  /**
   * Get remaining rate limit for a specific resource
   */
  getRateLimitRemaining(resource: string = "core"): number {
    return this.rateLimitInfo.get(resource)?.remaining ?? -1;
  }

  /**
   * Check if rate limited
   */
  isRateLimited(resource: string = "core"): boolean {
    const info = this.rateLimitInfo.get(resource);
    if (!info) return false;
    return info.remaining === 0 && Date.now() / 1000 < info.reset;
  }

  // ===========================================================================
  // CACHE MANAGEMENT
  // ===========================================================================

  /**
   * Invalidate cache entries matching pattern
   */
  invalidateCache(pattern: string | RegExp): number {
    return this.cache.invalidate(pattern);
  }

  /**
   * Clear entire cache
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { size: number; enabled: boolean; ttl: number } {
    return {
      size: this.cache.size,
      enabled: this.config.enableCache,
      ttl: this.config.cacheTtl,
    };
  }

  // ===========================================================================
  // AUTHENTICATION
  // ===========================================================================

  /**
   * Get authenticated user
   */
  async getAuthenticatedUser(): Promise<{
    login: string;
    id: number;
    name: string | null;
    email: string | null;
  }> {
    return this.rest("GET /user");
  }

  /**
   * Verify authentication
   */
  async verifyAuth(): Promise<boolean> {
    try {
      await this.getAuthenticatedUser();
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get available scopes for the token
   */
  async getTokenScopes(): Promise<string[]> {
    const response = await this.octokit.request("GET /user");
    const scopes = response.headers["x-oauth-scopes"];
    return scopes ? scopes.split(", ") : [];
  }

  // ===========================================================================
  // UTILITIES
  // ===========================================================================

  /**
   * Raw Octokit instance (for advanced use)
   */
  get raw(): InstanceType<typeof EnhancedOctokit> {
    return this.octokit;
  }

  /**
   * Get configuration
   */
  getConfig(): GitHubUniverseConfig {
    return { ...this.config };
  }

  // ===========================================================================
  // PRIVATE METHODS
  // ===========================================================================

  private getCacheKey(
    type: string,
    route: string,
    options?: Record<string, unknown>
  ): string {
    const optionsHash = options
      ? JSON.stringify(Object.entries(options).sort())
      : "";
    return `${type}:${route}:${optionsHash}`;
  }

  private interpolatePath(
    path: string,
    options?: Record<string, unknown>
  ): string {
    if (!options) return path;

    return path.replace(/{(\w+)}/g, (_, key) => {
      return String(options[key] ?? `{${key}}`);
    });
  }

  private updateRateLimitInfo(
    headers: Record<string, string | number | undefined>
  ): void {
    const limit = headers["x-ratelimit-limit"];
    const remaining = headers["x-ratelimit-remaining"];
    const reset = headers["x-ratelimit-reset"];
    const used = headers["x-ratelimit-used"];
    const resource = headers["x-ratelimit-resource"];

    if (limit !== undefined && remaining !== undefined) {
      const info: RateLimit = {
        limit: Number(limit),
        remaining: Number(remaining),
        reset: Number(reset),
        used: Number(used),
        resource: String(resource || "core"),
      };

      this.rateLimitInfo.set(info.resource, info);

      // Emit warning if low on quota
      if (info.remaining < 100) {
        this.emit(
          "ratelimit:warning",
          info.remaining,
          new Date(info.reset * 1000)
        );
      }
    }
  }

  private normalizeError(error: unknown): APIError {
    if (error && typeof error === "object") {
      const err = error as Record<string, unknown>;
      return {
        status: Number(err.status || 500),
        message: String(err.message || "Unknown error"),
        documentation_url: err.documentation_url as string | undefined,
        errors: err.errors as APIError["errors"],
      };
    }

    return {
      status: 500,
      message: String(error),
    };
  }

  private log(
    level: "debug" | "info" | "warn" | "error",
    message: string,
    data?: unknown
  ): void {
    const levels = ["debug", "info", "warn", "error", "silent"];
    const configLevel = levels.indexOf(this.config.logLevel);
    const messageLevel = levels.indexOf(level);

    if (messageLevel >= configLevel) {
      const timestamp = new Date().toISOString();
      const prefix = `[GitHubClient] [${level.toUpperCase()}] ${timestamp}`;

      if (data) {
        console[level](`${prefix} ${message}`, data);
      } else {
        console[level](`${prefix} ${message}`);
      }
    }
  }
}

// =============================================================================
// FACTORY FUNCTION
// =============================================================================

/**
 * Create a GitHub client with simplified configuration
 */
export function createGitHubClient(
  token: string,
  options?: Partial<Omit<GitHubUniverseConfig, "auth">>
): GitHubClient {
  return new GitHubClient({
    auth: { token, apiVersion: "2022-11-28" },
    ...options,
  });
}

/**
 * Create a GitHub client for GitHub Enterprise
 */
export function createEnterpriseClient(
  token: string,
  baseUrl: string,
  options?: Partial<Omit<GitHubUniverseConfig, "auth">>
): GitHubClient {
  return new GitHubClient({
    auth: { token, apiVersion: "2022-11-28", baseUrl },
    ...options,
  });
}

// Export config type
export type { GitHubUniverseConfig };
