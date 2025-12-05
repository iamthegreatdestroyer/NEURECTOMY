/**
 * API Versioning Strategy
 *
 * This module provides utilities for managing API versions in the client.
 * Supports URL path versioning (e.g., /api/v1/agents) and header-based versioning.
 */

export type ApiVersion = "v1" | "v2" | "v3";

export interface VersionedApiConfig {
  baseUrl: string;
  version: ApiVersion;
  strategy?: "path" | "header";
  headers?: Record<string, string>;
  timeout?: number;
}

/**
 * Default version to use when not specified.
 */
export const DEFAULT_API_VERSION: ApiVersion = "v1";

/**
 * Version header name for header-based versioning.
 */
export const API_VERSION_HEADER = "X-API-Version";

/**
 * Build versioned URL based on strategy.
 */
export function buildVersionedUrl(
  baseUrl: string,
  path: string,
  version: ApiVersion,
  strategy: "path" | "header" = "path"
): string {
  const normalizedBase = baseUrl.replace(/\/$/, "");
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;

  if (strategy === "path") {
    // URL path versioning: /api/v1/resource
    return `${normalizedBase}/api/${version}${normalizedPath}`;
  }

  // Header-based versioning: base URL unchanged
  return `${normalizedBase}${normalizedPath}`;
}

/**
 * Get version headers for header-based versioning.
 */
export function getVersionHeaders(
  version: ApiVersion,
  strategy: "path" | "header"
): Record<string, string> {
  if (strategy === "header") {
    return { [API_VERSION_HEADER]: version };
  }
  return {};
}

/**
 * Parse API version from URL or headers.
 */
export function parseVersion(
  url: string,
  headers?: Record<string, string>
): ApiVersion | null {
  // Try header first
  if (headers?.[API_VERSION_HEADER]) {
    return headers[API_VERSION_HEADER] as ApiVersion;
  }

  // Try URL path
  const match = url.match(/\/api\/(v\d+)\//);
  if (match) {
    return match[1] as ApiVersion;
  }

  return null;
}

/**
 * Check if a version is supported.
 */
export function isVersionSupported(version: string): version is ApiVersion {
  return ["v1", "v2", "v3"].includes(version);
}

/**
 * Version compatibility checker.
 */
export interface VersionCompatibility {
  current: ApiVersion;
  minimum: ApiVersion;
  deprecated?: ApiVersion[];
  sunset?: Record<ApiVersion, string>; // Version -> sunset date
}

/**
 * Check if client version is compatible with server requirements.
 */
export function checkVersionCompatibility(
  clientVersion: ApiVersion,
  serverCompatibility: VersionCompatibility
): { compatible: boolean; warnings: string[] } {
  const warnings: string[] = [];

  // Version comparison (simple numeric extraction)
  const versionNum = (v: ApiVersion) => parseInt(v.replace("v", ""), 10);
  const clientNum = versionNum(clientVersion);
  const minimumNum = versionNum(serverCompatibility.minimum);

  if (clientNum < minimumNum) {
    return {
      compatible: false,
      warnings: [
        `Client version ${clientVersion} is below minimum required ${serverCompatibility.minimum}`,
      ],
    };
  }

  // Check for deprecation
  if (serverCompatibility.deprecated?.includes(clientVersion)) {
    const sunset = serverCompatibility.sunset?.[clientVersion];
    warnings.push(
      `Version ${clientVersion} is deprecated${sunset ? ` and will be removed on ${sunset}` : ""}`
    );
  }

  return { compatible: true, warnings };
}

/**
 * Migration path between versions.
 */
export interface VersionMigration {
  from: ApiVersion;
  to: ApiVersion;
  transformRequest?: (data: unknown) => unknown;
  transformResponse?: (data: unknown) => unknown;
  notes?: string[];
}

/**
 * Common migrations between versions.
 */
export const VERSION_MIGRATIONS: VersionMigration[] = [
  {
    from: "v1",
    to: "v2",
    notes: [
      "Agent 'status' field renamed from 'state' in v1",
      "Pagination uses cursor-based instead of offset",
      "Error responses include 'requestId' field",
    ],
    transformResponse: (data: unknown) => {
      // Example transformation
      if (typeof data === "object" && data !== null && "state" in data) {
        const { state, ...rest } = data as Record<string, unknown>;
        return { ...rest, status: state };
      }
      return data;
    },
  },
  {
    from: "v2",
    to: "v3",
    notes: [
      "Workflow 'steps' renamed to 'nodes'",
      "Added 'edges' for node connections",
      "Timestamps use ISO 8601 format",
    ],
  },
];

/**
 * Get migration path between versions.
 */
export function getMigrationPath(
  from: ApiVersion,
  to: ApiVersion
): VersionMigration[] {
  const fromNum = parseInt(from.replace("v", ""), 10);
  const toNum = parseInt(to.replace("v", ""), 10);

  if (fromNum >= toNum) {
    return []; // No migration needed or downgrade not supported
  }

  return VERSION_MIGRATIONS.filter((m) => {
    const mFromNum = parseInt(m.from.replace("v", ""), 10);
    const mToNum = parseInt(m.to.replace("v", ""), 10);
    return mFromNum >= fromNum && mToNum <= toNum;
  });
}
