/**
 * Schema Registry - Version Management and Compatibility Checking
 *
 * Provides comprehensive schema governance through:
 * - Version tracking and comparison
 * - Compatibility assessment between versions
 * - Breaking change detection
 * - Schema evolution guidance
 *
 * @packageDocumentation
 * @module @neurectomy/api-client/graphql
 */

import { EventEmitter } from "events";
import type { GraphQLClient } from "./client";

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Semantic version representation
 */
export interface SemanticVersion {
  major: number;
  minor: number;
  patch: number;
  prerelease?: string;
  build?: string;
}

/**
 * Schema version information
 */
export interface SchemaVersionInfo {
  version: string;
  semver: SemanticVersion;
  releaseDate: Date;
  isLatest: boolean;
  isSupported: boolean;
  deprecatedAt?: Date;
  sunsetAt?: Date;
  changelog: ChangelogEntry[];
  breakingChanges: BreakingChange[];
  migrationGuideUrl?: string;
}

/**
 * Change type enumeration
 */
export enum ChangeType {
  ADDED = "ADDED",
  CHANGED = "CHANGED",
  REMOVED = "REMOVED",
  DEPRECATED = "DEPRECATED",
  FIXED = "FIXED",
  SECURITY = "SECURITY",
  PERFORMANCE = "PERFORMANCE",
}

/**
 * Changelog entry for version changes
 */
export interface ChangelogEntry {
  type: ChangeType;
  description: string;
  affectedPaths: string[];
  issueNumber?: string;
  author?: string;
}

/**
 * Breaking change type enumeration
 */
export enum BreakingChangeType {
  FIELD_REMOVED = "FIELD_REMOVED",
  TYPE_REMOVED = "TYPE_REMOVED",
  ENUM_VALUE_REMOVED = "ENUM_VALUE_REMOVED",
  REQUIRED_ARGUMENT_ADDED = "REQUIRED_ARGUMENT_ADDED",
  TYPE_CHANGED = "TYPE_CHANGED",
  NULLABLE_TO_NON_NULL = "NULLABLE_TO_NON_NULL",
  DEFAULT_VALUE_CHANGED = "DEFAULT_VALUE_CHANGED",
  INTERFACE_IMPLEMENTATION_REMOVED = "INTERFACE_IMPLEMENTATION_REMOVED",
  DIRECTIVE_REMOVED = "DIRECTIVE_REMOVED",
  DIRECTIVE_ARGUMENT_REMOVED = "DIRECTIVE_ARGUMENT_REMOVED",
  UNION_MEMBER_REMOVED = "UNION_MEMBER_REMOVED",
}

/**
 * Breaking change description
 */
export interface BreakingChange {
  path: string;
  type: BreakingChangeType;
  description: string;
  previousBehavior: string;
  newBehavior: string;
  migrationSteps: string[];
  codeExample?: CodeExample;
}

/**
 * Code example for migration guidance
 */
export interface CodeExample {
  before: string;
  after: string;
  language: string;
}

/**
 * Compatibility level enumeration
 */
export enum CompatibilityLevel {
  FULL = "FULL",
  BACKWARD = "BACKWARD",
  FORWARD = "FORWARD",
  BREAKING = "BREAKING",
  UNKNOWN = "UNKNOWN",
}

/**
 * Compatibility report between two schema versions
 */
export interface CompatibilityReport {
  sourceVersion: string;
  targetVersion: string;
  isCompatible: boolean;
  compatibilityLevel: CompatibilityLevel;
  breakingChanges: BreakingChange[];
  additions: string[];
  deprecations: string[];
  warnings: string[];
  migrationEstimate?: MigrationEstimate;
}

/**
 * Migration effort estimate
 */
export interface MigrationEstimate {
  hours: number;
  complexity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  automatedPercentage: number;
  requiredSteps: number;
  optionalSteps: number;
}

/**
 * Schema element type
 */
export enum SchemaElementType {
  TYPE = "TYPE",
  FIELD = "FIELD",
  ARGUMENT = "ARGUMENT",
  INPUT_FIELD = "INPUT_FIELD",
  ENUM_VALUE = "ENUM_VALUE",
  DIRECTIVE = "DIRECTIVE",
  UNION_MEMBER = "UNION_MEMBER",
  INTERFACE_IMPLEMENTATION = "INTERFACE_IMPLEMENTATION",
}

/**
 * Deprecation information
 */
export interface DeprecationInfo {
  path: string;
  elementType: SchemaElementType;
  reason: string;
  replacement?: string;
  deprecatedSince: string;
  sunsetDate?: Date;
  usageCount: number;
  usagePercentage: number;
  hasRecentUsage: boolean;
  affectedClients: string[];
}

/**
 * Schema health metrics
 */
export interface SchemaHealth {
  score: number;
  deprecatedCount: number;
  documentationCoverage: number;
  undocumentedTypes: string[];
  undocumentedFields: string[];
  complexTypes: ComplexTypeInfo[];
  circularReferences: string[];
}

/**
 * Complex type information
 */
export interface ComplexTypeInfo {
  path: string;
  fieldCount: number;
  depth: number;
  cyclomaticComplexity: number;
}

/**
 * Schema registry configuration
 */
export interface SchemaRegistryConfig {
  /** GraphQL client for fetching schema info */
  client: GraphQLClient;
  /** Enable automatic version checking */
  autoCheckVersion?: boolean;
  /** Version check interval in milliseconds */
  checkInterval?: number;
  /** Cache TTL for version info in milliseconds */
  cacheTTL?: number;
  /** Warn on deprecated field usage */
  warnOnDeprecation?: boolean;
  /** Strict mode - throw on incompatible versions */
  strictMode?: boolean;
}

/**
 * Schema registry events
 */
export interface SchemaRegistryEvents {
  "version:changed": (
    newVersion: SchemaVersionInfo,
    oldVersion: SchemaVersionInfo
  ) => void;
  "version:deprecated": (version: SchemaVersionInfo) => void;
  "version:sunset": (version: SchemaVersionInfo) => void;
  "deprecation:detected": (deprecation: DeprecationInfo) => void;
  "compatibility:warning": (report: CompatibilityReport) => void;
  error: (error: Error) => void;
}

// ============================================================================
// Schema Registry Implementation
// ============================================================================

/**
 * Schema Registry for managing GraphQL schema versions and compatibility
 *
 * @example
 * ```typescript
 * const registry = new SchemaRegistry({
 *   client: graphqlClient,
 *   autoCheckVersion: true,
 *   checkInterval: 60000,
 *   warnOnDeprecation: true
 * });
 *
 * // Get current version
 * const version = await registry.getCurrentVersion();
 *
 * // Check compatibility before upgrade
 * const report = await registry.checkCompatibility('1.0.0', '2.0.0');
 * if (report.isCompatible) {
 *   await registry.migrateToVersion('2.0.0');
 * }
 * ```
 */
export class SchemaRegistry extends EventEmitter {
  private readonly config: Required<SchemaRegistryConfig>;
  private currentVersion: SchemaVersionInfo | null = null;
  private versionCache: Map<string, SchemaVersionInfo> = new Map();
  private deprecationCache: Map<string, DeprecationInfo> = new Map();
  private checkIntervalId: NodeJS.Timeout | null = null;
  private lastCheckTime: number = 0;

  constructor(config: SchemaRegistryConfig) {
    super();
    this.config = {
      autoCheckVersion: true,
      checkInterval: 60000, // 1 minute
      cacheTTL: 300000, // 5 minutes
      warnOnDeprecation: true,
      strictMode: false,
      ...config,
    };

    if (this.config.autoCheckVersion) {
      this.startVersionCheck();
    }
  }

  // --------------------------------------------------------------------------
  // Version Management
  // --------------------------------------------------------------------------

  /**
   * Get the current schema version
   */
  async getCurrentVersion(): Promise<SchemaVersionInfo> {
    if (this.currentVersion && !this.isCacheExpired()) {
      return this.currentVersion;
    }

    const version = await this.fetchSchemaVersion();
    this.currentVersion = version;
    this.versionCache.set(version.version, version);
    this.lastCheckTime = Date.now();

    return version;
  }

  /**
   * Get all available schema versions
   */
  async getAvailableVersions(): Promise<SchemaVersionInfo[]> {
    const query = `
      query GetSchemaVersions {
        schemaVersions {
          version
          releaseDate
          isLatest
          isSupported
          deprecatedAt
          sunsetAt
          changelog {
            type
            description
            affectedPaths
            issueNumber
            author
          }
          breakingChanges {
            path
            breakingType
            description
            previousBehavior
            newBehavior
            migrationSteps
            codeExample {
              before
              after
              language
            }
          }
          migrationGuideUrl
        }
      }
    `;

    const result = await this.config.client.query<{
      schemaVersions: Array<{
        version: string;
        releaseDate: string;
        isLatest: boolean;
        isSupported: boolean;
        deprecatedAt?: string;
        sunsetAt?: string;
        changelog: Array<{
          type: string;
          description: string;
          affectedPaths: string[];
          issueNumber?: string;
          author?: string;
        }>;
        breakingChanges: Array<{
          path: string;
          breakingType: string;
          description: string;
          previousBehavior: string;
          newBehavior: string;
          migrationSteps: string[];
          codeExample?: {
            before: string;
            after: string;
            language: string;
          };
        }>;
        migrationGuideUrl?: string;
      }>;
    }>(query);

    const versions = result.schemaVersions.map((v) => this.mapSchemaVersion(v));

    // Update cache
    versions.forEach((version) => {
      this.versionCache.set(version.version, version);
    });

    return versions;
  }

  /**
   * Get a specific schema version
   */
  async getVersion(version: string): Promise<SchemaVersionInfo | null> {
    // Check cache first
    const cached = this.versionCache.get(version);
    if (cached) {
      return cached;
    }

    const query = `
      query GetSchemaVersion($version: String!) {
        schemaVersion(version: $version) {
          version
          releaseDate
          isLatest
          isSupported
          deprecatedAt
          sunsetAt
          changelog {
            type
            description
            affectedPaths
            issueNumber
            author
          }
          breakingChanges {
            path
            breakingType
            description
            previousBehavior
            newBehavior
            migrationSteps
            codeExample {
              before
              after
              language
            }
          }
          migrationGuideUrl
        }
      }
    `;

    const result = await this.config.client.query<{
      schemaVersion: {
        version: string;
        releaseDate: string;
        isLatest: boolean;
        isSupported: boolean;
        deprecatedAt?: string;
        sunsetAt?: string;
        changelog: Array<{
          type: string;
          description: string;
          affectedPaths: string[];
          issueNumber?: string;
          author?: string;
        }>;
        breakingChanges: Array<{
          path: string;
          breakingType: string;
          description: string;
          previousBehavior: string;
          newBehavior: string;
          migrationSteps: string[];
          codeExample?: {
            before: string;
            after: string;
            language: string;
          };
        }>;
        migrationGuideUrl?: string;
      } | null;
    }>(query, { version });

    if (!result.schemaVersion) {
      return null;
    }

    const versionInfo = this.mapSchemaVersion(result.schemaVersion);
    this.versionCache.set(version, versionInfo);

    return versionInfo;
  }

  // --------------------------------------------------------------------------
  // Compatibility Checking
  // --------------------------------------------------------------------------

  /**
   * Check compatibility between two schema versions
   */
  async checkCompatibility(
    sourceVersion: string,
    targetVersion: string
  ): Promise<CompatibilityReport> {
    const query = `
      query CheckCompatibility($from: String!, $to: String!) {
        checkCompatibility(fromVersion: $from, toVersion: $to) {
          isCompatible
          compatibilityLevel
          breakingChanges {
            path
            breakingType
            description
            previousBehavior
            newBehavior
            migrationSteps
            codeExample {
              before
              after
              language
            }
          }
          additions
          deprecations
          warnings
        }
      }
    `;

    const result = await this.config.client.query<{
      checkCompatibility: {
        isCompatible: boolean;
        compatibilityLevel: string;
        breakingChanges: Array<{
          path: string;
          breakingType: string;
          description: string;
          previousBehavior: string;
          newBehavior: string;
          migrationSteps: string[];
          codeExample?: {
            before: string;
            after: string;
            language: string;
          };
        }>;
        additions: string[];
        deprecations: string[];
        warnings: string[];
      };
    }>(query, { from: sourceVersion, to: targetVersion });

    const report: CompatibilityReport = {
      sourceVersion,
      targetVersion,
      isCompatible: result.checkCompatibility.isCompatible,
      compatibilityLevel: result.checkCompatibility
        .compatibilityLevel as CompatibilityLevel,
      breakingChanges: result.checkCompatibility.breakingChanges.map((bc) => ({
        path: bc.path,
        type: bc.breakingType as BreakingChangeType,
        description: bc.description,
        previousBehavior: bc.previousBehavior,
        newBehavior: bc.newBehavior,
        migrationSteps: bc.migrationSteps,
        codeExample: bc.codeExample,
      })),
      additions: result.checkCompatibility.additions,
      deprecations: result.checkCompatibility.deprecations,
      warnings: result.checkCompatibility.warnings,
      migrationEstimate: this.estimateMigrationEffort(
        result.checkCompatibility.breakingChanges
      ),
    };

    // Emit warning if breaking changes detected
    if (!report.isCompatible || report.breakingChanges.length > 0) {
      this.emit("compatibility:warning", report);
    }

    // Throw in strict mode if incompatible
    if (this.config.strictMode && !report.isCompatible) {
      throw new SchemaIncompatibilityError(report);
    }

    return report;
  }

  /**
   * Check if current client version is compatible with server schema
   */
  async validateClientCompatibility(
    clientVersion: string
  ): Promise<CompatibilityReport> {
    const serverVersion = await this.getCurrentVersion();
    return this.checkCompatibility(clientVersion, serverVersion.version);
  }

  // --------------------------------------------------------------------------
  // Deprecation Tracking
  // --------------------------------------------------------------------------

  /**
   * Get all deprecations in the current schema
   */
  async getDeprecations(): Promise<DeprecationInfo[]> {
    const query = `
      query GetDeprecations {
        deprecations {
          path
          elementType
          reason
          replacement
          deprecatedSince
          sunsetDate
          usageCount
          usagePercentage
          hasRecentUsage
          affectedClients
        }
      }
    `;

    const result = await this.config.client.query<{
      deprecations: Array<{
        path: string;
        elementType: string;
        reason: string;
        replacement?: string;
        deprecatedSince: string;
        sunsetDate?: string;
        usageCount: number;
        usagePercentage: number;
        hasRecentUsage: boolean;
        affectedClients: string[];
      }>;
    }>(query);

    const deprecations = result.deprecations.map((d) => ({
      path: d.path,
      elementType: d.elementType as SchemaElementType,
      reason: d.reason,
      replacement: d.replacement,
      deprecatedSince: d.deprecatedSince,
      sunsetDate: d.sunsetDate ? new Date(d.sunsetDate) : undefined,
      usageCount: d.usageCount,
      usagePercentage: d.usagePercentage,
      hasRecentUsage: d.hasRecentUsage,
      affectedClients: d.affectedClients,
    }));

    // Update cache
    deprecations.forEach((d) => {
      this.deprecationCache.set(d.path, d);
    });

    return deprecations;
  }

  /**
   * Check if a specific schema element is deprecated
   */
  async isDeprecated(path: string): Promise<DeprecationInfo | null> {
    // Check cache first
    const cached = this.deprecationCache.get(path);
    if (cached) {
      return cached;
    }

    const deprecations = await this.getDeprecations();
    return deprecations.find((d) => d.path === path) || null;
  }

  /**
   * Track usage of a deprecated field
   */
  async trackDeprecatedUsage(path: string, clientId: string): Promise<void> {
    const mutation = `
      mutation TrackDeprecatedUsage($path: String!, $clientId: String!) {
        trackDeprecatedUsage(path: $path, clientId: $clientId)
      }
    `;

    await this.config.client.mutate(mutation, { path, clientId });

    // Emit deprecation event
    const deprecation = await this.isDeprecated(path);
    if (deprecation && this.config.warnOnDeprecation) {
      this.emit("deprecation:detected", deprecation);
      console.warn(
        `[SchemaRegistry] Deprecated field used: ${path}. ${deprecation.reason}` +
          (deprecation.replacement
            ? ` Use ${deprecation.replacement} instead.`
            : "")
      );
    }
  }

  // --------------------------------------------------------------------------
  // Schema Health
  // --------------------------------------------------------------------------

  /**
   * Get schema health metrics
   */
  async getSchemaHealth(): Promise<SchemaHealth> {
    const query = `
      query GetSchemaHealth {
        schemaMetadata {
          health {
            score
            deprecatedCount
            documentationCoverage
            undocumentedTypes
            undocumentedFields
          }
        }
      }
    `;

    const result = await this.config.client.query<{
      schemaMetadata: {
        health: {
          score: number;
          deprecatedCount: number;
          documentationCoverage: number;
          undocumentedTypes: string[];
          undocumentedFields: string[];
        };
      };
    }>(query);

    return {
      score: result.schemaMetadata.health.score,
      deprecatedCount: result.schemaMetadata.health.deprecatedCount,
      documentationCoverage: result.schemaMetadata.health.documentationCoverage,
      undocumentedTypes: result.schemaMetadata.health.undocumentedTypes,
      undocumentedFields: result.schemaMetadata.health.undocumentedFields,
      complexTypes: [], // Would need additional query
      circularReferences: [], // Would need schema introspection
    };
  }

  // --------------------------------------------------------------------------
  // Version Comparison Utilities
  // --------------------------------------------------------------------------

  /**
   * Parse a version string into semantic version components
   */
  parseVersion(version: string): SemanticVersion {
    const match = version.match(
      /^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$/
    );

    if (!match) {
      throw new Error(`Invalid semantic version: ${version}`);
    }

    return {
      major: parseInt(match[1], 10),
      minor: parseInt(match[2], 10),
      patch: parseInt(match[3], 10),
      prerelease: match[4],
      build: match[5],
    };
  }

  /**
   * Compare two semantic versions
   * Returns: -1 if a < b, 0 if a == b, 1 if a > b
   */
  compareVersions(
    a: string | SemanticVersion,
    b: string | SemanticVersion
  ): number {
    const versionA = typeof a === "string" ? this.parseVersion(a) : a;
    const versionB = typeof b === "string" ? this.parseVersion(b) : b;

    // Compare major
    if (versionA.major !== versionB.major) {
      return versionA.major > versionB.major ? 1 : -1;
    }

    // Compare minor
    if (versionA.minor !== versionB.minor) {
      return versionA.minor > versionB.minor ? 1 : -1;
    }

    // Compare patch
    if (versionA.patch !== versionB.patch) {
      return versionA.patch > versionB.patch ? 1 : -1;
    }

    // Compare prerelease (no prerelease > prerelease)
    if (versionA.prerelease && !versionB.prerelease) return -1;
    if (!versionA.prerelease && versionB.prerelease) return 1;
    if (versionA.prerelease && versionB.prerelease) {
      return versionA.prerelease.localeCompare(versionB.prerelease);
    }

    return 0;
  }

  /**
   * Check if a version satisfies a range
   */
  satisfiesRange(version: string, range: string): boolean {
    // Simple range checking (supports: ^x.y.z, ~x.y.z, >=x.y.z, etc.)
    const parsed = this.parseVersion(version);

    if (range.startsWith("^")) {
      // Compatible with version (allows minor and patch updates)
      const rangeVersion = this.parseVersion(range.slice(1));
      return (
        parsed.major === rangeVersion.major &&
        (parsed.minor > rangeVersion.minor ||
          (parsed.minor === rangeVersion.minor &&
            parsed.patch >= rangeVersion.patch))
      );
    }

    if (range.startsWith("~")) {
      // Approximately equivalent (allows patch updates)
      const rangeVersion = this.parseVersion(range.slice(1));
      return (
        parsed.major === rangeVersion.major &&
        parsed.minor === rangeVersion.minor &&
        parsed.patch >= rangeVersion.patch
      );
    }

    if (range.startsWith(">=")) {
      const rangeVersion = this.parseVersion(range.slice(2));
      return this.compareVersions(parsed, rangeVersion) >= 0;
    }

    if (range.startsWith(">")) {
      const rangeVersion = this.parseVersion(range.slice(1));
      return this.compareVersions(parsed, rangeVersion) > 0;
    }

    if (range.startsWith("<=")) {
      const rangeVersion = this.parseVersion(range.slice(2));
      return this.compareVersions(parsed, rangeVersion) <= 0;
    }

    if (range.startsWith("<")) {
      const rangeVersion = this.parseVersion(range.slice(1));
      return this.compareVersions(parsed, rangeVersion) < 0;
    }

    // Exact match
    const rangeVersion = this.parseVersion(range);
    return this.compareVersions(parsed, rangeVersion) === 0;
  }

  // --------------------------------------------------------------------------
  // Lifecycle Management
  // --------------------------------------------------------------------------

  /**
   * Start automatic version checking
   */
  startVersionCheck(): void {
    if (this.checkIntervalId) return;

    this.checkIntervalId = setInterval(async () => {
      try {
        const newVersion = await this.fetchSchemaVersion();

        if (
          this.currentVersion &&
          newVersion.version !== this.currentVersion.version
        ) {
          const oldVersion = this.currentVersion;
          this.currentVersion = newVersion;
          this.emit("version:changed", newVersion, oldVersion);

          // Check if old version is now deprecated
          if (oldVersion.deprecatedAt) {
            this.emit("version:deprecated", oldVersion);
          }
        } else {
          this.currentVersion = newVersion;
        }

        this.lastCheckTime = Date.now();
      } catch (error) {
        this.emit("error", error as Error);
      }
    }, this.config.checkInterval);
  }

  /**
   * Stop automatic version checking
   */
  stopVersionCheck(): void {
    if (this.checkIntervalId) {
      clearInterval(this.checkIntervalId);
      this.checkIntervalId = null;
    }
  }

  /**
   * Clear all caches
   */
  clearCache(): void {
    this.versionCache.clear();
    this.deprecationCache.clear();
    this.currentVersion = null;
    this.lastCheckTime = 0;
  }

  /**
   * Dispose of the registry
   */
  dispose(): void {
    this.stopVersionCheck();
    this.clearCache();
    this.removeAllListeners();
  }

  // --------------------------------------------------------------------------
  // Private Methods
  // --------------------------------------------------------------------------

  private async fetchSchemaVersion(): Promise<SchemaVersionInfo> {
    const query = `
      query GetCurrentSchemaVersion {
        schemaVersion {
          version
          releaseDate
          isLatest
          isSupported
          deprecatedAt
          sunsetAt
          changelog {
            type
            description
            affectedPaths
            issueNumber
            author
          }
          breakingChanges {
            path
            breakingType
            description
            previousBehavior
            newBehavior
            migrationSteps
            codeExample {
              before
              after
              language
            }
          }
          migrationGuideUrl
        }
      }
    `;

    const result = await this.config.client.query<{
      schemaVersion: {
        version: string;
        releaseDate: string;
        isLatest: boolean;
        isSupported: boolean;
        deprecatedAt?: string;
        sunsetAt?: string;
        changelog: Array<{
          type: string;
          description: string;
          affectedPaths: string[];
          issueNumber?: string;
          author?: string;
        }>;
        breakingChanges: Array<{
          path: string;
          breakingType: string;
          description: string;
          previousBehavior: string;
          newBehavior: string;
          migrationSteps: string[];
          codeExample?: {
            before: string;
            after: string;
            language: string;
          };
        }>;
        migrationGuideUrl?: string;
      };
    }>(query);

    return this.mapSchemaVersion(result.schemaVersion);
  }

  private mapSchemaVersion(raw: {
    version: string;
    releaseDate: string;
    isLatest: boolean;
    isSupported: boolean;
    deprecatedAt?: string;
    sunsetAt?: string;
    changelog: Array<{
      type: string;
      description: string;
      affectedPaths: string[];
      issueNumber?: string;
      author?: string;
    }>;
    breakingChanges: Array<{
      path: string;
      breakingType: string;
      description: string;
      previousBehavior: string;
      newBehavior: string;
      migrationSteps: string[];
      codeExample?: {
        before: string;
        after: string;
        language: string;
      };
    }>;
    migrationGuideUrl?: string;
  }): SchemaVersionInfo {
    return {
      version: raw.version,
      semver: this.parseVersion(raw.version),
      releaseDate: new Date(raw.releaseDate),
      isLatest: raw.isLatest,
      isSupported: raw.isSupported,
      deprecatedAt: raw.deprecatedAt ? new Date(raw.deprecatedAt) : undefined,
      sunsetAt: raw.sunsetAt ? new Date(raw.sunsetAt) : undefined,
      changelog: raw.changelog.map((c) => ({
        type: c.type as ChangeType,
        description: c.description,
        affectedPaths: c.affectedPaths,
        issueNumber: c.issueNumber,
        author: c.author,
      })),
      breakingChanges: raw.breakingChanges.map((bc) => ({
        path: bc.path,
        type: bc.breakingType as BreakingChangeType,
        description: bc.description,
        previousBehavior: bc.previousBehavior,
        newBehavior: bc.newBehavior,
        migrationSteps: bc.migrationSteps,
        codeExample: bc.codeExample,
      })),
      migrationGuideUrl: raw.migrationGuideUrl,
    };
  }

  private isCacheExpired(): boolean {
    return Date.now() - this.lastCheckTime > this.config.cacheTTL;
  }

  private estimateMigrationEffort(
    breakingChanges: Array<{
      path: string;
      breakingType: string;
      description: string;
      previousBehavior: string;
      newBehavior: string;
      migrationSteps: string[];
      codeExample?: {
        before: string;
        after: string;
        language: string;
      };
    }>
  ): MigrationEstimate {
    const changeCount = breakingChanges.length;

    // Estimate based on number and type of changes
    let hours = 0;
    let automatedSteps = 0;
    let totalSteps = 0;

    for (const change of breakingChanges) {
      totalSteps += change.migrationSteps.length;

      // Simple heuristic based on change type
      switch (change.breakingType) {
        case "FIELD_REMOVED":
        case "ENUM_VALUE_REMOVED":
          hours += 0.5;
          if (change.codeExample) automatedSteps++;
          break;
        case "TYPE_REMOVED":
        case "TYPE_CHANGED":
          hours += 2;
          break;
        case "REQUIRED_ARGUMENT_ADDED":
        case "NULLABLE_TO_NON_NULL":
          hours += 1;
          if (change.codeExample) automatedSteps++;
          break;
        default:
          hours += 1;
      }
    }

    // Determine complexity
    let complexity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
    if (changeCount === 0) {
      complexity = "LOW";
    } else if (changeCount <= 3) {
      complexity = "LOW";
    } else if (changeCount <= 10) {
      complexity = "MEDIUM";
    } else if (changeCount <= 20) {
      complexity = "HIGH";
    } else {
      complexity = "CRITICAL";
    }

    return {
      hours: Math.max(hours, 0.5),
      complexity,
      automatedPercentage:
        totalSteps > 0 ? (automatedSteps / totalSteps) * 100 : 100,
      requiredSteps: totalSteps,
      optionalSteps: 0,
    };
  }
}

// ============================================================================
// Custom Errors
// ============================================================================

/**
 * Error thrown when schema incompatibility is detected in strict mode
 */
export class SchemaIncompatibilityError extends Error {
  constructor(public readonly report: CompatibilityReport) {
    super(
      `Schema incompatibility detected between versions ${report.sourceVersion} and ${report.targetVersion}. ` +
        `${report.breakingChanges.length} breaking change(s) found.`
    );
    this.name = "SchemaIncompatibilityError";
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a schema registry instance with default configuration
 */
export function createSchemaRegistry(client: GraphQLClient): SchemaRegistry {
  return new SchemaRegistry({ client });
}

/**
 * Create a schema registry with strict mode enabled
 */
export function createStrictSchemaRegistry(
  client: GraphQLClient
): SchemaRegistry {
  return new SchemaRegistry({
    client,
    strictMode: true,
    warnOnDeprecation: true,
  });
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Format a compatibility report as a human-readable string
 */
export function formatCompatibilityReport(report: CompatibilityReport): string {
  const lines: string[] = [];

  lines.push(`Schema Compatibility Report`);
  lines.push(`===========================`);
  lines.push(`From: ${report.sourceVersion}`);
  lines.push(`To: ${report.targetVersion}`);
  lines.push(`Compatible: ${report.isCompatible ? "Yes" : "No"}`);
  lines.push(`Level: ${report.compatibilityLevel}`);
  lines.push("");

  if (report.breakingChanges.length > 0) {
    lines.push(`Breaking Changes (${report.breakingChanges.length}):`);
    for (const change of report.breakingChanges) {
      lines.push(`  - [${change.type}] ${change.path}`);
      lines.push(`    ${change.description}`);
      if (change.migrationSteps.length > 0) {
        lines.push(`    Migration steps:`);
        change.migrationSteps.forEach((step, i) => {
          lines.push(`      ${i + 1}. ${step}`);
        });
      }
    }
    lines.push("");
  }

  if (report.additions.length > 0) {
    lines.push(`Additions (${report.additions.length}):`);
    report.additions.forEach((addition) => lines.push(`  + ${addition}`));
    lines.push("");
  }

  if (report.deprecations.length > 0) {
    lines.push(`Deprecations (${report.deprecations.length}):`);
    report.deprecations.forEach((deprecation) =>
      lines.push(`  ! ${deprecation}`)
    );
    lines.push("");
  }

  if (report.warnings.length > 0) {
    lines.push(`Warnings (${report.warnings.length}):`);
    report.warnings.forEach((warning) => lines.push(`  ⚠ ${warning}`));
    lines.push("");
  }

  if (report.migrationEstimate) {
    lines.push(`Migration Estimate:`);
    lines.push(`  Hours: ~${report.migrationEstimate.hours}`);
    lines.push(`  Complexity: ${report.migrationEstimate.complexity}`);
    lines.push(
      `  Automated: ${report.migrationEstimate.automatedPercentage.toFixed(0)}%`
    );
    lines.push(
      `  Steps: ${report.migrationEstimate.requiredSteps} required, ${report.migrationEstimate.optionalSteps} optional`
    );
  }

  return lines.join("\n");
}
