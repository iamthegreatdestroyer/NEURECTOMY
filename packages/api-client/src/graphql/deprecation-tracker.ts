/**
 * Deprecation Tracker - Track and Monitor Deprecated Field Usage
 *
 * Provides comprehensive deprecation lifecycle management:
 * - Real-time usage tracking of deprecated elements
 * - Sunset timeline management
 * - Client impact analysis
 * - Automated notifications and warnings
 *
 * @packageDocumentation
 * @module @neurectomy/api-client/graphql
 */

import { EventEmitter } from "events";
import type { GraphQLClient } from "./client";
import type {
  SchemaRegistry,
  DeprecationInfo,
  SchemaElementType,
} from "./schema-registry";

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Deprecation lifecycle state
 */
export enum DeprecationState {
  /** Newly deprecated, still fully functional */
  ANNOUNCED = "ANNOUNCED",
  /** Warning phase, deprecation warnings active */
  WARNING = "WARNING",
  /** Final warning phase, sunset imminent */
  FINAL_WARNING = "FINAL_WARNING",
  /** Sunset in progress, may have reduced functionality */
  SUNSETTING = "SUNSETTING",
  /** Fully removed from schema */
  REMOVED = "REMOVED",
}

/**
 * Deprecation entry with full lifecycle tracking
 */
export interface DeprecationEntry {
  id: string;
  path: string;
  elementType: SchemaElementType;
  reason: string;
  replacement?: string;
  deprecatedAt: Date;
  warningAt?: Date;
  finalWarningAt?: Date;
  sunsetAt?: Date;
  removedAt?: Date;
  state: DeprecationState;
  usageStats: UsageStats;
  affectedClients: ClientInfo[];
  migrationGuideUrl?: string;
  tags: string[];
}

/**
 * Usage statistics for a deprecated element
 */
export interface UsageStats {
  totalCalls: number;
  uniqueClients: number;
  last24Hours: number;
  last7Days: number;
  last30Days: number;
  trend: "INCREASING" | "STABLE" | "DECREASING";
  peakUsageDate?: Date;
  lastUsedAt?: Date;
  callsOverTime: TimeSeriesData[];
}

/**
 * Time series data point
 */
export interface TimeSeriesData {
  timestamp: Date;
  value: number;
}

/**
 * Client information for impact analysis
 */
export interface ClientInfo {
  id: string;
  name: string;
  version: string;
  usageCount: number;
  lastUsed: Date;
  contactEmail?: string;
  isInternalClient: boolean;
  migrationStatus: MigrationStatus;
}

/**
 * Migration status for a client
 */
export enum MigrationStatus {
  NOT_STARTED = "NOT_STARTED",
  IN_PROGRESS = "IN_PROGRESS",
  COMPLETED = "COMPLETED",
  BLOCKED = "BLOCKED",
}

/**
 * Sunset schedule definition
 */
export interface SunsetSchedule {
  deprecationId: string;
  phases: SunsetPhase[];
  notifications: NotificationSchedule[];
  automaticActions: AutomaticAction[];
}

/**
 * Sunset phase definition
 */
export interface SunsetPhase {
  phase: DeprecationState;
  startDate: Date;
  endDate?: Date;
  description: string;
  actions: string[];
}

/**
 * Notification schedule for deprecation
 */
export interface NotificationSchedule {
  type: NotificationType;
  triggerDays: number;
  recipients: string[];
  template: string;
  sent: boolean;
  sentAt?: Date;
}

/**
 * Notification type
 */
export enum NotificationType {
  EMAIL = "EMAIL",
  WEBHOOK = "WEBHOOK",
  IN_APP = "IN_APP",
  LOG = "LOG",
}

/**
 * Automatic action on sunset
 */
export interface AutomaticAction {
  type: AutomaticActionType;
  triggerDate: Date;
  executed: boolean;
  executedAt?: Date;
  result?: string;
}

/**
 * Automatic action types
 */
export enum AutomaticActionType {
  RETURN_WARNING = "RETURN_WARNING",
  RATE_LIMIT = "RATE_LIMIT",
  RETURN_ERROR = "RETURN_ERROR",
  REMOVE_FROM_SCHEMA = "REMOVE_FROM_SCHEMA",
}

/**
 * Usage tracking entry
 */
export interface UsageTrackingEntry {
  path: string;
  clientId: string;
  clientVersion?: string;
  timestamp: Date;
  operationName?: string;
  queryHash?: string;
  responseTime?: number;
}

/**
 * Impact analysis result
 */
export interface ImpactAnalysis {
  deprecation: DeprecationEntry;
  totalAffectedClients: number;
  totalAffectedCalls: number;
  internalClients: ClientInfo[];
  externalClients: ClientInfo[];
  highImpactClients: ClientInfo[];
  estimatedMigrationEffort: number;
  riskLevel: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  recommendations: string[];
}

/**
 * Deprecation report
 */
export interface DeprecationReport {
  generatedAt: Date;
  totalDeprecations: number;
  byState: Record<DeprecationState, number>;
  upcomingSunsets: DeprecationEntry[];
  highUsageDeprecations: DeprecationEntry[];
  clientsNeedingMigration: ClientInfo[];
  recommendations: string[];
}

/**
 * Tracker configuration
 */
export interface DeprecationTrackerConfig {
  client: GraphQLClient;
  schemaRegistry: SchemaRegistry;
  /** Enable real-time usage tracking */
  enableTracking?: boolean;
  /** Warning threshold in days before sunset */
  warningThresholdDays?: number;
  /** Final warning threshold in days */
  finalWarningThresholdDays?: number;
  /** Auto-notify affected clients */
  autoNotify?: boolean;
  /** High usage threshold (calls per day) */
  highUsageThreshold?: number;
  /** Client identifier for tracking */
  clientId?: string;
}

/**
 * Tracker events
 */
export interface DeprecationTrackerEvents {
  "deprecation:new": (entry: DeprecationEntry) => void;
  "deprecation:warning": (entry: DeprecationEntry) => void;
  "deprecation:final-warning": (entry: DeprecationEntry) => void;
  "deprecation:sunset": (entry: DeprecationEntry) => void;
  "deprecation:removed": (entry: DeprecationEntry) => void;
  "usage:tracked": (entry: UsageTrackingEntry) => void;
  "usage:high": (entry: DeprecationEntry) => void;
  "client:affected": (
    client: ClientInfo,
    deprecation: DeprecationEntry
  ) => void;
  "notification:sent": (notification: NotificationSchedule) => void;
  error: (error: Error) => void;
}

// ============================================================================
// Deprecation Tracker Implementation
// ============================================================================

/**
 * Deprecation Tracker for monitoring deprecated field usage
 *
 * @example
 * ```typescript
 * const tracker = new DeprecationTracker({
 *   client: graphqlClient,
 *   schemaRegistry: registry,
 *   enableTracking: true,
 *   warningThresholdDays: 30,
 *   autoNotify: true
 * });
 *
 * // Track usage of a deprecated field
 * await tracker.trackUsage('Query.oldField', 'client-123');
 *
 * // Get impact analysis
 * const impact = await tracker.analyzeImpact('Query.oldField');
 *
 * // Generate deprecation report
 * const report = await tracker.generateReport();
 * ```
 */
export class DeprecationTracker extends EventEmitter {
  private readonly config: Required<DeprecationTrackerConfig>;
  private readonly deprecations: Map<string, DeprecationEntry> = new Map();
  private readonly usageBuffer: UsageTrackingEntry[] = [];
  private readonly sunsetSchedules: Map<string, SunsetSchedule> = new Map();
  private flushIntervalId: NodeJS.Timeout | null = null;
  private checkIntervalId: NodeJS.Timeout | null = null;

  constructor(config: DeprecationTrackerConfig) {
    super();
    this.config = {
      enableTracking: true,
      warningThresholdDays: 30,
      finalWarningThresholdDays: 7,
      autoNotify: false,
      highUsageThreshold: 1000,
      clientId: "unknown",
      ...config,
    };

    // Start tracking if enabled
    if (this.config.enableTracking) {
      this.startTracking();
    }

    // Load initial deprecations
    this.loadDeprecations();
  }

  // --------------------------------------------------------------------------
  // Usage Tracking
  // --------------------------------------------------------------------------

  /**
   * Track usage of a deprecated field
   */
  async trackUsage(
    path: string,
    clientId?: string,
    metadata?: {
      clientVersion?: string;
      operationName?: string;
      queryHash?: string;
      responseTime?: number;
    }
  ): Promise<void> {
    const entry: UsageTrackingEntry = {
      path,
      clientId: clientId || this.config.clientId,
      clientVersion: metadata?.clientVersion,
      timestamp: new Date(),
      operationName: metadata?.operationName,
      queryHash: metadata?.queryHash,
      responseTime: metadata?.responseTime,
    };

    this.usageBuffer.push(entry);
    this.emit("usage:tracked", entry);

    // Check if this is a high-usage deprecation
    const deprecation = this.deprecations.get(path);
    if (
      deprecation &&
      deprecation.usageStats.last24Hours > this.config.highUsageThreshold
    ) {
      this.emit("usage:high", deprecation);
    }
  }

  /**
   * Track usage from a GraphQL request
   */
  trackFromRequest(
    query: string,
    variables?: Record<string, unknown>,
    clientId?: string
  ): void {
    // Extract deprecated fields from query
    const deprecatedPaths = this.extractDeprecatedPaths(query);

    for (const path of deprecatedPaths) {
      this.trackUsage(path, clientId, {
        queryHash: this.hashQuery(query),
      });
    }
  }

  /**
   * Create a request interceptor for automatic tracking
   */
  createRequestInterceptor(): (
    query: string,
    variables?: Record<string, unknown>
  ) => void {
    return (query: string, variables?: Record<string, unknown>) => {
      this.trackFromRequest(query, variables, this.config.clientId);
    };
  }

  // --------------------------------------------------------------------------
  // Deprecation Management
  // --------------------------------------------------------------------------

  /**
   * Get all deprecations
   */
  async getDeprecations(): Promise<DeprecationEntry[]> {
    await this.loadDeprecations();
    return Array.from(this.deprecations.values());
  }

  /**
   * Get a specific deprecation
   */
  async getDeprecation(path: string): Promise<DeprecationEntry | null> {
    await this.loadDeprecations();
    return this.deprecations.get(path) || null;
  }

  /**
   * Get deprecations by state
   */
  async getDeprecationsByState(
    state: DeprecationState
  ): Promise<DeprecationEntry[]> {
    const all = await this.getDeprecations();
    return all.filter((d) => d.state === state);
  }

  /**
   * Get upcoming sunsets within days
   */
  async getUpcomingSunsets(
    withinDays: number = 30
  ): Promise<DeprecationEntry[]> {
    const all = await this.getDeprecations();
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() + withinDays);

    return all
      .filter(
        (d) =>
          d.sunsetAt &&
          d.sunsetAt <= cutoff &&
          d.state !== DeprecationState.REMOVED
      )
      .sort(
        (a, b) => (a.sunsetAt?.getTime() || 0) - (b.sunsetAt?.getTime() || 0)
      );
  }

  /**
   * Add a new deprecation
   */
  async addDeprecation(
    path: string,
    elementType: SchemaElementType,
    options: {
      reason: string;
      replacement?: string;
      sunsetDate?: Date;
      migrationGuideUrl?: string;
      tags?: string[];
    }
  ): Promise<DeprecationEntry> {
    const entry: DeprecationEntry = {
      id: `dep-${Date.now()}-${path}`,
      path,
      elementType,
      reason: options.reason,
      replacement: options.replacement,
      deprecatedAt: new Date(),
      sunsetAt: options.sunsetDate,
      state: DeprecationState.ANNOUNCED,
      usageStats: {
        totalCalls: 0,
        uniqueClients: 0,
        last24Hours: 0,
        last7Days: 0,
        last30Days: 0,
        trend: "STABLE",
        callsOverTime: [],
      },
      affectedClients: [],
      migrationGuideUrl: options.migrationGuideUrl,
      tags: options.tags || [],
    };

    this.deprecations.set(path, entry);
    this.emit("deprecation:new", entry);

    // Create sunset schedule if date provided
    if (options.sunsetDate) {
      this.createSunsetSchedule(entry);
    }

    return entry;
  }

  /**
   * Update deprecation state
   */
  async updateDeprecationState(
    path: string,
    state: DeprecationState
  ): Promise<DeprecationEntry | null> {
    const entry = this.deprecations.get(path);
    if (!entry) return null;

    const oldState = entry.state;
    entry.state = state;

    // Update phase dates
    const now = new Date();
    switch (state) {
      case DeprecationState.WARNING:
        entry.warningAt = now;
        this.emit("deprecation:warning", entry);
        break;
      case DeprecationState.FINAL_WARNING:
        entry.finalWarningAt = now;
        this.emit("deprecation:final-warning", entry);
        break;
      case DeprecationState.SUNSETTING:
        entry.sunsetAt = entry.sunsetAt || now;
        this.emit("deprecation:sunset", entry);
        break;
      case DeprecationState.REMOVED:
        entry.removedAt = now;
        this.emit("deprecation:removed", entry);
        break;
    }

    this.deprecations.set(path, entry);
    return entry;
  }

  // --------------------------------------------------------------------------
  // Impact Analysis
  // --------------------------------------------------------------------------

  /**
   * Analyze impact of a deprecation
   */
  async analyzeImpact(path: string): Promise<ImpactAnalysis | null> {
    const deprecation = await this.getDeprecation(path);
    if (!deprecation) return null;

    // Fetch latest client data
    await this.refreshClientData(path);

    const internalClients = deprecation.affectedClients.filter(
      (c) => c.isInternalClient
    );
    const externalClients = deprecation.affectedClients.filter(
      (c) => !c.isInternalClient
    );
    const highImpactClients = deprecation.affectedClients.filter(
      (c) => c.usageCount > this.config.highUsageThreshold
    );

    // Calculate risk level
    let riskLevel: ImpactAnalysis["riskLevel"];
    if (
      highImpactClients.length > 5 ||
      deprecation.usageStats.last24Hours > 10000
    ) {
      riskLevel = "CRITICAL";
    } else if (
      highImpactClients.length > 2 ||
      deprecation.usageStats.last24Hours > 1000
    ) {
      riskLevel = "HIGH";
    } else if (
      deprecation.affectedClients.length > 5 ||
      deprecation.usageStats.last24Hours > 100
    ) {
      riskLevel = "MEDIUM";
    } else {
      riskLevel = "LOW";
    }

    // Generate recommendations
    const recommendations = this.generateRecommendations(
      deprecation,
      riskLevel
    );

    // Estimate migration effort
    const estimatedMigrationEffort = this.estimateMigrationEffort(deprecation);

    return {
      deprecation,
      totalAffectedClients: deprecation.affectedClients.length,
      totalAffectedCalls: deprecation.usageStats.totalCalls,
      internalClients,
      externalClients,
      highImpactClients,
      estimatedMigrationEffort,
      riskLevel,
      recommendations,
    };
  }

  /**
   * Get all affected clients for a deprecation
   */
  async getAffectedClients(path: string): Promise<ClientInfo[]> {
    const deprecation = await this.getDeprecation(path);
    if (!deprecation) return [];

    await this.refreshClientData(path);
    return deprecation.affectedClients;
  }

  // --------------------------------------------------------------------------
  // Sunset Scheduling
  // --------------------------------------------------------------------------

  /**
   * Create a sunset schedule for a deprecation
   */
  createSunsetSchedule(deprecation: DeprecationEntry): SunsetSchedule {
    if (!deprecation.sunsetAt) {
      throw new Error("Cannot create sunset schedule without sunset date");
    }

    const warningDate = new Date(deprecation.sunsetAt);
    warningDate.setDate(
      warningDate.getDate() - this.config.warningThresholdDays
    );

    const finalWarningDate = new Date(deprecation.sunsetAt);
    finalWarningDate.setDate(
      finalWarningDate.getDate() - this.config.finalWarningThresholdDays
    );

    const schedule: SunsetSchedule = {
      deprecationId: deprecation.id,
      phases: [
        {
          phase: DeprecationState.ANNOUNCED,
          startDate: deprecation.deprecatedAt,
          endDate: warningDate,
          description: "Deprecation announced, full functionality maintained",
          actions: [
            "Add deprecation notice to documentation",
            "Update changelog",
          ],
        },
        {
          phase: DeprecationState.WARNING,
          startDate: warningDate,
          endDate: finalWarningDate,
          description: "Warning phase, deprecation warnings in responses",
          actions: [
            "Enable deprecation warnings in responses",
            "Notify affected clients",
          ],
        },
        {
          phase: DeprecationState.FINAL_WARNING,
          startDate: finalWarningDate,
          endDate: deprecation.sunsetAt,
          description: "Final warning phase, sunset imminent",
          actions: ["Send final notifications", "Prepare removal"],
        },
        {
          phase: DeprecationState.REMOVED,
          startDate: deprecation.sunsetAt,
          description: "Element removed from schema",
          actions: ["Remove element", "Update schema version"],
        },
      ],
      notifications: this.generateNotificationSchedule(deprecation),
      automaticActions: [
        {
          type: AutomaticActionType.RETURN_WARNING,
          triggerDate: warningDate,
          executed: false,
        },
        {
          type: AutomaticActionType.RETURN_ERROR,
          triggerDate: deprecation.sunsetAt,
          executed: false,
        },
      ],
    };

    this.sunsetSchedules.set(deprecation.id, schedule);
    return schedule;
  }

  /**
   * Get sunset schedule for a deprecation
   */
  getSunsetSchedule(deprecationId: string): SunsetSchedule | null {
    return this.sunsetSchedules.get(deprecationId) || null;
  }

  /**
   * Update sunset date for a deprecation
   */
  async extendSunsetDate(
    path: string,
    newDate: Date
  ): Promise<DeprecationEntry | null> {
    const entry = this.deprecations.get(path);
    if (!entry) return null;

    const oldDate = entry.sunsetAt;
    entry.sunsetAt = newDate;

    // Recreate sunset schedule
    if (entry.sunsetAt) {
      this.createSunsetSchedule(entry);
    }

    this.deprecations.set(path, entry);

    // Log the extension
    console.info(
      `[DeprecationTracker] Extended sunset date for ${path} from ${oldDate?.toISOString()} to ${newDate.toISOString()}`
    );

    return entry;
  }

  // --------------------------------------------------------------------------
  // Reporting
  // --------------------------------------------------------------------------

  /**
   * Generate a comprehensive deprecation report
   */
  async generateReport(): Promise<DeprecationReport> {
    const deprecations = await this.getDeprecations();

    // Count by state
    const byState: Record<DeprecationState, number> = {
      [DeprecationState.ANNOUNCED]: 0,
      [DeprecationState.WARNING]: 0,
      [DeprecationState.FINAL_WARNING]: 0,
      [DeprecationState.SUNSETTING]: 0,
      [DeprecationState.REMOVED]: 0,
    };

    for (const dep of deprecations) {
      byState[dep.state]++;
    }

    // Get upcoming sunsets
    const upcomingSunsets = await this.getUpcomingSunsets(30);

    // Get high usage deprecations
    const highUsageDeprecations = deprecations.filter(
      (d) => d.usageStats.last24Hours > this.config.highUsageThreshold
    );

    // Get clients needing migration
    const clientsNeedingMigration: ClientInfo[] = [];
    const seenClients = new Set<string>();

    for (const dep of deprecations) {
      for (const client of dep.affectedClients) {
        if (
          !seenClients.has(client.id) &&
          client.migrationStatus !== MigrationStatus.COMPLETED
        ) {
          clientsNeedingMigration.push(client);
          seenClients.add(client.id);
        }
      }
    }

    // Generate recommendations
    const recommendations: string[] = [];

    if (upcomingSunsets.length > 0) {
      recommendations.push(
        `${upcomingSunsets.length} deprecation(s) sunsetting within 30 days - prioritize client communication`
      );
    }

    if (highUsageDeprecations.length > 0) {
      recommendations.push(
        `${highUsageDeprecations.length} deprecated element(s) with high usage - consider extending sunset dates`
      );
    }

    if (clientsNeedingMigration.length > 0) {
      recommendations.push(
        `${clientsNeedingMigration.length} client(s) still need to migrate - reach out proactively`
      );
    }

    return {
      generatedAt: new Date(),
      totalDeprecations: deprecations.length,
      byState,
      upcomingSunsets,
      highUsageDeprecations,
      clientsNeedingMigration,
      recommendations,
    };
  }

  /**
   * Export deprecation data
   */
  async exportData(): Promise<{
    deprecations: DeprecationEntry[];
    schedules: SunsetSchedule[];
    generatedAt: Date;
  }> {
    return {
      deprecations: await this.getDeprecations(),
      schedules: Array.from(this.sunsetSchedules.values()),
      generatedAt: new Date(),
    };
  }

  // --------------------------------------------------------------------------
  // Lifecycle Management
  // --------------------------------------------------------------------------

  /**
   * Start tracking and monitoring
   */
  startTracking(): void {
    // Flush usage buffer periodically
    this.flushIntervalId = setInterval(() => {
      this.flushUsageBuffer();
    }, 30000); // Every 30 seconds

    // Check deprecation states periodically
    this.checkIntervalId = setInterval(() => {
      this.checkDeprecationStates();
    }, 3600000); // Every hour
  }

  /**
   * Stop tracking
   */
  stopTracking(): void {
    if (this.flushIntervalId) {
      clearInterval(this.flushIntervalId);
      this.flushIntervalId = null;
    }

    if (this.checkIntervalId) {
      clearInterval(this.checkIntervalId);
      this.checkIntervalId = null;
    }
  }

  /**
   * Dispose of the tracker
   */
  dispose(): void {
    this.stopTracking();
    this.flushUsageBuffer();
    this.deprecations.clear();
    this.sunsetSchedules.clear();
    this.removeAllListeners();
  }

  // --------------------------------------------------------------------------
  // Private Methods
  // --------------------------------------------------------------------------

  /**
   * Load deprecations from schema registry
   */
  private async loadDeprecations(): Promise<void> {
    try {
      const deprecations = await this.config.schemaRegistry.getDeprecations();

      for (const dep of deprecations) {
        const existing = this.deprecations.get(dep.path);

        const entry: DeprecationEntry = {
          id: existing?.id || `dep-${Date.now()}-${dep.path}`,
          path: dep.path,
          elementType: dep.elementType,
          reason: dep.reason,
          replacement: dep.replacement,
          deprecatedAt: existing?.deprecatedAt || new Date(dep.deprecatedSince),
          warningAt: existing?.warningAt,
          finalWarningAt: existing?.finalWarningAt,
          sunsetAt: dep.sunsetDate,
          removedAt: existing?.removedAt,
          state: this.calculateState(dep),
          usageStats: existing?.usageStats || {
            totalCalls: dep.usageCount,
            uniqueClients: dep.affectedClients.length,
            last24Hours: 0,
            last7Days: 0,
            last30Days: 0,
            trend: "STABLE",
            callsOverTime: [],
          },
          affectedClients:
            existing?.affectedClients ||
            dep.affectedClients.map((id) => ({
              id,
              name: id,
              version: "unknown",
              usageCount: 0,
              lastUsed: new Date(),
              isInternalClient: false,
              migrationStatus: MigrationStatus.NOT_STARTED,
            })),
          migrationGuideUrl: existing?.migrationGuideUrl,
          tags: existing?.tags || [],
        };

        this.deprecations.set(dep.path, entry);
      }
    } catch (error) {
      this.emit("error", error as Error);
    }
  }

  /**
   * Calculate deprecation state based on dates
   */
  private calculateState(dep: DeprecationInfo): DeprecationState {
    const now = new Date();

    if (dep.sunsetDate && now >= dep.sunsetDate) {
      return DeprecationState.REMOVED;
    }

    if (dep.sunsetDate) {
      const daysUntilSunset = Math.ceil(
        (dep.sunsetDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24)
      );

      if (daysUntilSunset <= this.config.finalWarningThresholdDays) {
        return DeprecationState.FINAL_WARNING;
      }

      if (daysUntilSunset <= this.config.warningThresholdDays) {
        return DeprecationState.WARNING;
      }
    }

    return DeprecationState.ANNOUNCED;
  }

  /**
   * Flush usage buffer to storage
   */
  private async flushUsageBuffer(): Promise<void> {
    if (this.usageBuffer.length === 0) return;

    const entries = [...this.usageBuffer];
    this.usageBuffer.length = 0;

    // Aggregate usage by path
    const usageByPath = new Map<string, UsageTrackingEntry[]>();
    for (const entry of entries) {
      const existing = usageByPath.get(entry.path) || [];
      existing.push(entry);
      usageByPath.set(entry.path, existing);
    }

    // Update deprecation stats
    for (const [path, pathEntries] of usageByPath) {
      const deprecation = this.deprecations.get(path);
      if (deprecation) {
        deprecation.usageStats.totalCalls += pathEntries.length;

        // Update unique clients
        const uniqueClients = new Set(pathEntries.map((e) => e.clientId));
        deprecation.usageStats.uniqueClients = Math.max(
          deprecation.usageStats.uniqueClients,
          uniqueClients.size
        );

        // Update last used
        const maxTimestamp = Math.max(
          ...pathEntries.map((e) => e.timestamp.getTime())
        );
        deprecation.usageStats.lastUsedAt = new Date(maxTimestamp);

        // Add time series data point
        deprecation.usageStats.callsOverTime.push({
          timestamp: new Date(),
          value: pathEntries.length,
        });

        // Keep only last 30 days of time series
        const cutoff = new Date();
        cutoff.setDate(cutoff.getDate() - 30);
        deprecation.usageStats.callsOverTime =
          deprecation.usageStats.callsOverTime.filter(
            (d) => d.timestamp >= cutoff
          );

        this.deprecations.set(path, deprecation);
      }
    }

    // Optionally send to server
    // await this.sendUsageToServer(entries);
  }

  /**
   * Check and update deprecation states
   */
  private async checkDeprecationStates(): Promise<void> {
    for (const [path, entry] of this.deprecations) {
      const newState = this.calculateState({
        path: entry.path,
        elementType: entry.elementType,
        reason: entry.reason,
        replacement: entry.replacement,
        deprecatedSince: entry.deprecatedAt.toISOString(),
        sunsetDate: entry.sunsetAt,
        usageCount: entry.usageStats.totalCalls,
        usagePercentage: 0,
        hasRecentUsage: !!entry.usageStats.lastUsedAt,
        affectedClients: entry.affectedClients.map((c) => c.id),
      });

      if (newState !== entry.state) {
        await this.updateDeprecationState(path, newState);
      }
    }
  }

  /**
   * Refresh client data for a deprecation
   */
  private async refreshClientData(path: string): Promise<void> {
    // This would query the server for updated client usage data
    // For now, we'll use cached data
  }

  /**
   * Generate notification schedule for a deprecation
   */
  private generateNotificationSchedule(
    deprecation: DeprecationEntry
  ): NotificationSchedule[] {
    if (!deprecation.sunsetAt) return [];

    return [
      {
        type: NotificationType.EMAIL,
        triggerDays: 60,
        recipients: [],
        template: "deprecation-notice-60",
        sent: false,
      },
      {
        type: NotificationType.EMAIL,
        triggerDays: 30,
        recipients: [],
        template: "deprecation-warning-30",
        sent: false,
      },
      {
        type: NotificationType.EMAIL,
        triggerDays: 7,
        recipients: [],
        template: "deprecation-final-warning",
        sent: false,
      },
      {
        type: NotificationType.EMAIL,
        triggerDays: 1,
        recipients: [],
        template: "deprecation-sunset-tomorrow",
        sent: false,
      },
    ];
  }

  /**
   * Generate recommendations based on deprecation data
   */
  private generateRecommendations(
    deprecation: DeprecationEntry,
    riskLevel: ImpactAnalysis["riskLevel"]
  ): string[] {
    const recommendations: string[] = [];

    // High usage recommendations
    if (deprecation.usageStats.last24Hours > this.config.highUsageThreshold) {
      recommendations.push(
        "High usage detected - consider extending sunset date or providing migration assistance"
      );
    }

    // Usage trend recommendations
    if (deprecation.usageStats.trend === "INCREASING") {
      recommendations.push(
        "Usage is increasing despite deprecation - investigate why and improve migration documentation"
      );
    }

    // Client migration recommendations
    const unmigrated = deprecation.affectedClients.filter(
      (c) => c.migrationStatus !== MigrationStatus.COMPLETED
    );
    if (unmigrated.length > 0) {
      recommendations.push(
        `${unmigrated.length} clients still need to migrate - reach out directly`
      );
    }

    // Risk level recommendations
    switch (riskLevel) {
      case "CRITICAL":
        recommendations.push(
          "CRITICAL risk level - consider pausing sunset and creating dedicated migration support"
        );
        break;
      case "HIGH":
        recommendations.push(
          "HIGH risk level - ensure all affected clients have been notified and have migration plans"
        );
        break;
      case "MEDIUM":
        recommendations.push(
          "MEDIUM risk level - monitor client migration progress closely"
        );
        break;
    }

    // Replacement recommendations
    if (!deprecation.replacement) {
      recommendations.push(
        "No replacement specified - document alternatives or explain removal rationale"
      );
    }

    // Migration guide recommendations
    if (!deprecation.migrationGuideUrl) {
      recommendations.push(
        "No migration guide provided - create documentation to help clients migrate"
      );
    }

    return recommendations;
  }

  /**
   * Estimate migration effort for a deprecation
   */
  private estimateMigrationEffort(deprecation: DeprecationEntry): number {
    let effort = 0;

    // Base effort by element type
    const effortByType: Record<string, number> = {
      TYPE: 8,
      FIELD: 2,
      ARGUMENT: 1,
      INPUT_FIELD: 2,
      ENUM_VALUE: 1,
      DIRECTIVE: 4,
      UNION_MEMBER: 3,
      INTERFACE_IMPLEMENTATION: 6,
    };

    effort += effortByType[deprecation.elementType] || 2;

    // Scale by number of affected clients
    effort *= Math.log2(deprecation.affectedClients.length + 1);

    // Scale by usage
    if (deprecation.usageStats.last24Hours > 10000) {
      effort *= 2;
    } else if (deprecation.usageStats.last24Hours > 1000) {
      effort *= 1.5;
    }

    return Math.ceil(effort);
  }

  /**
   * Extract deprecated paths from a query
   */
  private extractDeprecatedPaths(query: string): string[] {
    const paths: string[] = [];

    // Simple extraction - in production would use proper GraphQL parsing
    for (const [path] of this.deprecations) {
      const fieldName = path.split(".").pop();
      if (fieldName && query.includes(fieldName)) {
        paths.push(path);
      }
    }

    return paths;
  }

  /**
   * Hash a query string
   */
  private hashQuery(query: string): string {
    let hash = 0;
    for (let i = 0; i < query.length; i++) {
      const char = query.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return hash.toString(36);
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a deprecation tracker with default configuration
 */
export function createDeprecationTracker(
  client: GraphQLClient,
  schemaRegistry: SchemaRegistry
): DeprecationTracker {
  return new DeprecationTracker({ client, schemaRegistry });
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Format a deprecation report as a human-readable string
 */
export function formatDeprecationReport(report: DeprecationReport): string {
  const lines: string[] = [];

  lines.push("# Deprecation Report");
  lines.push(`Generated: ${report.generatedAt.toISOString()}`);
  lines.push("");
  lines.push("## Summary");
  lines.push(`Total Deprecations: ${report.totalDeprecations}`);
  lines.push("");
  lines.push("### By State");
  for (const [state, count] of Object.entries(report.byState)) {
    if (count > 0) {
      lines.push(`- ${state}: ${count}`);
    }
  }
  lines.push("");

  if (report.upcomingSunsets.length > 0) {
    lines.push("## Upcoming Sunsets (Next 30 Days)");
    for (const dep of report.upcomingSunsets) {
      lines.push(
        `- **${dep.path}**: ${dep.sunsetAt?.toLocaleDateString()} - ${dep.reason}`
      );
    }
    lines.push("");
  }

  if (report.highUsageDeprecations.length > 0) {
    lines.push("## High Usage Deprecations");
    for (const dep of report.highUsageDeprecations) {
      lines.push(`- **${dep.path}**: ${dep.usageStats.last24Hours} calls/day`);
    }
    lines.push("");
  }

  if (report.clientsNeedingMigration.length > 0) {
    lines.push("## Clients Needing Migration");
    for (const client of report.clientsNeedingMigration) {
      lines.push(
        `- **${client.name}** (${client.id}): ${client.migrationStatus}`
      );
    }
    lines.push("");
  }

  if (report.recommendations.length > 0) {
    lines.push("## Recommendations");
    for (const rec of report.recommendations) {
      lines.push(`- ${rec}`);
    }
  }

  return lines.join("\n");
}

/**
 * Format impact analysis as a human-readable string
 */
export function formatImpactAnalysis(analysis: ImpactAnalysis): string {
  const lines: string[] = [];

  lines.push(`# Impact Analysis: ${analysis.deprecation.path}`);
  lines.push("");
  lines.push(`## Risk Level: ${analysis.riskLevel}`);
  lines.push("");
  lines.push("## Metrics");
  lines.push(`- Total Affected Clients: ${analysis.totalAffectedClients}`);
  lines.push(`- Total Calls: ${analysis.totalAffectedCalls}`);
  lines.push(`- Internal Clients: ${analysis.internalClients.length}`);
  lines.push(`- External Clients: ${analysis.externalClients.length}`);
  lines.push(`- High Impact Clients: ${analysis.highImpactClients.length}`);
  lines.push(
    `- Estimated Migration Effort: ${analysis.estimatedMigrationEffort} hours`
  );
  lines.push("");

  if (analysis.highImpactClients.length > 0) {
    lines.push("## High Impact Clients");
    for (const client of analysis.highImpactClients) {
      lines.push(
        `- ${client.name}: ${client.usageCount} calls, last used ${client.lastUsed.toLocaleDateString()}`
      );
    }
    lines.push("");
  }

  if (analysis.recommendations.length > 0) {
    lines.push("## Recommendations");
    for (const rec of analysis.recommendations) {
      lines.push(`- ${rec}`);
    }
  }

  return lines.join("\n");
}
