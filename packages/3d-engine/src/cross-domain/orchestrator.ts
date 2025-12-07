/**
 * Cross-Domain Orchestrator
 *
 * Central coordinator for all cross-domain operations.
 * Manages the unified state, event routing, and feature integration.
 *
 * This is the main entry point for cross-domain functionality,
 * providing a high-level API for the innovation features.
 *
 * @module @neurectomy/3d-engine/cross-domain/orchestrator
 * @agents @OMNISCIENT @NEXUS @ARCHITECT
 */

import type {
  UnifiedEntity,
  UnifiedGraph,
  UnifiedTimeline,
  UnifiedTemporalPoint,
  UnifiedEvent,
  UnifiedState,
  Domain,
  UniversalId,
  EventType,
  TrainingSession,
  TrainingMetrics,
  HealthStatus,
  ConnectionType,
  EventBusConfig,
  EventBusStatistics,
} from "./types";

import { CrossDomainEventBridge, defaultEventBusConfig } from "./event-bridge";

import {
  ForgeAdapter,
  TwinAdapter,
  FoundryAdapter,
  Adapters,
  toUnifiedEntity,
  fromUnifiedEntity,
} from "./adapters";

import {
  IsomorphismRegistry,
  detectPatterns,
  areGraphsIsomorphic,
  mapTimeline,
  transformEvent,
  type CrossDomainPattern,
} from "./isomorphisms";

// ============================================================================
// Orchestrator State
// ============================================================================

/**
 * Unified entity store
 */
interface EntityStore {
  entities: Map<UniversalId, UnifiedEntity>;
  byDomain: Map<Domain, Set<UniversalId>>;
  byType: Map<string, Set<UniversalId>>;
}

/**
 * Graph store
 */
interface GraphStore {
  graphs: Map<string, UnifiedGraph>;
  entityToGraph: Map<UniversalId, string>;
}

/**
 * Timeline store
 */
interface TimelineStore {
  timelines: Map<string, UnifiedTimeline>;
  currentTime: number;
  isPlaying: boolean;
  playbackSpeed: number;
}

/**
 * Training session store
 */
interface TrainingStore {
  sessions: Map<string, TrainingSession>;
  activeSession: string | null;
}

/**
 * Orchestrator configuration
 */
export interface OrchestratorConfig {
  /** Event bus configuration */
  eventBusConfig?: EventBusConfig;

  /** Enable automatic cross-domain sync */
  autoSync: boolean;

  /** Domains to include */
  enabledDomains: Domain[];

  /** Pattern detection interval (ms) */
  patternDetectionInterval: number;

  /** Maximum entities to track */
  maxEntities: number;

  /** Maximum timelines to track */
  maxTimelines: number;

  /** Enable debug logging */
  debug: boolean;
}

const defaultConfig: OrchestratorConfig = {
  eventBusConfig: defaultEventBusConfig,
  autoSync: true,
  enabledDomains: ["forge", "twin", "foundry"],
  patternDetectionInterval: 5000,
  maxEntities: 100000,
  maxTimelines: 1000,
  debug: false,
};

// ============================================================================
// Cross-Domain Orchestrator
// ============================================================================

/**
 * Main orchestrator for cross-domain operations
 */
export class CrossDomainOrchestrator {
  private static instance: CrossDomainOrchestrator | null = null;

  private config: OrchestratorConfig;
  private eventBridge: CrossDomainEventBridge;

  private entityStore: EntityStore;
  private graphStore: GraphStore;
  private timelineStore: TimelineStore;
  private trainingStore: TrainingStore;

  private patternCache: Map<string, CrossDomainPattern[]> = new Map();
  private patternDetectionTimer: ReturnType<typeof setInterval> | null = null;

  private adapters: typeof Adapters;
  private isomorphisms: typeof IsomorphismRegistry;

  private constructor(config: Partial<OrchestratorConfig> = {}) {
    this.config = { ...defaultConfig, ...config };

    this.eventBridge = CrossDomainEventBridge.getInstance(
      this.config.eventBusConfig
    );

    this.entityStore = {
      entities: new Map(),
      byDomain: new Map([
        ["forge", new Set()],
        ["twin", new Set()],
        ["foundry", new Set()],
      ]),
      byType: new Map(),
    };

    this.graphStore = {
      graphs: new Map(),
      entityToGraph: new Map(),
    };

    this.timelineStore = {
      timelines: new Map(),
      currentTime: Date.now(),
      isPlaying: false,
      playbackSpeed: 1.0,
    };

    this.trainingStore = {
      sessions: new Map(),
      activeSession: null,
    };

    this.adapters = Adapters;
    this.isomorphisms = IsomorphismRegistry;

    this.setupEventHandlers();
    this.startPatternDetection();
  }

  /**
   * Get singleton instance
   */
  static getInstance(
    config?: Partial<OrchestratorConfig>
  ): CrossDomainOrchestrator {
    if (!CrossDomainOrchestrator.instance) {
      CrossDomainOrchestrator.instance = new CrossDomainOrchestrator(config);
    }
    return CrossDomainOrchestrator.instance;
  }

  /**
   * Reset instance (for testing)
   */
  static resetInstance(): void {
    if (CrossDomainOrchestrator.instance) {
      CrossDomainOrchestrator.instance.dispose();
      CrossDomainOrchestrator.instance = null;
    }
  }

  /**
   * Dispose orchestrator
   */
  dispose(): void {
    if (this.patternDetectionTimer) {
      clearInterval(this.patternDetectionTimer);
    }
    this.eventBridge.dispose();
  }

  // ==========================================================================
  // Event Handlers
  // ==========================================================================

  private setupEventHandlers(): void {
    // Handle entity creation
    this.eventBridge.subscribe<UnifiedEntity>("entity:created", (event) =>
      this.handleEntityCreated(event)
    );

    // Handle entity updates
    this.eventBridge.subscribe<UnifiedEntity>("entity:updated", (event) =>
      this.handleEntityUpdated(event)
    );

    // Handle entity deletion
    this.eventBridge.subscribe<{ entityId: UniversalId }>(
      "entity:deleted",
      (event) => this.handleEntityDeleted(event)
    );

    // Handle graph modifications
    this.eventBridge.subscribe<UnifiedGraph>("graph:modified", (event) =>
      this.handleGraphModified(event)
    );

    // Handle training events
    this.eventBridge.subscribe<TrainingSession>("training:started", (event) =>
      this.handleTrainingStarted(event)
    );

    this.eventBridge.subscribe<{ sessionId: string; metrics: TrainingMetrics }>(
      "training:step",
      (event) => this.handleTrainingStep(event)
    );

    this.eventBridge.subscribe<{ sessionId: string }>(
      "training:completed",
      (event) => this.handleTrainingCompleted(event)
    );

    // Setup cross-domain routing if auto-sync enabled
    if (this.config.autoSync) {
      this.setupAutoSync();
    }
  }

  private handleEntityCreated(event: UnifiedEvent<UnifiedEntity>): void {
    const entity = event.payload;
    if (!entity) return;

    this.entityStore.entities.set(entity.id, entity);
    this.entityStore.byDomain.get(entity.domain)?.add(entity.id);

    if (!this.entityStore.byType.has(entity.type)) {
      this.entityStore.byType.set(entity.type, new Set());
    }
    this.entityStore.byType.get(entity.type)!.add(entity.id);

    this.log(`Entity created: ${entity.id} (${entity.domain}/${entity.type})`);
  }

  private handleEntityUpdated(event: UnifiedEvent<UnifiedEntity>): void {
    const entity = event.payload;
    if (!entity) return;

    this.entityStore.entities.set(entity.id, entity);
    this.log(`Entity updated: ${entity.id}`);
  }

  private handleEntityDeleted(
    event: UnifiedEvent<{ entityId: UniversalId }>
  ): void {
    const { entityId } = event.payload ?? {};
    if (!entityId) return;

    const entity = this.entityStore.entities.get(entityId);
    if (entity) {
      this.entityStore.byDomain.get(entity.domain)?.delete(entityId);
      this.entityStore.byType.get(entity.type)?.delete(entityId);
      this.entityStore.entities.delete(entityId);
    }

    this.log(`Entity deleted: ${entityId}`);
  }

  private handleGraphModified(event: UnifiedEvent<UnifiedGraph>): void {
    const graph = event.payload;
    if (!graph) return;

    this.graphStore.graphs.set(graph.id, graph);

    // Update entity-to-graph mappings
    for (const entityId of graph.nodes.keys()) {
      this.graphStore.entityToGraph.set(entityId, graph.id);
    }

    // Invalidate pattern cache for this graph
    this.patternCache.delete(graph.id);

    this.log(`Graph modified: ${graph.id}`);
  }

  private handleTrainingStarted(event: UnifiedEvent<TrainingSession>): void {
    const session = event.payload;
    if (!session) return;

    this.trainingStore.sessions.set(session.id, session);
    this.trainingStore.activeSession = session.id;

    this.log(`Training started: ${session.id}`);
  }

  private handleTrainingStep(
    event: UnifiedEvent<{ sessionId: string; metrics: TrainingMetrics }>
  ): void {
    const { sessionId, metrics } = event.payload ?? {};
    if (!sessionId || !metrics) return;

    const session = this.trainingStore.sessions.get(sessionId);
    if (session) {
      session.currentMetrics = metrics;
      session.history.push(metrics);
    }
  }

  private handleTrainingCompleted(
    event: UnifiedEvent<{ sessionId: string }>
  ): void {
    const { sessionId } = event.payload ?? {};
    if (!sessionId) return;

    const session = this.trainingStore.sessions.get(sessionId);
    if (session) {
      session.status = "completed";
      session.endTime = Date.now();
    }

    if (this.trainingStore.activeSession === sessionId) {
      this.trainingStore.activeSession = null;
    }

    this.log(`Training completed: ${sessionId}`);
  }

  private setupAutoSync(): void {
    // Route Forge events to Twin
    this.eventBridge.addRoute({
      sourceEvent: "component:updated",
      targetEvent: "state:changed",
      sourceDomain: "forge",
      targetDomains: ["twin"],
      enabled: true,
    });

    // Route Twin predictions to Forge visualization
    this.eventBridge.addRoute({
      sourceEvent: "prediction:completed",
      targetEvent: "component:updated",
      sourceDomain: "twin",
      targetDomains: ["forge"],
      enabled: true,
    });

    // Route Foundry training updates to Twin
    this.eventBridge.addRoute({
      sourceEvent: "training:step",
      targetEvent: "state:changed",
      sourceDomain: "foundry",
      targetDomains: ["twin"],
      enabled: true,
    });
  }

  private startPatternDetection(): void {
    if (this.config.patternDetectionInterval > 0) {
      this.patternDetectionTimer = setInterval(
        () => this.detectAllPatterns(),
        this.config.patternDetectionInterval
      );
    }
  }

  private detectAllPatterns(): void {
    for (const [graphId, graph] of this.graphStore.graphs) {
      if (!this.patternCache.has(graphId)) {
        const patterns = detectPatterns(graph);
        this.patternCache.set(graphId, patterns);

        if (patterns.length > 0) {
          this.eventBridge.publish({
            id: `pattern-${Date.now()}`,
            type: "metrics:updated",
            payload: { graphId, patterns },
            timestamp: Date.now(),
            sourceDomain: "twin",
            targetDomains: ["forge", "twin", "foundry"],
          });
        }
      }
    }
  }

  private log(message: string): void {
    if (this.config.debug) {
      console.log(`[CrossDomainOrchestrator] ${message}`);
    }
  }

  // ==========================================================================
  // Entity Operations
  // ==========================================================================

  /**
   * Register an entity from any domain
   */
  registerEntity(entity: unknown, domain: Domain): UnifiedEntity {
    const unified = toUnifiedEntity(entity, domain);

    this.eventBridge.publish({
      id: `create-${unified.id}`,
      type: "entity:created",
      payload: unified,
      timestamp: Date.now(),
      sourceDomain: domain,
      targetDomains: this.config.enabledDomains,
    });

    return unified;
  }

  /**
   * Update an entity
   */
  updateEntity(entityId: UniversalId, updates: Partial<UnifiedEntity>): void {
    const entity = this.entityStore.entities.get(entityId);
    if (!entity) return;

    const updated = { ...entity, ...updates, updatedAt: Date.now() };

    this.eventBridge.publish({
      id: `update-${entityId}-${Date.now()}`,
      type: "entity:updated",
      payload: updated,
      timestamp: Date.now(),
      sourceDomain: entity.domain,
      targetDomains: this.config.enabledDomains,
    });
  }

  /**
   * Delete an entity
   */
  deleteEntity(entityId: UniversalId): void {
    const entity = this.entityStore.entities.get(entityId);
    if (!entity) return;

    this.eventBridge.publish({
      id: `delete-${entityId}`,
      type: "entity:deleted",
      payload: { entityId },
      timestamp: Date.now(),
      sourceDomain: entity.domain,
      targetDomains: this.config.enabledDomains,
    });
  }

  /**
   * Get entity by ID
   */
  getEntity(entityId: UniversalId): UnifiedEntity | undefined {
    return this.entityStore.entities.get(entityId);
  }

  /**
   * Get entities by domain
   */
  getEntitiesByDomain(domain: Domain): UnifiedEntity[] {
    const ids = this.entityStore.byDomain.get(domain);
    if (!ids) return [];

    return Array.from(ids)
      .map((id) => this.entityStore.entities.get(id))
      .filter((e): e is UnifiedEntity => e !== undefined);
  }

  /**
   * Get entities by type
   */
  getEntitiesByType(type: string): UnifiedEntity[] {
    const ids = this.entityStore.byType.get(type);
    if (!ids) return [];

    return Array.from(ids)
      .map((id) => this.entityStore.entities.get(id))
      .filter((e): e is UnifiedEntity => e !== undefined);
  }

  /**
   * Convert entity to target domain
   */
  convertEntity(entityId: UniversalId, targetDomain: Domain): unknown {
    const entity = this.entityStore.entities.get(entityId);
    if (!entity) return null;

    return fromUnifiedEntity(entity, targetDomain);
  }

  // ==========================================================================
  // Graph Operations
  // ==========================================================================

  /**
   * Register a graph
   */
  registerGraph(graph: UnifiedGraph): void {
    this.eventBridge.publish({
      id: `graph-${graph.id}`,
      type: "graph:modified",
      payload: graph,
      timestamp: Date.now(),
      sourceDomain: "twin",
      targetDomains: this.config.enabledDomains,
    });
  }

  /**
   * Build graph from entities
   */
  buildGraph(entityIds: UniversalId[], graphId: string): UnifiedGraph {
    const nodes = new Map<UniversalId, UnifiedEntity>();
    const edges = new Map<
      string,
      {
        id: string;
        sourceId: UniversalId;
        targetId: UniversalId;
        type: ConnectionType;
      }
    >();

    for (const id of entityIds) {
      const entity = this.entityStore.entities.get(id);
      if (entity) {
        nodes.set(id, entity);

        // Create edges from connections
        for (const conn of entity.connections) {
          if (entityIds.includes(conn.targetId)) {
            edges.set(conn.id, {
              id: conn.id,
              sourceId: entity.id,
              targetId: conn.targetId,
              type: conn.type,
            });
          }
        }
      }
    }

    const graph: UnifiedGraph = {
      id: graphId,
      nodes,
      edges,
      metadata: {
        createdAt: Date.now(),
        modifiedAt: Date.now(),
        version: 1,
      },
    };

    this.registerGraph(graph);
    return graph;
  }

  /**
   * Get graph by ID
   */
  getGraph(graphId: string): UnifiedGraph | undefined {
    return this.graphStore.graphs.get(graphId);
  }

  /**
   * Check if two graphs are isomorphic
   */
  compareGraphs(graphId1: string, graphId2: string): boolean {
    const g1 = this.graphStore.graphs.get(graphId1);
    const g2 = this.graphStore.graphs.get(graphId2);

    if (!g1 || !g2) return false;
    return areGraphsIsomorphic(g1, g2);
  }

  /**
   * Get patterns detected in a graph
   */
  getGraphPatterns(graphId: string): CrossDomainPattern[] {
    if (!this.patternCache.has(graphId)) {
      const graph = this.graphStore.graphs.get(graphId);
      if (graph) {
        const patterns = detectPatterns(graph);
        this.patternCache.set(graphId, patterns);
      }
    }

    return this.patternCache.get(graphId) ?? [];
  }

  // ==========================================================================
  // Timeline Operations
  // ==========================================================================

  /**
   * Create a new timeline
   */
  createTimeline(id: string, entityIds: UniversalId[]): UnifiedTimeline {
    const entities = entityIds
      .map((id) => this.entityStore.entities.get(id))
      .filter((e): e is UnifiedEntity => e !== undefined);

    const initialPoint: UnifiedTemporalPoint = {
      timestamp: Date.now(),
      state: entities.reduce(
        (acc, e) => {
          acc[e.id] = e.state;
          return acc;
        },
        {} as Record<string, unknown>
      ),
      metadata: {
        description: "Initial state",
        tags: ["initial"],
        importance: 1,
        automated: false,
      },
    };

    const timeline: UnifiedTimeline = {
      id,
      entityIds,
      points: [initialPoint],
      currentIndex: 0,
      playbackState: {
        isPlaying: false,
        speed: 1.0,
        direction: "forward",
        loop: false,
      },
    };

    this.timelineStore.timelines.set(id, timeline);
    return timeline;
  }

  /**
   * Add point to timeline
   */
  addTimelinePoint(
    timelineId: string,
    state: Record<string, unknown>,
    metadata?: Partial<UnifiedTemporalPoint["metadata"]>
  ): void {
    const timeline = this.timelineStore.timelines.get(timelineId);
    if (!timeline) return;

    const point: UnifiedTemporalPoint = {
      timestamp: Date.now(),
      state,
      metadata: {
        description: "",
        tags: [],
        importance: 0.5,
        automated: true,
        ...metadata,
      },
    };

    timeline.points.push(point);
  }

  /**
   * Seek to timeline position
   */
  seekTimeline(timelineId: string, index: number): void {
    const timeline = this.timelineStore.timelines.get(timelineId);
    if (!timeline || index < 0 || index >= timeline.points.length) return;

    timeline.currentIndex = index;
    const point = timeline.points[index];

    // Restore state for each entity
    for (const [entityId, state] of Object.entries(point.state)) {
      this.updateEntity(entityId as UniversalId, { state });
    }

    this.eventBridge.publish({
      id: `timeline-seek-${Date.now()}`,
      type: "timeline:seek",
      payload: { timelineId, index, timestamp: point.timestamp },
      timestamp: Date.now(),
      sourceDomain: "forge",
      targetDomains: ["twin"],
    });
  }

  /**
   * Play timeline
   */
  playTimeline(timelineId: string): void {
    const timeline = this.timelineStore.timelines.get(timelineId);
    if (!timeline) return;

    timeline.playbackState.isPlaying = true;

    this.eventBridge.publish({
      id: `timeline-play-${Date.now()}`,
      type: "timeline:play",
      payload: { timelineId },
      timestamp: Date.now(),
      sourceDomain: "forge",
      targetDomains: ["twin"],
    });
  }

  /**
   * Pause timeline
   */
  pauseTimeline(timelineId: string): void {
    const timeline = this.timelineStore.timelines.get(timelineId);
    if (!timeline) return;

    timeline.playbackState.isPlaying = false;

    this.eventBridge.publish({
      id: `timeline-pause-${Date.now()}`,
      type: "timeline:pause",
      payload: { timelineId },
      timestamp: Date.now(),
      sourceDomain: "forge",
      targetDomains: ["twin"],
    });
  }

  /**
   * Get timeline
   */
  getTimeline(timelineId: string): UnifiedTimeline | undefined {
    return this.timelineStore.timelines.get(timelineId);
  }

  /**
   * Map timeline to another domain
   */
  mapTimelineToDomain(
    timelineId: string,
    targetDomain: Domain
  ): UnifiedTimeline | null {
    const timeline = this.timelineStore.timelines.get(timelineId);
    if (!timeline) return null;

    return mapTimeline(timeline, targetDomain);
  }

  // ==========================================================================
  // Training Operations
  // ==========================================================================

  /**
   * Start a training session
   */
  startTraining(
    config: Omit<TrainingSession, "id" | "status" | "history" | "startTime">
  ): TrainingSession {
    const session: TrainingSession = {
      ...config,
      id: `training-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      status: "running",
      history: [],
      startTime: Date.now(),
    };

    this.eventBridge.publish({
      id: `training-start-${session.id}`,
      type: "training:started",
      payload: session,
      timestamp: Date.now(),
      sourceDomain: "foundry",
      targetDomains: ["forge", "twin"],
    });

    return session;
  }

  /**
   * Record training step
   */
  recordTrainingStep(sessionId: string, metrics: TrainingMetrics): void {
    this.eventBridge.publish({
      id: `training-step-${sessionId}-${Date.now()}`,
      type: "training:step",
      payload: { sessionId, metrics },
      timestamp: Date.now(),
      sourceDomain: "foundry",
      targetDomains: ["forge", "twin"],
    });
  }

  /**
   * Complete training
   */
  completeTraining(sessionId: string): void {
    this.eventBridge.publish({
      id: `training-complete-${sessionId}`,
      type: "training:completed",
      payload: { sessionId },
      timestamp: Date.now(),
      sourceDomain: "foundry",
      targetDomains: ["forge", "twin"],
    });
  }

  /**
   * Get training session
   */
  getTrainingSession(sessionId: string): TrainingSession | undefined {
    return this.trainingStore.sessions.get(sessionId);
  }

  /**
   * Get active training session
   */
  getActiveTraining(): TrainingSession | null {
    if (!this.trainingStore.activeSession) return null;
    return (
      this.trainingStore.sessions.get(this.trainingStore.activeSession) ?? null
    );
  }

  // ==========================================================================
  // Cross-Domain Operations
  // ==========================================================================

  /**
   * Sync entity across domains
   */
  syncEntityAcrossDomains(entityId: UniversalId): void {
    const entity = this.entityStore.entities.get(entityId);
    if (!entity) return;

    for (const domain of this.config.enabledDomains) {
      if (domain !== entity.domain) {
        const transformed = fromUnifiedEntity(entity, domain);

        this.eventBridge.publish({
          id: `sync-${entityId}-${domain}`,
          type: "entity:updated",
          payload: toUnifiedEntity(transformed, domain),
          timestamp: Date.now(),
          sourceDomain: entity.domain,
          targetDomains: [domain],
        });
      }
    }
  }

  /**
   * Bridge event between domains
   */
  bridgeEvent<T>(event: UnifiedEvent<T>, targetDomain: Domain): void {
    const transformed = transformEvent(event, targetDomain);
    if (transformed) {
      this.eventBridge.publish(transformed);
    }
  }

  /**
   * Get orchestrator statistics
   */
  getStatistics(): {
    entities: { total: number; byDomain: Record<Domain, number> };
    graphs: number;
    timelines: number;
    training: { total: number; active: boolean };
    eventBus: EventBusStatistics;
    patterns: number;
  } {
    const byDomain: Record<Domain, number> = {
      forge: this.entityStore.byDomain.get("forge")?.size ?? 0,
      twin: this.entityStore.byDomain.get("twin")?.size ?? 0,
      foundry: this.entityStore.byDomain.get("foundry")?.size ?? 0,
    };

    return {
      entities: {
        total: this.entityStore.entities.size,
        byDomain,
      },
      graphs: this.graphStore.graphs.size,
      timelines: this.timelineStore.timelines.size,
      training: {
        total: this.trainingStore.sessions.size,
        active: this.trainingStore.activeSession !== null,
      },
      eventBus: this.eventBridge.getStatistics(),
      patterns: Array.from(this.patternCache.values()).reduce(
        (sum, p) => sum + p.length,
        0
      ),
    };
  }

  /**
   * Get health status
   */
  getHealth(): HealthStatus {
    const stats = this.getStatistics();
    const eventBusHealth = stats.eventBus.errors.length === 0;
    const entityHealth = stats.entities.total < this.config.maxEntities;
    const timelineHealth = stats.timelines < this.config.maxTimelines;

    const status: HealthStatus["status"] =
      eventBusHealth && entityHealth && timelineHealth
        ? "healthy"
        : eventBusHealth && (entityHealth || timelineHealth)
          ? "degraded"
          : "unhealthy";

    return {
      status,
      checks: {
        eventBus: { healthy: eventBusHealth },
        entities: { healthy: entityHealth },
        timelines: { healthy: timelineHealth },
      },
      lastCheck: Date.now(),
    };
  }

  // ==========================================================================
  // Accessors
  // ==========================================================================

  /**
   * Get event bridge
   */
  getEventBridge(): CrossDomainEventBridge {
    return this.eventBridge;
  }

  /**
   * Get adapters
   */
  getAdapters(): typeof Adapters {
    return this.adapters;
  }

  /**
   * Get isomorphisms
   */
  getIsomorphisms(): typeof IsomorphismRegistry {
    return this.isomorphisms;
  }
}

// ============================================================================
// Convenience Exports
// ============================================================================

/**
 * Get the orchestrator instance
 */
export const getOrchestrator = CrossDomainOrchestrator.getInstance;

/**
 * Quick access to orchestrator operations
 */
export const crossDomain = {
  get orchestrator() {
    return CrossDomainOrchestrator.getInstance();
  },

  get eventBridge() {
    return CrossDomainOrchestrator.getInstance().getEventBridge();
  },

  get adapters() {
    return CrossDomainOrchestrator.getInstance().getAdapters();
  },

  get isomorphisms() {
    return CrossDomainOrchestrator.getInstance().getIsomorphisms();
  },

  registerEntity: (entity: unknown, domain: Domain) =>
    CrossDomainOrchestrator.getInstance().registerEntity(entity, domain),

  getEntity: (id: UniversalId) =>
    CrossDomainOrchestrator.getInstance().getEntity(id),

  buildGraph: (entityIds: UniversalId[], graphId: string) =>
    CrossDomainOrchestrator.getInstance().buildGraph(entityIds, graphId),

  createTimeline: (id: string, entityIds: UniversalId[]) =>
    CrossDomainOrchestrator.getInstance().createTimeline(id, entityIds),

  startTraining: (
    config: Omit<TrainingSession, "id" | "status" | "history" | "startTime">
  ) => CrossDomainOrchestrator.getInstance().startTraining(config),

  getStatistics: () => CrossDomainOrchestrator.getInstance().getStatistics(),

  getHealth: () => CrossDomainOrchestrator.getInstance().getHealth(),
};

export default CrossDomainOrchestrator;
