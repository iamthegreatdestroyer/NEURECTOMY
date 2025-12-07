/**
 * Cross-Domain Event Bridge
 *
 * Enables seamless event propagation between Forge, Twin, and Foundry domains.
 * Implements publish-subscribe with domain routing, event transformation,
 * and cross-domain correlation tracking.
 *
 * @module @neurectomy/3d-engine/cross-domain/event-bridge
 * @agents @SYNAPSE @STREAM @APEX
 */

import type {
  UnifiedEvent,
  EventType,
  Domain,
  EventPriority,
  UniversalId,
  Timestamp,
} from "./types";

// ============================================================================
// Event Bridge Types
// ============================================================================

/**
 * Event handler function signature
 */
export type EventHandler<T = unknown> = (
  event: UnifiedEvent<T>
) => void | Promise<void>;

/**
 * Event filter predicate
 */
export type EventFilter = (event: UnifiedEvent) => boolean;

/**
 * Event transformer function
 */
export type EventTransformer<TIn = unknown, TOut = unknown> = (
  event: UnifiedEvent<TIn>,
  targetDomain: Domain
) => UnifiedEvent<TOut> | null;

/**
 * Subscription configuration
 */
export interface SubscriptionConfig {
  /** Event types to subscribe to */
  eventTypes?: EventType[];

  /** Source domains to listen to */
  sourceDomains?: Domain[];

  /** Priority filter */
  minPriority?: EventPriority;

  /** Custom filter predicate */
  filter?: EventFilter;

  /** Transform events before delivery */
  transformer?: EventTransformer;

  /** Handle errors */
  onError?: (error: Error, event: UnifiedEvent) => void;
}

/**
 * Subscription handle
 */
export interface Subscription {
  id: UniversalId;
  config: SubscriptionConfig;
  handler: EventHandler;
  unsubscribe: () => void;
}

/**
 * Event bridge statistics
 */
export interface BridgeStatistics {
  totalEventsPublished: number;
  totalEventsDelivered: number;
  totalEventsFailed: number;
  eventsByType: Record<EventType, number>;
  eventsByDomain: Record<Domain, number>;
  averageDeliveryTimeMs: number;
  activeSubscriptions: number;
}

// ============================================================================
// Domain Routing Rules
// ============================================================================

/**
 * Default event propagation rules between domains
 */
const DEFAULT_PROPAGATION_RULES: Record<EventType, Domain[]> = {
  // Forge events → Twin (state tracking) and Foundry (architecture changes)
  "component:created": ["twin", "foundry"],
  "component:updated": ["twin", "foundry"],
  "component:deleted": ["twin", "foundry"],
  "component:selected": ["twin"],
  "component:moved": ["twin"],
  "connection:created": ["twin", "foundry"],
  "connection:deleted": ["twin", "foundry"],
  "timeline:seek": ["twin"],
  "timeline:play": ["twin"],
  "timeline:pause": ["twin"],

  // Twin events → Forge (visualization) and Foundry (model sync)
  "state:changed": ["forge", "foundry"],
  "state:synced": ["forge"],
  "state:diverged": ["forge", "foundry"],
  "prediction:started": ["forge"],
  "prediction:completed": ["forge", "foundry"],
  "scenario:created": ["forge"],
  "scenario:evaluated": ["forge", "foundry"],

  // Foundry events → Forge (visualization) and Twin (state capture)
  "training:started": ["forge", "twin"],
  "training:step": ["forge", "twin"],
  "training:epoch": ["forge", "twin"],
  "training:completed": ["forge", "twin"],
  "training:failed": ["forge", "twin"],
  "checkpoint:saved": ["forge", "twin"],
  "checkpoint:loaded": ["forge", "twin"],
  "architecture:changed": ["forge", "twin"],

  // Universal events → All domains
  "entity:created": ["forge", "twin", "foundry"],
  "entity:updated": ["forge", "twin", "foundry"],
  "entity:deleted": ["forge", "twin", "foundry"],
  "graph:modified": ["forge", "twin", "foundry"],
  "metrics:updated": ["forge", "twin", "foundry"],
};

// ============================================================================
// Event Bridge Implementation
// ============================================================================

/**
 * CrossDomainEventBridge
 *
 * Central hub for cross-domain event propagation in NEURECTOMY.
 * Connects Forge, Twin, and Foundry modules through a unified event system.
 */
export class CrossDomainEventBridge {
  private static instance: CrossDomainEventBridge | null = null;

  private subscriptions: Map<UniversalId, Subscription> = new Map();
  private domainSubscriptions: Map<Domain, Set<UniversalId>> = new Map();
  private typeSubscriptions: Map<EventType, Set<UniversalId>> = new Map();
  private eventHistory: UnifiedEvent[] = [];
  private maxHistorySize: number = 1000;
  private statistics: BridgeStatistics;
  private eventQueue: UnifiedEvent[] = [];
  private processing: boolean = false;
  private propagationRules: Record<EventType, Domain[]>;

  /**
   * Get the singleton instance of CrossDomainEventBridge
   */
  static getInstance(options?: {
    maxHistorySize?: number;
    propagationRules?: Partial<Record<EventType, Domain[]>>;
  }): CrossDomainEventBridge {
    if (!CrossDomainEventBridge.instance) {
      CrossDomainEventBridge.instance = new CrossDomainEventBridge(options);
    }
    return CrossDomainEventBridge.instance;
  }

  constructor(
    options: {
      maxHistorySize?: number;
      propagationRules?: Partial<Record<EventType, Domain[]>>;
    } = {}
  ) {
    this.maxHistorySize = options.maxHistorySize ?? 1000;
    this.propagationRules = {
      ...DEFAULT_PROPAGATION_RULES,
      ...options.propagationRules,
    };

    this.statistics = {
      totalEventsPublished: 0,
      totalEventsDelivered: 0,
      totalEventsFailed: 0,
      eventsByType: {} as Record<EventType, number>,
      eventsByDomain: { forge: 0, twin: 0, foundry: 0 },
      averageDeliveryTimeMs: 0,
      activeSubscriptions: 0,
    };

    // Initialize domain subscription maps
    (["forge", "twin", "foundry"] as Domain[]).forEach((domain) => {
      this.domainSubscriptions.set(domain, new Set());
    });
  }

  // ==========================================================================
  // Subscription Management
  // ==========================================================================

  /**
   * Subscribe to events
   */
  subscribe(
    handler: EventHandler,
    config: SubscriptionConfig = {}
  ): Subscription {
    const id = this.generateId();

    const subscription: Subscription = {
      id,
      config,
      handler,
      unsubscribe: () => this.unsubscribe(id),
    };

    this.subscriptions.set(id, subscription);

    // Index by domains
    const domains =
      config.sourceDomains ?? (["forge", "twin", "foundry"] as Domain[]);
    domains.forEach((domain) => {
      this.domainSubscriptions.get(domain)?.add(id);
    });

    // Index by event types
    if (config.eventTypes) {
      config.eventTypes.forEach((type) => {
        if (!this.typeSubscriptions.has(type)) {
          this.typeSubscriptions.set(type, new Set());
        }
        this.typeSubscriptions.get(type)!.add(id);
      });
    }

    this.statistics.activeSubscriptions++;

    return subscription;
  }

  /**
   * Subscribe to a specific domain
   */
  subscribeToDomain(
    domain: Domain,
    handler: EventHandler,
    config: Omit<SubscriptionConfig, "sourceDomains"> = {}
  ): Subscription {
    return this.subscribe(handler, { ...config, sourceDomains: [domain] });
  }

  /**
   * Subscribe to specific event types
   */
  subscribeToTypes(
    eventTypes: EventType[],
    handler: EventHandler,
    config: Omit<SubscriptionConfig, "eventTypes"> = {}
  ): Subscription {
    return this.subscribe(handler, { ...config, eventTypes });
  }

  /**
   * Unsubscribe by ID
   */
  unsubscribe(subscriptionId: UniversalId): void {
    const subscription = this.subscriptions.get(subscriptionId);
    if (!subscription) return;

    // Remove from domain indexes
    this.domainSubscriptions.forEach((subs) => subs.delete(subscriptionId));

    // Remove from type indexes
    this.typeSubscriptions.forEach((subs) => subs.delete(subscriptionId));

    this.subscriptions.delete(subscriptionId);
    this.statistics.activeSubscriptions--;
  }

  /**
   * Remove all listeners/subscriptions
   */
  removeAllListeners(): void {
    this.subscriptions.clear();
    this.domainSubscriptions.forEach((subs) => subs.clear());
    this.typeSubscriptions.forEach((subs) => subs.clear());
    this.statistics.activeSubscriptions = 0;
  }

  // ==========================================================================
  // Event Publishing
  // ==========================================================================

  /**
   * Publish an event to the bridge
   */
  async publish<T>(event: UnifiedEvent<T>): Promise<void> {
    this.statistics.totalEventsPublished++;
    this.updateTypeStats(event.type);
    this.updateDomainStats(event.sourceDomain);

    // Add to history
    this.addToHistory(event);

    // Queue for processing
    this.eventQueue.push(event);

    // Process queue
    if (!this.processing) {
      await this.processQueue();
    }
  }

  /**
   * Create and publish an event
   */
  async emit<T>(
    type: EventType,
    sourceDomain: Domain,
    payload: T,
    options: {
      sourceEntityId?: UniversalId;
      priority?: EventPriority;
      propagatable?: boolean;
      correlationId?: UniversalId;
    } = {}
  ): Promise<UnifiedEvent<T>> {
    const event: UnifiedEvent<T> = {
      id: this.generateId(),
      type,
      sourceDomain,
      targetDomains: this.getTargetDomains(type, sourceDomain),
      timestamp: Date.now(),
      sourceEntityId: options.sourceEntityId,
      payload,
      propagatable: options.propagatable ?? true,
      priority: options.priority ?? "normal",
      correlationId: options.correlationId,
    };

    await this.publish(event);
    return event;
  }

  // ==========================================================================
  // Event Processing
  // ==========================================================================

  private async processQueue(): Promise<void> {
    this.processing = true;

    while (this.eventQueue.length > 0) {
      const event = this.eventQueue.shift()!;
      await this.deliverEvent(event);
    }

    this.processing = false;
  }

  private async deliverEvent(event: UnifiedEvent): Promise<void> {
    const startTime = Date.now();
    const matchingSubscriptions = this.findMatchingSubscriptions(event);

    const deliveryPromises = matchingSubscriptions.map(async (subscription) => {
      try {
        // Transform event if transformer is configured
        let deliveryEvent: UnifiedEvent | null = event;

        if (subscription.config.transformer) {
          // For each target domain, transform and deliver
          const targetDomains =
            subscription.config.sourceDomains ?? event.targetDomains;
          for (const targetDomain of targetDomains) {
            deliveryEvent = subscription.config.transformer(
              event,
              targetDomain
            );
            if (deliveryEvent) {
              await subscription.handler(deliveryEvent);
              this.statistics.totalEventsDelivered++;
            }
          }
        } else {
          await subscription.handler(event);
          this.statistics.totalEventsDelivered++;
        }
      } catch (error) {
        this.statistics.totalEventsFailed++;
        if (subscription.config.onError) {
          subscription.config.onError(error as Error, event);
        } else {
          console.error(
            `Event delivery failed for subscription ${subscription.id}:`,
            error
          );
        }
      }
    });

    await Promise.all(deliveryPromises);

    // Update average delivery time
    const deliveryTime = Date.now() - startTime;
    this.updateAverageDeliveryTime(deliveryTime);
  }

  private findMatchingSubscriptions(event: UnifiedEvent): Subscription[] {
    const matching: Subscription[] = [];

    for (const subscription of this.subscriptions.values()) {
      if (this.eventMatchesSubscription(event, subscription)) {
        matching.push(subscription);
      }
    }

    // Sort by priority (handlers for high-priority events first)
    matching.sort((a, b) => {
      const priorityOrder: Record<EventPriority, number> = {
        critical: 0,
        high: 1,
        normal: 2,
        low: 3,
      };
      const aPriority = a.config.minPriority ?? "low";
      const bPriority = b.config.minPriority ?? "low";
      return priorityOrder[aPriority] - priorityOrder[bPriority];
    });

    return matching;
  }

  private eventMatchesSubscription(
    event: UnifiedEvent,
    subscription: Subscription
  ): boolean {
    const { config } = subscription;

    // Check event type filter
    if (config.eventTypes && !config.eventTypes.includes(event.type)) {
      return false;
    }

    // Check source domain filter
    if (
      config.sourceDomains &&
      !config.sourceDomains.includes(event.sourceDomain)
    ) {
      // Also check if this subscription's domain is in the event's target domains
      const hasTargetMatch = config.sourceDomains.some((d) =>
        event.targetDomains.includes(d)
      );
      if (!hasTargetMatch) {
        return false;
      }
    }

    // Check priority filter
    if (config.minPriority) {
      const priorityLevel: Record<EventPriority, number> = {
        low: 0,
        normal: 1,
        high: 2,
        critical: 3,
      };
      if (priorityLevel[event.priority] < priorityLevel[config.minPriority]) {
        return false;
      }
    }

    // Check custom filter
    if (config.filter && !config.filter(event)) {
      return false;
    }

    return true;
  }

  // ==========================================================================
  // Cross-Domain Routing
  // ==========================================================================

  private getTargetDomains(type: EventType, source: Domain): Domain[] {
    const defaultTargets = this.propagationRules[type] ?? [];
    // Filter out the source domain
    return defaultTargets.filter((d) => d !== source);
  }

  /**
   * Configure custom propagation rules
   */
  setEventPropagation(type: EventType, targets: Domain[]): void {
    this.propagationRules[type] = targets;
  }

  /**
   * Get current propagation rules
   */
  getPropagationRules(): Record<EventType, Domain[]> {
    return { ...this.propagationRules };
  }

  // ==========================================================================
  // History & Statistics
  // ==========================================================================

  private addToHistory(event: UnifiedEvent): void {
    this.eventHistory.push(event);

    // Trim history if needed
    while (this.eventHistory.length > this.maxHistorySize) {
      this.eventHistory.shift();
    }
  }

  /**
   * Get event history
   */
  getHistory(
    options: {
      domain?: Domain;
      type?: EventType;
      since?: Timestamp;
      limit?: number;
    } = {}
  ): UnifiedEvent[] {
    let history = [...this.eventHistory];

    if (options.domain) {
      history = history.filter(
        (e) =>
          e.sourceDomain === options.domain ||
          e.targetDomains.includes(options.domain!)
      );
    }

    if (options.type) {
      history = history.filter((e) => e.type === options.type);
    }

    if (options.since) {
      history = history.filter((e) => e.timestamp >= options.since!);
    }

    if (options.limit) {
      history = history.slice(-options.limit);
    }

    return history;
  }

  /**
   * Get bridge statistics
   */
  getStatistics(): BridgeStatistics {
    return { ...this.statistics };
  }

  /**
   * Reset statistics
   */
  resetStatistics(): void {
    this.statistics = {
      totalEventsPublished: 0,
      totalEventsDelivered: 0,
      totalEventsFailed: 0,
      eventsByType: {} as Record<EventType, number>,
      eventsByDomain: { forge: 0, twin: 0, foundry: 0 },
      averageDeliveryTimeMs: 0,
      activeSubscriptions: this.subscriptions.size,
    };
  }

  /**
   * Clear event history
   */
  clearHistory(): void {
    this.eventHistory = [];
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  private generateId(): UniversalId {
    if (typeof crypto !== "undefined" && crypto.randomUUID) {
      return crypto.randomUUID();
    }
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
      const r = (Math.random() * 16) | 0;
      const v = c === "x" ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }

  private updateTypeStats(type: EventType): void {
    this.statistics.eventsByType[type] =
      (this.statistics.eventsByType[type] ?? 0) + 1;
  }

  private updateDomainStats(domain: Domain): void {
    this.statistics.eventsByDomain[domain]++;
  }

  private updateAverageDeliveryTime(newTime: number): void {
    const total = this.statistics.totalEventsDelivered;
    const current = this.statistics.averageDeliveryTimeMs;
    this.statistics.averageDeliveryTimeMs =
      (current * (total - 1) + newTime) / total;
  }

  // ==========================================================================
  // Lifecycle
  // ==========================================================================

  /**
   * Dispose of the event bridge
   */
  dispose(): void {
    this.subscriptions.clear();
    this.domainSubscriptions.clear();
    this.typeSubscriptions.clear();
    this.eventHistory = [];
    this.eventQueue = [];
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let globalBridge: CrossDomainEventBridge | null = null;

/**
 * Get the global event bridge instance
 */
export function getEventBridge(): CrossDomainEventBridge {
  if (!globalBridge) {
    globalBridge = new CrossDomainEventBridge();
  }
  return globalBridge;
}

/**
 * Create a new event bridge instance
 */
export function createEventBridge(
  options?: ConstructorParameters<typeof CrossDomainEventBridge>[0]
): CrossDomainEventBridge {
  return new CrossDomainEventBridge(options);
}

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * Emit an event to the global bridge
 */
export async function emit<T>(
  type: EventType,
  sourceDomain: Domain,
  payload: T,
  options?: Parameters<CrossDomainEventBridge["emit"]>[3]
): Promise<UnifiedEvent<T>> {
  return getEventBridge().emit(type, sourceDomain, payload, options);
}

/**
 * Subscribe to the global bridge
 */
export function subscribe(
  handler: EventHandler,
  config?: SubscriptionConfig
): Subscription {
  return getEventBridge().subscribe(handler, config);
}

/**
 * Subscribe to Forge events
 */
export function subscribeToForge(
  handler: EventHandler,
  config?: Omit<SubscriptionConfig, "sourceDomains">
): Subscription {
  return getEventBridge().subscribeToDomain("forge", handler, config);
}

/**
 * Subscribe to Twin events
 */
export function subscribeToTwin(
  handler: EventHandler,
  config?: Omit<SubscriptionConfig, "sourceDomains">
): Subscription {
  return getEventBridge().subscribeToDomain("twin", handler, config);
}

/**
 * Subscribe to Foundry events
 */
export function subscribeToFoundry(
  handler: EventHandler,
  config?: Omit<SubscriptionConfig, "sourceDomains">
): Subscription {
  return getEventBridge().subscribeToDomain("foundry", handler, config);
}
