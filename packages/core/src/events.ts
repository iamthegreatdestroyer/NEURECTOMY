/**
 * Event Sourcing Module
 *
 * Implements event sourcing patterns for state management.
 * Events are immutable records of state changes that can be replayed.
 */

import { generateId, createTimestamp } from "./utils/identifiers";

// =============================================================================
// Event Types
// =============================================================================

/**
 * Base event interface - all events must have these properties.
 */
export interface BaseEvent<T extends string = string, P = unknown> {
  /** Unique event identifier */
  id: string;
  /** Event type (e.g., "agent.created", "workflow.started") */
  type: T;
  /** Event payload */
  payload: P;
  /** Event metadata */
  metadata: EventMetadata;
  /** ISO 8601 timestamp */
  timestamp: string;
  /** Event version for schema evolution */
  version: number;
}

/**
 * Event metadata for tracing and auditing.
 */
export interface EventMetadata {
  /** ID of the aggregate this event belongs to */
  aggregateId: string;
  /** Type of aggregate (e.g., "agent", "workflow") */
  aggregateType: string;
  /** User who triggered the event */
  userId?: string;
  /** Correlation ID for tracing across services */
  correlationId?: string;
  /** Causation ID (ID of event that caused this one) */
  causationId?: string;
}

/**
 * Event store record (persisted format).
 */
export interface EventRecord<E extends BaseEvent = BaseEvent> {
  /** Global sequence number */
  sequenceNumber: number;
  /** Stream position (per aggregate) */
  streamPosition: number;
  /** The event data */
  event: E;
  /** When the event was stored */
  storedAt: string;
}

// =============================================================================
// Event Store
// =============================================================================

/**
 * Event store interface - backends must implement this.
 */
export interface EventStore {
  /** Append events to a stream */
  append(streamId: string, events: BaseEvent[]): Promise<void>;

  /** Read events from a stream */
  read(
    streamId: string,
    fromPosition?: number,
    limit?: number
  ): Promise<EventRecord[]>;

  /** Read all events (global ordering) */
  readAll(fromSequence?: number, limit?: number): Promise<EventRecord[]>;

  /** Subscribe to new events */
  subscribe(handler: EventHandler): () => void;

  /** Subscribe to a specific stream */
  subscribeToStream(streamId: string, handler: EventHandler): () => void;
}

/**
 * Event handler function type.
 */
export type EventHandler<E extends BaseEvent = BaseEvent> = (
  event: E,
  record: EventRecord<E>
) => void | Promise<void>;

// =============================================================================
// In-Memory Event Store (for development/testing)
// =============================================================================

/**
 * In-memory event store implementation.
 * Use for development and testing only.
 */
export class InMemoryEventStore implements EventStore {
  private events: EventRecord[] = [];
  private streams: Map<string, EventRecord[]> = new Map();
  private handlers: Set<EventHandler> = new Set();
  private streamHandlers: Map<string, Set<EventHandler>> = new Map();
  private sequenceNumber = 0;

  async append(streamId: string, events: BaseEvent[]): Promise<void> {
    const stream = this.streams.get(streamId) || [];
    let streamPosition = stream.length;

    for (const event of events) {
      this.sequenceNumber++;
      streamPosition++;

      const record: EventRecord = {
        sequenceNumber: this.sequenceNumber,
        streamPosition,
        event,
        storedAt: createTimestamp(),
      };

      this.events.push(record);
      stream.push(record);

      // Notify handlers
      await this.notifyHandlers(record);
    }

    this.streams.set(streamId, stream);
  }

  async read(
    streamId: string,
    fromPosition = 0,
    limit = 100
  ): Promise<EventRecord[]> {
    const stream = this.streams.get(streamId) || [];
    return stream.slice(fromPosition, fromPosition + limit);
  }

  async readAll(fromSequence = 0, limit = 100): Promise<EventRecord[]> {
    return this.events
      .filter((r) => r.sequenceNumber > fromSequence)
      .slice(0, limit);
  }

  subscribe(handler: EventHandler): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  subscribeToStream(streamId: string, handler: EventHandler): () => void {
    const handlers = this.streamHandlers.get(streamId) || new Set();
    handlers.add(handler);
    this.streamHandlers.set(streamId, handlers);

    return () => handlers.delete(handler);
  }

  private async notifyHandlers(record: EventRecord): Promise<void> {
    // Global handlers
    for (const handler of this.handlers) {
      await handler(record.event, record);
    }

    // Stream-specific handlers
    const streamId = record.event.metadata.aggregateId;
    const streamHandlers = this.streamHandlers.get(streamId);
    if (streamHandlers) {
      for (const handler of streamHandlers) {
        await handler(record.event, record);
      }
    }
  }

  /** Clear all events (for testing) */
  clear(): void {
    this.events = [];
    this.streams.clear();
    this.sequenceNumber = 0;
  }
}

// =============================================================================
// Aggregate
// =============================================================================

/**
 * Base aggregate class - domain objects that emit events.
 */
export abstract class Aggregate<TState, TEvent extends BaseEvent> {
  protected state: TState;
  private uncommittedEvents: TEvent[] = [];
  private version = 0;

  constructor(
    public readonly id: string,
    initialState: TState
  ) {
    this.state = initialState;
  }

  /**
   * Get current state.
   */
  getState(): TState {
    return this.state;
  }

  /**
   * Get current version.
   */
  getVersion(): number {
    return this.version;
  }

  /**
   * Get uncommitted events.
   */
  getUncommittedEvents(): TEvent[] {
    return [...this.uncommittedEvents];
  }

  /**
   * Mark events as committed.
   */
  markEventsAsCommitted(): void {
    this.uncommittedEvents = [];
  }

  /**
   * Apply an event to update state.
   * Must be implemented by concrete aggregates.
   */
  protected abstract applyEvent(event: TEvent): void;

  /**
   * Record a new event.
   */
  protected recordEvent(event: TEvent): void {
    this.applyEvent(event);
    this.uncommittedEvents.push(event);
    this.version++;
  }

  /**
   * Rebuild state from events.
   */
  loadFromHistory(events: TEvent[]): void {
    for (const event of events) {
      this.applyEvent(event);
      this.version++;
    }
  }
}

// =============================================================================
// Event Factory
// =============================================================================

/**
 * Create a new event with standard fields populated.
 */
export function createEvent<T extends string, P>(
  type: T,
  payload: P,
  metadata: Omit<EventMetadata, "correlationId" | "causationId"> & {
    correlationId?: string;
    causationId?: string;
  },
  version = 1
): BaseEvent<T, P> {
  return {
    id: generateId(),
    type,
    payload,
    metadata,
    timestamp: createTimestamp(),
    version,
  };
}

// =============================================================================
// Projections
// =============================================================================

/**
 * Projection function type - transforms events into read models.
 */
export type Projection<TReadModel, TEvent extends BaseEvent = BaseEvent> = {
  name: string;
  initialState: TReadModel;
  apply: (state: TReadModel, event: TEvent) => TReadModel;
  shouldHandle: (event: BaseEvent) => event is TEvent;
};

/**
 * Run a projection over a set of events.
 */
export function runProjection<TReadModel, TEvent extends BaseEvent>(
  projection: Projection<TReadModel, TEvent>,
  events: BaseEvent[]
): TReadModel {
  let state = projection.initialState;

  for (const event of events) {
    if (projection.shouldHandle(event)) {
      state = projection.apply(state, event);
    }
  }

  return state;
}

/**
 * Create a live projection that updates on new events.
 */
export function createLiveProjection<TReadModel, TEvent extends BaseEvent>(
  projection: Projection<TReadModel, TEvent>,
  eventStore: EventStore,
  initialEvents?: BaseEvent[]
): {
  getState: () => TReadModel;
  stop: () => void;
} {
  let state = runProjection(projection, initialEvents || []);

  const unsubscribe = eventStore.subscribe((event) => {
    if (projection.shouldHandle(event)) {
      state = projection.apply(state, event);
    }
  });

  return {
    getState: () => state,
    stop: unsubscribe,
  };
}

// =============================================================================
// Snapshots
// =============================================================================

/**
 * Snapshot for aggregate state.
 */
export interface Snapshot<TState> {
  aggregateId: string;
  state: TState;
  version: number;
  timestamp: string;
}

/**
 * Create a snapshot of aggregate state.
 */
export function createSnapshot<TState>(
  aggregate: Aggregate<TState, BaseEvent>
): Snapshot<TState> {
  return {
    aggregateId: aggregate.id,
    state: aggregate.getState(),
    version: aggregate.getVersion(),
    timestamp: createTimestamp(),
  };
}
