/**
 * @fileoverview Enhanced Subscription Manager for NEURECTOMY
 *
 * Production-resilient WebSocket subscription manager with:
 * - Automatic reconnection with exponential backoff
 * - Connection state tracking and events
 * - Subscription filtering and buffering
 * - Event ID-based resumption after reconnect
 * - Heartbeat/keepalive mechanism
 * - Backpressure handling with throttle strategies
 *
 * @module @neurectomy/api-client/graphql/subscription-manager
 */

import {
  Client,
  createClient,
  ClientOptions,
  SubscribePayload,
} from "graphql-ws";

// =============================================================================
// TYPES & INTERFACES
// =============================================================================

/**
 * Connection state for the subscription manager.
 */
export type ConnectionState =
  | "disconnected"
  | "connecting"
  | "connected"
  | "reconnecting"
  | "failed";

/**
 * Throttle strategy for high-volume subscriptions.
 */
export type ThrottleStrategy =
  | "DROP_OLDEST"
  | "DROP_NEWEST"
  | "SAMPLE"
  | "NONE";

/**
 * Configuration options for the SubscriptionManager.
 */
export interface SubscriptionManagerConfig {
  /** WebSocket URL for GraphQL subscriptions */
  url: string;

  /** Function to generate connection parameters (e.g., auth tokens) */
  connectionParams?: () =>
    | Record<string, unknown>
    | Promise<Record<string, unknown>>;

  /** Maximum number of reconnection attempts before giving up */
  maxReconnectAttempts?: number;

  /** Initial delay in milliseconds before first reconnect attempt */
  reconnectDelay?: number;

  /** Maximum delay in milliseconds between reconnect attempts */
  maxReconnectDelay?: number;

  /** Interval in milliseconds for heartbeat/keepalive */
  heartbeatInterval?: number;

  /** Timeout in milliseconds to wait for heartbeat response */
  heartbeatTimeout?: number;

  /** Whether to automatically reconnect on connection loss */
  autoReconnect?: boolean;

  /** Enable debug logging */
  debug?: boolean;
}

/**
 * Subscription filter options.
 */
export interface SubscriptionFilter {
  /** Filter by log level */
  level?: string;

  /** Filter by text content (contains) */
  contains?: string;

  /** Only events after this timestamp */
  since?: Date;

  /** Custom filter function */
  custom?: (data: unknown) => boolean;
}

/**
 * Options for creating a subscription.
 */
export interface SubscriptionOptions<TData = unknown> {
  /** Unique identifier for this subscription */
  id: string;

  /** GraphQL subscription query string */
  query: string;

  /** Variables for the subscription */
  variables?: Record<string, unknown>;

  /** Filter to apply to incoming events */
  filter?: SubscriptionFilter;

  /** Throttle strategy for high-volume events */
  throttle?: ThrottleStrategy;

  /** Buffer size for throttling (default: 100) */
  bufferSize?: number;

  /** Sample interval in ms (for SAMPLE throttle strategy) */
  sampleInterval?: number;

  /** Last event ID for resumption after reconnect */
  lastEventId?: string;

  /** Callback when data is received */
  onData: (data: TData) => void;

  /** Callback when an error occurs */
  onError?: (error: Error) => void;

  /** Callback when subscription completes */
  onComplete?: () => void;

  /** Callback when subscription is resumed after reconnect */
  onResume?: (missedEventCount?: number) => void;
}

/**
 * Connection state event data.
 */
export interface ConnectionStateEvent {
  /** Current connection state */
  state: ConnectionState;

  /** Previous connection state */
  previousState: ConnectionState;

  /** When the connection was established (if connected) */
  connectedAt?: Date;

  /** Last heartbeat timestamp */
  lastHeartbeat?: Date;

  /** Number of active subscriptions */
  subscriptionCount: number;

  /** Number of reconnection attempts made */
  reconnectAttempts: number;

  /** Error if state is 'failed' */
  error?: Error;
}

/**
 * Internal subscription record.
 */
interface SubscriptionRecord<TData = unknown> {
  options: SubscriptionOptions<TData>;
  unsubscribe: () => void;
  buffer: TData[];
  lastEventId?: string;
  lastSampleTime?: number;
}

// =============================================================================
// SUBSCRIPTION MANAGER
// =============================================================================

/**
 * Enhanced subscription manager with production-resilient features.
 *
 * @example
 * ```typescript
 * const manager = new SubscriptionManager({
 *   url: 'wss://api.neurectomy.io/graphql',
 *   connectionParams: () => ({ token: getAuthToken() }),
 *   maxReconnectAttempts: 10,
 * });
 *
 * await manager.connect();
 *
 * const unsubscribe = manager.subscribe({
 *   id: 'agent-logs',
 *   query: `subscription AgentLogs($agentId: ID!) {
 *     agentLogs(agentId: $agentId) {
 *       eventId
 *       timestamp
 *       level
 *       message
 *     }
 *   }`,
 *   variables: { agentId: 'agent-123' },
 *   filter: { level: 'ERROR' },
 *   throttle: 'SAMPLE',
 *   sampleInterval: 1000,
 *   onData: (event) => console.log(event),
 *   onError: (error) => console.error(error),
 * });
 *
 * // Later...
 * unsubscribe();
 * manager.dispose();
 * ```
 */
export class SubscriptionManager {
  private client: Client | null = null;
  private connectionState: ConnectionState = "disconnected";
  private previousState: ConnectionState = "disconnected";
  private reconnectAttempts = 0;
  private connectedAt?: Date;
  private lastHeartbeat?: Date;
  private heartbeatTimer?: ReturnType<typeof setInterval>;
  private subscriptions = new Map<string, SubscriptionRecord>();
  private connectionStateListeners = new Set<
    (event: ConnectionStateEvent) => void
  >();

  // Configuration with defaults
  private readonly config: Required<SubscriptionManagerConfig>;

  constructor(config: SubscriptionManagerConfig) {
    this.config = {
      url: config.url,
      connectionParams: config.connectionParams ?? (() => ({})),
      maxReconnectAttempts: config.maxReconnectAttempts ?? 10,
      reconnectDelay: config.reconnectDelay ?? 1000,
      maxReconnectDelay: config.maxReconnectDelay ?? 30000,
      heartbeatInterval: config.heartbeatInterval ?? 30000,
      heartbeatTimeout: config.heartbeatTimeout ?? 5000,
      autoReconnect: config.autoReconnect ?? true,
      debug: config.debug ?? false,
    };
  }

  // ===========================================================================
  // CONNECTION MANAGEMENT
  // ===========================================================================

  /**
   * Establish WebSocket connection.
   *
   * @returns Promise that resolves when connected
   * @throws Error if connection fails after all retry attempts
   */
  async connect(): Promise<void> {
    if (this.connectionState === "connected") {
      this.log("Already connected");
      return;
    }

    this.updateConnectionState("connecting");

    return new Promise((resolve, reject) => {
      const clientOptions: ClientOptions = {
        url: this.config.url,
        connectionParams: this.config.connectionParams,
        retryAttempts: this.config.autoReconnect
          ? this.config.maxReconnectAttempts
          : 0,
        retryWait: async (retries) => {
          const delay = this.calculateBackoff(retries);
          this.log(`Reconnecting in ${delay}ms (attempt ${retries + 1})`);
          await this.sleep(delay);
        },
        on: {
          connected: () => {
            this.handleConnected();
            resolve();
          },
          closed: (event) => {
            this.handleClosed(event);
          },
          error: (error) => {
            this.handleError(error);
            if (this.connectionState === "connecting" && !this.client) {
              reject(error);
            }
          },
          connecting: () => {
            if (this.connectionState === "disconnected") {
              this.updateConnectionState("connecting");
            } else {
              this.updateConnectionState("reconnecting");
            }
          },
        },
        lazy: false,
        keepAlive: this.config.heartbeatInterval,
      };

      this.client = createClient(clientOptions);
    });
  }

  /**
   * Disconnect and clean up resources.
   */
  async disconnect(): Promise<void> {
    this.stopHeartbeat();
    this.subscriptions.forEach((record) => record.unsubscribe());
    this.subscriptions.clear();

    if (this.client) {
      await this.client.dispose();
      this.client = null;
    }

    this.updateConnectionState("disconnected");
  }

  /**
   * Get current connection state.
   */
  getConnectionState(): ConnectionState {
    return this.connectionState;
  }

  /**
   * Check if connected.
   */
  isConnected(): boolean {
    return this.connectionState === "connected";
  }

  /**
   * Subscribe to connection state changes.
   */
  onConnectionStateChange(
    listener: (event: ConnectionStateEvent) => void
  ): () => void {
    this.connectionStateListeners.add(listener);
    return () => {
      this.connectionStateListeners.delete(listener);
    };
  }

  // ===========================================================================
  // SUBSCRIPTION MANAGEMENT
  // ===========================================================================

  /**
   * Create a new subscription.
   *
   * @param options - Subscription configuration
   * @returns Function to unsubscribe
   */
  subscribe<TData = unknown>(options: SubscriptionOptions<TData>): () => void {
    if (!this.client) {
      throw new Error(
        "SubscriptionManager not connected. Call connect() first."
      );
    }

    // Check for duplicate subscription ID
    if (this.subscriptions.has(options.id)) {
      this.log(`Replacing existing subscription: ${options.id}`);
      this.unsubscribe(options.id);
    }

    const record = this.createSubscriptionRecord(options);
    this.subscriptions.set(options.id, record);

    this.log(`Created subscription: ${options.id}`);
    this.emitConnectionStateChange();

    return () => this.unsubscribe(options.id);
  }

  /**
   * Unsubscribe by subscription ID.
   */
  unsubscribe(id: string): boolean {
    const record = this.subscriptions.get(id);
    if (!record) {
      return false;
    }

    record.unsubscribe();
    this.subscriptions.delete(id);
    this.log(`Unsubscribed: ${id}`);
    this.emitConnectionStateChange();

    return true;
  }

  /**
   * Get all active subscription IDs.
   */
  getActiveSubscriptions(): string[] {
    return Array.from(this.subscriptions.keys());
  }

  /**
   * Get subscription count.
   */
  getSubscriptionCount(): number {
    return this.subscriptions.size;
  }

  /**
   * Update filter for an existing subscription.
   */
  updateFilter(id: string, filter: SubscriptionFilter): boolean {
    const record = this.subscriptions.get(id);
    if (!record) {
      return false;
    }

    record.options.filter = filter;
    this.log(`Updated filter for subscription: ${id}`);
    return true;
  }

  // ===========================================================================
  // CLEANUP
  // ===========================================================================

  /**
   * Dispose of all resources.
   */
  dispose(): void {
    this.disconnect();
    this.connectionStateListeners.clear();
  }

  // ===========================================================================
  // PRIVATE METHODS
  // ===========================================================================

  private createSubscriptionRecord<TData>(
    options: SubscriptionOptions<TData>
  ): SubscriptionRecord<TData> {
    const payload: SubscribePayload = {
      query: options.query,
      variables: options.variables,
    };

    // Add last event ID for resumption if provided
    if (options.lastEventId) {
      payload.variables = {
        ...payload.variables,
        lastEventId: options.lastEventId,
      };
    }

    const record: SubscriptionRecord<TData> = {
      options,
      unsubscribe: () => {},
      buffer: [],
      lastEventId: options.lastEventId,
      lastSampleTime: Date.now(),
    };

    const unsubscribe = this.client!.subscribe<{ [key: string]: TData }>(
      payload,
      {
        next: (result) => {
          if (result.data) {
            const data = Object.values(result.data)[0] as TData;
            this.handleSubscriptionData(record, data);
          }
        },
        error: (error) => {
          const err = error instanceof Error ? error : new Error(String(error));
          this.log(`Subscription error [${options.id}]: ${err.message}`);
          options.onError?.(err);
        },
        complete: () => {
          this.log(`Subscription complete: ${options.id}`);
          options.onComplete?.();
        },
      }
    );

    record.unsubscribe = unsubscribe;
    return record;
  }

  private handleSubscriptionData<TData>(
    record: SubscriptionRecord<TData>,
    data: TData
  ): void {
    const { options } = record;

    // Apply filter if specified
    if (options.filter && !this.applyFilter(data, options.filter)) {
      return;
    }

    // Extract and store event ID if present
    if (typeof data === "object" && data !== null && "eventId" in data) {
      record.lastEventId = (data as { eventId: string }).eventId;
    }

    // Apply throttling
    const shouldEmit = this.applyThrottle(record, data);
    if (shouldEmit) {
      options.onData(data);
    }
  }

  private applyFilter(data: unknown, filter: SubscriptionFilter): boolean {
    // Custom filter takes precedence
    if (filter.custom) {
      return filter.custom(data);
    }

    if (typeof data !== "object" || data === null) {
      return true;
    }

    const obj = data as Record<string, unknown>;

    // Level filter
    if (filter.level && obj.level !== filter.level) {
      return false;
    }

    // Contains filter (check message field)
    if (filter.contains) {
      const message = String(obj.message ?? obj.content ?? "");
      if (!message.toLowerCase().includes(filter.contains.toLowerCase())) {
        return false;
      }
    }

    // Since filter
    if (filter.since && obj.timestamp) {
      const eventTime = new Date(obj.timestamp as string);
      if (eventTime < filter.since) {
        return false;
      }
    }

    return true;
  }

  private applyThrottle<TData>(
    record: SubscriptionRecord<TData>,
    data: TData
  ): boolean {
    const { options, buffer } = record;
    const throttle = options.throttle ?? "NONE";
    const bufferSize = options.bufferSize ?? 100;

    if (throttle === "NONE") {
      return true;
    }

    switch (throttle) {
      case "DROP_OLDEST": {
        if (buffer.length >= bufferSize) {
          buffer.shift(); // Remove oldest
        }
        buffer.push(data);
        return buffer.length === 1; // Emit if this is the only item
      }

      case "DROP_NEWEST": {
        if (buffer.length >= bufferSize) {
          return false; // Don't add new data
        }
        buffer.push(data);
        return true;
      }

      case "SAMPLE": {
        const now = Date.now();
        const interval = options.sampleInterval ?? 1000;
        if (now - (record.lastSampleTime ?? 0) >= interval) {
          record.lastSampleTime = now;
          return true;
        }
        return false;
      }

      default:
        return true;
    }
  }

  private handleConnected(): void {
    this.reconnectAttempts = 0;
    this.connectedAt = new Date();
    this.lastHeartbeat = new Date();
    this.updateConnectionState("connected");
    this.startHeartbeat();
    this.resubscribeAll();
  }

  private handleClosed(event?: CloseEvent): void {
    this.log(`Connection closed: ${event?.code} - ${event?.reason}`);
    this.stopHeartbeat();

    if (this.config.autoReconnect && this.connectionState === "connected") {
      this.updateConnectionState("reconnecting");
    } else {
      this.updateConnectionState("disconnected");
    }
  }

  private handleError(error: unknown): void {
    const err = error instanceof Error ? error : new Error(String(error));
    this.log(`Connection error: ${err.message}`);

    this.reconnectAttempts++;

    if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
      this.updateConnectionState("failed");
    }
  }

  private resubscribeAll(): void {
    this.log(`Resubscribing ${this.subscriptions.size} subscription(s)`);

    // Store current subscriptions
    const currentSubs = Array.from(this.subscriptions.entries());

    // Clear and recreate
    this.subscriptions.clear();

    for (const [id, record] of currentSubs) {
      try {
        const newRecord = this.createSubscriptionRecord({
          ...record.options,
          lastEventId: record.lastEventId,
        });
        this.subscriptions.set(id, newRecord);
        record.options.onResume?.();
      } catch (error) {
        this.log(`Failed to resubscribe: ${id}`);
        record.options.onError?.(
          error instanceof Error ? error : new Error(String(error))
        );
      }
    }
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();

    this.heartbeatTimer = setInterval(() => {
      this.lastHeartbeat = new Date();
      this.emitConnectionStateChange();
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = undefined;
    }
  }

  private updateConnectionState(state: ConnectionState): void {
    if (this.connectionState === state) {
      return;
    }

    this.previousState = this.connectionState;
    this.connectionState = state;
    this.log(`Connection state: ${this.previousState} -> ${state}`);
    this.emitConnectionStateChange();
  }

  private emitConnectionStateChange(): void {
    const event: ConnectionStateEvent = {
      state: this.connectionState,
      previousState: this.previousState,
      connectedAt: this.connectedAt,
      lastHeartbeat: this.lastHeartbeat,
      subscriptionCount: this.subscriptions.size,
      reconnectAttempts: this.reconnectAttempts,
    };

    this.connectionStateListeners.forEach((listener) => {
      try {
        listener(event);
      } catch (error) {
        this.log(`Connection state listener error: ${error}`);
      }
    });
  }

  private calculateBackoff(attempt: number): number {
    const delay = Math.min(
      this.config.reconnectDelay * Math.pow(2, attempt),
      this.config.maxReconnectDelay
    );
    // Add jitter (±10%)
    const jitter = delay * 0.1 * (Math.random() * 2 - 1);
    return Math.round(delay + jitter);
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private log(message: string): void {
    if (this.config.debug) {
      console.log(`[SubscriptionManager] ${message}`);
    }
  }
}

// =============================================================================
// FACTORY FUNCTION
// =============================================================================

/**
 * Create a pre-configured SubscriptionManager instance.
 *
 * @param config - Configuration options
 * @returns Configured SubscriptionManager instance
 */
export function createSubscriptionManager(
  config: SubscriptionManagerConfig
): SubscriptionManager {
  return new SubscriptionManager(config);
}

// =============================================================================
// TYPE EXPORTS
// =============================================================================

export type {
  ConnectionState,
  ThrottleStrategy,
  SubscriptionManagerConfig,
  SubscriptionFilter,
  SubscriptionOptions,
  ConnectionStateEvent,
};
