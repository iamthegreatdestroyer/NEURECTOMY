/**
 * WebSocket Client for Intelligence Foundry
 *
 * Provides real-time updates for training metrics, experiment status,
 * trial completion, and optimization progress.
 *
 * @module intelligence-foundry/websocket
 */

// ============================================================================
// Type Definitions
// ============================================================================

export type WebSocketEventType =
  | "training:metrics"
  | "experiment:status"
  | "trial:complete"
  | "optimization:progress"
  | "connection:open"
  | "connection:close"
  | "connection:error";

export interface TrainingMetricsEvent {
  type: "training:metrics";
  run_id: string;
  epoch: number;
  metrics: {
    train_loss?: number;
    val_loss?: number;
    train_accuracy?: number;
    val_accuracy?: number;
    learning_rate?: number;
    gpu_memory?: number;
    throughput?: number;
  };
  timestamp: number;
}

export interface ExperimentStatusEvent {
  type: "experiment:status";
  experiment_id: string;
  run_id: string;
  status: "running" | "completed" | "failed" | "stopped";
  progress?: number;
  message?: string;
  timestamp: number;
}

export interface TrialCompleteEvent {
  type: "trial:complete";
  study_id: string;
  trial_number: number;
  trial_id: string;
  params: Record<string, any>;
  value: number | null;
  state: "complete" | "pruned" | "failed";
  duration: number;
  timestamp: number;
}

export interface OptimizationProgressEvent {
  type: "optimization:progress";
  study_id: string;
  n_trials_completed: number;
  n_trials_total: number;
  best_value: number | null;
  best_params: Record<string, any> | null;
  estimated_time_remaining?: number;
  timestamp: number;
}

export interface ConnectionEvent {
  type: "connection:open" | "connection:close" | "connection:error";
  message?: string;
  timestamp: number;
}

export type WebSocketEvent =
  | TrainingMetricsEvent
  | ExperimentStatusEvent
  | TrialCompleteEvent
  | OptimizationProgressEvent
  | ConnectionEvent;

export type WebSocketEventHandler<T extends WebSocketEvent = WebSocketEvent> = (
  event: T
) => void | Promise<void>;

export interface WebSocketClientOptions {
  url?: string;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  debug?: boolean;
}

// ============================================================================
// WebSocket Client Class
// ============================================================================

export class IntelligenceFoundryWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectInterval: number;
  private maxReconnectAttempts: number;
  private heartbeatInterval: number;
  private debug: boolean;
  private reconnectAttempts = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private eventHandlers: Map<WebSocketEventType, Set<WebSocketEventHandler>> =
    new Map();
  private isIntentionallyClosed = false;

  constructor(options: WebSocketClientOptions = {}) {
    this.url = options.url || this.getWebSocketUrl();
    this.reconnectInterval = options.reconnectInterval || 3000;
    this.maxReconnectAttempts = options.maxReconnectAttempts || 10;
    this.heartbeatInterval = options.heartbeatInterval || 30000;
    this.debug = options.debug || false;
  }

  /**
   * Connect to the WebSocket server
   *
   * @example
   * ```typescript
   * const ws = new IntelligenceFoundryWebSocket();
   * await ws.connect();
   * ```
   */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.isIntentionallyClosed = false;
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          this.log("WebSocket connected");
          this.reconnectAttempts = 0;
          this.startHeartbeat();
          this.emit({
            type: "connection:open",
            timestamp: Date.now(),
          });
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event.data);
        };

        this.ws.onerror = (error) => {
          this.log("WebSocket error:", error);
          this.emit({
            type: "connection:error",
            message: "WebSocket error occurred",
            timestamp: Date.now(),
          });
          reject(error);
        };

        this.ws.onclose = () => {
          this.log("WebSocket closed");
          this.stopHeartbeat();
          this.emit({
            type: "connection:close",
            timestamp: Date.now(),
          });

          if (!this.isIntentionallyClosed) {
            this.attemptReconnect();
          }
        };
      } catch (error) {
        this.log("Failed to create WebSocket:", error);
        reject(error);
      }
    });
  }

  /**
   * Disconnect from the WebSocket server
   *
   * @example
   * ```typescript
   * await ws.disconnect();
   * ```
   */
  disconnect(): void {
    this.isIntentionallyClosed = true;
    this.stopHeartbeat();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * Subscribe to a specific event type
   *
   * @param eventType - Type of event to listen for
   * @param handler - Callback function to handle the event
   * @returns Unsubscribe function
   *
   * @example
   * ```typescript
   * const unsubscribe = ws.on('training:metrics', (event) => {
   *   console.log(`Epoch ${event.epoch}: loss=${event.metrics.train_loss}`);
   * });
   *
   * // Later, to unsubscribe:
   * unsubscribe();
   * ```
   */
  on<T extends WebSocketEventType>(
    eventType: T,
    handler: WebSocketEventHandler<Extract<WebSocketEvent, { type: T }>>
  ): () => void {
    if (!this.eventHandlers.has(eventType)) {
      this.eventHandlers.set(eventType, new Set());
    }
    this.eventHandlers.get(eventType)!.add(handler as WebSocketEventHandler);

    // Return unsubscribe function
    return () => {
      const handlers = this.eventHandlers.get(eventType);
      if (handlers) {
        handlers.delete(handler as WebSocketEventHandler);
      }
    };
  }

  /**
   * Subscribe to multiple event types at once
   *
   * @param eventTypes - Array of event types to listen for
   * @param handler - Callback function to handle all events
   * @returns Unsubscribe function
   *
   * @example
   * ```typescript
   * const unsubscribe = ws.onMultiple(
   *   ['training:metrics', 'experiment:status'],
   *   (event) => {
   *     console.log('Event received:', event.type);
   *   }
   * );
   * ```
   */
  onMultiple(
    eventTypes: WebSocketEventType[],
    handler: WebSocketEventHandler
  ): () => void {
    const unsubscribers = eventTypes.map((type) =>
      this.on(type, handler as any)
    );
    return () => {
      unsubscribers.forEach((unsub) => unsub());
    };
  }

  /**
   * Remove a specific event handler
   *
   * @param eventType - Type of event
   * @param handler - Handler to remove
   */
  off(eventType: WebSocketEventType, handler: WebSocketEventHandler): void {
    const handlers = this.eventHandlers.get(eventType);
    if (handlers) {
      handlers.delete(handler);
    }
  }

  /**
   * Remove all handlers for a specific event type
   *
   * @param eventType - Type of event to clear handlers for
   */
  clearHandlers(eventType: WebSocketEventType): void {
    this.eventHandlers.delete(eventType);
  }

  /**
   * Remove all event handlers
   */
  clearAllHandlers(): void {
    this.eventHandlers.clear();
  }

  /**
   * Send a message to the server
   *
   * @param data - Data to send
   */
  send(data: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      this.log("Cannot send message: WebSocket not connected");
    }
  }

  /**
   * Check if WebSocket is currently connected
   */
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Get current connection state
   */
  getReadyState(): number | null {
    return this.ws?.readyState ?? null;
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleMessage(data: string): void {
    try {
      const event = JSON.parse(data) as WebSocketEvent;
      this.emit(event);
    } catch (error) {
      this.log("Failed to parse WebSocket message:", error);
    }
  }

  /**
   * Emit an event to all registered handlers
   */
  private emit(event: WebSocketEvent): void {
    const handlers = this.eventHandlers.get(event.type);
    if (handlers) {
      handlers.forEach((handler) => {
        try {
          handler(event);
        } catch (error) {
          this.log(`Error in event handler for ${event.type}:`, error);
        }
      });
    }
  }

  /**
   * Attempt to reconnect after connection loss
   */
  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.log(
        `Max reconnect attempts (${this.maxReconnectAttempts}) reached. Giving up.`
      );
      return;
    }

    this.reconnectAttempts++;
    this.log(
      `Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`
    );

    this.reconnectTimer = setTimeout(() => {
      this.connect().catch((error) => {
        this.log("Reconnect failed:", error);
      });
    }, this.reconnectInterval);
  }

  /**
   * Start heartbeat to keep connection alive
   */
  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected()) {
        this.send({ type: "ping", timestamp: Date.now() });
      }
    }, this.heartbeatInterval);
  }

  /**
   * Stop heartbeat timer
   */
  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * Get WebSocket URL based on current location
   */
  private getWebSocketUrl(): string {
    if (typeof window === "undefined") {
      return "ws://localhost:16083/ws/intelligence-foundry";
    }
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    return `${protocol}//${host}/ws/intelligence-foundry`;
  }

  /**
   * Debug logging
   */
  private log(...args: any[]): void {
    if (this.debug) {
      console.log("[IntelligenceFoundryWS]", ...args);
    }
  }
}

// ============================================================================
// Singleton Instance Factory
// ============================================================================

let websocketInstance: IntelligenceFoundryWebSocket | null = null;

/**
 * Get or create WebSocket client singleton instance
 *
 * @param options - WebSocket client options
 * @returns WebSocket client instance
 *
 * @example
 * ```typescript
 * const ws = getIntelligenceFoundryWebSocket({ debug: true });
 * await ws.connect();
 *
 * ws.on('training:metrics', (event) => {
 *   console.log(`Train loss: ${event.metrics.train_loss}`);
 * });
 * ```
 */
export function getIntelligenceFoundryWebSocket(
  options?: WebSocketClientOptions
): IntelligenceFoundryWebSocket {
  if (!websocketInstance) {
    websocketInstance = new IntelligenceFoundryWebSocket(options);
  }
  return websocketInstance;
}

/**
 * Reset the singleton instance (useful for testing)
 */
export function resetIntelligenceFoundryWebSocket(): void {
  if (websocketInstance) {
    websocketInstance.disconnect();
    websocketInstance = null;
  }
}

// ============================================================================
// React Hook (Optional - for convenience)
// ============================================================================

/**
 * React hook for using Intelligence Foundry WebSocket
 *
 * @param autoConnect - Whether to connect automatically on mount
 * @returns WebSocket client instance
 *
 * @example
 * ```typescript
 * function MyComponent() {
 *   const ws = useIntelligenceFoundryWebSocket();
 *
 *   useEffect(() => {
 *     const unsubscribe = ws.on('training:metrics', (event) => {
 *       setMetrics(event.metrics);
 *     });
 *     return unsubscribe;
 *   }, [ws]);
 *
 *   return <div>Training metrics display</div>;
 * }
 * ```
 */
export function useIntelligenceFoundryWebSocket(
  autoConnect = true
): IntelligenceFoundryWebSocket {
  // This would typically use React hooks, but keeping it simple for now
  const ws = getIntelligenceFoundryWebSocket();

  if (autoConnect && !ws.isConnected()) {
    ws.connect().catch((error) => {
      console.error("Failed to connect WebSocket:", error);
    });
  }

  return ws;
}
