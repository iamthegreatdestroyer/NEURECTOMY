/**
 * @neurectomy/test-config - Event Mocks
 *
 * Mock implementations for event emitters, message queues, and pub/sub systems.
 */

import { vi } from "vitest";

// ============================================================================
// Types
// ============================================================================

export type EventHandler<T = unknown> = (data: T) => void | Promise<void>;

export interface MockEventEmitter<
  Events extends Record<string, unknown> = Record<string, unknown>,
> {
  on: <K extends keyof Events>(
    event: K,
    handler: EventHandler<Events[K]>
  ) => () => void;
  off: <K extends keyof Events>(
    event: K,
    handler: EventHandler<Events[K]>
  ) => void;
  emit: <K extends keyof Events>(event: K, data: Events[K]) => Promise<void>;
  once: <K extends keyof Events>(
    event: K,
    handler: EventHandler<Events[K]>
  ) => () => void;
  listenerCount: <K extends keyof Events>(event: K) => number;
  removeAllListeners: <K extends keyof Events>(event?: K) => void;
  getEmittedEvents: () => Array<{
    event: string;
    data: unknown;
    timestamp: number;
  }>;
  reset: () => void;
}

export interface MockMessageQueue<T = unknown> {
  messages: T[];
  publish: (message: T) => Promise<void>;
  subscribe: (handler: (message: T) => void | Promise<void>) => () => void;
  peek: () => T | undefined;
  ack: () => T | undefined;
  nack: (message: T) => void;
  size: () => number;
  clear: () => void;
  reset: () => void;
}

export interface MockPubSub<T = unknown> {
  channels: Map<string, Set<(message: T) => void>>;
  publish: (channel: string, message: T) => Promise<number>;
  subscribe: (channel: string, handler: (message: T) => void) => () => void;
  unsubscribe: (channel: string, handler?: (message: T) => void) => void;
  getSubscriberCount: (channel: string) => number;
  reset: () => void;
}

// ============================================================================
// Event Emitter Mock
// ============================================================================

/**
 * Create a typed event emitter mock
 *
 * @example
 * interface MyEvents {
 *   userCreated: { id: string; name: string };
 *   userDeleted: { id: string };
 * }
 *
 * const emitter = createMockEventEmitter<MyEvents>();
 * emitter.on('userCreated', (data) => console.log(data.name));
 * await emitter.emit('userCreated', { id: '1', name: 'Test' });
 */
export function createMockEventEmitter<
  Events extends Record<string, unknown> = Record<string, unknown>,
>(): MockEventEmitter<Events> {
  const listeners = new Map<keyof Events, Set<EventHandler<unknown>>>();
  const emittedEvents: Array<{
    event: string;
    data: unknown;
    timestamp: number;
  }> = [];

  return {
    on: vi.fn(
      <K extends keyof Events>(event: K, handler: EventHandler<Events[K]>) => {
        if (!listeners.has(event)) {
          listeners.set(event, new Set());
        }
        listeners.get(event)!.add(handler as EventHandler<unknown>);

        // Return unsubscribe function
        return () => {
          listeners.get(event)?.delete(handler as EventHandler<unknown>);
        };
      }
    ),

    off: vi.fn(
      <K extends keyof Events>(event: K, handler: EventHandler<Events[K]>) => {
        listeners.get(event)?.delete(handler as EventHandler<unknown>);
      }
    ),

    emit: vi.fn(async <K extends keyof Events>(event: K, data: Events[K]) => {
      emittedEvents.push({
        event: event as string,
        data,
        timestamp: Date.now(),
      });

      const handlers = listeners.get(event);
      if (handlers) {
        const promises = Array.from(handlers).map((handler) => handler(data));
        await Promise.all(promises);
      }
    }),

    once: vi.fn(
      <K extends keyof Events>(event: K, handler: EventHandler<Events[K]>) => {
        const wrappedHandler: EventHandler<Events[K]> = async (data) => {
          listeners.get(event)?.delete(wrappedHandler as EventHandler<unknown>);
          await handler(data);
        };

        if (!listeners.has(event)) {
          listeners.set(event, new Set());
        }
        listeners.get(event)!.add(wrappedHandler as EventHandler<unknown>);

        return () => {
          listeners.get(event)?.delete(wrappedHandler as EventHandler<unknown>);
        };
      }
    ),

    listenerCount: vi.fn(<K extends keyof Events>(event: K) => {
      return listeners.get(event)?.size ?? 0;
    }),

    removeAllListeners: vi.fn(<K extends keyof Events>(event?: K) => {
      if (event) {
        listeners.delete(event);
      } else {
        listeners.clear();
      }
    }),

    getEmittedEvents() {
      return [...emittedEvents];
    },

    reset() {
      listeners.clear();
      emittedEvents.length = 0;
    },
  };
}

// ============================================================================
// Message Queue Mock
// ============================================================================

/**
 * Create a simple message queue mock
 *
 * @example
 * const queue = createMockMessageQueue<{ type: string; payload: unknown }>();
 * await queue.publish({ type: 'task', payload: { id: 1 } });
 * queue.subscribe((msg) => console.log(msg.type));
 */
export function createMockMessageQueue<T = unknown>(): MockMessageQueue<T> {
  const messages: T[] = [];
  const subscribers: Set<(message: T) => void | Promise<void>> = new Set();
  const pendingMessages: T[] = [];

  return {
    messages,

    publish: vi.fn(async (message: T) => {
      messages.push(message);
      pendingMessages.push(message);

      // Notify all subscribers
      const promises = Array.from(subscribers).map((handler) =>
        handler(message)
      );
      await Promise.all(promises);
    }),

    subscribe: vi.fn((handler: (message: T) => void | Promise<void>) => {
      subscribers.add(handler);

      return () => {
        subscribers.delete(handler);
      };
    }),

    peek() {
      return pendingMessages[0];
    },

    ack: vi.fn(() => {
      return pendingMessages.shift();
    }),

    nack: vi.fn((message: T) => {
      // Re-queue the message at the end
      pendingMessages.push(message);
    }),

    size() {
      return pendingMessages.length;
    },

    clear() {
      pendingMessages.length = 0;
    },

    reset() {
      messages.length = 0;
      pendingMessages.length = 0;
      subscribers.clear();
    },
  };
}

// ============================================================================
// Pub/Sub Mock
// ============================================================================

/**
 * Create a pub/sub mock for testing distributed messaging
 *
 * @example
 * const pubsub = createMockPubSub<string>();
 * pubsub.subscribe('notifications', (msg) => console.log(msg));
 * await pubsub.publish('notifications', 'Hello!'); // Returns 1 (subscriber count)
 */
export function createMockPubSub<T = unknown>(): MockPubSub<T> {
  const channels = new Map<string, Set<(message: T) => void>>();

  return {
    channels,

    publish: vi.fn(async (channel: string, message: T) => {
      const subscribers = channels.get(channel);
      if (!subscribers || subscribers.size === 0) {
        return 0;
      }

      const promises = Array.from(subscribers).map((handler) => {
        try {
          return Promise.resolve(handler(message));
        } catch (error) {
          console.error(`Error in subscriber for channel ${channel}:`, error);
          return Promise.resolve();
        }
      });

      await Promise.all(promises);
      return subscribers.size;
    }),

    subscribe: vi.fn((channel: string, handler: (message: T) => void) => {
      if (!channels.has(channel)) {
        channels.set(channel, new Set());
      }
      channels.get(channel)!.add(handler);

      return () => {
        channels.get(channel)?.delete(handler);
      };
    }),

    unsubscribe: vi.fn((channel: string, handler?: (message: T) => void) => {
      if (handler) {
        channels.get(channel)?.delete(handler);
      } else {
        channels.delete(channel);
      }
    }),

    getSubscriberCount(channel: string) {
      return channels.get(channel)?.size ?? 0;
    },

    reset() {
      channels.clear();
    },
  };
}

// ============================================================================
// NATS JetStream Mock (for NEURECTOMY event sourcing)
// ============================================================================

export interface MockJetStreamMessage<T = unknown> {
  subject: string;
  data: T;
  seq: number;
  time: Date;
  ack: () => void;
  nak: (delay?: number) => void;
  working: () => void;
  term: () => void;
}

export interface MockJetStreamConsumer<T = unknown> {
  messages: MockJetStreamMessage<T>[];
  consume: () => AsyncIterable<MockJetStreamMessage<T>>;
  fetch: (opts?: {
    max_messages?: number;
    expires?: number;
  }) => Promise<MockJetStreamMessage<T>[]>;
}

export interface MockJetStream<T = unknown> {
  streams: Map<
    string,
    { subjects: string[]; messages: MockJetStreamMessage<T>[] }
  >;
  publish: (
    subject: string,
    data: T
  ) => Promise<{ seq: number; stream: string }>;
  subscribe: (subject: string) => MockJetStreamConsumer<T>;
  addStream: (name: string, subjects: string[]) => void;
  deleteStream: (name: string) => void;
  getStream: (
    name: string
  ) => { subjects: string[]; messages: MockJetStreamMessage<T>[] } | undefined;
  reset: () => void;
}

/**
 * Create a NATS JetStream mock for testing event sourcing
 *
 * @example
 * const js = createMockJetStream();
 * js.addStream('AGENTS', ['agents.>']);
 * await js.publish('agents.created', { id: '123', name: 'Test Agent' });
 */
export function createMockJetStream<T = unknown>(): MockJetStream<T> {
  const streams = new Map<
    string,
    { subjects: string[]; messages: MockJetStreamMessage<T>[] }
  >();
  let globalSeq = 0;

  function matchSubject(pattern: string, subject: string): boolean {
    const patternParts = pattern.split(".");
    const subjectParts = subject.split(".");

    for (let i = 0; i < patternParts.length; i++) {
      if (patternParts[i] === ">") {
        return true; // Match all remaining
      }
      if (patternParts[i] === "*") {
        continue; // Match single token
      }
      if (patternParts[i] !== subjectParts[i]) {
        return false;
      }
    }

    return patternParts.length === subjectParts.length;
  }

  function findStreamForSubject(subject: string): string | undefined {
    for (const [name, stream] of streams) {
      if (stream.subjects.some((pattern) => matchSubject(pattern, subject))) {
        return name;
      }
    }
    return undefined;
  }

  return {
    streams,

    publish: vi.fn(async (subject: string, data: T) => {
      const streamName = findStreamForSubject(subject);
      if (!streamName) {
        throw new Error(`No stream found for subject: ${subject}`);
      }

      const stream = streams.get(streamName)!;
      const seq = ++globalSeq;

      const message: MockJetStreamMessage<T> = {
        subject,
        data,
        seq,
        time: new Date(),
        ack: vi.fn(),
        nak: vi.fn(),
        working: vi.fn(),
        term: vi.fn(),
      };

      stream.messages.push(message);

      return { seq, stream: streamName };
    }),

    subscribe: vi.fn((subject: string) => {
      const streamName = findStreamForSubject(subject);
      const stream = streamName ? streams.get(streamName) : undefined;

      const consumer: MockJetStreamConsumer<T> = {
        messages:
          stream?.messages.filter((m) => matchSubject(subject, m.subject)) ??
          [],

        async *consume() {
          const messages =
            stream?.messages.filter((m) => matchSubject(subject, m.subject)) ??
            [];
          for (const message of messages) {
            yield message;
          }
        },

        fetch: vi.fn(async (opts?: { max_messages?: number }) => {
          const messages =
            stream?.messages.filter((m) => matchSubject(subject, m.subject)) ??
            [];
          const max = opts?.max_messages ?? messages.length;
          return messages.slice(0, max);
        }),
      };

      return consumer;
    }),

    addStream: vi.fn((name: string, subjects: string[]) => {
      streams.set(name, { subjects, messages: [] });
    }),

    deleteStream: vi.fn((name: string) => {
      streams.delete(name);
    }),

    getStream(name: string) {
      return streams.get(name);
    },

    reset() {
      streams.clear();
      globalSeq = 0;
    },
  };
}

// ============================================================================
// DOM Event Simulation
// ============================================================================

/**
 * Create and dispatch a custom DOM event
 */
export function dispatchCustomEvent<T>(
  element: Element | Document | Window,
  eventName: string,
  detail?: T
): boolean {
  const event = new CustomEvent(eventName, {
    detail,
    bubbles: true,
    cancelable: true,
  });
  return element.dispatchEvent(event);
}

/**
 * Create a keyboard event for testing
 */
export function createKeyboardEvent(
  type: "keydown" | "keyup" | "keypress",
  key: string,
  options: Partial<KeyboardEventInit> = {}
): KeyboardEvent {
  return new KeyboardEvent(type, {
    key,
    code: key.length === 1 ? `Key${key.toUpperCase()}` : key,
    bubbles: true,
    cancelable: true,
    ...options,
  });
}

/**
 * Create a mouse event for testing
 */
export function createMouseEvent(
  type:
    | "click"
    | "dblclick"
    | "mousedown"
    | "mouseup"
    | "mousemove"
    | "mouseenter"
    | "mouseleave",
  options: Partial<MouseEventInit> = {}
): MouseEvent {
  return new MouseEvent(type, {
    bubbles: true,
    cancelable: true,
    view: window,
    ...options,
  });
}

/**
 * Create a touch event for testing mobile interactions
 */
export function createTouchEvent(
  type: "touchstart" | "touchmove" | "touchend" | "touchcancel",
  touches: Array<{ clientX: number; clientY: number }> = []
): TouchEvent {
  const touchList = touches.map((touch, i) => ({
    identifier: i,
    target: document.body,
    clientX: touch.clientX,
    clientY: touch.clientY,
    pageX: touch.clientX,
    pageY: touch.clientY,
    screenX: touch.clientX,
    screenY: touch.clientY,
    radiusX: 1,
    radiusY: 1,
    rotationAngle: 0,
    force: 1,
  }));

  return new TouchEvent(type, {
    bubbles: true,
    cancelable: true,
    touches: touchList as unknown as TouchList,
    targetTouches: touchList as unknown as TouchList,
    changedTouches: touchList as unknown as TouchList,
  });
}
