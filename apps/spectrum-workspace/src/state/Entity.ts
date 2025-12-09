/**
 * Entity System
 *
 * Reactive entity-based state management inspired by Zed's GPUI.
 * Provides observable entities with automatic subscription management.
 *
 * @module @neurectomy/state
 * @author @APEX @PRISM
 */

import type { Subscription, Disposable } from "../interfaces/types";

// ============================================================================
// Entity ID System
// ============================================================================

/**
 * Global entity ID counter for unique identification
 */
let entityIdCounter = 0;

/**
 * Generate a unique entity ID
 */
export function generateEntityId(prefix: string = "entity"): string {
  return `${prefix}_${++entityIdCounter}_${Date.now().toString(36)}`;
}

// ============================================================================
// Observable Value
// ============================================================================

/**
 * Listener callback type
 */
export type Listener<T> = (value: T, previous: T) => void;

/**
 * Observable value with subscription support.
 * Similar to Zed's Model<T> but adapted for React/TypeScript.
 */
export class Observable<T> implements Disposable {
  protected _value: T;
  protected listeners: Set<Listener<T>> = new Set();
  protected disposed = false;

  constructor(initialValue: T) {
    this._value = initialValue;
  }

  /**
   * Get the current value
   */
  get(): T {
    return this._value;
  }

  /**
   * Subscribe to value changes
   */
  subscribe(callback: Listener<T>): Subscription {
    if (this.disposed) {
      throw new Error("Cannot subscribe to disposed Observable");
    }

    this.listeners.add(callback);
    const id = generateEntityId("subscription");

    return {
      id,
      unsubscribe: () => {
        this.listeners.delete(callback);
      },
    };
  }

  /**
   * Map to a derived observable
   */
  map<U>(mapper: (value: T) => U): DerivedObservable<T, U> {
    return new DerivedObservable(this, mapper);
  }

  /**
   * Notify all listeners of a change
   */
  protected notify(previous: T): void {
    for (const listener of this.listeners) {
      try {
        listener(this._value, previous);
      } catch (error) {
        console.error("Observable listener error:", error);
      }
    }
  }

  /**
   * Dispose the observable
   */
  dispose(): void {
    if (!this.disposed) {
      this.disposed = true;
      this.listeners.clear();
    }
  }
}

/**
 * Mutable observable that can be updated
 */
export class MutableObservable<T> extends Observable<T> {
  /**
   * Set a new value
   */
  set(value: T): void {
    if (this.disposed) {
      throw new Error("Cannot set value on disposed Observable");
    }

    const previous = this._value;
    if (!Object.is(previous, value)) {
      this._value = value;
      this.notify(previous);
    }
  }

  /**
   * Update value using a function
   */
  update(updater: (current: T) => T): void {
    this.set(updater(this._value));
  }

  /**
   * Silently set value without notifying
   */
  setSilent(value: T): void {
    this._value = value;
  }
}

/**
 * Derived observable that transforms values from a source
 */
export class DerivedObservable<S, T> extends Observable<T> {
  private sourceSubscription: Subscription;

  constructor(source: Observable<S>, mapper: (value: S) => T) {
    super(mapper(source.get()));

    this.sourceSubscription = source.subscribe((value) => {
      const previous = this._value;
      this._value = mapper(value);
      if (!Object.is(previous, this._value)) {
        this.notify(previous);
      }
    });
  }

  dispose(): void {
    this.sourceSubscription.unsubscribe();
    super.dispose();
  }
}

// ============================================================================
// Entity Definition
// ============================================================================

/**
 * Entity base interface
 */
export interface EntityDefinition {
  id: string;
}

/**
 * Entity with observable state
 */
export class Entity<T extends EntityDefinition> implements Disposable {
  readonly id: string;
  private state: MutableObservable<T>;
  private disposed = false;

  constructor(initialState: T) {
    this.id = initialState.id;
    this.state = new MutableObservable(initialState);
  }

  /**
   * Get the current state
   */
  get(): T {
    return this.state.get();
  }

  /**
   * Update the entity state
   */
  update(updater: Partial<T> | ((current: T) => Partial<T>)): void {
    if (this.disposed) {
      throw new Error("Cannot update disposed Entity");
    }

    const current = this.state.get();
    const updates = typeof updater === "function" ? updater(current) : updater;
    this.state.set({ ...current, ...updates });
  }

  /**
   * Replace the entire state
   */
  set(state: T): void {
    if (this.disposed) {
      throw new Error("Cannot set disposed Entity");
    }
    this.state.set(state);
  }

  /**
   * Subscribe to state changes
   */
  subscribe(callback: (state: T, previous: T) => void): Subscription {
    return this.state.subscribe(callback);
  }

  /**
   * Select a derived value from state
   */
  select<U>(selector: (state: T) => U): Observable<U> {
    return this.state.map(selector);
  }

  /**
   * Dispose the entity
   */
  dispose(): void {
    if (!this.disposed) {
      this.disposed = true;
      this.state.dispose();
    }
  }
}

// ============================================================================
// Entity Store
// ============================================================================

/**
 * Entity change event
 */
export interface EntityChangeEvent<T extends EntityDefinition> {
  type: "added" | "updated" | "removed";
  entity: Entity<T>;
  entityId: string;
}

/**
 * Store for managing a collection of entities
 */
export class EntityStore<T extends EntityDefinition> implements Disposable {
  private entities: Map<string, Entity<T>> = new Map();
  private listeners: Set<(event: EntityChangeEvent<T>) => void> = new Set();
  private entitySubscriptions: Map<string, Subscription> = new Map();
  private disposed = false;

  /**
   * Get entity by ID
   */
  get(id: string): Entity<T> | undefined {
    return this.entities.get(id);
  }

  /**
   * Get all entities
   */
  getAll(): Entity<T>[] {
    return Array.from(this.entities.values());
  }

  /**
   * Get all entity IDs
   */
  getIds(): string[] {
    return Array.from(this.entities.keys());
  }

  /**
   * Check if entity exists
   */
  has(id: string): boolean {
    return this.entities.has(id);
  }

  /**
   * Get entity count
   */
  get size(): number {
    return this.entities.size;
  }

  /**
   * Add an entity to the store
   */
  add(entity: Entity<T>): Entity<T> {
    if (this.disposed) {
      throw new Error("Cannot add to disposed EntityStore");
    }

    if (this.entities.has(entity.id)) {
      throw new Error(`Entity with id ${entity.id} already exists`);
    }

    this.entities.set(entity.id, entity);

    // Subscribe to entity changes
    const subscription = entity.subscribe(() => {
      this.notify({ type: "updated", entity, entityId: entity.id });
    });
    this.entitySubscriptions.set(entity.id, subscription);

    this.notify({ type: "added", entity, entityId: entity.id });
    return entity;
  }

  /**
   * Create and add a new entity
   */
  create(initialState: T): Entity<T> {
    const entity = new Entity(initialState);
    return this.add(entity);
  }

  /**
   * Remove an entity from the store
   */
  remove(id: string): boolean {
    const entity = this.entities.get(id);
    if (!entity) return false;

    // Unsubscribe from entity changes
    const subscription = this.entitySubscriptions.get(id);
    subscription?.unsubscribe();
    this.entitySubscriptions.delete(id);

    this.entities.delete(id);
    entity.dispose();

    this.notify({ type: "removed", entity, entityId: id });
    return true;
  }

  /**
   * Clear all entities
   */
  clear(): void {
    for (const id of this.entities.keys()) {
      this.remove(id);
    }
  }

  /**
   * Subscribe to store changes
   */
  onChange(callback: (event: EntityChangeEvent<T>) => void): Subscription {
    if (this.disposed) {
      throw new Error("Cannot subscribe to disposed EntityStore");
    }

    this.listeners.add(callback);
    const id = generateEntityId("store-subscription");

    return {
      id,
      unsubscribe: () => {
        this.listeners.delete(callback);
      },
    };
  }

  /**
   * Subscribe to entity changes
   */
  onEntityChange(
    entityId: string,
    callback: (state: T, previous: T) => void
  ): Subscription | undefined {
    const entity = this.entities.get(entityId);
    return entity?.subscribe(callback);
  }

  /**
   * Query entities by predicate
   */
  query(predicate: (entity: Entity<T>) => boolean): Entity<T>[] {
    return this.getAll().filter(predicate);
  }

  /**
   * Find first entity matching predicate
   */
  find(predicate: (entity: Entity<T>) => boolean): Entity<T> | undefined {
    return this.getAll().find(predicate);
  }

  /**
   * Notify listeners of a change
   */
  private notify(event: EntityChangeEvent<T>): void {
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch (error) {
        console.error("EntityStore listener error:", error);
      }
    }
  }

  /**
   * Dispose the store and all entities
   */
  dispose(): void {
    if (!this.disposed) {
      this.disposed = true;

      for (const subscription of this.entitySubscriptions.values()) {
        subscription.unsubscribe();
      }
      this.entitySubscriptions.clear();

      for (const entity of this.entities.values()) {
        entity.dispose();
      }
      this.entities.clear();

      this.listeners.clear();
    }
  }
}

// ============================================================================
// Batch Updates
// ============================================================================

/**
 * Batch multiple state updates into a single notification
 */
export class BatchUpdate implements Disposable {
  private pending: (() => void)[] = [];
  private scheduled = false;
  private disposed = false;

  /**
   * Queue an update for batching
   */
  queue(update: () => void): void {
    if (this.disposed) return;

    this.pending.push(update);
    this.schedule();
  }

  /**
   * Schedule flush on next microtask
   */
  private schedule(): void {
    if (this.scheduled || this.disposed) return;

    this.scheduled = true;
    queueMicrotask(() => {
      this.flush();
    });
  }

  /**
   * Flush all pending updates
   */
  flush(): void {
    if (this.disposed) return;

    const updates = this.pending;
    this.pending = [];
    this.scheduled = false;

    for (const update of updates) {
      try {
        update();
      } catch (error) {
        console.error("BatchUpdate error:", error);
      }
    }
  }

  /**
   * Dispose the batch updater
   */
  dispose(): void {
    this.disposed = true;
    this.pending = [];
  }
}

/**
 * Global batch updater singleton
 */
export const globalBatch = new BatchUpdate();

/**
 * Run updates in a batch
 */
export function batch(fn: () => void): void {
  globalBatch.queue(fn);
}

// ============================================================================
// Computed Values
// ============================================================================

/**
 * Computed value that derives from multiple sources
 */
export class Computed<T> extends Observable<T> {
  private sources: Observable<unknown>[];
  private compute: () => T;
  private subscriptions: Subscription[] = [];
  private dirty = true;
  private cachedValue: T;

  constructor(sources: Observable<unknown>[], compute: () => T) {
    super(undefined as T);
    this.sources = sources;
    this.compute = compute;
    this.cachedValue = compute();
    this._value = this.cachedValue;

    // Subscribe to all sources
    for (const source of sources) {
      this.subscriptions.push(
        source.subscribe(() => {
          this.dirty = true;
          const previous = this.cachedValue;
          this.cachedValue = this.compute();
          this._value = this.cachedValue;
          if (!Object.is(previous, this.cachedValue)) {
            this.notify(previous);
          }
        })
      );
    }
  }

  get(): T {
    if (this.dirty) {
      this.cachedValue = this.compute();
      this._value = this.cachedValue;
      this.dirty = false;
    }
    return this.cachedValue;
  }

  dispose(): void {
    for (const sub of this.subscriptions) {
      sub.unsubscribe();
    }
    this.subscriptions = [];
    super.dispose();
  }
}

/**
 * Create a computed value from multiple sources
 */
export function computed<T>(
  sources: Observable<unknown>[],
  compute: () => T
): Computed<T> {
  return new Computed(sources, compute);
}
