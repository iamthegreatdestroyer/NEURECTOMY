/**
 * React Hooks for Entity System
 *
 * Custom hooks for subscribing to entity-based reactive state.
 * Provides automatic cleanup and efficient re-rendering.
 *
 * @module @neurectomy/state
 * @author @APEX @PRISM
 */

import {
  useEffect,
  useState,
  useMemo,
  useCallback,
  useRef,
  useSyncExternalStore,
} from "react";
import type { Subscription } from "../interfaces";
import {
  type Observable,
  type Entity,
  type EntityStore,
  type EntityDefinition,
  type EntityChangeEvent,
} from "./Entity";
import {
  getGlobalStore,
  type GlobalStore,
  type WorkspaceGlobalState,
} from "./Store";

// ============================================================================
// Observable Hooks
// ============================================================================

/**
 * Subscribe to an observable value with automatic cleanup
 */
export function useObservable<T>(observable: Observable<T>): T {
  const getSnapshot = useCallback(() => observable.get(), [observable]);

  const subscribe = useCallback(
    (onStoreChange: () => void) => {
      const subscription = observable.subscribe(onStoreChange);
      return () => subscription.unsubscribe();
    },
    [observable]
  );

  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

/**
 * Subscribe to a derived value from an observable
 */
export function useObservableSelector<T, U>(
  observable: Observable<T>,
  selector: (value: T) => U,
  deps: unknown[] = []
): U {
  const memoizedSelector = useCallback(selector, deps);
  const getSnapshot = useCallback(
    () => memoizedSelector(observable.get()),
    [observable, memoizedSelector]
  );

  const previousValueRef = useRef<U | undefined>(undefined);
  const getSnapshotWithMemo = useCallback(() => {
    const newValue = getSnapshot();
    // Only update if value actually changed (shallow compare)
    if (Object.is(previousValueRef.current, newValue)) {
      return previousValueRef.current!;
    }
    previousValueRef.current = newValue;
    return newValue;
  }, [getSnapshot]);

  const subscribe = useCallback(
    (onStoreChange: () => void) => {
      const subscription = observable.subscribe(onStoreChange);
      return () => subscription.unsubscribe();
    },
    [observable]
  );

  return useSyncExternalStore(
    subscribe,
    getSnapshotWithMemo,
    getSnapshotWithMemo
  );
}

// ============================================================================
// Entity Hooks
// ============================================================================

/**
 * Subscribe to an entity's state
 */
export function useEntity<T extends EntityDefinition>(
  entity: Entity<T> | undefined
): T | undefined {
  const [state, setState] = useState<T | undefined>(entity?.get());

  useEffect(() => {
    if (!entity) {
      setState(undefined);
      return;
    }

    setState(entity.get());
    const subscription = entity.subscribe((newState) => {
      setState(newState);
    });

    return () => subscription.unsubscribe();
  }, [entity]);

  return state;
}

/**
 * Subscribe to a selected value from an entity
 */
export function useEntitySelector<T extends EntityDefinition, U>(
  entity: Entity<T> | undefined,
  selector: (state: T) => U,
  deps: unknown[] = []
): U | undefined {
  const memoizedSelector = useCallback(selector, deps);
  const [value, setValue] = useState<U | undefined>(() => {
    return entity ? memoizedSelector(entity.get()) : undefined;
  });

  useEffect(() => {
    if (!entity) {
      setValue(undefined);
      return;
    }

    setValue(memoizedSelector(entity.get()));
    const subscription = entity.subscribe((state) => {
      setValue(memoizedSelector(state));
    });

    return () => subscription.unsubscribe();
  }, [entity, memoizedSelector]);

  return value;
}

// ============================================================================
// Entity Store Hooks
// ============================================================================

/**
 * Subscribe to entity store changes
 */
export function useEntityStore<T extends EntityDefinition>(
  store: EntityStore<T>
): Entity<T>[] {
  const [entities, setEntities] = useState<Entity<T>[]>(() => store.getAll());

  useEffect(() => {
    setEntities(store.getAll());
    const subscription = store.onChange(() => {
      setEntities(store.getAll());
    });

    return () => subscription.unsubscribe();
  }, [store]);

  return entities;
}

/**
 * Subscribe to a specific entity by ID from a store
 */
export function useEntityById<T extends EntityDefinition>(
  store: EntityStore<T>,
  id: string | null | undefined
): Entity<T> | undefined {
  const [entity, setEntity] = useState<Entity<T> | undefined>(() =>
    id ? store.get(id) : undefined
  );

  useEffect(() => {
    if (!id) {
      setEntity(undefined);
      return;
    }

    setEntity(store.get(id));
    const subscription = store.onChange((event) => {
      if (event.entityId === id) {
        if (event.type === "removed") {
          setEntity(undefined);
        } else {
          setEntity(store.get(id));
        }
      }
    });

    return () => subscription.unsubscribe();
  }, [store, id]);

  return entity;
}

/**
 * Query entities by predicate with live updates
 */
export function useEntityQuery<T extends EntityDefinition>(
  store: EntityStore<T>,
  predicate: (entity: Entity<T>) => boolean,
  deps: unknown[] = []
): Entity<T>[] {
  const memoizedPredicate = useCallback(predicate, deps);
  const [entities, setEntities] = useState<Entity<T>[]>(() =>
    store.query(memoizedPredicate)
  );

  useEffect(() => {
    setEntities(store.query(memoizedPredicate));
    const subscription = store.onChange(() => {
      setEntities(store.query(memoizedPredicate));
    });

    return () => subscription.unsubscribe();
  }, [store, memoizedPredicate]);

  return entities;
}

// ============================================================================
// Global Store Hooks
// ============================================================================

/**
 * Get the global store instance
 */
export function useGlobalStore(): GlobalStore {
  return useMemo(() => getGlobalStore(), []);
}

/**
 * Subscribe to global workspace state
 */
export function useWorkspaceState(): WorkspaceGlobalState {
  const store = useGlobalStore();
  return useObservable(store.globalState);
}

/**
 * Select a value from global workspace state
 */
export function useWorkspaceStateSelector<T>(
  selector: (state: WorkspaceGlobalState) => T,
  deps: unknown[] = []
): T {
  const store = useGlobalStore();
  return useObservableSelector(store.globalState, selector, deps);
}

/**
 * Subscribe to active pane
 */
export function useActivePane() {
  const store = useGlobalStore();
  const activePaneId = useWorkspaceStateSelector((s) => s.activePaneId, []);
  return useEntityById(store.panes, activePaneId);
}

/**
 * Subscribe to items in a pane
 */
export function usePaneItems(paneId: string | null | undefined) {
  const store = useGlobalStore();
  return useEntityQuery(
    store.items,
    (entity) => entity.get().paneId === paneId,
    [paneId]
  );
}

/**
 * Subscribe to panels in a dock position
 */
export function usePanelsByPosition(position: "left" | "right" | "bottom") {
  const store = useGlobalStore();
  return useEntityQuery(
    store.panels,
    (entity) => entity.get().position === position,
    [position]
  );
}

/**
 * Subscribe to dock visibility
 */
export function useDockVisible(position: "left" | "right" | "bottom"): boolean {
  const store = useGlobalStore();
  const observable = useMemo(() => {
    switch (position) {
      case "left":
        return store.leftDockVisible;
      case "right":
        return store.rightDockVisible;
      case "bottom":
        return store.bottomDockVisible;
    }
  }, [store, position]);

  return useObservable(observable);
}

/**
 * Subscribe to active panel in a dock
 */
export function useActivePanelInDock(
  position: "left" | "right" | "bottom"
): string | null {
  const store = useGlobalStore();
  const observable = useMemo(() => {
    switch (position) {
      case "left":
        return store.leftDockActivePanel;
      case "right":
        return store.rightDockActivePanel;
      case "bottom":
        return store.bottomDockActivePanel;
    }
  }, [store, position]);

  return useObservable(observable);
}

// ============================================================================
// Action Hooks
// ============================================================================

/**
 * Get workspace actions
 */
export function useWorkspaceActions() {
  const store = useGlobalStore();

  return useMemo(
    () => ({
      // Pane actions
      createPane: () => store.createPane(),
      setActivePane: (id: string) => store.setActivePane(id),
      focusPane: (id: string) => store.focusPane(id),
      splitPane: (id: string, direction: "horizontal" | "vertical") =>
        store.splitPane(id, direction),

      // Item actions
      addItemToPane: (paneId: string, itemId: string, activate?: boolean) =>
        store.addItemToPane(paneId, itemId, activate),
      removeItemFromPane: (paneId: string, itemId: string) =>
        store.removeItemFromPane(paneId, itemId),
      setActiveItem: (paneId: string, itemId: string) =>
        store.setActiveItem(paneId, itemId),
      setItemDirty: (itemId: string, isDirty: boolean) =>
        store.setItemDirty(itemId, isDirty),
      createItem: store.createItem.bind(store),
      findItemByPath: (path: string) => store.findItemByPath(path),

      // Panel actions
      togglePanel: (id: string) => store.togglePanel(id),
      setActivePanelForDock: (
        position: "left" | "right" | "bottom",
        id: string | null
      ) => store.setActivePanelForDock(position, id),

      // Focus actions
      focusDock: (position: "left" | "right" | "bottom") =>
        store.focusDock(position),
      focusPanel: (id: string) => store.focusPanel(id),

      // UI actions
      toggleSidebar: () => store.toggleSidebar(),
      toggleBottomPanel: () => store.toggleBottomPanel(),
      toggleZenMode: () => store.toggleZenMode(),
      setTheme: (theme: "light" | "dark" | "system") => store.setTheme(theme),
    }),
    [store]
  );
}

// ============================================================================
// Subscription Management Hook
// ============================================================================

/**
 * Manage multiple subscriptions with automatic cleanup
 */
export function useSubscriptions(): {
  add: (subscription: Subscription) => void;
  clear: () => void;
} {
  const subscriptionsRef = useRef<Subscription[]>([]);

  useEffect(() => {
    return () => {
      for (const sub of subscriptionsRef.current) {
        sub.unsubscribe();
      }
      subscriptionsRef.current = [];
    };
  }, []);

  return useMemo(
    () => ({
      add: (subscription: Subscription) => {
        subscriptionsRef.current.push(subscription);
      },
      clear: () => {
        for (const sub of subscriptionsRef.current) {
          sub.unsubscribe();
        }
        subscriptionsRef.current = [];
      },
    }),
    []
  );
}
