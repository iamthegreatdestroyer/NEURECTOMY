/**
 * State Module
 *
 * Central export point for the entity-based reactive state system.
 *
 * @module @neurectomy/state
 * @author @APEX @PRISM
 */

// Entity System
export {
  generateEntityId,
  Observable,
  MutableObservable,
  DerivedObservable,
  Entity,
  EntityStore,
  BatchUpdate,
  globalBatch,
  batch,
  Computed,
  computed,
  type Listener,
  type EntityDefinition,
  type EntityChangeEvent,
} from "./Entity";

// Global Store
export {
  GlobalStore,
  getGlobalStore,
  resetGlobalStore,
  type PanelEntity,
  type PaneEntity,
  type ItemEntity,
  type ProjectEntity,
  type LayoutEntity,
  type WorkspaceGlobalState,
} from "./Store";

// React Hooks
export {
  // Observable hooks
  useObservable,
  useObservableSelector,

  // Entity hooks
  useEntity,
  useEntitySelector,
  useEntityStore,
  useEntityById,
  useEntityQuery,

  // Global store hooks
  useGlobalStore,
  useWorkspaceState,
  useWorkspaceStateSelector,
  useActivePane,
  usePaneItems,
  usePanelsByPosition,
  useDockVisible,
  useActivePanelInDock,

  // Action hooks
  useWorkspaceActions,

  // Subscription management
  useSubscriptions,
} from "./hooks";
