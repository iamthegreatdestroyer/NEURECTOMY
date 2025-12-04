/**
 * @file Graph3D Component
 * @description Main 3D force-directed graph visualization component
 * @module @neurectomy/3d-engine/visualization/graph
 * @agents @CANVAS @APEX @ARCHITECT
 */

import React, {
  useRef,
  useCallback,
  useEffect,
  useMemo,
  Suspense,
  forwardRef,
  useImperativeHandle,
} from "react";
import { Canvas, useFrame, useThree, ThreeEvent } from "@react-three/fiber";
import {
  OrbitControls,
  PerspectiveCamera,
  Environment,
  Stats,
  GizmoHelper,
  GizmoViewport,
  Grid,
  AdaptiveDpr,
  AdaptiveEvents,
  Preload,
  PerformanceMonitor,
  BakeShadows,
} from "@react-three/drei";
import { EffectComposer, Bloom, SMAA } from "@react-three/postprocessing";
import * as THREE from "three";
import type {
  GraphNode,
  GraphEdge,
  GraphConfig,
  NodeEvent,
  EdgeEvent,
  SelectionEvent,
  DragNodeEvent,
} from "./types";
import {
  useGraphStore,
  createGraphStore,
  type GraphStoreHook,
  type GraphStore,
} from "./store";
import { NodeMesh, InstancedNodes } from "./NodeMesh";
import { EdgeLine, InstancedEdges } from "./EdgeLine";

// ============================================================================
// Types
// ============================================================================

export interface Graph3DProps {
  /** Initial nodes */
  initialNodes?: GraphNode[];
  /** Initial edges */
  initialEdges?: GraphEdge[];
  /** Graph configuration */
  config?: Partial<GraphConfig>;

  /** External store instance (for controlled mode) */
  store?: ReturnType<typeof createGraphStore>;

  // Events
  onNodeClick?: (event: NodeEvent) => void;
  onNodeDoubleClick?: (event: NodeEvent) => void;
  onNodeHover?: (event: NodeEvent | null) => void;
  onNodeDragStart?: (event: DragNodeEvent) => void;
  onNodeDrag?: (event: DragNodeEvent) => void;
  onNodeDragEnd?: (event: DragNodeEvent) => void;
  onEdgeClick?: (event: EdgeEvent) => void;
  onEdgeHover?: (event: EdgeEvent | null) => void;
  onSelectionChange?: (event: SelectionEvent) => void;
  onSimulationTick?: (alpha: number) => void;
  onSimulationEnd?: () => void;

  // UI Options
  showStats?: boolean;
  showGizmo?: boolean;
  showGrid?: boolean;
  enablePostProcessing?: boolean;
  useInstancing?: boolean;
  instanceThreshold?: number;

  // Styling
  className?: string;
  style?: React.CSSProperties;
}

export interface Graph3DRef {
  /** Get the internal store state */
  getStore: () => GraphStore;
  /** Add nodes */
  addNodes: (nodes: GraphNode[]) => void;
  /** Add edges */
  addEdges: (edges: GraphEdge[]) => void;
  /** Remove nodes */
  removeNodes: (ids: string[]) => void;
  /** Remove edges */
  removeEdges: (ids: string[]) => void;
  /** Clear the graph */
  clear: () => void;
  /** Start simulation */
  startSimulation: () => void;
  /** Stop simulation */
  stopSimulation: () => void;
  /** Reset camera */
  resetCamera: () => void;
  /** Fit graph to view */
  fitToView: () => void;
  /** Focus on node */
  focusOnNode: (id: string) => void;
  /** Get camera state */
  getCameraState: () => { position: THREE.Vector3; target: THREE.Vector3 };
  /** Export as PNG */
  exportPNG: () => Promise<string>;
  /** Export graph data */
  exportData: () => { nodes: GraphNode[]; edges: GraphEdge[] };
}

// ============================================================================
// Loading Component
// ============================================================================

const LoadingFallback: React.FC = () => (
  <mesh>
    <boxGeometry args={[0.5, 0.5, 0.5]} />
    <meshStandardMaterial color="#4a90d9" wireframe />
  </mesh>
);

// ============================================================================
// Scene Content Component
// ============================================================================

interface SceneContentProps {
  store: GraphStoreHook;
  config: GraphConfig;
  showGrid: boolean;
  enablePostProcessing: boolean;
  useInstancing: boolean;
  instanceThreshold: number;
  onNodeClick?: (event: NodeEvent) => void;
  onNodeDoubleClick?: (event: NodeEvent) => void;
  onNodeHover?: (event: NodeEvent | null) => void;
  onNodeDragStart?: (event: DragNodeEvent) => void;
  onNodeDrag?: (event: DragNodeEvent) => void;
  onNodeDragEnd?: (event: DragNodeEvent) => void;
  onEdgeClick?: (event: EdgeEvent) => void;
  onEdgeHover?: (event: EdgeEvent | null) => void;
  onSimulationTick?: (alpha: number) => void;
}

const SceneContent: React.FC<SceneContentProps> = ({
  store,
  config,
  showGrid,
  enablePostProcessing,
  useInstancing,
  instanceThreshold,
  onNodeClick,
  onNodeDoubleClick,
  onNodeHover,
  onNodeDragStart,
  onNodeDrag,
  onNodeDragEnd,
  onEdgeClick,
  onEdgeHover,
  onSimulationTick,
}) => {
  const { gl, camera } = useThree();

  // Subscribe to store state
  const nodes = store((s) => s.nodes);
  const edges = store((s) => s.edges);
  const selection = store((s) => s.selection);
  const hover = store((s) => s.hover);
  const simulation = store((s) => s.simulation);
  const showLabels = store((s) => s.showLabels);
  const showEdges = store((s) => s.showEdges);

  // Convert maps to arrays
  const nodesArray = useMemo(() => Array.from(nodes.values()), [nodes]);
  const edgesArray = useMemo(() => Array.from(edges.values()), [edges]);

  // Resolve edges with their nodes
  const resolvedEdges = useMemo(() => {
    return edgesArray
      .map((edge) => ({
        edge,
        source: nodes.get(edge.sourceId),
        target: nodes.get(edge.targetId),
      }))
      .filter(
        (e): e is { edge: GraphEdge; source: GraphNode; target: GraphNode } =>
          e.source !== undefined && e.target !== undefined
      );
  }, [edgesArray, nodes]);

  // Simulation tick
  useFrame((_, delta) => {
    if (simulation.running) {
      store.getState().tickSimulation(delta);
      onSimulationTick?.(simulation.alpha);
    }
  });

  // Camera sync
  useEffect(() => {
    const camState = store.getState().camera;
    camera.position.set(
      camState.position.x,
      camState.position.y,
      camState.position.z
    );
    camera.lookAt(camState.target.x, camState.target.y, camState.target.z);
  }, []);

  // Node event handlers
  const handleNodeClick = useCallback(
    (nodeId: string, event: ThreeEvent<MouseEvent>) => {
      const node = nodes.get(nodeId);
      if (!node) return;

      // Update selection
      if (event.nativeEvent.shiftKey || event.nativeEvent.ctrlKey) {
        if (selection.nodeIds.has(nodeId)) {
          store.getState().deselectNode(nodeId);
        } else {
          store.getState().selectNode(nodeId, true);
        }
      } else {
        store.getState().selectNode(nodeId, false);
      }

      onNodeClick?.({
        type: "node:click",
        timestamp: Date.now(),
        originalEvent: event.nativeEvent,
        nodeId,
        node,
        position: node.position,
        screenPosition: {
          x: event.nativeEvent.clientX,
          y: event.nativeEvent.clientY,
        },
      });
    },
    [nodes, selection, store, onNodeClick]
  );

  const handleNodeDoubleClick = useCallback(
    (nodeId: string, event: ThreeEvent<MouseEvent>) => {
      const node = nodes.get(nodeId);
      if (!node) return;

      store.getState().focusOnNode(nodeId);

      onNodeDoubleClick?.({
        type: "node:doubleclick",
        timestamp: Date.now(),
        originalEvent: event.nativeEvent,
        nodeId,
        node,
        position: node.position,
        screenPosition: {
          x: event.nativeEvent.clientX,
          y: event.nativeEvent.clientY,
        },
      });
    },
    [nodes, store, onNodeDoubleClick]
  );

  const handleNodePointerEnter = useCallback(
    (nodeId: string, event: ThreeEvent<PointerEvent>) => {
      const node = nodes.get(nodeId);
      if (!node) return;

      store.getState().setHoveredNode(nodeId);

      onNodeHover?.({
        type: "node:hover",
        timestamp: Date.now(),
        originalEvent: event.nativeEvent,
        nodeId,
        node,
        position: node.position,
        screenPosition: {
          x: event.nativeEvent.clientX,
          y: event.nativeEvent.clientY,
        },
      });
    },
    [nodes, store, onNodeHover]
  );

  const handleNodePointerLeave = useCallback(
    (nodeId: string) => {
      store.getState().setHoveredNode(null);
      onNodeHover?.(null);
    },
    [store, onNodeHover]
  );

  const handleNodeDragStart = useCallback(
    (nodeId: string, position: THREE.Vector3) => {
      const node = nodes.get(nodeId);
      if (!node) return;

      store
        .getState()
        .startDrag(nodeId, { x: position.x, y: position.y, z: position.z });

      onNodeDragStart?.({
        type: "node:drag:start",
        timestamp: Date.now(),
        nodeId,
        node,
        position: node.position,
        screenPosition: { x: 0, y: 0 },
        delta: { x: 0, y: 0, z: 0 },
        totalDelta: { x: 0, y: 0, z: 0 },
      });
    },
    [nodes, store, onNodeDragStart]
  );

  const handleNodeDrag = useCallback(
    (nodeId: string, position: THREE.Vector3) => {
      const node = nodes.get(nodeId);
      if (!node) return;

      const startPos = store.getState().drag.startPosition;
      const delta = startPos
        ? {
            x: position.x - startPos.x,
            y: position.y - startPos.y,
            z: position.z - startPos.z,
          }
        : { x: 0, y: 0, z: 0 };

      store
        .getState()
        .updateDrag({ x: position.x, y: position.y, z: position.z });

      onNodeDrag?.({
        type: "node:drag",
        timestamp: Date.now(),
        nodeId,
        node,
        position: { x: position.x, y: position.y, z: position.z },
        screenPosition: { x: 0, y: 0 },
        delta,
        totalDelta: delta,
      });
    },
    [nodes, store, onNodeDrag]
  );

  const handleNodeDragEnd = useCallback(
    (nodeId: string) => {
      const node = nodes.get(nodeId);
      if (!node) return;

      store.getState().endDrag();

      onNodeDragEnd?.({
        type: "node:drag:end",
        timestamp: Date.now(),
        nodeId,
        node,
        position: node.position,
        screenPosition: { x: 0, y: 0 },
        delta: { x: 0, y: 0, z: 0 },
        totalDelta: { x: 0, y: 0, z: 0 },
      });
    },
    [nodes, store, onNodeDragEnd]
  );

  // Edge event handlers
  const handleEdgeClick = useCallback(
    (edgeId: string, event: ThreeEvent<MouseEvent>) => {
      const edge = edges.get(edgeId);
      const source = edge ? nodes.get(edge.sourceId) : undefined;
      const target = edge ? nodes.get(edge.targetId) : undefined;

      if (!edge || !source || !target) return;

      if (event.nativeEvent.shiftKey || event.nativeEvent.ctrlKey) {
        if (selection.edgeIds.has(edgeId)) {
          store.getState().deselectEdge(edgeId);
        } else {
          store.getState().selectEdge(edgeId, true);
        }
      } else {
        store.getState().selectEdge(edgeId, false);
      }

      onEdgeClick?.({
        type: "edge:click",
        timestamp: Date.now(),
        originalEvent: event.nativeEvent,
        edgeId,
        edge,
        sourceNode: source,
        targetNode: target,
      });
    },
    [edges, nodes, selection, store, onEdgeClick]
  );

  const handleEdgePointerEnter = useCallback(
    (edgeId: string, event: ThreeEvent<PointerEvent>) => {
      const edge = edges.get(edgeId);
      const source = edge ? nodes.get(edge.sourceId) : undefined;
      const target = edge ? nodes.get(edge.targetId) : undefined;

      if (!edge || !source || !target) return;

      store.getState().setHoveredEdge(edgeId);

      onEdgeHover?.({
        type: "edge:hover",
        timestamp: Date.now(),
        originalEvent: event.nativeEvent,
        edgeId,
        edge,
        sourceNode: source,
        targetNode: target,
      });
    },
    [edges, nodes, store, onEdgeHover]
  );

  const handleEdgePointerLeave = useCallback(
    (edgeId: string) => {
      store.getState().setHoveredEdge(null);
      onEdgeHover?.(null);
    },
    [store, onEdgeHover]
  );

  // Canvas click to deselect
  const handleCanvasClick = useCallback(
    (event: ThreeEvent<MouseEvent>) => {
      // Only if clicking on background
      if (event.object.type === "Scene") {
        store.getState().clearSelection();
      }
    },
    [store]
  );

  // Decide whether to use instancing
  const shouldUseInstancing =
    useInstancing && nodesArray.length > instanceThreshold;

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={config.visual.ambientLight} />
      <directionalLight
        position={[10, 20, 10]}
        intensity={0.8}
        castShadow
        shadow-mapSize={[2048, 2048]}
      />
      <pointLight position={[-10, -10, -10]} intensity={0.3} />

      {/* Environment */}
      <Environment preset="city" background={false} />

      {/* Grid */}
      {showGrid && (
        <Grid
          args={[100, 100]}
          cellSize={1}
          cellThickness={0.5}
          cellColor="#303050"
          sectionSize={5}
          sectionThickness={1}
          sectionColor="#404070"
          fadeDistance={50}
          fadeStrength={1}
          followCamera
          infiniteGrid
        />
      )}

      {/* Edges */}
      {showEdges && (
        <group name="edges">
          {shouldUseInstancing ? (
            <InstancedEdges
              edges={resolvedEdges}
              config={config.visual.edge}
              selectedIds={selection.edgeIds}
              hoveredId={hover.edgeId}
            />
          ) : (
            resolvedEdges.map(({ edge, source, target }) => (
              <EdgeLine
                key={edge.id}
                edge={edge}
                sourceNode={source}
                targetNode={target}
                config={config.visual.edge}
                onClick={handleEdgeClick}
                onPointerEnter={handleEdgePointerEnter}
                onPointerLeave={handleEdgePointerLeave}
              />
            ))
          )}
        </group>
      )}

      {/* Nodes */}
      <group name="nodes">
        {shouldUseInstancing ? (
          <InstancedNodes
            nodes={nodesArray}
            config={config.visual.node}
            selectedIds={selection.nodeIds}
            hoveredId={hover.nodeId}
            onClick={handleNodeClick}
          />
        ) : (
          nodesArray.map((node) => (
            <NodeMesh
              key={node.id}
              node={node}
              config={config.visual.node}
              showLabel={showLabels}
              onClick={handleNodeClick}
              onDoubleClick={handleNodeDoubleClick}
              onPointerEnter={handleNodePointerEnter}
              onPointerLeave={handleNodePointerLeave}
              onDragStart={handleNodeDragStart}
              onDrag={handleNodeDrag}
              onDragEnd={handleNodeDragEnd}
            />
          ))
        )}
      </group>

      {/* Post-processing */}
      {enablePostProcessing && config.visual.bloom && (
        <EffectComposer>
          <Bloom
            intensity={config.visual.bloomIntensity}
            luminanceThreshold={0.9}
            luminanceSmoothing={0.025}
          />
          <SMAA />
        </EffectComposer>
      )}
    </>
  );
};

// ============================================================================
// Main Graph3D Component
// ============================================================================

export const Graph3D = forwardRef<Graph3DRef, Graph3DProps>(
  (
    {
      initialNodes = [],
      initialEdges = [],
      config: configOverrides,
      store: externalStore,
      onNodeClick,
      onNodeDoubleClick,
      onNodeHover,
      onNodeDragStart,
      onNodeDrag,
      onNodeDragEnd,
      onEdgeClick,
      onEdgeHover,
      onSelectionChange,
      onSimulationTick,
      onSimulationEnd,
      showStats = false,
      showGizmo = true,
      showGrid = true,
      enablePostProcessing = true,
      useInstancing = true,
      instanceThreshold = 100,
      className,
      style,
    },
    ref
  ) => {
    // Create or use external store
    const internalStore = useMemo(
      () => externalStore || createGraphStore(),
      [externalStore]
    );
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Merge config with defaults
    const config = useMemo((): GraphConfig => {
      const state = internalStore.getState();
      return {
        ...state.config,
        ...configOverrides,
        physics: { ...state.config.physics, ...configOverrides?.physics },
        visual: {
          ...state.config.visual,
          ...configOverrides?.visual,
          node: {
            ...state.config.visual.node,
            ...configOverrides?.visual?.node,
          },
          edge: {
            ...state.config.visual.edge,
            ...configOverrides?.visual?.edge,
          },
        },
        interaction: {
          ...state.config.interaction,
          ...configOverrides?.interaction,
        },
        layout: { ...state.config.layout, ...configOverrides?.layout },
      };
    }, [internalStore, configOverrides]);

    // Initialize with data
    useEffect(() => {
      if (initialNodes.length > 0 || initialEdges.length > 0) {
        internalStore.getState().setGraph(initialNodes, initialEdges);
      }
    }, []); // Only on mount

    // Subscribe to selection changes
    useEffect(() => {
      if (!onSelectionChange) return;

      return internalStore.subscribe(
        (state) => ({
          nodeIds: state.selection.nodeIds,
          edgeIds: state.selection.edgeIds,
        }),
        (
          current: { nodeIds: Set<string>; edgeIds: Set<string> },
          previous: { nodeIds: Set<string>; edgeIds: Set<string> }
        ) => {
          onSelectionChange({
            type: "node:select",
            timestamp: Date.now(),
            selectedNodeIds: Array.from(current.nodeIds),
            selectedEdgeIds: Array.from(current.edgeIds),
            previousSelection: {
              nodeIds: Array.from(previous.nodeIds),
              edgeIds: Array.from(previous.edgeIds),
            },
          });
        }
      );
    }, [internalStore, onSelectionChange]);

    // Subscribe to simulation end
    useEffect(() => {
      if (!onSimulationEnd) return;

      return internalStore.subscribe(
        (state) => state.simulation.running,
        (running: boolean, wasRunning: boolean) => {
          if (wasRunning && !running) {
            onSimulationEnd();
          }
        }
      );
    }, [internalStore, onSimulationEnd]);

    // Expose ref methods
    useImperativeHandle(
      ref,
      () => ({
        getStore: () => internalStore.getState(),

        addNodes: (nodes) => internalStore.getState().addNodes(nodes),
        addEdges: (edges) => internalStore.getState().addEdges(edges),
        removeNodes: (ids) => internalStore.getState().removeNodes(ids),
        removeEdges: (ids) => internalStore.getState().removeEdges(ids),
        clear: () => internalStore.getState().clearGraph(),

        startSimulation: () => internalStore.getState().startSimulation(),
        stopSimulation: () => internalStore.getState().stopSimulation(),

        resetCamera: () => internalStore.getState().resetCamera(),
        fitToView: () => internalStore.getState().fitToView(),
        focusOnNode: (id) => internalStore.getState().focusOnNode(id),

        getCameraState: () => {
          const state = internalStore.getState().camera;
          return {
            position: new THREE.Vector3(
              state.position.x,
              state.position.y,
              state.position.z
            ),
            target: new THREE.Vector3(
              state.target.x,
              state.target.y,
              state.target.z
            ),
          };
        },

        exportPNG: async () => {
          if (!canvasRef.current) return "";
          return canvasRef.current.toDataURL("image/png");
        },

        exportData: () => {
          const state = internalStore.getState();
          return {
            nodes: Array.from(state.nodes.values()),
            edges: Array.from(state.edges.values()),
          };
        },
      }),
      [internalStore]
    );

    return (
      <div
        className={className}
        style={{
          width: "100%",
          height: "100%",
          position: "relative",
          ...style,
        }}
      >
        <Canvas
          ref={canvasRef}
          shadows={config.visual.shadows}
          dpr={[1, 2]}
          gl={{
            antialias: config.visual.antialias,
            toneMapping: THREE.ACESFilmicToneMapping,
            toneMappingExposure: 1,
            preserveDrawingBuffer: true,
          }}
          camera={{
            fov: 45,
            near: 0.1,
            far: 1000,
            position: [10, 10, 10],
          }}
          style={{ background: config.visual.backgroundColor }}
        >
          <Suspense fallback={<LoadingFallback />}>
            {/* Performance optimization */}
            <AdaptiveDpr pixelated />
            <AdaptiveEvents />
            <Preload all />

            {/* Performance monitoring */}
            <PerformanceMonitor
              onIncline={() => {
                // Increase quality when performance is good
              }}
              onDecline={() => {
                // Decrease quality when performance is poor
              }}
            />

            {/* Camera controls */}
            <PerspectiveCamera makeDefault position={[10, 10, 10]} fov={45} />
            <OrbitControls
              makeDefault
              enableDamping
              dampingFactor={0.05}
              rotateSpeed={0.5}
              zoomSpeed={0.8}
              panSpeed={0.8}
              minDistance={2}
              maxDistance={200}
            />

            {/* Scene content */}
            <SceneContent
              store={internalStore}
              config={config}
              showGrid={showGrid}
              enablePostProcessing={enablePostProcessing}
              useInstancing={useInstancing}
              instanceThreshold={instanceThreshold}
              onNodeClick={onNodeClick}
              onNodeDoubleClick={onNodeDoubleClick}
              onNodeHover={onNodeHover}
              onNodeDragStart={onNodeDragStart}
              onNodeDrag={onNodeDrag}
              onNodeDragEnd={onNodeDragEnd}
              onEdgeClick={onEdgeClick}
              onEdgeHover={onEdgeHover}
              onSimulationTick={onSimulationTick}
            />

            {/* Gizmo */}
            {showGizmo && (
              <GizmoHelper alignment="bottom-right" margin={[80, 80]}>
                <GizmoViewport
                  axisColors={["#ff4040", "#40ff40", "#4040ff"]}
                  labelColor="white"
                />
              </GizmoHelper>
            )}
          </Suspense>

          {/* Stats */}
          {showStats && <Stats />}
        </Canvas>
      </div>
    );
  }
);

Graph3D.displayName = "Graph3D";

export default Graph3D;
