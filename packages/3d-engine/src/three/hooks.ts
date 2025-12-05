/**
 * React Three Fiber Hooks for NEURECTOMY 3D Engine
 *
 * Custom hooks providing reactive access to the 3D engine systems.
 * Built for agent architecture visualization with real-time updates.
 *
 * @module @neurectomy/3d-engine/three/hooks
 * @agents @APEX @CANVAS
 * @phase Phase 3 - Dimensional Forge
 * @step Step 2 - Three.js Integration Layer
 *
 * NOTE: This is a simplified version. Full implementation pending
 * completion of SceneGraph, SceneManager, and MaterialFactory APIs.
 */

import {
  useRef,
  useEffect,
  useState,
  useCallback,
  useContext,
  createContext,
} from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { SceneGraph, SceneGraphConfig } from "./scene-graph";
import {
  CameraSystem,
  CameraMode,
  CameraConfig,
  CameraState,
} from "./camera-system";
import { MaterialFactory, AgentStatus, AgentMaterialConfig } from "./materials";
import { WebGPUBridge, WebGPUBridgeConfig } from "./webgpu-bridge";
import { SceneManager, SceneManagerConfig } from "./scene-manager";

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================

/**
 * Agent node data for visualization
 */
export interface AgentNodeData {
  id: string;
  type: string;
  position: THREE.Vector3;
  status: AgentStatus;
  [key: string]: unknown;
}

/**
 * Connection data between agents
 */
export interface AgentConnectionData {
  id: string;
  sourceId: string;
  targetId: string;
  [key: string]: unknown;
}

/**
 * Engine performance statistics
 */
export interface EngineStats {
  fps: number;
  frameTime: number;
  triangles: number;
  drawCalls: number;
  agentCount: number;
  connectionCount: number;
  visibleNodes: number;
  gpuMemory: number;
}

/**
 * Engine context for accessing 3D engine systems
 */
export interface EngineContextValue {
  sceneGraph: SceneGraph | null;
  cameraSystem: CameraSystem | null;
  materialFactory: MaterialFactory | null;
  webgpuBridge: WebGPUBridge | null;
  sceneManager: SceneManager | null;
  isInitialized: boolean;
  stats: EngineStats;
}

// =============================================================================
// CONSTANTS
// =============================================================================

const defaultStats: EngineStats = {
  fps: 0,
  frameTime: 0,
  triangles: 0,
  drawCalls: 0,
  agentCount: 0,
  connectionCount: 0,
  visibleNodes: 0,
  gpuMemory: 0,
};

const defaultEngineContext: EngineContextValue = {
  sceneGraph: null,
  cameraSystem: null,
  materialFactory: null,
  webgpuBridge: null,
  sceneManager: null,
  isInitialized: false,
  stats: defaultStats,
};

// =============================================================================
// CONTEXT
// =============================================================================

const EngineContext = createContext<EngineContextValue>(defaultEngineContext);

export const EngineProvider = EngineContext.Provider;
export const useEngineContext = () => useContext(EngineContext);

// =============================================================================
// INITIALIZATION HOOKS
// =============================================================================

/**
 * Options for initializing the 3D engine
 */
export interface UseEngineOptions {
  sceneGraphConfig?: Partial<SceneGraphConfig>;
  cameraConfig?: Partial<CameraConfig>;
  webgpuConfig?: Partial<WebGPUBridgeConfig>;
  sceneManagerConfig?: Partial<SceneManagerConfig>;
  enableWebGPU?: boolean;
}

/**
 * Hook to initialize the 3D engine systems
 */
export function useEngine(options: UseEngineOptions = {}): EngineContextValue {
  const { gl, scene, camera } = useThree();
  const [isInitialized, setIsInitialized] = useState(false);
  const [stats, setStats] = useState<EngineStats>(defaultStats);

  const sceneGraphRef = useRef<SceneGraph | null>(null);
  const cameraSystemRef = useRef<CameraSystem | null>(null);
  const materialFactoryRef = useRef<MaterialFactory | null>(null);
  const webgpuBridgeRef = useRef<WebGPUBridge | null>(null);
  const sceneManagerRef = useRef<SceneManager | null>(null);

  useEffect(() => {
    // Initialize SceneGraph
    sceneGraphRef.current = new SceneGraph(options.sceneGraphConfig);

    // Initialize CameraSystem
    cameraSystemRef.current = new CameraSystem(
      camera as THREE.PerspectiveCamera,
      gl.domElement,
      options.cameraConfig
    );

    // Initialize MaterialFactory
    materialFactoryRef.current = new MaterialFactory();

    // Initialize SceneManager
    sceneManagerRef.current = new SceneManager(
      scene,
      options.sceneManagerConfig
    );

    // Initialize WebGPU Bridge if enabled
    if (options.enableWebGPU && options.webgpuConfig) {
      webgpuBridgeRef.current = new WebGPUBridge(options.webgpuConfig);
    }

    setIsInitialized(true);

    return () => {
      // Cleanup
      sceneGraphRef.current?.dispose();
      cameraSystemRef.current?.dispose();
      materialFactoryRef.current?.dispose();
      webgpuBridgeRef.current?.dispose();
      sceneManagerRef.current?.dispose();
    };
  }, [gl, scene, camera, options.enableWebGPU]);

  // Update stats on each frame
  useFrame((_state, delta) => {
    if (!isInitialized) return;

    cameraSystemRef.current?.update(delta);

    const info = gl.info;
    setStats({
      fps: Math.round(1 / delta),
      frameTime: delta * 1000,
      triangles: info.render.triangles,
      drawCalls: info.render.calls,
      agentCount: 0, // TODO: Get from scene manager
      connectionCount: 0, // TODO: Get from scene manager
      visibleNodes: 0, // TODO: Get from scene graph
      gpuMemory: info.memory.geometries + info.memory.textures,
    });
  });

  return {
    sceneGraph: sceneGraphRef.current,
    cameraSystem: cameraSystemRef.current,
    materialFactory: materialFactoryRef.current,
    webgpuBridge: webgpuBridgeRef.current,
    sceneManager: sceneManagerRef.current,
    isInitialized,
    stats,
  };
}

// =============================================================================
// CAMERA HOOKS
// =============================================================================

/**
 * Hook for camera system access and control
 */
export function useCamera() {
  const { cameraSystem } = useEngineContext();
  const { camera } = useThree();

  const setMode = useCallback(
    (mode: CameraMode) => {
      cameraSystem?.setMode(mode);
    },
    [cameraSystem]
  );

  const lookAt = useCallback(
    (target: THREE.Vector3) => {
      if (cameraSystem) {
        camera.lookAt(target);
      }
    },
    [cameraSystem, camera]
  );

  const resetCamera = useCallback(() => {
    cameraSystem?.resetToDefault();
  }, [cameraSystem]);

  const getState = useCallback((): CameraState | null => {
    return cameraSystem?.getState() ?? null;
  }, [cameraSystem]);

  return {
    camera,
    cameraSystem,
    setMode,
    lookAt,
    resetCamera,
    getState,
    isReady: cameraSystem !== null,
  };
}

// =============================================================================
// SCENE GRAPH HOOKS
// =============================================================================

/**
 * Hook for scene graph access
 */
export function useSceneGraph() {
  const { sceneGraph } = useEngineContext();

  const getStatistics = useCallback(() => {
    return sceneGraph?.getStatistics() ?? null;
  }, [sceneGraph]);

  return {
    sceneGraph,
    getStatistics,
    isReady: sceneGraph !== null,
  };
}

// =============================================================================
// MATERIAL HOOKS
// =============================================================================

/**
 * Hook for material factory access
 */
export function useMaterials() {
  const { materialFactory } = useEngineContext();

  const createAgentMaterial = useCallback(
    (config: AgentMaterialConfig) => {
      return materialFactory?.createAgentMaterial(config) ?? null;
    },
    [materialFactory]
  );

  const createConnectionMaterial = useCallback(
    (config: { color?: number; opacity?: number } = {}) => {
      return materialFactory?.createConnectionMaterial(config) ?? null;
    },
    [materialFactory]
  );

  const createGridMaterial = useCallback(
    (config: { color?: number; opacity?: number } = {}) => {
      return materialFactory?.createGridMaterial(config) ?? null;
    },
    [materialFactory]
  );

  return {
    materialFactory,
    createAgentMaterial,
    createConnectionMaterial,
    createGridMaterial,
    isReady: materialFactory !== null,
  };
}

// =============================================================================
// SCENE MANAGER HOOKS
// =============================================================================

/**
 * Hook for scene manager access
 */
export function useSceneManager() {
  const { sceneManager } = useEngineContext();

  const getStats = useCallback(() => {
    return sceneManager?.getStats() ?? null;
  }, [sceneManager]);

  return {
    sceneManager,
    getStats,
    isReady: sceneManager !== null,
  };
}

// =============================================================================
// FRAME HOOKS
// =============================================================================

/**
 * Hook to run logic on each frame
 */
export function useFrameCallback(
  callback: (delta: number, elapsed: number) => void,
  priority: number = 0
) {
  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  useFrame((state, delta) => {
    callbackRef.current(delta, state.clock.elapsedTime);
  }, priority);
}

// =============================================================================
// UTILITY HOOKS
// =============================================================================

/**
 * Hook to track a Three.js object's world position
 */
export function useWorldPosition(object: THREE.Object3D | null) {
  const [position, setPosition] = useState(new THREE.Vector3());

  useFrame(() => {
    if (object) {
      object.getWorldPosition(position);
      setPosition(position.clone());
    }
  });

  return position;
}

/**
 * Hook to track mouse position in normalized device coordinates
 */
export function useNormalizedMouse() {
  const [mouse, setMouse] = useState(new THREE.Vector2());
  const { gl } = useThree();

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      const rect = gl.domElement.getBoundingClientRect();
      setMouse(
        new THREE.Vector2(
          ((event.clientX - rect.left) / rect.width) * 2 - 1,
          -((event.clientY - rect.top) / rect.height) * 2 + 1
        )
      );
    };

    gl.domElement.addEventListener("mousemove", handleMouseMove);
    return () =>
      gl.domElement.removeEventListener("mousemove", handleMouseMove);
  }, [gl.domElement]);

  return mouse;
}

/**
 * Hook to track if WebGPU is available and initialized
 */
export function useWebGPU() {
  const { webgpuBridge } = useEngineContext();

  return {
    isAvailable: webgpuBridge !== null,
    bridge: webgpuBridge,
  };
}

// =============================================================================
// EXPORTS
// =============================================================================

export type {
  SceneGraphConfig,
  CameraConfig,
  CameraState,
  CameraMode,
  WebGPUBridgeConfig,
  SceneManagerConfig,
};
