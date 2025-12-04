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
 */

import { useRef, useEffect, useState, useCallback, useMemo, useContext, createContext } from 'react';
import { useFrame, useThree, RootState } from '@react-three/fiber';
import * as THREE from 'three';
import { SceneGraph, SceneGraphConfig, SceneNodeType, SceneNode } from './scene-graph';
import { CameraSystem, CameraMode, CameraConfig, CameraState } from './camera-system';
import { MaterialLibrary, AgentMaterialOptions, AgentStatus } from './materials';
import { WebGPUBridge, WebGPUBridgeConfig } from './webgpu-bridge';
import { SceneManager, AgentNodeData, AgentConnectionData } from './scene-manager';

// =============================================================================
// CONTEXT PROVIDERS
// =============================================================================

/**
 * Engine context for accessing 3D engine systems
 */
export interface EngineContextValue {
  sceneGraph: SceneGraph | null;
  cameraSystem: CameraSystem | null;
  materialLibrary: MaterialLibrary | null;
  webgpuBridge: WebGPUBridge | null;
  sceneManager: SceneManager | null;
  isInitialized: boolean;
  stats: EngineStats;
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

const defaultStats: EngineStats = {
  fps: 0,
  frameTime: 0,
  triangles: 0,
  drawCalls: 0,
  agentCount: 0,
  connectionCount: 0,
  visibleNodes: 0,
  gpuMemory: 0
};

const EngineContext = createContext<EngineContextValue>({
  sceneGraph: null,
  cameraSystem: null,
  materialLibrary: null,
  webgpuBridge: null,
  sceneManager: null,
  isInitialized: false,
  stats: defaultStats
});

export const EngineProvider = EngineContext.Provider;
export const useEngineContext = () => useContext(EngineContext);

// =============================================================================
// INITIALIZATION HOOKS
// =============================================================================

/**
 * Hook to initialize and manage the complete 3D engine
 */
export interface UseEngineOptions {
  enableWebGPU?: boolean;
  sceneGraphConfig?: Partial<SceneGraphConfig>;
  cameraConfig?: Partial<CameraConfig>;
  webgpuConfig?: Partial<WebGPUBridgeConfig>;
  onInitialized?: () => void;
  onError?: (error: Error) => void;
}

export function useEngine(options: UseEngineOptions = {}): EngineContextValue {
  const { scene, camera, gl } = useThree();
  
  const [isInitialized, setIsInitialized] = useState(false);
  const [stats, setStats] = useState<EngineStats>(defaultStats);
  
  const sceneGraphRef = useRef<SceneGraph | null>(null);
  const cameraSystemRef = useRef<CameraSystem | null>(null);
  const materialLibraryRef = useRef<MaterialLibrary | null>(null);
  const webgpuBridgeRef = useRef<WebGPUBridge | null>(null);
  const sceneManagerRef = useRef<SceneManager | null>(null);
  
  // Initialize all systems
  useEffect(() => {
    const init = async () => {
      try {
        // Initialize Material Library
        materialLibraryRef.current = new MaterialLibrary();
        
        // Initialize Scene Graph
        sceneGraphRef.current = new SceneGraph({
          maxDepth: options.sceneGraphConfig?.maxDepth ?? 8,
          minNodeSize: options.sceneGraphConfig?.minNodeSize ?? 10,
          worldBounds: options.sceneGraphConfig?.worldBounds ?? {
            min: new THREE.Vector3(-1000, -1000, -1000),
            max: new THREE.Vector3(1000, 1000, 1000)
          }
        });
        
        // Initialize Camera System
        cameraSystemRef.current = new CameraSystem(
          camera as THREE.PerspectiveCamera,
          gl.domElement,
          options.cameraConfig
        );
        
        // Initialize Scene Manager
        sceneManagerRef.current = new SceneManager(scene, {
          instanceBatchSize: 1000,
          lodDistances: [50, 150, 400],
          frustumCulling: true,
          gpuInstancing: true
        });
        
        // Initialize WebGPU Bridge if enabled
        if (options.enableWebGPU) {
          webgpuBridgeRef.current = new WebGPUBridge(options.webgpuConfig);
          await webgpuBridgeRef.current.initialize();
        }
        
        setIsInitialized(true);
        options.onInitialized?.();
      } catch (error) {
        console.error('Engine initialization failed:', error);
        options.onError?.(error as Error);
      }
    };
    
    init();
    
    return () => {
      sceneGraphRef.current?.dispose();
      cameraSystemRef.current?.dispose();
      materialLibraryRef.current?.dispose();
      webgpuBridgeRef.current?.dispose();
      sceneManagerRef.current?.dispose();
    };
  }, []);
  
  // Update stats each frame
  useFrame((state, delta) => {
    if (!isInitialized) return;
    
    const info = state.gl.info;
    const manager = sceneManagerRef.current;
    const sceneGraph = sceneGraphRef.current;
    
    setStats({
      fps: Math.round(1 / delta),
      frameTime: delta * 1000,
      triangles: info.render?.triangles ?? 0,
      drawCalls: info.render?.calls ?? 0,
      agentCount: manager?.getAgentCount() ?? 0,
      connectionCount: manager?.getConnectionCount() ?? 0,
      visibleNodes: sceneGraph?.getVisibleNodeCount() ?? 0,
      gpuMemory: info.memory?.geometries ?? 0
    });
    
    // Update camera system
    cameraSystemRef.current?.update(delta);
  });
  
  return {
    sceneGraph: sceneGraphRef.current,
    cameraSystem: cameraSystemRef.current,
    materialLibrary: materialLibraryRef.current,
    webgpuBridge: webgpuBridgeRef.current,
    sceneManager: sceneManagerRef.current,
    isInitialized,
    stats
  };
}

// =============================================================================
// SCENE GRAPH HOOKS
// =============================================================================

/**
 * Hook for scene graph operations
 */
export function useSceneGraph() {
  const { sceneGraph, isInitialized } = useEngineContext();
  
  const addNode = useCallback((
    type: SceneNodeType,
    transform?: { position?: THREE.Vector3; rotation?: THREE.Euler; scale?: THREE.Vector3 },
    parentId?: string
  ): SceneNode | null => {
    if (!sceneGraph) return null;
    return sceneGraph.addNode(type, transform, parentId);
  }, [sceneGraph]);
  
  const removeNode = useCallback((nodeId: string): boolean => {
    if (!sceneGraph) return false;
    return sceneGraph.removeNode(nodeId);
  }, [sceneGraph]);
  
  const getNode = useCallback((nodeId: string): SceneNode | null => {
    if (!sceneGraph) return null;
    return sceneGraph.getNode(nodeId);
  }, [sceneGraph]);
  
  const queryFrustum = useCallback((camera: THREE.Camera): SceneNode[] => {
    if (!sceneGraph) return [];
    return sceneGraph.queryFrustum(camera);
  }, [sceneGraph]);
  
  const queryRadius = useCallback((center: THREE.Vector3, radius: number): SceneNode[] => {
    if (!sceneGraph) return [];
    return sceneGraph.queryRadius(center, radius);
  }, [sceneGraph]);
  
  const queryRay = useCallback((origin: THREE.Vector3, direction: THREE.Vector3): SceneNode[] => {
    if (!sceneGraph) return [];
    return sceneGraph.queryRay(origin, direction);
  }, [sceneGraph]);
  
  return {
    sceneGraph,
    isInitialized,
    addNode,
    removeNode,
    getNode,
    queryFrustum,
    queryRadius,
    queryRay
  };
}

/**
 * Hook for observing scene graph changes
 */
export function useSceneGraphObserver(
  onNodeAdded?: (node: SceneNode) => void,
  onNodeRemoved?: (nodeId: string) => void,
  onNodeUpdated?: (node: SceneNode) => void
) {
  const { sceneGraph, isInitialized } = useEngineContext();
  
  useEffect(() => {
    if (!sceneGraph || !isInitialized) return;
    
    // Subscribe to scene graph events
    const handleAdded = onNodeAdded ? (e: CustomEvent<SceneNode>) => onNodeAdded(e.detail) : undefined;
    const handleRemoved = onNodeRemoved ? (e: CustomEvent<string>) => onNodeRemoved(e.detail) : undefined;
    const handleUpdated = onNodeUpdated ? (e: CustomEvent<SceneNode>) => onNodeUpdated(e.detail) : undefined;
    
    const target = sceneGraph as unknown as EventTarget;
    
    if (handleAdded) target.addEventListener('nodeAdded', handleAdded as EventListener);
    if (handleRemoved) target.addEventListener('nodeRemoved', handleRemoved as EventListener);
    if (handleUpdated) target.addEventListener('nodeUpdated', handleUpdated as EventListener);
    
    return () => {
      if (handleAdded) target.removeEventListener('nodeAdded', handleAdded as EventListener);
      if (handleRemoved) target.removeEventListener('nodeRemoved', handleRemoved as EventListener);
      if (handleUpdated) target.removeEventListener('nodeUpdated', handleUpdated as EventListener);
    };
  }, [sceneGraph, isInitialized, onNodeAdded, onNodeRemoved, onNodeUpdated]);
}

// =============================================================================
// CAMERA HOOKS
// =============================================================================

/**
 * Hook for camera control
 */
export function useCamera() {
  const { cameraSystem, isInitialized } = useEngineContext();
  const [cameraState, setCameraState] = useState<CameraState | null>(null);
  
  // Update camera state each frame
  useFrame(() => {
    if (!cameraSystem) return;
    setCameraState(cameraSystem.getState());
  });
  
  const setMode = useCallback((mode: CameraMode, options?: {
    target?: THREE.Vector3;
    duration?: number;
  }) => {
    if (!cameraSystem) return;
    cameraSystem.setMode(mode, options);
  }, [cameraSystem]);
  
  const lookAt = useCallback((target: THREE.Vector3, duration?: number) => {
    if (!cameraSystem) return;
    cameraSystem.lookAt(target, duration);
  }, [cameraSystem]);
  
  const moveTo = useCallback((position: THREE.Vector3, duration?: number) => {
    if (!cameraSystem) return;
    cameraSystem.moveTo(position, duration);
  }, [cameraSystem]);
  
  const focusOn = useCallback((target: THREE.Object3D | THREE.Vector3, padding?: number) => {
    if (!cameraSystem) return;
    cameraSystem.focusOn(target, padding);
  }, [cameraSystem]);
  
  const setFollowTarget = useCallback((target: THREE.Object3D | null) => {
    if (!cameraSystem) return;
    cameraSystem.setFollowTarget(target);
  }, [cameraSystem]);
  
  const startCinematic = useCallback((keyframes: Array<{
    position: THREE.Vector3;
    target: THREE.Vector3;
    duration: number;
    easing?: string;
  }>) => {
    if (!cameraSystem) return;
    cameraSystem.startCinematic(keyframes);
  }, [cameraSystem]);
  
  const stopCinematic = useCallback(() => {
    if (!cameraSystem) return;
    cameraSystem.stopCinematic();
  }, [cameraSystem]);
  
  return {
    cameraSystem,
    cameraState,
    isInitialized,
    setMode,
    lookAt,
    moveTo,
    focusOn,
    setFollowTarget,
    startCinematic,
    stopCinematic
  };
}

/**
 * Hook for camera constraints
 */
export function useCameraConstraints(constraints: {
  minDistance?: number;
  maxDistance?: number;
  minPolarAngle?: number;
  maxPolarAngle?: number;
  boundingBox?: THREE.Box3;
}) {
  const { cameraSystem } = useEngineContext();
  
  useEffect(() => {
    if (!cameraSystem) return;
    cameraSystem.setConstraints(constraints);
  }, [cameraSystem, constraints]);
}

// =============================================================================
// MATERIAL HOOKS
// =============================================================================

/**
 * Hook for material management
 */
export function useMaterials() {
  const { materialLibrary, isInitialized } = useEngineContext();
  
  const getAgentMaterial = useCallback((status: AgentStatus, options?: Partial<AgentMaterialOptions>) => {
    if (!materialLibrary) return null;
    return materialLibrary.getAgentMaterial(status, options);
  }, [materialLibrary]);
  
  const getConnectionMaterial = useCallback((type?: 'data' | 'control' | 'dependency') => {
    if (!materialLibrary) return null;
    return materialLibrary.getConnectionMaterial(type);
  }, [materialLibrary]);
  
  const getHolographicMaterial = useCallback((options?: {
    baseColor?: THREE.Color;
    scanlineSpeed?: number;
    glitchIntensity?: number;
  }) => {
    if (!materialLibrary) return null;
    return materialLibrary.getHolographicMaterial(options);
  }, [materialLibrary]);
  
  const getEnergyMaterial = useCallback((options?: {
    color?: THREE.Color;
    pulseSpeed?: number;
    intensity?: number;
  }) => {
    if (!materialLibrary) return null;
    return materialLibrary.getEnergyMaterial(options);
  }, [materialLibrary]);
  
  const getGridMaterial = useCallback((options?: {
    gridColor?: THREE.Color;
    cellSize?: number;
    lineWidth?: number;
  }) => {
    if (!materialLibrary) return null;
    return materialLibrary.getGridMaterial(options);
  }, [materialLibrary]);
  
  const updateTime = useCallback((time: number) => {
    if (!materialLibrary) return;
    materialLibrary.updateTime(time);
  }, [materialLibrary]);
  
  // Auto-update time each frame
  useFrame((state) => {
    if (!materialLibrary) return;
    materialLibrary.updateTime(state.clock.elapsedTime);
  });
  
  return {
    materialLibrary,
    isInitialized,
    getAgentMaterial,
    getConnectionMaterial,
    getHolographicMaterial,
    getEnergyMaterial,
    getGridMaterial,
    updateTime
  };
}

// =============================================================================
// SCENE MANAGER HOOKS
// =============================================================================

/**
 * Hook for agent management
 */
export function useAgents() {
  const { sceneManager, isInitialized } = useEngineContext();
  const [agentCount, setAgentCount] = useState(0);
  
  // Track agent count
  useFrame(() => {
    if (!sceneManager) return;
    setAgentCount(sceneManager.getAgentCount());
  });
  
  const addAgent = useCallback((data: AgentNodeData): string | null => {
    if (!sceneManager) return null;
    return sceneManager.addAgent(data);
  }, [sceneManager]);
  
  const removeAgent = useCallback((agentId: string): boolean => {
    if (!sceneManager) return false;
    return sceneManager.removeAgent(agentId);
  }, [sceneManager]);
  
  const updateAgent = useCallback((agentId: string, data: Partial<AgentNodeData>): boolean => {
    if (!sceneManager) return false;
    return sceneManager.updateAgent(agentId, data);
  }, [sceneManager]);
  
  const getAgent = useCallback((agentId: string) => {
    if (!sceneManager) return null;
    return sceneManager.getAgent(agentId);
  }, [sceneManager]);
  
  const getAgentAtPosition = useCallback((position: THREE.Vector3, radius?: number) => {
    if (!sceneManager) return null;
    return sceneManager.getAgentAtPosition(position, radius);
  }, [sceneManager]);
  
  const getAgentsInFrustum = useCallback((camera: THREE.Camera) => {
    if (!sceneManager) return [];
    return sceneManager.getAgentsInFrustum(camera);
  }, [sceneManager]);
  
  const highlightAgent = useCallback((agentId: string, highlighted: boolean) => {
    if (!sceneManager) return;
    sceneManager.highlightAgent(agentId, highlighted);
  }, [sceneManager]);
  
  const selectAgent = useCallback((agentId: string | null) => {
    if (!sceneManager) return;
    sceneManager.selectAgent(agentId);
  }, [sceneManager]);
  
  return {
    sceneManager,
    agentCount,
    isInitialized,
    addAgent,
    removeAgent,
    updateAgent,
    getAgent,
    getAgentAtPosition,
    getAgentsInFrustum,
    highlightAgent,
    selectAgent
  };
}

/**
 * Hook for connection management
 */
export function useConnections() {
  const { sceneManager, isInitialized } = useEngineContext();
  const [connectionCount, setConnectionCount] = useState(0);
  
  // Track connection count
  useFrame(() => {
    if (!sceneManager) return;
    setConnectionCount(sceneManager.getConnectionCount());
  });
  
  const addConnection = useCallback((data: AgentConnectionData): string | null => {
    if (!sceneManager) return null;
    return sceneManager.addConnection(data);
  }, [sceneManager]);
  
  const removeConnection = useCallback((connectionId: string): boolean => {
    if (!sceneManager) return false;
    return sceneManager.removeConnection(connectionId);
  }, [sceneManager]);
  
  const updateConnection = useCallback((connectionId: string, data: Partial<AgentConnectionData>): boolean => {
    if (!sceneManager) return false;
    return sceneManager.updateConnection(connectionId, data);
  }, [sceneManager]);
  
  const getConnection = useCallback((connectionId: string) => {
    if (!sceneManager) return null;
    return sceneManager.getConnection(connectionId);
  }, [sceneManager]);
  
  const getConnectionsForAgent = useCallback((agentId: string) => {
    if (!sceneManager) return [];
    return sceneManager.getConnectionsForAgent(agentId);
  }, [sceneManager]);
  
  const animateConnection = useCallback((connectionId: string, options?: {
    speed?: number;
    direction?: 'forward' | 'backward' | 'bidirectional';
  }) => {
    if (!sceneManager) return;
    sceneManager.animateConnection(connectionId, options);
  }, [sceneManager]);
  
  return {
    sceneManager,
    connectionCount,
    isInitialized,
    addConnection,
    removeConnection,
    updateConnection,
    getConnection,
    getConnectionsForAgent,
    animateConnection
  };
}

// =============================================================================
// SELECTION & INTERACTION HOOKS
// =============================================================================

/**
 * Hook for object selection
 */
export function useSelection() {
  const { sceneManager } = useEngineContext();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  
  const select = useCallback((id: string | null) => {
    setSelectedId(id);
    sceneManager?.selectAgent(id);
  }, [sceneManager]);
  
  const hover = useCallback((id: string | null) => {
    if (hoveredId !== id) {
      if (hoveredId) sceneManager?.highlightAgent(hoveredId, false);
      if (id) sceneManager?.highlightAgent(id, true);
      setHoveredId(id);
    }
  }, [sceneManager, hoveredId]);
  
  const clearSelection = useCallback(() => {
    select(null);
  }, [select]);
  
  return {
    selectedId,
    hoveredId,
    select,
    hover,
    clearSelection
  };
}

/**
 * Hook for raycasting and picking
 */
export function useRaycast() {
  const { camera, scene, raycaster } = useThree();
  const { sceneManager } = useEngineContext();
  
  const raycastFromMouse = useCallback((
    event: React.MouseEvent | MouseEvent,
    objects?: THREE.Object3D[]
  ): THREE.Intersection[] => {
    const rect = (event.target as HTMLElement).getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    raycaster.setFromCamera(new THREE.Vector2(x, y), camera);
    
    const targets = objects ?? scene.children;
    return raycaster.intersectObjects(targets, true);
  }, [camera, scene, raycaster]);
  
  const getAgentUnderMouse = useCallback((event: React.MouseEvent | MouseEvent) => {
    const intersects = raycastFromMouse(event);
    
    for (const intersect of intersects) {
      const agentId = intersect.object.userData?.agentId;
      if (agentId) {
        return {
          agentId,
          point: intersect.point,
          distance: intersect.distance,
          object: intersect.object
        };
      }
    }
    
    return null;
  }, [raycastFromMouse]);
  
  return {
    raycastFromMouse,
    getAgentUnderMouse
  };
}

// =============================================================================
// ANIMATION HOOKS
// =============================================================================

/**
 * Hook for frame-based animation
 */
export function useAnimationLoop(
  callback: (state: RootState, delta: number) => void,
  priority?: number
) {
  useFrame(callback, priority);
}

/**
 * Hook for interpolated value animation
 */
export function useInterpolatedValue(
  target: number,
  options: {
    speed?: number;
    threshold?: number;
    easing?: 'linear' | 'ease-in' | 'ease-out' | 'ease-in-out';
  } = {}
) {
  const { speed = 5, threshold = 0.001, easing = 'ease-out' } = options;
  const currentRef = useRef(target);
  const [value, setValue] = useState(target);
  
  const easingFunctions = {
    'linear': (t: number) => t,
    'ease-in': (t: number) => t * t,
    'ease-out': (t: number) => t * (2 - t),
    'ease-in-out': (t: number) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t
  };
  
  useFrame((_, delta) => {
    const diff = target - currentRef.current;
    if (Math.abs(diff) < threshold) {
      if (currentRef.current !== target) {
        currentRef.current = target;
        setValue(target);
      }
      return;
    }
    
    const t = Math.min(1, speed * delta);
    const easedT = easingFunctions[easing](t);
    currentRef.current += diff * easedT;
    setValue(currentRef.current);
  });
  
  return value;
}

/**
 * Hook for spring physics animation
 */
export function useSpring(
  target: THREE.Vector3,
  options: {
    stiffness?: number;
    damping?: number;
    mass?: number;
  } = {}
) {
  const { stiffness = 100, damping = 10, mass = 1 } = options;
  const positionRef = useRef(target.clone());
  const velocityRef = useRef(new THREE.Vector3());
  const [position, setPosition] = useState(target.clone());
  
  useFrame((_, delta) => {
    // Spring force: F = -k * (current - target)
    const displacement = positionRef.current.clone().sub(target);
    const springForce = displacement.multiplyScalar(-stiffness);
    
    // Damping force: F = -c * velocity
    const dampingForce = velocityRef.current.clone().multiplyScalar(-damping);
    
    // Total acceleration: a = F / m
    const acceleration = springForce.add(dampingForce).divideScalar(mass);
    
    // Update velocity and position
    velocityRef.current.addScaledVector(acceleration, delta);
    positionRef.current.addScaledVector(velocityRef.current, delta);
    
    setPosition(positionRef.current.clone());
  });
  
  return position;
}

// =============================================================================
// PERFORMANCE HOOKS
// =============================================================================

/**
 * Hook for performance monitoring
 */
export function usePerformanceMonitor(options: {
  targetFPS?: number;
  onPerformanceIssue?: (fps: number) => void;
  sampleSize?: number;
} = {}) {
  const { targetFPS = 60, onPerformanceIssue, sampleSize = 60 } = options;
  const { stats } = useEngineContext();
  const fpsHistory = useRef<number[]>([]);
  const [averageFPS, setAverageFPS] = useState(60);
  const [performanceLevel, setPerformanceLevel] = useState<'high' | 'medium' | 'low'>('high');
  
  useFrame(() => {
    fpsHistory.current.push(stats.fps);
    if (fpsHistory.current.length > sampleSize) {
      fpsHistory.current.shift();
    }
    
    const avg = fpsHistory.current.reduce((a, b) => a + b, 0) / fpsHistory.current.length;
    setAverageFPS(Math.round(avg));
    
    // Determine performance level
    if (avg >= targetFPS * 0.9) {
      setPerformanceLevel('high');
    } else if (avg >= targetFPS * 0.5) {
      setPerformanceLevel('medium');
    } else {
      setPerformanceLevel('low');
      onPerformanceIssue?.(avg);
    }
  });
  
  return {
    currentFPS: stats.fps,
    averageFPS,
    performanceLevel,
    stats
  };
}

/**
 * Hook for automatic quality adjustment
 */
export function useAdaptiveQuality(options: {
  minQuality?: number;
  maxQuality?: number;
  targetFPS?: number;
} = {}) {
  const { minQuality = 0.5, maxQuality = 1.0, targetFPS = 60 } = options;
  const { averageFPS, performanceLevel } = usePerformanceMonitor({ targetFPS });
  const [quality, setQuality] = useState(maxQuality);
  
  useEffect(() => {
    if (performanceLevel === 'low') {
      setQuality(q => Math.max(minQuality, q - 0.1));
    } else if (performanceLevel === 'high' && quality < maxQuality) {
      setQuality(q => Math.min(maxQuality, q + 0.05));
    }
  }, [performanceLevel, minQuality, maxQuality, quality]);
  
  return {
    quality,
    performanceLevel,
    averageFPS
  };
}

// =============================================================================
// UTILITY HOOKS
// =============================================================================

/**
 * Hook for keyboard shortcuts
 */
export function useKeyboardShortcuts(shortcuts: Record<string, () => void>) {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Build shortcut string
      const parts: string[] = [];
      if (event.ctrlKey || event.metaKey) parts.push('ctrl');
      if (event.shiftKey) parts.push('shift');
      if (event.altKey) parts.push('alt');
      parts.push(event.key.toLowerCase());
      
      const shortcut = parts.join('+');
      
      if (shortcuts[shortcut]) {
        event.preventDefault();
        shortcuts[shortcut]();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts]);
}

/**
 * Hook for window resize handling
 */
export function useWindowResize(callback: (width: number, height: number) => void) {
  useEffect(() => {
    const handleResize = () => {
      callback(window.innerWidth, window.innerHeight);
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [callback]);
}

/**
 * Hook for capturing screenshots
 */
export function useScreenshot() {
  const { gl, scene, camera } = useThree();
  
  const capture = useCallback((options?: {
    width?: number;
    height?: number;
    format?: 'png' | 'jpeg';
    quality?: number;
  }) => {
    const { width, height, format = 'png', quality = 0.95 } = options ?? {};
    
    // Store original size
    const originalSize = gl.getSize(new THREE.Vector2());
    
    // Resize if needed
    if (width && height) {
      gl.setSize(width, height);
    }
    
    // Render frame
    gl.render(scene, camera);
    
    // Capture canvas
    const mimeType = format === 'png' ? 'image/png' : 'image/jpeg';
    const dataURL = gl.domElement.toDataURL(mimeType, quality);
    
    // Restore original size
    if (width && height) {
      gl.setSize(originalSize.x, originalSize.y);
    }
    
    return dataURL;
  }, [gl, scene, camera]);
  
  const download = useCallback((filename: string, options?: {
    width?: number;
    height?: number;
    format?: 'png' | 'jpeg';
    quality?: number;
  }) => {
    const dataURL = capture(options);
    
    const link = document.createElement('a');
    link.download = filename;
    link.href = dataURL;
    link.click();
  }, [capture]);
  
  return { capture, download };
}

// Export default for convenience
export default {
  useEngine,
  useSceneGraph,
  useSceneGraphObserver,
  useCamera,
  useCameraConstraints,
  useMaterials,
  useAgents,
  useConnections,
  useSelection,
  useRaycast,
  useAnimationLoop,
  useInterpolatedValue,
  useSpring,
  usePerformanceMonitor,
  useAdaptiveQuality,
  useKeyboardShortcuts,
  useWindowResize,
  useScreenshot
};
