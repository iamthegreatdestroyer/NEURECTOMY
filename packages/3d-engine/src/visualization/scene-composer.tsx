/**
 * @file Scene Composer - Main Visualization Container
 * @description High-level component for composing agent architecture visualizations
 * @module @neurectomy/3d-engine/visualization
 * @agents @CANVAS @ARCHITECT
 */

import React, { 
  useRef, 
  useState, 
  useCallback, 
  useEffect, 
  useMemo,
  Suspense,
} from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { 
  OrbitControls, 
  PerspectiveCamera,
  Environment,
  Stats,
  GizmoHelper,
  GizmoViewport,
  ContactShadows,
  Html,
  useProgress,
  AdaptiveDpr,
  AdaptiveEvents,
  Preload,
} from '@react-three/drei';
import {
  Vector3,
  Color,
  PCFSoftShadowMap,
  ACESFilmicToneMapping,
} from 'three';
import type {
  AgentComponent,
  AgentConnection,
  VisualizationMode,
  ViewportConfig,
  SelectionState,
  RenderStats,
  Vector3D,
} from './types';
import { AgentGroup } from './agent-renderer';
import { ConnectionGroup } from './connection-renderer';
import { BlueprintMode } from './blueprint-mode';
import { InteractionManager, SelectionChangeEvent, DragEvent } from './interaction-system';

// ============================================================================
// Types & Interfaces
// ============================================================================

export interface SceneComposerProps {
  // Data
  components: AgentComponent[];
  connections: AgentConnection[];
  
  // Configuration
  mode?: VisualizationMode;
  viewportConfig?: Partial<ViewportConfig>;
  
  // State
  selectedIds?: string[];
  hoveredId?: string | null;
  
  // Events
  onSelectionChange?: (selection: SelectionState) => void;
  onHover?: (componentId: string | null) => void;
  onComponentClick?: (componentId: string, event: React.MouseEvent) => void;
  onConnectionClick?: (connectionId: string, event: React.MouseEvent) => void;
  onComponentMove?: (componentId: string, position: Vector3D) => void;
  onConnectionCreate?: (sourceId: string, targetId: string) => void;
  onRenderStats?: (stats: RenderStats) => void;
  
  // UI Options
  showStats?: boolean;
  showGizmo?: boolean;
  showGrid?: boolean;
  showShadows?: boolean;
  enableInteraction?: boolean;
  
  // Styling
  className?: string;
  style?: React.CSSProperties;
}

const DEFAULT_VIEWPORT_CONFIG: ViewportConfig = {
  fov: 45,
  near: 0.1,
  far: 1000,
  initialPosition: { x: 10, y: 10, z: 10 },
  initialTarget: { x: 0, y: 0, z: 0 },
  enableOrbitControls: true,
  enablePan: true,
  enableZoom: true,
  enableRotate: true,
  minDistance: 2,
  maxDistance: 100,
  maxPolarAngle: Math.PI / 2 + 0.2,
};

// ============================================================================
// Loading Fallback
// ============================================================================

const Loader: React.FC = () => {
  const { progress } = useProgress();
  
  return (
    <Html center>
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '12px',
          color: '#60a5fa',
          fontFamily: 'system-ui, -apple-system, sans-serif',
        }}
      >
        <div
          style={{
            width: '200px',
            height: '4px',
            background: '#1e3a5f',
            borderRadius: '2px',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              width: `${progress}%`,
              height: '100%',
              background: '#3b82f6',
              transition: 'width 0.3s ease',
            }}
          />
        </div>
        <span style={{ fontSize: '12px' }}>
          Loading... {progress.toFixed(0)}%
        </span>
      </div>
    </Html>
  );
};

// ============================================================================
// Scene Setup Component
// ============================================================================

interface SceneSetupProps {
  viewportConfig: ViewportConfig;
  showShadows: boolean;
  showGizmo: boolean;
  mode: VisualizationMode;
}

const SceneSetup: React.FC<SceneSetupProps> = ({
  viewportConfig,
  showShadows,
  showGizmo,
  mode,
}) => {
  const { camera } = useThree();

  // Set initial camera position
  useEffect(() => {
    camera.position.set(
      viewportConfig.initialPosition.x,
      viewportConfig.initialPosition.y,
      viewportConfig.initialPosition.z
    );
    camera.lookAt(
      viewportConfig.initialTarget.x,
      viewportConfig.initialTarget.y,
      viewportConfig.initialTarget.z
    );
  }, [camera, viewportConfig]);

  // Background color based on mode
  const backgroundColor = useMemo(() => {
    switch (mode) {
      case 'blueprint':
        return '#0a1929';
      case 'dark':
        return '#0f172a';
      case 'light':
        return '#f8fafc';
      default:
        return '#1e293b';
    }
  }, [mode]);

  return (
    <>
      {/* Background */}
      <color attach="background" args={[backgroundColor]} />
      
      {/* Ambient light */}
      <ambientLight intensity={mode === 'light' ? 0.6 : 0.4} />
      
      {/* Key light */}
      <directionalLight
        position={[10, 15, 10]}
        intensity={mode === 'light' ? 1.2 : 0.8}
        castShadow={showShadows}
        shadow-mapSize={[2048, 2048]}
        shadow-camera-far={50}
        shadow-camera-left={-20}
        shadow-camera-right={20}
        shadow-camera-top={20}
        shadow-camera-bottom={-20}
      />
      
      {/* Fill light */}
      <directionalLight
        position={[-5, 5, -5]}
        intensity={0.3}
      />
      
      {/* Rim light */}
      <directionalLight
        position={[0, 10, -10]}
        intensity={0.2}
      />

      {/* Environment for reflections */}
      {mode !== 'blueprint' && (
        <Environment preset="city" background={false} />
      )}

      {/* Contact shadows */}
      {showShadows && mode !== 'blueprint' && (
        <ContactShadows
          position={[0, -0.01, 0]}
          opacity={0.4}
          scale={30}
          blur={2}
          far={10}
        />
      )}

      {/* Gizmo helper */}
      {showGizmo && (
        <GizmoHelper
          alignment="bottom-right"
          margin={[80, 80]}
        >
          <GizmoViewport
            axisColors={['#ef4444', '#22c55e', '#3b82f6']}
            labelColor="white"
          />
        </GizmoHelper>
      )}

      {/* Orbit controls */}
      {viewportConfig.enableOrbitControls && (
        <OrbitControls
          makeDefault
          enablePan={viewportConfig.enablePan}
          enableZoom={viewportConfig.enableZoom}
          enableRotate={viewportConfig.enableRotate}
          minDistance={viewportConfig.minDistance}
          maxDistance={viewportConfig.maxDistance}
          maxPolarAngle={viewportConfig.maxPolarAngle}
          target={[
            viewportConfig.initialTarget.x,
            viewportConfig.initialTarget.y,
            viewportConfig.initialTarget.z,
          ]}
        />
      )}
    </>
  );
};

// ============================================================================
// Render Stats Tracker
// ============================================================================

interface StatsTrackerProps {
  onStats?: (stats: RenderStats) => void;
}

const StatsTracker: React.FC<StatsTrackerProps> = ({ onStats }) => {
  const { gl, scene } = useThree();
  const frameCount = useRef(0);
  const lastTime = useRef(performance.now());

  useFrame(() => {
    frameCount.current++;
    
    const now = performance.now();
    if (now - lastTime.current >= 1000) {
      const info = gl.info;
      
      onStats?.({
        fps: frameCount.current,
        frameTime: (now - lastTime.current) / frameCount.current,
        triangles: info.render.triangles,
        drawCalls: info.render.calls,
        geometries: info.memory.geometries,
        textures: info.memory.textures,
        programs: info.programs?.length || 0,
      });

      frameCount.current = 0;
      lastTime.current = now;
    }
  });

  return null;
};

// ============================================================================
// Scene Content
// ============================================================================

interface SceneContentProps {
  components: AgentComponent[];
  connections: AgentConnection[];
  mode: VisualizationMode;
  selectedIds: Set<string>;
  hoveredId: string | null;
  showGrid: boolean;
  onSelectionChange: (selection: SelectionChangeEvent) => void;
  onHover: (id: string | null) => void;
  onDrag: (event: DragEvent) => void;
}

const SceneContent: React.FC<SceneContentProps> = ({
  components,
  connections,
  mode,
  selectedIds,
  hoveredId,
  showGrid,
  onSelectionChange,
  onHover,
  onDrag,
}) => {
  // Build position map for connections
  const componentPositions = useMemo(() => {
    const map = new Map<string, Vector3D>();
    components.forEach(c => map.set(c.id, c.position));
    return map;
  }, [components]);

  return (
    <group name="scene-content">
      {/* Blueprint mode overlay */}
      {mode === 'blueprint' && (
        <BlueprintMode
          components={components}
          connections={connections}
          selectedIds={selectedIds}
          showMeasurements={false}
        />
      )}

      {/* Regular visualization modes */}
      {mode !== 'blueprint' && (
        <>
          {/* Grid floor */}
          {showGrid && (
            <gridHelper
              args={[40, 40, '#3b82f6', '#1e3a5f']}
              position={[0, -0.01, 0]}
            />
          )}

          {/* Agent components */}
          <AgentGroup
            components={components}
            selectedIds={selectedIds}
            hoveredId={hoveredId}
            lodEnabled={components.length > 50}
          />

          {/* Connections */}
          <ConnectionGroup
            connections={connections}
            componentPositions={componentPositions}
            selectedIds={selectedIds}
            showDataFlow
            showLabels={mode !== 'dark'}
          />
        </>
      )}
    </group>
  );
};

// ============================================================================
// Main Scene Composer Component
// ============================================================================

export const SceneComposer: React.FC<SceneComposerProps> = ({
  components,
  connections,
  mode = 'default',
  viewportConfig: userViewportConfig,
  selectedIds: externalSelectedIds = [],
  hoveredId: externalHoveredId = null,
  onSelectionChange,
  onHover,
  onComponentClick,
  onConnectionClick,
  onComponentMove,
  onConnectionCreate,
  onRenderStats,
  showStats = false,
  showGizmo = true,
  showGrid = true,
  showShadows = true,
  enableInteraction = true,
  className,
  style,
}) => {
  // Merge viewport config with defaults
  const viewportConfig = useMemo(
    () => ({ ...DEFAULT_VIEWPORT_CONFIG, ...userViewportConfig }),
    [userViewportConfig]
  );

  // Internal state
  const [internalSelectedIds, setInternalSelectedIds] = useState<Set<string>>(
    new Set(externalSelectedIds)
  );
  const [internalHoveredId, setInternalHoveredId] = useState<string | null>(
    externalHoveredId
  );

  // Sync external state
  useEffect(() => {
    setInternalSelectedIds(new Set(externalSelectedIds));
  }, [externalSelectedIds]);

  useEffect(() => {
    setInternalHoveredId(externalHoveredId);
  }, [externalHoveredId]);

  // Event handlers
  const handleSelectionChange = useCallback(
    (event: SelectionChangeEvent) => {
      const newSelection = new Set(event.componentIds);
      setInternalSelectedIds(newSelection);
      
      onSelectionChange?.({
        componentIds: event.componentIds,
        connectionIds: event.connectionIds,
      });
    },
    [onSelectionChange]
  );

  const handleHover = useCallback(
    (id: string | null) => {
      setInternalHoveredId(id);
      onHover?.(id);
    },
    [onHover]
  );

  const handleDrag = useCallback(
    (event: DragEvent) => {
      if (event.phase === 'end') {
        onComponentMove?.(event.componentId, event.currentPosition);
      }
    },
    [onComponentMove]
  );

  // Canvas configuration
  const canvasConfig = useMemo(
    () => ({
      shadows: showShadows ? { type: PCFSoftShadowMap } : false,
      dpr: [1, 2] as [number, number],
      gl: {
        antialias: true,
        alpha: false,
        powerPreference: 'high-performance' as const,
        toneMapping: ACESFilmicToneMapping,
        toneMappingExposure: 1.2,
      },
      camera: {
        fov: viewportConfig.fov,
        near: viewportConfig.near,
        far: viewportConfig.far,
        position: [
          viewportConfig.initialPosition.x,
          viewportConfig.initialPosition.y,
          viewportConfig.initialPosition.z,
        ] as [number, number, number],
      },
    }),
    [showShadows, viewportConfig]
  );

  return (
    <div
      className={className}
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
        ...style,
      }}
    >
      <Canvas {...canvasConfig}>
        <Suspense fallback={<Loader />}>
          {/* Performance optimizations */}
          <AdaptiveDpr pixelated />
          <AdaptiveEvents />
          <Preload all />

          {/* Scene setup */}
          <SceneSetup
            viewportConfig={viewportConfig}
            showShadows={showShadows}
            showGizmo={showGizmo}
            mode={mode}
          />

          {/* Main content */}
          <SceneContent
            components={components}
            connections={connections}
            mode={mode}
            selectedIds={internalSelectedIds}
            hoveredId={internalHoveredId}
            showGrid={showGrid}
            onSelectionChange={handleSelectionChange}
            onHover={handleHover}
            onDrag={handleDrag}
          />

          {/* Stats tracking */}
          {onRenderStats && <StatsTracker onStats={onRenderStats} />}
        </Suspense>

        {/* Performance stats overlay */}
        {showStats && <Stats />}
      </Canvas>

      {/* Mode indicator */}
      <ModeIndicator mode={mode} />

      {/* Keyboard shortcuts help */}
      {enableInteraction && <KeyboardShortcutsHint />}
    </div>
  );
};

// ============================================================================
// UI Overlays
// ============================================================================

interface ModeIndicatorProps {
  mode: VisualizationMode;
}

const ModeIndicator: React.FC<ModeIndicatorProps> = ({ mode }) => {
  const modeLabels: Record<VisualizationMode, string> = {
    default: 'Default',
    dark: 'Dark Mode',
    light: 'Light Mode',
    blueprint: 'Blueprint',
    wireframe: 'Wireframe',
  };

  return (
    <div
      style={{
        position: 'absolute',
        top: '12px',
        left: '12px',
        padding: '6px 12px',
        background: 'rgba(0, 0, 0, 0.6)',
        borderRadius: '4px',
        color: '#fff',
        fontSize: '12px',
        fontFamily: 'system-ui, -apple-system, sans-serif',
        pointerEvents: 'none',
      }}
    >
      {modeLabels[mode]}
    </div>
  );
};

const KeyboardShortcutsHint: React.FC = () => {
  const [showHelp, setShowHelp] = useState(false);

  return (
    <>
      <button
        onClick={() => setShowHelp(!showHelp)}
        style={{
          position: 'absolute',
          bottom: '12px',
          right: '12px',
          width: '32px',
          height: '32px',
          borderRadius: '50%',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          background: 'rgba(0, 0, 0, 0.6)',
          color: '#fff',
          fontSize: '14px',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
        title="Keyboard shortcuts"
      >
        ?
      </button>

      {showHelp && (
        <div
          style={{
            position: 'absolute',
            bottom: '52px',
            right: '12px',
            padding: '12px 16px',
            background: 'rgba(0, 0, 0, 0.85)',
            borderRadius: '8px',
            color: '#fff',
            fontSize: '12px',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            minWidth: '200px',
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: '8px' }}>
            Keyboard Shortcuts
          </div>
          <div style={{ display: 'grid', gap: '4px' }}>
            <ShortcutItem keys={['Click']} action="Select" />
            <ShortcutItem keys={['Shift', 'Click']} action="Multi-select" />
            <ShortcutItem keys={['Ctrl/⌘', 'A']} action="Select all" />
            <ShortcutItem keys={['Ctrl/⌘', 'D']} action="Duplicate" />
            <ShortcutItem keys={['Delete']} action="Delete selected" />
            <ShortcutItem keys={['Escape']} action="Cancel/Deselect" />
            <ShortcutItem keys={['Scroll']} action="Zoom" />
            <ShortcutItem keys={['RMB Drag']} action="Rotate view" />
            <ShortcutItem keys={['MMB Drag']} action="Pan view" />
          </div>
        </div>
      )}
    </>
  );
};

interface ShortcutItemProps {
  keys: string[];
  action: string;
}

const ShortcutItem: React.FC<ShortcutItemProps> = ({ keys, action }) => (
  <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px' }}>
    <span style={{ opacity: 0.7 }}>
      {keys.map((key, i) => (
        <React.Fragment key={key}>
          {i > 0 && ' + '}
          <kbd
            style={{
              padding: '2px 6px',
              background: 'rgba(255, 255, 255, 0.1)',
              borderRadius: '3px',
              fontSize: '11px',
            }}
          >
            {key}
          </kbd>
        </React.Fragment>
      ))}
    </span>
    <span>{action}</span>
  </div>
);

// ============================================================================
// Exports
// ============================================================================

export default SceneComposer;
