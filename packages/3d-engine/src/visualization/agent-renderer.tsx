/**
 * @file Agent Component 3D Renderer
 * @description React Three Fiber components for rendering agent architecture
 * @module @neurectomy/3d-engine/visualization
 * @agents @ARCHITECT @CANVAS
 */

import React, { useRef, useMemo, useState, useCallback } from 'react';
import { useFrame, ThreeEvent } from '@react-three/fiber';
import { Text, Billboard, Html } from '@react-three/drei';
import * as THREE from 'three';
import {
  AgentComponent,
  AgentComponentType,
  ComponentStatus,
  ComponentStyle,
  DEFAULT_COMPONENT_STYLES,
  ComponentMetrics,
} from './types';

// ============================================================================
// Agent Component Geometry
// ============================================================================

interface ComponentGeometryProps {
  type: AgentComponentType;
  style: ComponentStyle;
}

/**
 * Get geometry based on component type and style
 */
function useComponentGeometry(style: ComponentStyle): THREE.BufferGeometry {
  return useMemo(() => {
    switch (style.geometry) {
      case 'sphere':
        return new THREE.SphereGeometry(0.5, 32, 32);
      case 'box':
        return new THREE.BoxGeometry(0.8, 0.8, 0.8);
      case 'cylinder':
        return new THREE.CylinderGeometry(0.4, 0.4, 0.8, 32);
      case 'cone':
        return new THREE.ConeGeometry(0.4, 0.8, 32);
      case 'torus':
        return new THREE.TorusGeometry(0.35, 0.15, 16, 32);
      case 'octahedron':
        return new THREE.OctahedronGeometry(0.5);
      case 'icosahedron':
        return new THREE.IcosahedronGeometry(0.5);
      default:
        return new THREE.BoxGeometry(0.8, 0.8, 0.8);
    }
  }, [style.geometry]);
}

// ============================================================================
// Status Indicator
// ============================================================================

interface StatusIndicatorProps {
  status: ComponentStatus;
  position: [number, number, number];
}

/**
 * Visual indicator for component status
 */
const StatusIndicator: React.FC<StatusIndicatorProps> = ({ status, position }) => {
  const color = useMemo(() => {
    switch (status) {
      case 'active':
        return '#22c55e';
      case 'idle':
        return '#64748b';
      case 'processing':
        return '#3b82f6';
      case 'error':
        return '#ef4444';
      case 'disabled':
        return '#6b7280';
      case 'warning':
        return '#f59e0b';
      default:
        return '#64748b';
    }
  }, [status]);

  const ref = useRef<THREE.Mesh>(null);
  
  // Pulse animation for processing status
  useFrame((state) => {
    if (ref.current && status === 'processing') {
      const scale = 1 + Math.sin(state.clock.elapsedTime * 4) * 0.2;
      ref.current.scale.setScalar(scale);
    }
  });

  return (
    <mesh ref={ref} position={position}>
      <sphereGeometry args={[0.08, 16, 16]} />
      <meshBasicMaterial color={color} />
    </mesh>
  );
};

// ============================================================================
// Metrics Display
// ============================================================================

interface MetricsDisplayProps {
  metrics?: ComponentMetrics;
  visible: boolean;
}

/**
 * HTML overlay showing component metrics
 */
const MetricsDisplay: React.FC<MetricsDisplayProps> = ({ metrics, visible }) => {
  if (!visible || !metrics) return null;

  return (
    <Html
      position={[0, 1.2, 0]}
      center
      style={{
        background: 'rgba(15, 23, 42, 0.9)',
        padding: '8px 12px',
        borderRadius: '6px',
        fontSize: '11px',
        color: '#e2e8f0',
        border: '1px solid rgba(100, 116, 139, 0.3)',
        pointerEvents: 'none',
        whiteSpace: 'nowrap',
      }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        {metrics.avgLatencyMs !== undefined && (
          <div>Latency: <span style={{ color: '#22d3ee' }}>{metrics.avgLatencyMs.toFixed(1)}ms</span></div>
        )}
        {metrics.throughput !== undefined && (
          <div>Throughput: <span style={{ color: '#34d399' }}>{metrics.throughput.toFixed(1)}/s</span></div>
        )}
        {metrics.errorRate !== undefined && (
          <div>Error Rate: <span style={{ color: metrics.errorRate > 0.05 ? '#f87171' : '#34d399' }}>
            {(metrics.errorRate * 100).toFixed(1)}%
          </span></div>
        )}
        {metrics.tokenUsage !== undefined && (
          <div>Tokens: <span style={{ color: '#a78bfa' }}>{metrics.tokenUsage.toLocaleString()}</span></div>
        )}
      </div>
    </Html>
  );
};

// ============================================================================
// Agent Component Node
// ============================================================================

export interface AgentNodeProps {
  component: AgentComponent;
  selected: boolean;
  hovered: boolean;
  onSelect: (id: string, multi: boolean) => void;
  onHover: (id: string | null) => void;
  showMetrics: boolean;
  showLabels: boolean;
}

/**
 * 3D representation of an agent component
 */
export const AgentNode: React.FC<AgentNodeProps> = ({
  component,
  selected,
  hovered,
  onSelect,
  onHover,
  showMetrics,
  showLabels,
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [localHover, setLocalHover] = useState(false);
  
  // Merge default style with component style
  const style = useMemo(() => ({
    ...DEFAULT_COMPONENT_STYLES[component.type],
    ...component.style,
    opacity: component.style.opacity ?? 1,
    wireframe: component.style.wireframe ?? false,
    emissiveIntensity: component.style.emissiveIntensity ?? 0.2,
  }), [component.type, component.style]);

  const geometry = useComponentGeometry(style as ComponentStyle);

  // Material with selection/hover effects
  const material = useMemo(() => {
    const baseEmissive = style.emissiveColor || style.primaryColor;
    let emissiveIntensity = style.emissiveIntensity || 0.2;
    
    if (selected) emissiveIntensity += 0.4;
    if (hovered || localHover) emissiveIntensity += 0.2;

    return new THREE.MeshStandardMaterial({
      color: style.primaryColor,
      emissive: baseEmissive,
      emissiveIntensity,
      metalness: 0.3,
      roughness: 0.4,
      transparent: style.opacity < 1,
      opacity: style.opacity,
      wireframe: style.wireframe,
    });
  }, [style, selected, hovered, localHover]);

  // Animation
  useFrame((state) => {
    if (!meshRef.current) return;
    
    // Subtle floating animation
    const yOffset = Math.sin(state.clock.elapsedTime * 0.5 + component.position.x) * 0.02;
    meshRef.current.position.y = component.position.y + yOffset;

    // Rotation for processing status
    if (component.metadata.status === 'processing') {
      meshRef.current.rotation.y += 0.02;
    }

    // Selection outline pulse
    if (selected) {
      const pulse = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.03;
      meshRef.current.scale.setScalar(pulse);
    } else {
      meshRef.current.scale.setScalar(1);
    }
  });

  // Event handlers
  const handlePointerOver = useCallback((e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    setLocalHover(true);
    onHover(component.id);
    document.body.style.cursor = 'pointer';
  }, [component.id, onHover]);

  const handlePointerOut = useCallback((e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    setLocalHover(false);
    onHover(null);
    document.body.style.cursor = 'auto';
  }, [onHover]);

  const handleClick = useCallback((e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    onSelect(component.id, e.shiftKey || e.ctrlKey || e.metaKey);
  }, [component.id, onSelect]);

  return (
    <group
      position={[component.position.x, component.position.y, component.position.z]}
      rotation={[component.rotation.x, component.rotation.y, component.rotation.z]}
    >
      {/* Main mesh */}
      <mesh
        ref={meshRef}
        geometry={geometry}
        material={material}
        scale={[component.scale.x, component.scale.y, component.scale.z]}
        onPointerOver={handlePointerOver}
        onPointerOut={handlePointerOut}
        onClick={handleClick}
        castShadow
        receiveShadow
      />

      {/* Selection outline */}
      {selected && (
        <mesh scale={[
          component.scale.x * 1.1,
          component.scale.y * 1.1,
          component.scale.z * 1.1
        ]}>
          <boxGeometry args={[1, 1, 1]} />
          <meshBasicMaterial
            color="#ffffff"
            wireframe
            transparent
            opacity={0.5}
          />
        </mesh>
      )}

      {/* Status indicator */}
      <StatusIndicator
        status={component.metadata.status}
        position={[0.5, 0.5, 0.5]}
      />

      {/* Label */}
      {showLabels && (
        <Billboard follow lockX={false} lockY={false} lockZ={false}>
          <Text
            position={[0, -0.8, 0]}
            fontSize={0.15}
            color="#e2e8f0"
            anchorX="center"
            anchorY="top"
            outlineWidth={0.02}
            outlineColor="#0f172a"
          >
            {component.name}
          </Text>
          <Text
            position={[0, -0.95, 0]}
            fontSize={0.1}
            color="#94a3b8"
            anchorX="center"
            anchorY="top"
          >
            {component.type}
          </Text>
        </Billboard>
      )}

      {/* Metrics display */}
      <MetricsDisplay
        metrics={component.metadata.metrics}
        visible={showMetrics && (hovered || localHover)}
      />
    </group>
  );
};

// ============================================================================
// Agent Architecture Renderer
// ============================================================================

export interface AgentArchitectureProps {
  components: AgentComponent[];
  selectedIds: Set<string>;
  hoveredId: string | null;
  onSelect: (id: string, multi: boolean) => void;
  onHover: (id: string | null) => void;
  showMetrics?: boolean;
  showLabels?: boolean;
}

/**
 * Render full agent architecture with all components
 */
export const AgentArchitecture: React.FC<AgentArchitectureProps> = ({
  components,
  selectedIds,
  hoveredId,
  onSelect,
  onHover,
  showMetrics = true,
  showLabels = true,
}) => {
  return (
    <group name="agent-architecture">
      {components.map((component) => (
        <AgentNode
          key={component.id}
          component={component}
          selected={selectedIds.has(component.id)}
          hovered={hoveredId === component.id}
          onSelect={onSelect}
          onHover={onHover}
          showMetrics={showMetrics}
          showLabels={showLabels}
        />
      ))}
    </group>
  );
};

// ============================================================================
// Component Type Icons (2D overlay)
// ============================================================================

interface ComponentIconProps {
  type: AgentComponentType;
  size?: number;
}

/**
 * Icon representation for component type
 */
export const ComponentIcon: React.FC<ComponentIconProps> = ({
  type,
  size = 24,
}) => {
  const icons: Record<AgentComponentType, string> = {
    agent: '🤖',
    llm: '🧠',
    tool: '🔧',
    memory: '💾',
    embedding: '📊',
    retriever: '🔍',
    router: '🔀',
    executor: '⚡',
    planner: '📋',
    evaluator: '✅',
    guardrail: '🛡️',
    connector: '🔌',
    custom: '⚙️',
  };

  return (
    <span style={{ fontSize: size }}>
      {icons[type] || '⚙️'}
    </span>
  );
};

export default AgentArchitecture;
