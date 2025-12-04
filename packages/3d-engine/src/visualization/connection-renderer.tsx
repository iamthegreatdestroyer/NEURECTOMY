/**
 * @file Connection Renderer for Agent Components
 * @description Renders connections between agent components with data flow visualization
 * @module @neurectomy/3d-engine/visualization
 * @agents @VERTEX @CANVAS
 */

import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Line, QuadraticBezierLine, CubicBezierLine, Text } from '@react-three/drei';
import * as THREE from 'three';
import {
  AgentConnection,
  ConnectionType,
  AgentComponent,
  Vector3D,
} from './types';

// ============================================================================
// Connection Styles
// ============================================================================

interface ConnectionStyle {
  color: string;
  lineWidth: number;
  dashSize?: number;
  gapSize?: number;
  animated?: boolean;
  animationSpeed?: number;
  flowDirection?: 'forward' | 'backward' | 'bidirectional';
}

const CONNECTION_STYLES: Record<ConnectionType, ConnectionStyle> = {
  data: {
    color: '#3b82f6',
    lineWidth: 2,
    animated: true,
    animationSpeed: 1,
    flowDirection: 'forward',
  },
  control: {
    color: '#8b5cf6',
    lineWidth: 2,
    dashSize: 0.1,
    gapSize: 0.05,
    animated: false,
  },
  feedback: {
    color: '#22c55e',
    lineWidth: 1.5,
    animated: true,
    animationSpeed: 0.5,
    flowDirection: 'backward',
  },
  bidirectional: {
    color: '#f59e0b',
    lineWidth: 2,
    animated: true,
    animationSpeed: 1,
    flowDirection: 'bidirectional',
  },
  dependency: {
    color: '#64748b',
    lineWidth: 1,
    dashSize: 0.05,
    gapSize: 0.03,
    animated: false,
  },
};

// ============================================================================
// Data Flow Particle
// ============================================================================

interface DataFlowParticleProps {
  startPoint: Vector3D;
  endPoint: Vector3D;
  controlPoint?: Vector3D;
  color: string;
  speed: number;
  direction: 'forward' | 'backward' | 'bidirectional';
  particleCount?: number;
}

/**
 * Animated particles showing data flow along connection
 */
const DataFlowParticles: React.FC<DataFlowParticleProps> = ({
  startPoint,
  endPoint,
  controlPoint,
  color,
  speed,
  direction,
  particleCount = 3,
}) => {
  const particlesRef = useRef<THREE.Points>(null);
  
  // Create particle geometry
  const particles = useMemo(() => {
    const positions = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);
    const offsets = new Float32Array(particleCount);
    
    for (let i = 0; i < particleCount; i++) {
      offsets[i] = i / particleCount;
      sizes[i] = 0.08 - i * 0.01;
    }
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    geometry.setAttribute('offset', new THREE.BufferAttribute(offsets, 1));
    
    return { geometry, offsets };
  }, [particleCount]);

  // Animate particles along path
  useFrame((state) => {
    if (!particlesRef.current) return;
    
    const positions = particlesRef.current.geometry.attributes.position.array as Float32Array;
    const time = state.clock.elapsedTime * speed;
    
    for (let i = 0; i < particleCount; i++) {
      let t = (time + particles.offsets[i]) % 1;
      
      // Reverse for backward direction
      if (direction === 'backward') {
        t = 1 - t;
      }
      
      // Calculate position along bezier curve or line
      let x, y, z;
      if (controlPoint) {
        // Quadratic bezier
        const t1 = 1 - t;
        x = t1 * t1 * startPoint.x + 2 * t1 * t * controlPoint.x + t * t * endPoint.x;
        y = t1 * t1 * startPoint.y + 2 * t1 * t * controlPoint.y + t * t * endPoint.y;
        z = t1 * t1 * startPoint.z + 2 * t1 * t * controlPoint.z + t * t * endPoint.z;
      } else {
        // Linear interpolation
        x = startPoint.x + (endPoint.x - startPoint.x) * t;
        y = startPoint.y + (endPoint.y - startPoint.y) * t;
        z = startPoint.z + (endPoint.z - startPoint.z) * t;
      }
      
      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;
    }
    
    particlesRef.current.geometry.attributes.position.needsUpdate = true;
  });

  return (
    <points ref={particlesRef} geometry={particles.geometry}>
      <pointsMaterial
        color={color}
        size={0.1}
        transparent
        opacity={0.8}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
};

// ============================================================================
// Connection Arrow
// ============================================================================

interface ConnectionArrowProps {
  position: Vector3D;
  direction: THREE.Vector3;
  color: string;
  size?: number;
}

/**
 * Arrow indicator showing direction
 */
const ConnectionArrow: React.FC<ConnectionArrowProps> = ({
  position,
  direction,
  color,
  size = 0.15,
}) => {
  const arrowRef = useRef<THREE.Mesh>(null);

  // Rotate to face direction
  useMemo(() => {
    if (arrowRef.current) {
      const up = new THREE.Vector3(0, 1, 0);
      const quaternion = new THREE.Quaternion().setFromUnitVectors(up, direction.clone().normalize());
      arrowRef.current.setRotationFromQuaternion(quaternion);
    }
  }, [direction]);

  return (
    <mesh
      ref={arrowRef}
      position={[position.x, position.y, position.z]}
    >
      <coneGeometry args={[size * 0.5, size, 8]} />
      <meshBasicMaterial color={color} transparent opacity={0.8} />
    </mesh>
  );
};

// ============================================================================
// Single Connection
// ============================================================================

export interface ConnectionRendererProps {
  connection: AgentConnection;
  sourceComponent: AgentComponent;
  targetComponent: AgentComponent;
  selected?: boolean;
  hovered?: boolean;
  showLabel?: boolean;
  showDataFlow?: boolean;
}

/**
 * Render a single connection between components
 */
export const ConnectionRenderer: React.FC<ConnectionRendererProps> = ({
  connection,
  sourceComponent,
  targetComponent,
  selected = false,
  hovered = false,
  showLabel = false,
  showDataFlow = true,
}) => {
  const style = useMemo(() => ({
    ...CONNECTION_STYLES[connection.type],
    ...connection.style,
  }), [connection.type, connection.style]);

  // Calculate connection points
  const points = useMemo(() => {
    const start: Vector3D = sourceComponent.position;
    const end: Vector3D = targetComponent.position;
    
    // Calculate control point for curved connections
    const midX = (start.x + end.x) / 2;
    const midY = (start.y + end.y) / 2;
    const midZ = (start.z + end.z) / 2;
    
    // Offset control point perpendicular to line
    const dx = end.x - start.x;
    const dz = end.z - start.z;
    const len = Math.sqrt(dx * dx + dz * dz);
    const offset = Math.min(len * 0.3, 1);
    
    const control: Vector3D = {
      x: midX - (dz / len) * offset,
      y: midY + 0.5,
      z: midZ + (dx / len) * offset,
    };
    
    return { start, end, control };
  }, [sourceComponent.position, targetComponent.position]);

  // Calculate arrow direction at endpoint
  const arrowDirection = useMemo(() => {
    const dir = new THREE.Vector3(
      points.end.x - points.control.x,
      points.end.y - points.control.y,
      points.end.z - points.control.z
    ).normalize();
    return dir;
  }, [points]);

  // Line color based on state
  const lineColor = useMemo(() => {
    if (selected) return '#ffffff';
    if (hovered) return '#60a5fa';
    return style.color;
  }, [selected, hovered, style.color]);

  // Line width based on state
  const lineWidth = useMemo(() => {
    if (selected) return style.lineWidth + 1;
    if (hovered) return style.lineWidth + 0.5;
    return style.lineWidth;
  }, [selected, hovered, style.lineWidth]);

  return (
    <group name={`connection-${connection.id}`}>
      {/* Main connection line */}
      <QuadraticBezierLine
        start={[points.start.x, points.start.y, points.start.z]}
        end={[points.end.x, points.end.y, points.end.z]}
        mid={[points.control.x, points.control.y, points.control.z]}
        color={lineColor}
        lineWidth={lineWidth}
        transparent
        opacity={selected || hovered ? 1 : 0.7}
        dashed={!!style.dashSize}
        dashSize={style.dashSize}
        gapSize={style.gapSize}
      />

      {/* Direction arrow */}
      <ConnectionArrow
        position={{
          x: points.end.x - arrowDirection.x * 0.6,
          y: points.end.y - arrowDirection.y * 0.6,
          z: points.end.z - arrowDirection.z * 0.6,
        }}
        direction={arrowDirection}
        color={lineColor}
      />

      {/* Data flow particles */}
      {showDataFlow && style.animated && (
        <DataFlowParticles
          startPoint={points.start}
          endPoint={points.end}
          controlPoint={points.control}
          color={style.color}
          speed={style.animationSpeed || 1}
          direction={style.flowDirection || 'forward'}
          particleCount={4}
        />
      )}

      {/* Bidirectional reverse flow */}
      {showDataFlow && style.flowDirection === 'bidirectional' && (
        <DataFlowParticles
          startPoint={points.end}
          endPoint={points.start}
          controlPoint={points.control}
          color={style.color}
          speed={(style.animationSpeed || 1) * 0.7}
          direction="forward"
          particleCount={3}
        />
      )}

      {/* Connection label */}
      {showLabel && connection.metadata?.label && (
        <Text
          position={[points.control.x, points.control.y + 0.2, points.control.z]}
          fontSize={0.12}
          color="#94a3b8"
          anchorX="center"
          anchorY="bottom"
        >
          {connection.metadata.label}
        </Text>
      )}

      {/* Bandwidth indicator */}
      {hovered && connection.metadata?.bandwidth && (
        <Text
          position={[points.control.x, points.control.y - 0.15, points.control.z]}
          fontSize={0.1}
          color="#64748b"
          anchorX="center"
          anchorY="top"
        >
          {`${connection.metadata.bandwidth} msg/s`}
        </Text>
      )}
    </group>
  );
};

// ============================================================================
// Connection Graph Renderer
// ============================================================================

export interface ConnectionGraphProps {
  connections: AgentConnection[];
  components: Map<string, AgentComponent>;
  selectedConnectionIds?: Set<string>;
  hoveredConnectionId?: string | null;
  showLabels?: boolean;
  showDataFlow?: boolean;
}

/**
 * Render all connections in an agent architecture
 */
export const ConnectionGraph: React.FC<ConnectionGraphProps> = ({
  connections,
  components,
  selectedConnectionIds = new Set(),
  hoveredConnectionId = null,
  showLabels = false,
  showDataFlow = true,
}) => {
  return (
    <group name="connection-graph">
      {connections.map((connection) => {
        const sourceComponent = components.get(connection.sourceId);
        const targetComponent = components.get(connection.targetId);
        
        if (!sourceComponent || !targetComponent) {
          console.warn(`Missing component for connection ${connection.id}`);
          return null;
        }
        
        return (
          <ConnectionRenderer
            key={connection.id}
            connection={connection}
            sourceComponent={sourceComponent}
            targetComponent={targetComponent}
            selected={selectedConnectionIds.has(connection.id)}
            hovered={hoveredConnectionId === connection.id}
            showLabel={showLabels}
            showDataFlow={showDataFlow}
          />
        );
      })}
    </group>
  );
};

// ============================================================================
// Connection Highlight Effect
// ============================================================================

interface ConnectionHighlightProps {
  connections: AgentConnection[];
  components: Map<string, AgentComponent>;
  highlightedComponentId: string;
}

/**
 * Highlight all connections to/from a component
 */
export const ConnectionHighlight: React.FC<ConnectionHighlightProps> = ({
  connections,
  components,
  highlightedComponentId,
}) => {
  // Find all connections involving the highlighted component
  const relevantConnections = useMemo(() => {
    return connections.filter(
      (conn) =>
        conn.sourceId === highlightedComponentId ||
        conn.targetId === highlightedComponentId
    );
  }, [connections, highlightedComponentId]);

  return (
    <group name="connection-highlight">
      {relevantConnections.map((connection) => {
        const sourceComponent = components.get(connection.sourceId);
        const targetComponent = components.get(connection.targetId);
        
        if (!sourceComponent || !targetComponent) return null;
        
        const isSource = connection.sourceId === highlightedComponentId;
        
        return (
          <ConnectionRenderer
            key={`highlight-${connection.id}`}
            connection={{
              ...connection,
              style: {
                ...connection.style,
                color: isSource ? '#22d3ee' : '#34d399',
              },
            }}
            sourceComponent={sourceComponent}
            targetComponent={targetComponent}
            selected
            showLabel
            showDataFlow
          />
        );
      })}
    </group>
  );
};

export default ConnectionGraph;
