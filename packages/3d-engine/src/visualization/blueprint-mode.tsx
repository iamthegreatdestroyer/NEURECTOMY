/**
 * @file Blueprint Mode Visualization
 * @description Technical schematic view with grid, measurements, and CAD-style rendering
 * @module @neurectomy/3d-engine/visualization
 * @agents @CANVAS @SCRIBE
 */

import React, { useMemo, useRef, useEffect, useState } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { Line, Text, Grid, Html } from '@react-three/drei';
import {
  Vector3,
  Color,
  Group,
  BufferGeometry,
  Float32BufferAttribute,
  LineBasicMaterial,
  LineDashedMaterial,
  LineSegments,
  ShaderMaterial,
  DoubleSide,
} from 'three';
import type { AgentComponent, AgentConnection, Vector3D, BlueprintConfig } from './types';

// ============================================================================
// Blueprint Configuration
// ============================================================================

const DEFAULT_BLUEPRINT_CONFIG: BlueprintConfig = {
  enabled: true,
  gridEnabled: true,
  gridSize: 20,
  gridDivisions: 40,
  gridColor: '#1e3a5f',
  gridCenterLineColor: '#3b82f6',
  showDimensions: true,
  showLabels: true,
  showConnectorLabels: true,
  lineColor: '#60a5fa',
  textColor: '#93c5fd',
  backgroundColor: '#0a1929',
  scale: 1.0,
};

export interface BlueprintModeProps {
  config?: Partial<BlueprintConfig>;
  components: AgentComponent[];
  connections: AgentConnection[];
  selectedIds?: Set<string>;
  showMeasurements?: boolean;
  onMeasure?: (distance: number, from: Vector3D, to: Vector3D) => void;
}

// ============================================================================
// Blueprint Grid Component
// ============================================================================

interface BlueprintGridProps {
  size: number;
  divisions: number;
  color: string;
  centerLineColor: string;
}

export const BlueprintGrid: React.FC<BlueprintGridProps> = ({
  size,
  divisions,
  color,
  centerLineColor,
}) => {
  const gridRef = useRef<Group>(null);

  const { majorLines, minorLines } = useMemo(() => {
    const majorPositions: number[] = [];
    const minorPositions: number[] = [];
    const halfSize = size / 2;
    const step = size / divisions;
    const majorStep = step * 5; // Major line every 5 minor lines

    for (let i = -halfSize; i <= halfSize; i += step) {
      const isMajor = Math.abs(i % majorStep) < 0.001 || i === 0;
      const positions = isMajor ? majorPositions : minorPositions;
      
      // Lines parallel to Z
      positions.push(-halfSize, 0, i, halfSize, 0, i);
      // Lines parallel to X
      positions.push(i, 0, -halfSize, i, 0, halfSize);
    }

    return {
      majorLines: new Float32Array(majorPositions),
      minorLines: new Float32Array(minorPositions),
    };
  }, [size, divisions]);

  return (
    <group ref={gridRef}>
      {/* Minor grid lines */}
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={minorLines.length / 3}
            array={minorLines}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color={color} transparent opacity={0.3} />
      </lineSegments>

      {/* Major grid lines */}
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={majorLines.length / 3}
            array={majorLines}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color={centerLineColor} transparent opacity={0.5} />
      </lineSegments>

      {/* Center cross (axes) */}
      <Line
        points={[[-size / 2, 0.01, 0], [size / 2, 0.01, 0]]}
        color="#ef4444"
        lineWidth={2}
      />
      <Line
        points={[[0, 0.01, -size / 2], [0, 0.01, size / 2]]}
        color="#22c55e"
        lineWidth={2}
      />
    </group>
  );
};

// ============================================================================
// Blueprint Component Renderer
// ============================================================================

interface BlueprintComponentProps {
  component: AgentComponent;
  config: BlueprintConfig;
  isSelected: boolean;
  showDimensions: boolean;
}

export const BlueprintComponent: React.FC<BlueprintComponentProps> = ({
  component,
  config,
  isSelected,
  showDimensions,
}) => {
  const groupRef = useRef<Group>(null);
  const { position, size = { x: 1, y: 0.5, z: 1 } } = component;

  // Calculate bounding box corners
  const corners = useMemo(() => {
    const halfX = size.x / 2;
    const halfY = size.y / 2;
    const halfZ = size.z / 2;

    return {
      topFace: [
        [-halfX, halfY, -halfZ],
        [halfX, halfY, -halfZ],
        [halfX, halfY, halfZ],
        [-halfX, halfY, halfZ],
        [-halfX, halfY, -halfZ],
      ] as [number, number, number][],
      bottomFace: [
        [-halfX, -halfY, -halfZ],
        [halfX, -halfY, -halfZ],
        [halfX, -halfY, halfZ],
        [-halfX, -halfY, halfZ],
        [-halfX, -halfY, -halfZ],
      ] as [number, number, number][],
      verticals: [
        [[-halfX, -halfY, -halfZ], [-halfX, halfY, -halfZ]],
        [[halfX, -halfY, -halfZ], [halfX, halfY, -halfZ]],
        [[halfX, -halfY, halfZ], [halfX, halfY, halfZ]],
        [[-halfX, -halfY, halfZ], [-halfX, halfY, halfZ]],
      ] as [[number, number, number], [number, number, number]][],
    };
  }, [size]);

  const lineColor = isSelected ? '#fbbf24' : config.lineColor;
  const lineWidth = isSelected ? 2 : 1;

  return (
    <group ref={groupRef} position={[position.x, position.y, position.z]}>
      {/* Top face outline */}
      <Line
        points={corners.topFace}
        color={lineColor}
        lineWidth={lineWidth}
      />

      {/* Bottom face outline */}
      <Line
        points={corners.bottomFace}
        color={lineColor}
        lineWidth={lineWidth}
        opacity={0.5}
        transparent
      />

      {/* Vertical edges */}
      {corners.verticals.map((edge, idx) => (
        <Line
          key={`vertical-${idx}`}
          points={edge}
          color={lineColor}
          lineWidth={lineWidth}
        />
      ))}

      {/* Center point marker */}
      <group position={[0, size.y / 2 + 0.05, 0]}>
        <Line
          points={[[-0.1, 0, 0], [0.1, 0, 0]]}
          color={lineColor}
          lineWidth={1}
        />
        <Line
          points={[[0, 0, -0.1], [0, 0, 0.1]]}
          color={lineColor}
          lineWidth={1}
        />
      </group>

      {/* Component label */}
      {config.showLabels && (
        <Text
          position={[0, size.y / 2 + 0.3, 0]}
          fontSize={0.15}
          color={config.textColor}
          anchorX="center"
          anchorY="bottom"
          font="/fonts/roboto-mono.woff"
        >
          {component.metadata?.name || component.id.slice(0, 8)}
        </Text>
      )}

      {/* Type badge */}
      {config.showLabels && (
        <Html
          position={[0, size.y / 2 + 0.5, 0]}
          center
          distanceFactor={10}
        >
          <div
            style={{
              padding: '2px 6px',
              background: getTypeColor(component.type),
              borderRadius: '2px',
              fontSize: '10px',
              fontFamily: 'monospace',
              color: '#fff',
              whiteSpace: 'nowrap',
              textTransform: 'uppercase',
            }}
          >
            {component.type}
          </div>
        </Html>
      )}

      {/* Dimension annotations */}
      {showDimensions && (
        <DimensionAnnotations
          size={size}
          color={config.textColor}
        />
      )}

      {/* Input/Output connectors */}
      <ConnectorPorts
        component={component}
        config={config}
      />
    </group>
  );
};

// ============================================================================
// Dimension Annotations
// ============================================================================

interface DimensionAnnotationsProps {
  size: Vector3D;
  color: string;
}

const DimensionAnnotations: React.FC<DimensionAnnotationsProps> = ({
  size,
  color,
}) => {
  const offset = 0.2;
  const fontSize = 0.08;

  return (
    <>
      {/* Width dimension (X axis) */}
      <group position={[0, -size.y / 2 - offset, size.z / 2 + offset]}>
        <Line
          points={[[-size.x / 2, 0, 0], [size.x / 2, 0, 0]]}
          color={color}
          lineWidth={1}
        />
        <Line
          points={[[-size.x / 2, -0.05, 0], [-size.x / 2, 0.05, 0]]}
          color={color}
        />
        <Line
          points={[[size.x / 2, -0.05, 0], [size.x / 2, 0.05, 0]]}
          color={color}
        />
        <Text
          position={[0, 0.1, 0]}
          fontSize={fontSize}
          color={color}
          anchorX="center"
        >
          {size.x.toFixed(2)}
        </Text>
      </group>

      {/* Depth dimension (Z axis) */}
      <group position={[size.x / 2 + offset, -size.y / 2 - offset, 0]}>
        <Line
          points={[[0, 0, -size.z / 2], [0, 0, size.z / 2]]}
          color={color}
          lineWidth={1}
        />
        <Text
          position={[0.1, 0, 0]}
          fontSize={fontSize}
          color={color}
          anchorX="left"
          rotation={[0, Math.PI / 2, 0]}
        >
          {size.z.toFixed(2)}
        </Text>
      </group>

      {/* Height dimension (Y axis) */}
      <group position={[size.x / 2 + offset, 0, size.z / 2 + offset]}>
        <Line
          points={[[0, -size.y / 2, 0], [0, size.y / 2, 0]]}
          color={color}
          lineWidth={1}
        />
        <Text
          position={[0.1, 0, 0]}
          fontSize={fontSize}
          color={color}
          anchorX="left"
          rotation={[0, 0, Math.PI / 2]}
        >
          {size.y.toFixed(2)}
        </Text>
      </group>
    </>
  );
};

// ============================================================================
// Connector Ports
// ============================================================================

interface ConnectorPortsProps {
  component: AgentComponent;
  config: BlueprintConfig;
}

const ConnectorPorts: React.FC<ConnectorPortsProps> = ({
  component,
  config,
}) => {
  const { size = { x: 1, y: 0.5, z: 1 }, inputs = [], outputs = [] } = component;

  // Calculate port positions
  const inputPorts = useMemo(() => {
    return inputs.map((input, idx) => {
      const spacing = size.z / (inputs.length + 1);
      return {
        ...input,
        position: [-size.x / 2 - 0.05, 0, -size.z / 2 + spacing * (idx + 1)] as [number, number, number],
      };
    });
  }, [inputs, size]);

  const outputPorts = useMemo(() => {
    return outputs.map((output, idx) => {
      const spacing = size.z / (outputs.length + 1);
      return {
        ...output,
        position: [size.x / 2 + 0.05, 0, -size.z / 2 + spacing * (idx + 1)] as [number, number, number],
      };
    });
  }, [outputs, size]);

  const portSize = 0.08;

  return (
    <>
      {/* Input ports (left side, circles) */}
      {inputPorts.map((port, idx) => (
        <group key={`input-${idx}`} position={port.position}>
          {/* Port circle */}
          <mesh rotation={[0, 0, Math.PI / 2]}>
            <ringGeometry args={[portSize * 0.5, portSize, 16]} />
            <meshBasicMaterial color="#22c55e" side={DoubleSide} />
          </mesh>
          
          {/* Port label */}
          {config.showConnectorLabels && (
            <Text
              position={[-0.15, 0, 0]}
              fontSize={0.06}
              color={config.textColor}
              anchorX="right"
              anchorY="middle"
            >
              {port.name || `IN${idx}`}
            </Text>
          )}
        </group>
      ))}

      {/* Output ports (right side, triangles) */}
      {outputPorts.map((port, idx) => (
        <group key={`output-${idx}`} position={port.position}>
          {/* Port triangle pointing right */}
          <mesh rotation={[0, 0, -Math.PI / 2]}>
            <coneGeometry args={[portSize, portSize * 1.5, 3]} />
            <meshBasicMaterial color="#ef4444" />
          </mesh>
          
          {/* Port label */}
          {config.showConnectorLabels && (
            <Text
              position={[0.15, 0, 0]}
              fontSize={0.06}
              color={config.textColor}
              anchorX="left"
              anchorY="middle"
            >
              {port.name || `OUT${idx}`}
            </Text>
          )}
        </group>
      ))}
    </>
  );
};

// ============================================================================
// Blueprint Connection Renderer
// ============================================================================

interface BlueprintConnectionProps {
  connection: AgentConnection;
  sourcePos: Vector3D;
  targetPos: Vector3D;
  config: BlueprintConfig;
  isSelected: boolean;
}

export const BlueprintConnection: React.FC<BlueprintConnectionProps> = ({
  connection,
  sourcePos,
  targetPos,
  config,
  isSelected,
}) => {
  // Create orthogonal routing (L-shaped or S-shaped)
  const routePoints = useMemo(() => {
    const midX = (sourcePos.x + targetPos.x) / 2;
    
    // Simple L-routing
    return [
      [sourcePos.x, sourcePos.y, sourcePos.z],
      [midX, sourcePos.y, sourcePos.z],
      [midX, targetPos.y, targetPos.z],
      [targetPos.x, targetPos.y, targetPos.z],
    ] as [number, number, number][];
  }, [sourcePos, targetPos]);

  const lineColor = isSelected ? '#fbbf24' : config.lineColor;

  return (
    <group>
      {/* Main connection line */}
      <Line
        points={routePoints}
        color={lineColor}
        lineWidth={isSelected ? 2 : 1}
        dashed={connection.type === 'event'}
        dashSize={0.1}
        gapSize={0.05}
      />

      {/* Direction arrow at midpoint */}
      <group
        position={[
          (sourcePos.x + targetPos.x) / 2,
          (sourcePos.y + targetPos.y) / 2,
          (sourcePos.z + targetPos.z) / 2,
        ]}
      >
        <mesh
          rotation={[
            0,
            Math.atan2(
              targetPos.x - sourcePos.x,
              targetPos.z - sourcePos.z
            ),
            0,
          ]}
        >
          <coneGeometry args={[0.05, 0.1, 3]} />
          <meshBasicMaterial color={lineColor} />
        </mesh>
      </group>

      {/* Connection label */}
      {config.showConnectorLabels && connection.metadata?.label && (
        <Text
          position={[
            (sourcePos.x + targetPos.x) / 2,
            (sourcePos.y + targetPos.y) / 2 + 0.15,
            (sourcePos.z + targetPos.z) / 2,
          ]}
          fontSize={0.08}
          color={config.textColor}
          anchorX="center"
        >
          {connection.metadata.label}
        </Text>
      )}
    </group>
  );
};

// ============================================================================
// Measurement Tool
// ============================================================================

interface MeasurementToolProps {
  active: boolean;
  startPoint?: Vector3D;
  endPoint?: Vector3D;
  color?: string;
}

export const MeasurementTool: React.FC<MeasurementToolProps> = ({
  active,
  startPoint,
  endPoint,
  color = '#fbbf24',
}) => {
  if (!active || !startPoint) return null;

  const end = endPoint || startPoint;
  const distance = Math.sqrt(
    (end.x - startPoint.x) ** 2 +
    (end.y - startPoint.y) ** 2 +
    (end.z - startPoint.z) ** 2
  );

  const midPoint: [number, number, number] = [
    (startPoint.x + end.x) / 2,
    (startPoint.y + end.y) / 2 + 0.2,
    (startPoint.z + end.z) / 2,
  ];

  return (
    <group>
      {/* Measurement line */}
      <Line
        points={[
          [startPoint.x, startPoint.y, startPoint.z],
          [end.x, end.y, end.z],
        ]}
        color={color}
        lineWidth={2}
        dashed
        dashSize={0.1}
        gapSize={0.05}
      />

      {/* Start marker */}
      <mesh position={[startPoint.x, startPoint.y, startPoint.z]}>
        <sphereGeometry args={[0.05, 16, 16]} />
        <meshBasicMaterial color={color} />
      </mesh>

      {/* End marker */}
      {endPoint && (
        <mesh position={[end.x, end.y, end.z]}>
          <sphereGeometry args={[0.05, 16, 16]} />
          <meshBasicMaterial color={color} />
        </mesh>
      )}

      {/* Distance label */}
      {endPoint && distance > 0.01 && (
        <Html position={midPoint} center>
          <div
            style={{
              padding: '4px 8px',
              background: 'rgba(0, 0, 0, 0.8)',
              border: `1px solid ${color}`,
              borderRadius: '4px',
              color,
              fontSize: '12px',
              fontFamily: 'monospace',
              whiteSpace: 'nowrap',
            }}
          >
            {distance.toFixed(3)} units
          </div>
        </Html>
      )}
    </group>
  );
};

// ============================================================================
// Blueprint Mode Container
// ============================================================================

export const BlueprintMode: React.FC<BlueprintModeProps> = ({
  config: userConfig,
  components,
  connections,
  selectedIds = new Set(),
  showMeasurements = false,
  onMeasure,
}) => {
  const config = { ...DEFAULT_BLUEPRINT_CONFIG, ...userConfig };
  const [measureStart, setMeasureStart] = useState<Vector3D | null>(null);
  const [measureEnd, setMeasureEnd] = useState<Vector3D | null>(null);

  // Build position map for connections
  const positionMap = useMemo(() => {
    const map = new Map<string, Vector3D>();
    components.forEach(comp => map.set(comp.id, comp.position));
    return map;
  }, [components]);

  if (!config.enabled) return null;

  return (
    <group name="blueprint-mode">
      {/* Blueprint grid */}
      {config.gridEnabled && (
        <BlueprintGrid
          size={config.gridSize}
          divisions={config.gridDivisions}
          color={config.gridColor}
          centerLineColor={config.gridCenterLineColor}
        />
      )}

      {/* Components */}
      {components.map(component => (
        <BlueprintComponent
          key={component.id}
          component={component}
          config={config}
          isSelected={selectedIds.has(component.id)}
          showDimensions={config.showDimensions && selectedIds.has(component.id)}
        />
      ))}

      {/* Connections */}
      {connections.map(connection => {
        const sourcePos = positionMap.get(connection.sourceId);
        const targetPos = positionMap.get(connection.targetId);
        
        if (!sourcePos || !targetPos) return null;

        return (
          <BlueprintConnection
            key={connection.id}
            connection={connection}
            sourcePos={sourcePos}
            targetPos={targetPos}
            config={config}
            isSelected={selectedIds.has(connection.id)}
          />
        );
      })}

      {/* Measurement tool */}
      <MeasurementTool
        active={showMeasurements}
        startPoint={measureStart || undefined}
        endPoint={measureEnd || undefined}
      />

      {/* Coordinate display */}
      <CoordinateDisplay config={config} />
    </group>
  );
};

// ============================================================================
// Coordinate Display (bottom-left corner of view)
// ============================================================================

interface CoordinateDisplayProps {
  config: BlueprintConfig;
}

const CoordinateDisplay: React.FC<CoordinateDisplayProps> = ({ config }) => {
  const { camera } = useThree();
  const [coords, setCoords] = useState({ x: 0, y: 0, z: 0 });

  useFrame(() => {
    setCoords({
      x: camera.position.x,
      y: camera.position.y,
      z: camera.position.z,
    });
  });

  return (
    <Html
      position={[0, 0, 0]}
      style={{
        position: 'fixed',
        bottom: '10px',
        left: '10px',
        zIndex: 100,
      }}
      calculatePosition={() => [10, window.innerHeight - 60, 0]}
    >
      <div
        style={{
          padding: '8px 12px',
          background: 'rgba(10, 25, 41, 0.9)',
          border: `1px solid ${config.lineColor}`,
          borderRadius: '4px',
          fontFamily: 'monospace',
          fontSize: '11px',
          color: config.textColor,
        }}
      >
        <div>Camera: [{coords.x.toFixed(2)}, {coords.y.toFixed(2)}, {coords.z.toFixed(2)}]</div>
        <div style={{ marginTop: '4px', color: config.lineColor }}>
          Scale: {config.scale}x
        </div>
      </div>
    </Html>
  );
};

// ============================================================================
// Helper Functions
// ============================================================================

function getTypeColor(type: string): string {
  const colors: Record<string, string> = {
    orchestrator: '#8b5cf6',
    worker: '#3b82f6',
    tool: '#22c55e',
    memory: '#f59e0b',
    router: '#ec4899',
    transformer: '#06b6d4',
    validator: '#84cc16',
    default: '#6b7280',
  };
  return colors[type] || colors.default;
}

export default BlueprintMode;
