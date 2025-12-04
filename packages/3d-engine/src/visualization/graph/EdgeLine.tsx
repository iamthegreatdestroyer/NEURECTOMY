/**
 * @file EdgeLine Component
 * @description 3D line representation for graph edges
 * @module @neurectomy/3d-engine/visualization/graph
 * @agents @CANVAS @APEX
 */

/// <reference types="@react-three/fiber" />

import React, { useRef, useMemo, useCallback } from "react";
import { useFrame, ThreeEvent, extend } from "@react-three/fiber";
import { Line, QuadraticBezierLine, CubicBezierLine } from "@react-three/drei";
import * as THREE from "three";
import type {
  GraphEdge,
  GraphNode,
  EdgeVisualConfig,
  EdgeDirection,
} from "./types";

// ============================================================================
// Types
// ============================================================================

export interface EdgeLineProps {
  /** Edge data */
  edge: GraphEdge;
  /** Source node */
  sourceNode: GraphNode;
  /** Target node */
  targetNode: GraphNode;
  /** Visual configuration */
  config: EdgeVisualConfig;
  /** On click handler */
  onClick?: (edgeId: string, event: ThreeEvent<MouseEvent>) => void;
  /** On pointer enter handler */
  onPointerEnter?: (edgeId: string, event: ThreeEvent<PointerEvent>) => void;
  /** On pointer leave handler */
  onPointerLeave?: (edgeId: string, event: ThreeEvent<PointerEvent>) => void;
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Calculate curve midpoint for bezier edge
 */
function calculateMidpoint(
  start: THREE.Vector3,
  end: THREE.Vector3,
  curveFactor: number
): THREE.Vector3 {
  const mid = new THREE.Vector3().lerpVectors(start, end, 0.5);

  // Calculate perpendicular offset
  const direction = new THREE.Vector3().subVectors(end, start).normalize();
  const perpendicular = new THREE.Vector3(-direction.y, direction.x, 0);

  // Add curve based on distance
  const distance = start.distanceTo(end);
  mid.add(perpendicular.multiplyScalar(distance * curveFactor));

  return mid;
}

/**
 * Calculate arrow position and rotation
 */
function calculateArrow(
  start: THREE.Vector3,
  end: THREE.Vector3,
  targetRadius: number
): { position: THREE.Vector3; rotation: THREE.Euler } {
  const direction = new THREE.Vector3().subVectors(end, start).normalize();

  // Position arrow at edge of target node
  const position = end.clone().sub(direction.multiplyScalar(targetRadius));

  // Calculate rotation to point toward target
  const rotation = new THREE.Euler();
  rotation.setFromRotationMatrix(
    new THREE.Matrix4().lookAt(position, end, new THREE.Vector3(0, 1, 0))
  );

  return { position, rotation };
}

// ============================================================================
// Edge Line Component
// ============================================================================

export const EdgeLine: React.FC<EdgeLineProps> = React.memo(
  ({
    edge,
    sourceNode,
    targetNode,
    config,
    onClick,
    onPointerEnter,
    onPointerLeave,
  }) => {
    const lineRef = useRef<THREE.Line>(null);
    const animationRef = useRef(0);

    // Calculate positions
    const startPos = useMemo(
      () =>
        new THREE.Vector3(
          sourceNode.position.x,
          sourceNode.position.y,
          sourceNode.position.z
        ),
      [sourceNode.position]
    );

    const endPos = useMemo(
      () =>
        new THREE.Vector3(
          targetNode.position.x,
          targetNode.position.y,
          targetNode.position.z
        ),
      [targetNode.position]
    );

    // Calculate color based on state
    const color = useMemo(() => {
      if (edge.state.selected) return config.selectedColor;
      if (edge.state.hovered) return config.hoveredColor;
      return edge.color || config.defaultColor;
    }, [edge.state.selected, edge.state.hovered, edge.color, config]);

    // Calculate line width based on state
    const lineWidth = useMemo(() => {
      let width = edge.width || config.defaultWidth;
      if (edge.state.hovered) width *= 1.5;
      if (edge.state.selected) width *= 1.3;
      return Math.min(Math.max(width, config.minWidth), config.maxWidth);
    }, [edge.width, edge.state.hovered, edge.state.selected, config]);

    // Calculate opacity
    const opacity = useMemo(() => {
      if (edge.state.dimmed) return 0.2;
      if (!edge.state.visible) return 0;
      return config.opacity;
    }, [edge.state.dimmed, edge.state.visible, config.opacity]);

    // Calculate midpoint for curved edges
    const midPoint = useMemo(
      () => calculateMidpoint(startPos, endPos, config.curveFactor),
      [startPos, endPos, config.curveFactor]
    );

    // Animate edge
    useFrame((_, delta) => {
      if (config.animated && lineRef.current) {
        animationRef.current += delta * config.animationSpeed;
        if (animationRef.current > 1) animationRef.current = 0;

        // Update dash offset for animation
        // dashOffset is a runtime property that may not be in type definitions
        const material = lineRef.current
          .material as THREE.LineDashedMaterial & { dashOffset?: number };
        if (material.dashOffset !== undefined) {
          material.dashOffset = -animationRef.current * 2;
        }
      }
    });

    // Event handlers
    const handleClick = useCallback(
      (event: ThreeEvent<MouseEvent>) => {
        event.stopPropagation();
        onClick?.(edge.id, event);
      },
      [edge.id, onClick]
    );

    const handlePointerEnter = useCallback(
      (event: ThreeEvent<PointerEvent>) => {
        event.stopPropagation();
        document.body.style.cursor = "pointer";
        onPointerEnter?.(edge.id, event);
      },
      [edge.id, onPointerEnter]
    );

    const handlePointerLeave = useCallback(
      (event: ThreeEvent<PointerEvent>) => {
        event.stopPropagation();
        document.body.style.cursor = "auto";
        onPointerLeave?.(edge.id, event);
      },
      [edge.id, onPointerLeave]
    );

    if (!edge.state.visible || opacity === 0) return null;

    // Use curved line if curveFactor > 0
    const useCurve = Math.abs(config.curveFactor) > 0.01;

    return (
      <group>
        {/* Main edge line */}
        {useCurve ? (
          <QuadraticBezierLine
            start={startPos}
            end={endPos}
            mid={midPoint}
            color={color}
            lineWidth={lineWidth * 100}
            transparent
            opacity={opacity}
            onClick={handleClick}
            onPointerEnter={handlePointerEnter}
            onPointerLeave={handlePointerLeave}
          />
        ) : (
          <Line
            ref={lineRef as any}
            points={[startPos, endPos]}
            color={color}
            lineWidth={lineWidth * 100}
            transparent
            opacity={opacity}
            onClick={handleClick}
            onPointerEnter={handlePointerEnter}
            onPointerLeave={handlePointerLeave}
            dashed={config.animated}
            dashSize={0.1}
            dashScale={10}
          />
        )}

        {/* Arrow heads */}
        {config.showArrows && (
          <>
            {/* Forward arrow */}
            {(edge.direction === "forward" ||
              edge.direction === "bidirectional") && (
              <ArrowHead
                start={startPos}
                end={endPos}
                targetRadius={targetNode.radius}
                size={config.arrowSize}
                color={color}
                opacity={opacity}
              />
            )}

            {/* Backward arrow */}
            {(edge.direction === "backward" ||
              edge.direction === "bidirectional") && (
              <ArrowHead
                start={endPos}
                end={startPos}
                targetRadius={sourceNode.radius}
                size={config.arrowSize}
                color={color}
                opacity={opacity}
              />
            )}
          </>
        )}

        {/* Edge label (if exists) */}
        {edge.metadata.label && edge.state.hovered && (
          <EdgeLabel
            position={midPoint}
            label={edge.metadata.label}
            color={color}
          />
        )}

        {/* Highlight effect */}
        {edge.state.highlighted && (
          <Line
            points={[startPos, endPos]}
            color={color}
            lineWidth={lineWidth * 200}
            transparent
            opacity={0.1}
          />
        )}

        {/* Animation particles */}
        {config.animated && (
          <AnimatedParticles
            start={startPos}
            end={endPos}
            mid={useCurve ? midPoint : undefined}
            color={color}
            speed={config.animationSpeed}
            progress={animationRef.current}
          />
        )}
      </group>
    );
  }
);

EdgeLine.displayName = "EdgeLine";

// ============================================================================
// Arrow Head Component
// ============================================================================

interface ArrowHeadProps {
  start: THREE.Vector3;
  end: THREE.Vector3;
  targetRadius: number;
  size: number;
  color: string;
  opacity: number;
}

const ArrowHead: React.FC<ArrowHeadProps> = React.memo(
  ({ start, end, targetRadius, size, color, opacity }) => {
    const { position, rotation } = useMemo(
      () => calculateArrow(start, end, targetRadius),
      [start, end, targetRadius]
    );

    return (
      <mesh position={position} rotation={rotation}>
        <coneGeometry args={[size * 0.5, size, 8]} />
        <meshStandardMaterial color={color} transparent opacity={opacity} />
      </mesh>
    );
  }
);

ArrowHead.displayName = "ArrowHead";

// ============================================================================
// Edge Label Component
// ============================================================================

interface EdgeLabelProps {
  position: THREE.Vector3;
  label: string;
  color: string;
}

const EdgeLabel: React.FC<EdgeLabelProps> = React.memo(
  ({ position, label, color }) => {
    return (
      <group position={position}>
        {/* Background */}
        <mesh>
          <planeGeometry args={[label.length * 0.08 + 0.2, 0.2]} />
          <meshBasicMaterial color="#000000" opacity={0.7} transparent />
        </mesh>

        {/* Text would go here - using Html for simplicity */}
        <group position={[0, 0, 0.01]}>
          <mesh>
            <planeGeometry args={[0.01, 0.01]} />
            <meshBasicMaterial color={color} transparent opacity={0} />
          </mesh>
        </group>
      </group>
    );
  }
);

EdgeLabel.displayName = "EdgeLabel";

// ============================================================================
// Animated Particles Component
// ============================================================================

interface AnimatedParticlesProps {
  start: THREE.Vector3;
  end: THREE.Vector3;
  mid?: THREE.Vector3;
  color: string;
  speed: number;
  progress: number;
}

const AnimatedParticles: React.FC<AnimatedParticlesProps> = React.memo(
  ({ start, end, mid, color, speed, progress }) => {
    const particleCount = 3;

    const particles = useMemo(() => {
      const result: THREE.Vector3[] = [];

      for (let i = 0; i < particleCount; i++) {
        const t = (progress + i / particleCount) % 1;
        let pos: THREE.Vector3;

        if (mid) {
          // Quadratic bezier interpolation
          const oneMinusT = 1 - t;
          pos = new THREE.Vector3()
            .addScaledVector(start, oneMinusT * oneMinusT)
            .addScaledVector(mid, 2 * oneMinusT * t)
            .addScaledVector(end, t * t);
        } else {
          // Linear interpolation
          pos = new THREE.Vector3().lerpVectors(start, end, t);
        }

        result.push(pos);
      }

      return result;
    }, [start, end, mid, progress]);

    return (
      <group>
        {particles.map((pos, i) => (
          <mesh key={i} position={pos}>
            <sphereGeometry args={[0.03, 8, 8]} />
            <meshBasicMaterial
              color={color}
              transparent
              opacity={0.8 - i * 0.2}
            />
          </mesh>
        ))}
      </group>
    );
  }
);

AnimatedParticles.displayName = "AnimatedParticles";

// ============================================================================
// Instanced Edges (for performance with many edges)
// ============================================================================

export interface InstancedEdgesProps {
  /** Array of edges with resolved nodes */
  edges: Array<{
    edge: GraphEdge;
    source: GraphNode;
    target: GraphNode;
  }>;
  /** Visual configuration */
  config: EdgeVisualConfig;
  /** Selection set */
  selectedIds: Set<string>;
  /** Hovered ID */
  hoveredId: string | null;
}

export const InstancedEdges: React.FC<InstancedEdgesProps> = React.memo(
  ({ edges, config, selectedIds, hoveredId }) => {
    // For simple straight lines, use LineSegments for batch rendering
    const positions = useMemo(() => {
      const points: number[] = [];

      for (const { edge, source, target } of edges) {
        if (!edge.state.visible) continue;

        points.push(
          source.position.x,
          source.position.y,
          source.position.z,
          target.position.x,
          target.position.y,
          target.position.z
        );
      }

      return new Float32Array(points);
    }, [edges]);

    const colors = useMemo(() => {
      const colorArray: number[] = [];
      const tempColor = new THREE.Color();

      for (const { edge } of edges) {
        if (!edge.state.visible) continue;

        if (selectedIds.has(edge.id)) {
          tempColor.set(config.selectedColor);
        } else if (hoveredId === edge.id) {
          tempColor.set(config.hoveredColor);
        } else {
          tempColor.set(edge.color || config.defaultColor);
        }

        // Two vertices per edge
        colorArray.push(tempColor.r, tempColor.g, tempColor.b);
        colorArray.push(tempColor.r, tempColor.g, tempColor.b);
      }

      return new Float32Array(colorArray);
    }, [edges, selectedIds, hoveredId, config]);

    if (edges.length === 0) return null;

    return (
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={positions.length / 3}
            array={positions}
            itemSize={3}
          />
          <bufferAttribute
            attach="attributes-color"
            count={colors.length / 3}
            array={colors}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial
          vertexColors
          transparent
          opacity={config.opacity}
          linewidth={1} // Note: linewidth only works in WebGL with certain conditions
        />
      </lineSegments>
    );
  }
);

InstancedEdges.displayName = "InstancedEdges";

export default EdgeLine;
