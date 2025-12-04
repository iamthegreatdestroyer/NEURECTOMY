/**
 * @file NodeMesh Component
 * @description 3D mesh representation for graph nodes
 * @module @neurectomy/3d-engine/visualization/graph
 * @agents @CANVAS @APEX
 */

/// <reference types="@react-three/fiber" />

import React, { useRef, useMemo, useCallback, useState } from "react";
import { useFrame, ThreeEvent } from "@react-three/fiber";
import { Html, Billboard, Text } from "@react-three/drei";
import * as THREE from "three";
import type { GraphNode, NodeVisualConfig } from "./types";

// ============================================================================
// Types
// ============================================================================

export interface NodeMeshProps {
  /** Node data */
  node: GraphNode;
  /** Visual configuration */
  config: NodeVisualConfig;
  /** Show label */
  showLabel?: boolean;
  /** On click handler */
  onClick?: (nodeId: string, event: ThreeEvent<MouseEvent>) => void;
  /** On double click handler */
  onDoubleClick?: (nodeId: string, event: ThreeEvent<MouseEvent>) => void;
  /** On pointer enter handler */
  onPointerEnter?: (nodeId: string, event: ThreeEvent<PointerEvent>) => void;
  /** On pointer leave handler */
  onPointerLeave?: (nodeId: string, event: ThreeEvent<PointerEvent>) => void;
  /** On drag start handler */
  onDragStart?: (nodeId: string, position: THREE.Vector3) => void;
  /** On drag handler */
  onDrag?: (nodeId: string, position: THREE.Vector3) => void;
  /** On drag end handler */
  onDragEnd?: (nodeId: string) => void;
}

// ============================================================================
// Node Geometry Factory
// ============================================================================

const NODE_GEOMETRIES: Record<string, THREE.BufferGeometry> = {};

function getNodeGeometry(type: string, radius: number): THREE.BufferGeometry {
  const key = `${type}-${radius.toFixed(2)}`;

  if (!NODE_GEOMETRIES[key]) {
    switch (type) {
      case "agent":
        NODE_GEOMETRIES[key] = new THREE.IcosahedronGeometry(radius, 2);
        break;
      case "llm":
        NODE_GEOMETRIES[key] = new THREE.OctahedronGeometry(radius, 1);
        break;
      case "tool":
        NODE_GEOMETRIES[key] = new THREE.BoxGeometry(
          radius * 1.5,
          radius * 1.5,
          radius * 1.5
        );
        break;
      case "memory":
        NODE_GEOMETRIES[key] = new THREE.CylinderGeometry(
          radius,
          radius,
          radius * 1.5,
          32
        );
        break;
      case "database":
        NODE_GEOMETRIES[key] = new THREE.CylinderGeometry(
          radius * 0.8,
          radius,
          radius * 1.8,
          6
        );
        break;
      case "api":
        NODE_GEOMETRIES[key] = new THREE.TorusGeometry(
          radius * 0.8,
          radius * 0.3,
          16,
          32
        );
        break;
      case "user":
        NODE_GEOMETRIES[key] = new THREE.CapsuleGeometry(
          radius * 0.5,
          radius,
          4,
          16
        );
        break;
      case "system":
        NODE_GEOMETRIES[key] = new THREE.DodecahedronGeometry(radius, 0);
        break;
      case "process":
        NODE_GEOMETRIES[key] = new THREE.ConeGeometry(radius, radius * 2, 32);
        break;
      case "service":
        NODE_GEOMETRIES[key] = new THREE.TetrahedronGeometry(radius * 1.2, 0);
        break;
      default:
        NODE_GEOMETRIES[key] = new THREE.SphereGeometry(radius, 32, 32);
    }
  }

  return NODE_GEOMETRIES[key];
}

// ============================================================================
// Node Icon Mapping
// ============================================================================

const NODE_ICONS: Record<string, string> = {
  agent: "🤖",
  llm: "🧠",
  tool: "🔧",
  memory: "💾",
  database: "🗄️",
  api: "🔌",
  user: "👤",
  system: "⚙️",
  process: "▶️",
  service: "📡",
  custom: "⬡",
};

// ============================================================================
// Component
// ============================================================================

export const NodeMesh: React.FC<NodeMeshProps> = React.memo(
  ({
    node,
    config,
    showLabel = true,
    onClick,
    onDoubleClick,
    onPointerEnter,
    onPointerLeave,
    onDragStart,
    onDrag,
    onDragEnd,
  }) => {
    const meshRef = useRef<THREE.Mesh>(null);
    const [isDragging, setIsDragging] = useState(false);
    const dragStartRef = useRef<THREE.Vector3 | null>(null);

    // Compute color based on state
    const color = useMemo(() => {
      if (node.state.selected) return config.selectedColor;
      if (node.state.hovered) return config.hoveredColor;
      return node.color || config.defaultColor;
    }, [node.state.selected, node.state.hovered, node.color, config]);

    // Compute scale based on state
    const scale = useMemo(() => {
      let s = 1;
      if (node.state.hovered) s *= 1.1;
      if (node.state.selected) s *= 1.05;
      if (node.state.dragging) s *= 1.15;
      return s;
    }, [node.state.hovered, node.state.selected, node.state.dragging]);

    // Compute opacity based on state
    const opacity = useMemo(() => {
      if (node.state.dimmed) return 0.3;
      if (!node.state.visible) return 0;
      return config.opacity;
    }, [node.state.dimmed, node.state.visible, config.opacity]);

    // Geometry
    const geometry = useMemo(
      () => getNodeGeometry(node.type, node.radius),
      [node.type, node.radius]
    );

    // Material with emissive for highlighting
    const material = useMemo(() => {
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(color),
        metalness: config.metalness,
        roughness: config.roughness,
        transparent: opacity < 1,
        opacity,
        emissive: node.state.highlighted
          ? new THREE.Color(color)
          : new THREE.Color(0x000000),
        emissiveIntensity: node.state.highlighted ? 0.3 : 0,
      });

      return mat;
    }, [
      color,
      config.metalness,
      config.roughness,
      opacity,
      node.state.highlighted,
    ]);

    // Animation frame
    useFrame((_, delta) => {
      if (!meshRef.current) return;

      // Smooth position interpolation
      const targetPos = new THREE.Vector3(
        node.position.x,
        node.position.y,
        node.position.z
      );

      meshRef.current.position.lerp(targetPos, isDragging ? 1 : 0.3);

      // Activity pulsing
      if (node.state.activity > 0) {
        const pulse =
          1 + Math.sin(Date.now() * 0.005) * 0.05 * node.state.activity;
        meshRef.current.scale.setScalar(scale * pulse);
      } else {
        meshRef.current.scale.setScalar(scale);
      }

      // Rotation for visual interest
      if (!node.pinned && !isDragging) {
        meshRef.current.rotation.y += delta * 0.1;
      }
    });

    // Event handlers
    const handleClick = useCallback(
      (event: ThreeEvent<MouseEvent>) => {
        event.stopPropagation();
        onClick?.(node.id, event);
      },
      [node.id, onClick]
    );

    const handleDoubleClick = useCallback(
      (event: ThreeEvent<MouseEvent>) => {
        event.stopPropagation();
        onDoubleClick?.(node.id, event);
      },
      [node.id, onDoubleClick]
    );

    const handlePointerEnter = useCallback(
      (event: ThreeEvent<PointerEvent>) => {
        event.stopPropagation();
        document.body.style.cursor = "pointer";
        onPointerEnter?.(node.id, event);
      },
      [node.id, onPointerEnter]
    );

    const handlePointerLeave = useCallback(
      (event: ThreeEvent<PointerEvent>) => {
        event.stopPropagation();
        document.body.style.cursor = "auto";
        onPointerLeave?.(node.id, event);
      },
      [node.id, onPointerLeave]
    );

    const handlePointerDown = useCallback(
      (event: ThreeEvent<PointerEvent>) => {
        event.stopPropagation();

        if (event.button === 0) {
          // Left button
          setIsDragging(true);
          dragStartRef.current = event.point.clone();
          onDragStart?.(node.id, event.point);

          // Capture pointer
          (event.target as HTMLElement).setPointerCapture(event.pointerId);
        }
      },
      [node.id, onDragStart]
    );

    const handlePointerMove = useCallback(
      (event: ThreeEvent<PointerEvent>) => {
        if (isDragging && dragStartRef.current) {
          event.stopPropagation();
          onDrag?.(node.id, event.point);
        }
      },
      [isDragging, node.id, onDrag]
    );

    const handlePointerUp = useCallback(
      (event: ThreeEvent<PointerEvent>) => {
        if (isDragging) {
          event.stopPropagation();
          setIsDragging(false);
          dragStartRef.current = null;
          onDragEnd?.(node.id);

          // Release pointer
          (event.target as HTMLElement).releasePointerCapture(event.pointerId);
        }
      },
      [isDragging, node.id, onDragEnd]
    );

    if (!node.state.visible) return null;

    return (
      <group>
        {/* Main node mesh */}
        <mesh
          ref={meshRef}
          geometry={geometry}
          material={material}
          onClick={handleClick}
          onDoubleClick={handleDoubleClick}
          onPointerEnter={handlePointerEnter}
          onPointerLeave={handlePointerLeave}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          castShadow
          receiveShadow
        />

        {/* Glow effect for highlighted nodes */}
        {node.state.highlighted && (
          <mesh
            position={[node.position.x, node.position.y, node.position.z]}
            scale={scale * 1.3}
          >
            <sphereGeometry args={[node.radius, 16, 16]} />
            <meshBasicMaterial
              color={color}
              transparent
              opacity={0.15}
              depthWrite={false}
            />
          </mesh>
        )}

        {/* Ring for selected nodes */}
        {node.state.selected && (
          <mesh
            position={[node.position.x, node.position.y, node.position.z]}
            rotation={[Math.PI / 2, 0, 0]}
          >
            <ringGeometry args={[node.radius * 1.3, node.radius * 1.5, 32]} />
            <meshBasicMaterial
              color={config.selectedColor}
              transparent
              opacity={0.8}
              side={THREE.DoubleSide}
            />
          </mesh>
        )}

        {/* Pin indicator */}
        {node.pinned && (
          <mesh
            position={[
              node.position.x,
              node.position.y + node.radius + 0.3,
              node.position.z,
            ]}
          >
            <coneGeometry args={[0.1, 0.2, 8]} />
            <meshBasicMaterial color="#ff6b6b" />
          </mesh>
        )}

        {/* Label */}
        {showLabel && config.showLabels && (
          <Billboard
            position={[
              node.position.x,
              node.position.y + node.radius + config.labelOffset,
              node.position.z,
            ]}
            follow={true}
            lockX={false}
            lockY={false}
            lockZ={false}
          >
            <Text
              fontSize={config.labelFontSize / 100}
              color={node.state.dimmed ? "#666666" : "#ffffff"}
              anchorX="center"
              anchorY="bottom"
              outlineWidth={0.02}
              outlineColor="#000000"
            >
              {NODE_ICONS[node.type] || "⬡"} {node.label}
            </Text>
          </Billboard>
        )}
      </group>
    );
  }
);

NodeMesh.displayName = "NodeMesh";

// ============================================================================
// Instanced Nodes (for performance with many nodes)
// ============================================================================

export interface InstancedNodesProps {
  /** Array of nodes */
  nodes: GraphNode[];
  /** Visual configuration */
  config: NodeVisualConfig;
  /** Selection set */
  selectedIds: Set<string>;
  /** Hovered ID */
  hoveredId: string | null;
  /** On click handler */
  onClick?: (nodeId: string, event: ThreeEvent<MouseEvent>) => void;
}

export const InstancedNodes: React.FC<InstancedNodesProps> = React.memo(
  ({ nodes, config, selectedIds, hoveredId, onClick }) => {
    const meshRef = useRef<THREE.InstancedMesh>(null);
    const tempObject = useMemo(() => new THREE.Object3D(), []);
    const tempColor = useMemo(() => new THREE.Color(), []);

    // Update instance matrices and colors
    useFrame(() => {
      if (!meshRef.current) return;

      const mesh = meshRef.current;

      nodes.forEach((node, i) => {
        // Position
        tempObject.position.set(
          node.position.x,
          node.position.y,
          node.position.z
        );

        // Scale based on state
        let scale = node.radius;
        if (hoveredId === node.id) scale *= 1.1;
        if (selectedIds.has(node.id)) scale *= 1.05;
        tempObject.scale.setScalar(scale);

        tempObject.updateMatrix();
        mesh.setMatrixAt(i, tempObject.matrix);

        // Color based on state
        if (selectedIds.has(node.id)) {
          tempColor.set(config.selectedColor);
        } else if (hoveredId === node.id) {
          tempColor.set(config.hoveredColor);
        } else {
          tempColor.set(node.color || config.defaultColor);
        }
        mesh.setColorAt(i, tempColor);
      });

      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    });

    // Handle click - need to find which instance was clicked
    const handleClick = useCallback(
      (event: ThreeEvent<MouseEvent>) => {
        if (event.instanceId !== undefined && onClick) {
          const node = nodes[event.instanceId];
          if (node) {
            onClick(node.id, event);
          }
        }
      },
      [nodes, onClick]
    );

    if (nodes.length === 0) return null;

    return (
      <instancedMesh
        ref={meshRef}
        args={[undefined, undefined, nodes.length]}
        onClick={handleClick}
        castShadow
        receiveShadow
      >
        <sphereGeometry args={[1, 16, 16]} />
        <meshStandardMaterial
          metalness={config.metalness}
          roughness={config.roughness}
        />
      </instancedMesh>
    );
  }
);

InstancedNodes.displayName = "InstancedNodes";

export default NodeMesh;
