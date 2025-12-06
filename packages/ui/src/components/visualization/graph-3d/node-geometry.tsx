/**
 * @fileoverview Instanced node geometry for high-performance 3D graph rendering
 * @description Self-contained component that renders graph nodes using GPU instancing
 * @copyright Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
 */

import React, { useRef, useMemo, useEffect, useCallback } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

// ============================================================================
// Types
// ============================================================================

/** LOD (Level of Detail) levels for node rendering */
export type LODLevel = "high" | "medium" | "low" | "ultra-low";

/** Visual theme colors for different node types */
export interface NodeTheme {
  concept?: string;
  entity?: string;
  relationship?: string;
  cluster?: string;
  document?: string;
  default?: string;
  selected?: string;
  hovered?: string;
}

/** Simplified node data structure for rendering */
export interface NodeData {
  id: string;
  type?: string;
  position: { x: number; y: number; z: number };
  size?: number;
  color?: string;
}

/** Props for InstancedNodeGeometry component */
export interface InstancedNodeGeometryProps {
  /** Array of nodes to render */
  nodes: NodeData[];
  /** Set of selected node IDs */
  selectedNodes?: Set<string>;
  /** Currently hovered node ID */
  hoveredNode?: string | null;
  /** Theme colors for node types */
  theme?: NodeTheme;
  /** Camera distance for LOD calculation */
  cameraDistance?: number;
  /** Enable LOD system */
  enableLOD?: boolean;
  /** Maximum number of instances */
  maxInstanceCount?: number;
  /** Callback when a node is clicked */
  onNodeClick?: (nodeId: string) => void;
  /** Callback when a node is hovered */
  onNodeHover?: (nodeId: string | null) => void;
}

interface InstancedNodeData {
  positions: Float32Array;
  colors: Float32Array;
  scales: Float32Array;
  nodeIds: string[];
}

interface LODConfig {
  level: LODLevel;
  geometry: THREE.BufferGeometry;
  segments: number;
  distanceThreshold: number;
}

// ============================================================================
// Constants
// ============================================================================

/** LOD configurations with geometry complexity and distance thresholds */
export const LOD_CONFIGS: LODConfig[] = [
  {
    level: "high",
    geometry: new THREE.SphereGeometry(1, 32, 32),
    segments: 32,
    distanceThreshold: 50,
  },
  {
    level: "medium",
    geometry: new THREE.SphereGeometry(1, 16, 16),
    segments: 16,
    distanceThreshold: 150,
  },
  {
    level: "low",
    geometry: new THREE.SphereGeometry(1, 8, 8),
    segments: 8,
    distanceThreshold: 400,
  },
  {
    level: "ultra-low",
    geometry: new THREE.OctahedronGeometry(1),
    segments: 3,
    distanceThreshold: Infinity,
  },
];

export const DEFAULT_THEME: NodeTheme = {
  concept: "#4a9eff",
  entity: "#ff6b6b",
  relationship: "#6bff8a",
  cluster: "#ffd93d",
  document: "#d4a5ff",
  default: "#888888",
  selected: "#ff6b6b",
  hovered: "#ffd93d",
};

const NODE_VERTEX_SHADER = `
  attribute vec3 instanceColor;
  attribute float instanceScale;
  attribute float instanceSelected;
  attribute float instanceHovered;
  
  varying vec3 vColor;
  varying vec3 vNormal;
  varying vec3 vViewPosition;
  varying float vSelected;
  varying float vHovered;
  
  void main() {
    vColor = instanceColor;
    vNormal = normalMatrix * normal;
    vSelected = instanceSelected;
    vHovered = instanceHovered;
    
    vec3 transformed = position * instanceScale;
    vec4 mvPosition = modelViewMatrix * instanceMatrix * vec4(transformed, 1.0);
    vViewPosition = -mvPosition.xyz;
    
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const NODE_FRAGMENT_SHADER = `
  uniform vec3 ambientColor;
  uniform vec3 specularColor;
  uniform float shininess;
  uniform vec3 lightPosition;
  uniform float glowIntensity;
  uniform vec3 selectedGlowColor;
  uniform vec3 hoveredGlowColor;
  
  varying vec3 vColor;
  varying vec3 vNormal;
  varying vec3 vViewPosition;
  varying float vSelected;
  varying float vHovered;
  
  void main() {
    vec3 normal = normalize(vNormal);
    vec3 viewDir = normalize(vViewPosition);
    vec3 lightDir = normalize(lightPosition - vViewPosition);
    
    // Ambient
    vec3 ambient = ambientColor * vColor;
    
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * vColor;
    
    // Specular (Blinn-Phong)
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), shininess);
    vec3 specular = specularColor * spec;
    
    vec3 result = ambient + diffuse + specular;
    
    // Selection/hover glow
    float glow = max(vSelected, vHovered * 0.5) * glowIntensity;
    vec3 glowColor = vSelected > 0.5 ? selectedGlowColor : hoveredGlowColor;
    result = mix(result, glowColor, glow);
    
    // Rim lighting for depth perception
    float rim = 1.0 - max(dot(viewDir, normal), 0.0);
    rim = pow(rim, 3.0);
    result += rim * 0.15 * vColor;
    
    gl_FragColor = vec4(result, 1.0);
  }
`;

// ============================================================================
// Utility Functions
// ============================================================================

function getNodeColor(node: NodeData, theme: NodeTheme): THREE.Color {
  // If node has explicit color, use it
  if (node.color) {
    return new THREE.Color(node.color);
  }

  // Otherwise use theme color based on type
  const typeColor =
    theme[node.type as keyof NodeTheme] || theme.default || "#888888";
  return new THREE.Color(typeColor);
}

function calculateLODLevel(distance: number): LODConfig {
  for (const config of LOD_CONFIGS) {
    if (distance < config.distanceThreshold) {
      return config;
    }
  }
  return LOD_CONFIGS[LOD_CONFIGS.length - 1];
}

// ============================================================================
// Component
// ============================================================================

/**
 * High-performance instanced node geometry for 3D graph visualization.
 *
 * Uses GPU instancing to efficiently render thousands of nodes with:
 * - Automatic LOD (Level of Detail) based on camera distance
 * - Custom shaders for selection/hover highlighting
 * - Blinn-Phong lighting with rim effects
 * - Raycasting for node interaction
 *
 * @example
 * ```tsx
 * <InstancedNodeGeometry
 *   nodes={graphNodes}
 *   selectedNodes={new Set(['node-1', 'node-2'])}
 *   hoveredNode={null}
 *   onNodeClick={(id) => console.log('Clicked:', id)}
 *   onNodeHover={(id) => console.log('Hovered:', id)}
 * />
 * ```
 */
export const InstancedNodeGeometry: React.FC<InstancedNodeGeometryProps> = ({
  nodes,
  selectedNodes = new Set(),
  hoveredNode = null,
  theme = DEFAULT_THEME,
  cameraDistance,
  enableLOD = true,
  maxInstanceCount = 100000,
  onNodeClick,
  onNodeHover,
}) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { camera, raycaster, pointer } = useThree();

  // Track internal hover state for raycasting
  const internalHoveredRef = useRef<string | null>(null);

  // Create instanced node data
  const instanceData = useMemo<InstancedNodeData>(() => {
    const count = Math.min(nodes.length, maxInstanceCount);

    return {
      positions: new Float32Array(count * 3),
      colors: new Float32Array(count * 3),
      scales: new Float32Array(count),
      nodeIds: nodes.slice(0, count).map((n) => n.id),
    };
  }, [nodes, maxInstanceCount]);

  // Calculate effective camera distance
  const effectiveDistance = useMemo(() => {
    if (cameraDistance !== undefined) {
      return cameraDistance;
    }
    // Calculate from camera position if not provided
    return camera.position.length();
  }, [cameraDistance, camera.position]);

  // Determine current LOD geometry
  const currentLOD = useMemo((): LODConfig => {
    if (!enableLOD) return LOD_CONFIGS[0];
    return calculateLODLevel(effectiveDistance);
  }, [effectiveDistance, enableLOD]);

  // Create shader material
  const material = useMemo(() => {
    return new THREE.ShaderMaterial({
      uniforms: {
        ambientColor: { value: new THREE.Color(0.3, 0.3, 0.3) },
        specularColor: { value: new THREE.Color(0.5, 0.5, 0.5) },
        shininess: { value: 64.0 },
        lightPosition: { value: new THREE.Vector3(50, 50, 50) },
        glowIntensity: { value: 0.4 },
        selectedGlowColor: {
          value: new THREE.Color(theme.selected || "#ff6b6b"),
        },
        hoveredGlowColor: {
          value: new THREE.Color(theme.hovered || "#ffd93d"),
        },
      },
      vertexShader: NODE_VERTEX_SHADER,
      fragmentShader: NODE_FRAGMENT_SHADER,
    });
  }, [theme.selected, theme.hovered]);

  // Update instance attributes
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;

    const count = Math.min(nodes.length, maxInstanceCount);
    const matrix = new THREE.Matrix4();
    const color = new THREE.Color();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();

    // Create instance attributes if not exist
    if (!mesh.geometry.getAttribute("instanceColor")) {
      mesh.geometry.setAttribute(
        "instanceColor",
        new THREE.InstancedBufferAttribute(
          new Float32Array(maxInstanceCount * 3),
          3
        )
      );
      mesh.geometry.setAttribute(
        "instanceScale",
        new THREE.InstancedBufferAttribute(
          new Float32Array(maxInstanceCount),
          1
        )
      );
      mesh.geometry.setAttribute(
        "instanceSelected",
        new THREE.InstancedBufferAttribute(
          new Float32Array(maxInstanceCount),
          1
        )
      );
      mesh.geometry.setAttribute(
        "instanceHovered",
        new THREE.InstancedBufferAttribute(
          new Float32Array(maxInstanceCount),
          1
        )
      );
    }

    const colorAttr = mesh.geometry.getAttribute(
      "instanceColor"
    ) as THREE.InstancedBufferAttribute;
    const scaleAttr = mesh.geometry.getAttribute(
      "instanceScale"
    ) as THREE.InstancedBufferAttribute;
    const selectedAttr = mesh.geometry.getAttribute(
      "instanceSelected"
    ) as THREE.InstancedBufferAttribute;
    const hoveredAttr = mesh.geometry.getAttribute(
      "instanceHovered"
    ) as THREE.InstancedBufferAttribute;

    // Update each instance
    for (let i = 0; i < count; i++) {
      const node = nodes[i];
      if (!node) continue;

      // Position
      position.set(
        node.position?.x || 0,
        node.position?.y || 0,
        node.position?.z || 0
      );

      // Scale based on node size
      const nodeScale = node.size || 1;
      scale.set(nodeScale, nodeScale, nodeScale);

      // Build matrix
      matrix.compose(position, quaternion, scale);
      mesh.setMatrixAt(i, matrix);

      // Color
      color.copy(getNodeColor(node, theme));
      colorAttr.setXYZ(i, color.r, color.g, color.b);

      // Scale attribute for shader
      scaleAttr.setX(i, nodeScale);

      // Selection state
      selectedAttr.setX(i, selectedNodes.has(node.id) ? 1.0 : 0.0);

      // Hover state
      hoveredAttr.setX(i, hoveredNode === node.id ? 1.0 : 0.0);
    }

    mesh.instanceMatrix.needsUpdate = true;
    colorAttr.needsUpdate = true;
    scaleAttr.needsUpdate = true;
    selectedAttr.needsUpdate = true;
    hoveredAttr.needsUpdate = true;
    mesh.count = count;
  }, [nodes, selectedNodes, hoveredNode, theme, maxInstanceCount]);

  // Raycasting for interaction
  const handlePointerMove = useCallback(() => {
    const mesh = meshRef.current;
    if (!mesh) return;

    raycaster.setFromCamera(pointer, camera);
    const intersects = raycaster.intersectObject(mesh);

    if (intersects.length > 0) {
      const instanceId = intersects[0]?.instanceId;
      if (
        instanceId !== undefined &&
        instanceId < instanceData.nodeIds.length
      ) {
        const nodeId = instanceData.nodeIds[instanceId];
        if (nodeId !== undefined && nodeId !== internalHoveredRef.current) {
          internalHoveredRef.current = nodeId;
          onNodeHover?.(nodeId);
        }
      }
    } else if (internalHoveredRef.current !== null) {
      internalHoveredRef.current = null;
      onNodeHover?.(null);
    }
  }, [camera, raycaster, pointer, instanceData.nodeIds, onNodeHover]);

  const handleClick = useCallback(() => {
    const mesh = meshRef.current;
    if (!mesh) return;

    raycaster.setFromCamera(pointer, camera);
    const intersects = raycaster.intersectObject(mesh);

    if (intersects.length > 0) {
      const instanceId = intersects[0]?.instanceId;
      if (
        instanceId !== undefined &&
        instanceId < instanceData.nodeIds.length
      ) {
        const nodeId = instanceData.nodeIds[instanceId];
        if (nodeId !== undefined) {
          onNodeClick?.(nodeId);
        }
      }
    }
  }, [camera, raycaster, pointer, instanceData.nodeIds, onNodeClick]);

  // Animation frame updates
  useFrame(() => {
    handlePointerMove();
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[currentLOD.geometry, material, maxInstanceCount]}
      frustumCulled
      onClick={handleClick}
    />
  );
};

export default InstancedNodeGeometry;
