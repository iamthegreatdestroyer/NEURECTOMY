/**
 * AgentNodeMesh Component
 *
 * 3D representation of an AI agent node in the Dimensional Forge.
 * Features physics-based interactions, animations, and real-time status updates.
 */

import { useRef, useState, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import { Text, Billboard, RoundedBox } from "@react-three/drei";
import * as THREE from "three";
import type { AgentNode } from "@/stores/agent-store";

export interface AgentNodeMeshProps {
  node: AgentNode;
  isSelected?: boolean;
  onClick?: (event: any) => void;
  onContextMenu?: (event: any) => void;
  onPositionChange?: (position: { x: number; y: number; z: number }) => void;
  onHover?: (isHovering: boolean) => void;
}

export function AgentNodeMesh({
  node,
  isSelected = false,
  onClick,
  onContextMenu,
  onPositionChange,
  onHover,
}: AgentNodeMeshProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const outlineRef = useRef<THREE.Mesh>(null);
  const groupRef = useRef<THREE.Group>(null);
  const [hovered, setHovered] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const dragPlaneRef = useRef(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0));
  const intersectionPointRef = useRef(new THREE.Vector3());
  const offsetRef = useRef(new THREE.Vector3());

  // Status color mapping
  const statusColors = useMemo(
    () => ({
      idle: "#6366f1", // Indigo
      running: "#22c55e", // Green
      paused: "#f59e0b", // Orange
      error: "#ef4444", // Red
      completed: "#8b5cf6", // Purple
    }),
    []
  );

  const color = node.color || statusColors[node.status];

  // Pulse animation for running agents
  useFrame((state) => {
    if (!meshRef.current) return;

    if (node.status === "running") {
      const pulse = Math.sin(state.clock.elapsedTime * 2) * 0.1 + 1;
      meshRef.current.scale.setScalar((node.scale || 1) * pulse);
    } else {
      meshRef.current.scale.setScalar(node.scale || 1);
    }

    // Selection outline glow
    if (outlineRef.current) {
      if (isSelected || hovered) {
        const glowIntensity = Math.sin(state.clock.elapsedTime * 3) * 0.2 + 0.8;
        (outlineRef.current.material as THREE.MeshBasicMaterial).opacity =
          glowIntensity * 0.5;
      }
    }
  });

  const handleClick = (e: any) => {
    if (!isDragging) {
      e.stopPropagation();
      onClick?.(e);
    }
  };

  const handlePointerDown = (e: any) => {
    if (!isSelected) return; // Only allow dragging selected nodes

    e.stopPropagation();
    setIsDragging(true);

    // Calculate offset between pointer and object position
    const { point } = e;
    if (groupRef.current) {
      offsetRef.current.copy(point).sub(groupRef.current.position);
    }

    // Prevent camera controls from interfering
    (e.target as any).setPointerCapture?.(e.pointerId);
  };

  const handlePointerMove = (e: any) => {
    if (!isDragging || !groupRef.current) return;

    e.stopPropagation();

    // Project pointer onto horizontal plane
    const { ray } = e;
    ray.intersectPlane(dragPlaneRef.current, intersectionPointRef.current);

    // Update position with offset
    const newPosition = intersectionPointRef.current
      .clone()
      .sub(offsetRef.current);
    groupRef.current.position.copy(newPosition);
  };

  const handlePointerUp = (e: any) => {
    if (isDragging) {
      e.stopPropagation();
      setIsDragging(false);

      // Notify parent of position change
      if (groupRef.current && onPositionChange) {
        onPositionChange({
          x: groupRef.current.position.x,
          y: groupRef.current.position.y,
          z: groupRef.current.position.z,
        });
      }

      (e.target as any).releasePointerCapture?.(e.pointerId);
    }
  };

  const handlePointerOver = (e: any) => {
    setHovered(true);
    onHover?.(true);
    document.body.style.cursor = isSelected ? "grab" : "pointer";
  };

  const handlePointerOut = (e: any) => {
    setHovered(false);
    onHover?.(false);
    if (!isDragging) {
      document.body.style.cursor = "auto";
    }
  };

  const handleContextMenu = (e: any) => {
    e.stopPropagation();
    onContextMenu?.(e);
  };

  return (
    <group
      ref={groupRef}
      position={[node.position.x, node.position.y, node.position.z]}
      visible={node.visible}
    >
      {/* Selection outline */}
      {(isSelected || hovered) && (
        <RoundedBox
          ref={outlineRef}
          args={[2.2, 2.2, 2.2]}
          radius={0.2}
          smoothness={4}
        >
          <meshBasicMaterial
            color={isSelected ? "#ffffff" : color}
            transparent
            opacity={0.3}
            side={THREE.BackSide}
          />
        </RoundedBox>
      )}

      {/* Main node mesh */}
      <RoundedBox
        ref={meshRef}
        args={[2, 2, 2]}
        radius={0.15}
        smoothness={4}
        onClick={handleClick}
        onContextMenu={handleContextMenu}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerOver={handlePointerOver}
        onPointerOut={handlePointerOut}
        castShadow
        receiveShadow
      >
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={node.status === "running" ? 0.3 : 0.1}
          metalness={0.6}
          roughness={0.4}
        />
      </RoundedBox>

      {/* Agent name label */}
      <Billboard follow={true} lockX={false} lockY={false} lockZ={false}>
        <Text
          position={[0, 1.5, 0]}
          fontSize={0.3}
          color="#ffffff"
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.02}
          outlineColor="#000000"
        >
          {node.codename || node.name}
        </Text>
      </Billboard>

      {/* Status indicator */}
      <Billboard follow={true} lockX={false} lockY={false} lockZ={false}>
        <Text
          position={[0, -1.5, 0]}
          fontSize={0.2}
          color={statusColors[node.status]}
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.01}
          outlineColor="#000000"
        >
          {node.status.toUpperCase()}
        </Text>
      </Billboard>

      {/* Activity particles for running agents */}
      {node.status === "running" && <ActivityParticles color={color} />}
    </group>
  );
}

/**
 * ActivityParticles Component
 *
 * Particle effect for active agents.
 */
interface ActivityParticlesProps {
  color: string;
}

function ActivityParticles({ color }: ActivityParticlesProps) {
  const particlesRef = useRef<THREE.Points>(null);

  const particleCount = 50;
  const positions = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      const radius = 1.5 + Math.random() * 0.5;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.random() * Math.PI;

      pos[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = radius * Math.cos(phi);
    }
    return pos;
  }, [particleCount]);

  useFrame((state) => {
    if (!particlesRef.current) return;

    const positions = particlesRef.current.geometry.attributes.position;
    const time = state.clock.elapsedTime;

    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3;
      const x = positions.array[i3];
      const y = positions.array[i3 + 1];
      const z = positions.array[i3 + 2];

      // Orbit animation
      const angle = time + i * 0.1;
      const radius = Math.sqrt(x * x + z * z);

      positions.array[i3] = radius * Math.cos(angle);
      positions.array[i3 + 2] = radius * Math.sin(angle);
    }

    positions.needsUpdate = true;

    // Rotate the entire particle system
    particlesRef.current.rotation.y = time * 0.2;
  });

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particleCount}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.1}
        color={color}
        transparent
        opacity={0.6}
        sizeAttenuation
        depthWrite={false}
      />
    </points>
  );
}
