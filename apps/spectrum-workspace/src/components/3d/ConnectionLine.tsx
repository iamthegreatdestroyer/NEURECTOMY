/**
 * ConnectionLine Component
 *
 * 3D animated line representing connections between agent nodes.
 * Features flow animations, colors based on connection type, and interactive selection.
 */

import { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { AgentConnection } from "@/stores/agent-store";

export interface ConnectionLineProps {
  connection: AgentConnection;
  startPosition: THREE.Vector3;
  endPosition: THREE.Vector3;
  isSelected?: boolean;
  onClick?: (connection: AgentConnection) => void;
}

export function ConnectionLine({
  connection,
  startPosition,
  endPosition,
  isSelected = false,
  onClick,
}: ConnectionLineProps) {
  const lineRef = useRef<THREE.Line>(null);
  const flowRef = useRef<THREE.Line>(null);

  // Connection type color mapping
  const typeColors = useMemo(
    () => ({
      data: "#3b82f6", // Blue
      control: "#8b5cf6", // Purple
      feedback: "#f59e0b", // Orange
    }),
    []
  );

  const color = typeColors[connection.type];

  // Create curve geometry
  const { curve, points } = useMemo(() => {
    // Create a smooth curve from start to end
    const start = startPosition.clone();
    const end = endPosition.clone();

    // Add curve height based on distance
    const distance = start.distanceTo(end);
    const midPoint = new THREE.Vector3()
      .addVectors(start, end)
      .multiplyScalar(0.5);
    midPoint.y += distance * 0.2; // Curve height

    const curve = new THREE.QuadraticBezierCurve3(start, midPoint, end);
    const points = curve.getPoints(50);

    return { curve, points };
  }, [startPosition, endPosition]);

  // Animated flow particles
  useFrame((state) => {
    if (!flowRef.current) return;

    // Animate dash offset for flow effect
    const speed = 0.5;
    const offset = (state.clock.elapsedTime * speed) % 1;

    const material = flowRef.current.material as THREE.LineDashedMaterial;
    (material as any).dashOffset = -offset * 2;
  });

  const handleClick = (e: any) => {
    e?.stopPropagation?.();
    onClick?.(connection);
  };

  return (
    <group>
      {/* Main connection line */}
      <line ref={lineRef as any} onClick={handleClick}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={points.length}
            array={new Float32Array(points.flatMap((p) => [p.x, p.y, p.z]))}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial
          color={color}
          linewidth={isSelected ? 3 : 1}
          transparent
          opacity={isSelected ? 1 : 0.6}
        />
      </line>

      {/* Flow animation line */}
      <line ref={flowRef as any}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={points.length}
            array={new Float32Array(points.flatMap((p) => [p.x, p.y, p.z]))}
            itemSize={3}
          />
        </bufferGeometry>
        <lineDashedMaterial
          color={color}
          linewidth={2}
          dashSize={0.5}
          gapSize={0.5}
          transparent
          opacity={0.8}
        />
      </line>

      {/* Arrow head at end */}
      <ArrowHead
        position={endPosition}
        direction={getDirection(points)}
        color={color}
      />
    </group>
  );
}

/**
 * ArrowHead Component
 *
 * Directional arrow indicating data flow.
 */
interface ArrowHeadProps {
  position: THREE.Vector3;
  direction: THREE.Vector3;
  color: string;
}

function ArrowHead({ position, direction, color }: ArrowHeadProps) {
  const meshRef = useRef<THREE.Mesh>(null);

  // Orient arrow to face direction
  useMemo(() => {
    if (meshRef.current) {
      const quaternion = new THREE.Quaternion();
      quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        direction.normalize()
      );
      meshRef.current.quaternion.copy(quaternion);
    }
  }, [direction]);

  return (
    <mesh ref={meshRef} position={position}>
      <coneGeometry args={[0.2, 0.4, 8]} />
      <meshBasicMaterial color={color} />
    </mesh>
  );
}

/**
 * Get direction vector from curve points
 */
function getDirection(points: THREE.Vector3[]): THREE.Vector3 {
  const lastPoint = points[points.length - 1];
  const secondLastPoint = points[points.length - 2];

  return new THREE.Vector3().subVectors(lastPoint, secondLastPoint).normalize();
}
