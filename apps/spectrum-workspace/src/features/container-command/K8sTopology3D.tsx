/**
 * K8sTopology3D Component
 *
 * 3D visualization of Kubernetes cluster topology.
 * Displays nodes, pods, and services in an interactive 3D space.
 */

import { Suspense, useCallback, useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Stars, Grid, Text, Html } from "@react-three/drei";
import { Vector3 } from "three";
import {
  Boxes,
  Server,
  Eye,
  EyeOff,
  Layers,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { useContainerStore } from "@/stores/container-store";
import type { KubernetesNode, KubernetesPod } from "@/stores/container-store";

/**
 * K8sTopology3D - 3D Kubernetes cluster visualization
 *
 * Features:
 * - 3D positioned nodes with resource usage
 * - Pods grouped by node with status colors
 * - Service connections overlay
 * - Interactive camera controls
 * - Real-time metrics display
 *
 * @example
 * ```tsx
 * <K8sTopology3D />
 * ```
 */
export function K8sTopology3D() {
  const {
    clusters,
    activeClusterId,
    selectedPodId,
    selectedNodeId,
    selectPod,
    selectNode,
    showMetrics,
    showConnections,
    toggleMetrics,
    toggleConnections,
  } = useContainerStore();

  const activeCluster = useMemo(
    () => clusters.find((c) => c.id === activeClusterId),
    [clusters, activeClusterId]
  );

  const handleNodeClick = useCallback(
    (nodeId: string) => {
      selectNode(nodeId === selectedNodeId ? null : nodeId);
    },
    [selectNode, selectedNodeId]
  );

  const handlePodClick = useCallback(
    (podId: string) => {
      selectPod(podId === selectedPodId ? null : podId);
    },
    [selectPod, selectedPodId]
  );

  if (!activeCluster) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <Boxes className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">No Cluster Selected</h3>
          <p className="text-muted-foreground text-sm">
            Select a Kubernetes cluster to view topology
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Controls */}
      <div className="h-12 px-4 flex items-center justify-between border-b border-border bg-card">
        <div className="flex items-center gap-2">
          <Boxes className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium">{activeCluster.name}</span>
          <span className="text-xs text-muted-foreground">
            {activeCluster.nodes.length} nodes • {activeCluster.pods.length}{" "}
            pods
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={toggleMetrics}
            className={`
              px-3 py-1.5 rounded-lg flex items-center gap-2 text-sm transition-colors
              ${showMetrics ? "bg-primary text-primary-foreground" : "hover:bg-muted"}
            `}
          >
            {showMetrics ? (
              <Eye className="w-4 h-4" />
            ) : (
              <EyeOff className="w-4 h-4" />
            )}
            Metrics
          </button>
          <button
            onClick={toggleConnections}
            className={`
              px-3 py-1.5 rounded-lg flex items-center gap-2 text-sm transition-colors
              ${showConnections ? "bg-primary text-primary-foreground" : "hover:bg-muted"}
            `}
          >
            <Layers className="w-4 h-4" />
            Connections
          </button>
        </div>
      </div>

      {/* 3D Canvas */}
      <div className="flex-1 relative">
        <Canvas
          camera={{ position: [20, 15, 20], fov: 60 }}
          className="bg-gradient-to-b from-slate-950 to-slate-900"
        >
          <Suspense fallback={null}>
            {/* Lighting */}
            <ambientLight intensity={0.5} />
            <directionalLight
              position={[10, 10, 5]}
              intensity={0.8}
              castShadow
            />
            <pointLight
              position={[-10, -10, -5]}
              intensity={0.3}
              color="#4f46e5"
            />

            {/* Environment */}
            <Stars
              radius={100}
              depth={50}
              count={5000}
              factor={4}
              fade
              speed={1}
            />
            <Grid
              args={[50, 50]}
              cellSize={1}
              cellThickness={0.5}
              cellColor="#334155"
              sectionSize={5}
              sectionThickness={1}
              sectionColor="#475569"
              fadeDistance={30}
              fadeStrength={1}
              followCamera={false}
              infiniteGrid
            />

            {/* Cluster Nodes */}
            {activeCluster.nodes.map((node) => (
              <NodeMesh
                key={node.id}
                node={node}
                isSelected={selectedNodeId === node.id}
                showMetrics={showMetrics}
                onClick={() => handleNodeClick(node.id)}
              />
            ))}

            {/* Pods */}
            {activeCluster.pods.map((pod) => {
              const node = activeCluster.nodes.find(
                (n) => n.name === pod.nodeName
              );
              if (!node || !node.position) return null;

              return (
                <PodMesh
                  key={pod.id}
                  pod={pod}
                  nodePosition={node.position}
                  isSelected={selectedPodId === pod.id}
                  onClick={() => handlePodClick(pod.id)}
                />
              );
            })}

            {/* Service Connections */}
            {showConnections &&
              activeCluster.services.map((service) => (
                <ServiceConnections
                  key={service.id}
                  service={service}
                  pods={activeCluster.pods}
                  nodes={activeCluster.nodes}
                />
              ))}

            {/* Camera Controls */}
            <OrbitControls
              enableDamping
              dampingFactor={0.05}
              minDistance={5}
              maxDistance={50}
            />
          </Suspense>
        </Canvas>

        {/* Legend */}
        <div className="absolute bottom-4 left-4 bg-card/90 backdrop-blur-sm border border-border rounded-lg p-4 space-y-2">
          <h4 className="text-xs font-semibold mb-2">Status</h4>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span className="text-muted-foreground">Ready / Running</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-3 h-3 rounded-full bg-yellow-500" />
            <span className="text-muted-foreground">Pending / Warning</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <span className="text-muted-foreground">Failed / Error</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-3 h-3 rounded-full bg-gray-500" />
            <span className="text-muted-foreground">Unknown</span>
          </div>
        </div>

        {/* Controls Guide */}
        <div className="absolute top-4 right-4 bg-card/90 backdrop-blur-sm border border-border rounded-lg p-3 space-y-1.5">
          <h4 className="text-xs font-semibold mb-2">Controls</h4>
          <p className="text-xs text-muted-foreground">
            <kbd className="px-1.5 py-0.5 bg-muted rounded text-xs">
              Left Click
            </kbd>{" "}
            + Drag to rotate
          </p>
          <p className="text-xs text-muted-foreground">
            <kbd className="px-1.5 py-0.5 bg-muted rounded text-xs">
              Right Click
            </kbd>{" "}
            + Drag to pan
          </p>
          <p className="text-xs text-muted-foreground">
            <kbd className="px-1.5 py-0.5 bg-muted rounded text-xs">Scroll</kbd>{" "}
            to zoom
          </p>
          <p className="text-xs text-muted-foreground">
            <kbd className="px-1.5 py-0.5 bg-muted rounded text-xs">Click</kbd>{" "}
            node/pod to select
          </p>
        </div>
      </div>
    </div>
  );
}

/**
 * NodeMesh - 3D representation of K8s node
 */
function NodeMesh({
  node,
  isSelected,
  showMetrics,
  onClick,
}: {
  node: KubernetesNode;
  isSelected: boolean;
  showMetrics: boolean;
  onClick: () => void;
}) {
  const meshRef = useRef<any>(null);
  const position = node.position || { x: 0, y: 0, z: 0 };

  // Pulse animation when selected
  useFrame((state) => {
    if (meshRef.current && isSelected) {
      meshRef.current.scale.setScalar(
        1 + Math.sin(state.clock.elapsedTime * 2) * 0.05
      );
    } else if (meshRef.current) {
      meshRef.current.scale.setScalar(1);
    }
  });

  const statusColor = node.status === "Ready" ? "#10b981" : "#ef4444";
  const cpuPercent = (node.cpu.used / node.cpu.total) * 100;
  const memoryPercent = (node.memory.used / node.memory.total) * 100;

  return (
    <group position={[position.x, position.y, position.z]}>
      {/* Node box */}
      <mesh ref={meshRef} onClick={onClick}>
        <boxGeometry args={[2, 2, 2]} />
        <meshStandardMaterial
          color={statusColor}
          metalness={0.6}
          roughness={0.4}
          emissive={statusColor}
          emissiveIntensity={isSelected ? 0.3 : 0.1}
        />
      </mesh>

      {/* Selection outline */}
      {isSelected && (
        <mesh>
          <boxGeometry args={[2.2, 2.2, 2.2]} />
          <meshBasicMaterial color="#6366f1" wireframe />
        </mesh>
      )}

      {/* Node label */}
      <Text
        position={[0, 1.5, 0]}
        fontSize={0.3}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
      >
        {node.name}
      </Text>

      {/* Role badge */}
      <Text
        position={[0, -1.5, 0]}
        fontSize={0.2}
        color={node.role === "master" ? "#f59e0b" : "#6b7280"}
        anchorX="center"
        anchorY="middle"
      >
        {node.role.toUpperCase()}
      </Text>

      {/* Metrics overlay */}
      {showMetrics && (
        <Html position={[0, 2.5, 0]} center>
          <div className="bg-card/95 backdrop-blur-sm border border-border rounded-lg p-2 text-xs space-y-1 min-w-[120px]">
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">CPU:</span>
              <span className="font-medium">{cpuPercent.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Memory:</span>
              <span className="font-medium">{memoryPercent.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Pods:</span>
              <span className="font-medium">
                {node.pods.used}/{node.pods.total}
              </span>
            </div>
          </div>
        </Html>
      )}
    </group>
  );
}

/**
 * PodMesh - 3D representation of K8s pod
 */
function PodMesh({
  pod,
  nodePosition,
  isSelected,
  onClick,
}: {
  pod: KubernetesPod;
  nodePosition: { x: number; y: number; z: number };
  isSelected: boolean;
  onClick: () => void;
}) {
  const meshRef = useRef<any>(null);

  // Orbit around parent node
  useFrame((state) => {
    if (meshRef.current) {
      const time = state.clock.elapsedTime * 0.5;
      const radius = 3;
      const angle = (parseInt(pod.id.slice(-2), 16) / 256) * Math.PI * 2 + time;

      meshRef.current.position.x = nodePosition.x + Math.cos(angle) * radius;
      meshRef.current.position.y = nodePosition.y + Math.sin(time) * 0.5;
      meshRef.current.position.z = nodePosition.z + Math.sin(angle) * radius;

      // Rotate pod
      meshRef.current.rotation.y = time;
    }
  });

  const statusColor = {
    Running: "#10b981",
    Pending: "#f59e0b",
    Succeeded: "#3b82f6",
    Failed: "#ef4444",
    Unknown: "#6b7280",
  }[pod.status];

  return (
    <group>
      <mesh ref={meshRef} onClick={onClick}>
        <sphereGeometry args={[0.3, 16, 16]} />
        <meshStandardMaterial
          color={statusColor}
          metalness={0.5}
          roughness={0.5}
          emissive={statusColor}
          emissiveIntensity={isSelected ? 0.5 : 0.2}
        />
      </mesh>

      {isSelected && (
        <Html position={[0, 0, 0]} center>
          <div className="bg-card/95 backdrop-blur-sm border border-border rounded-lg p-2 text-xs space-y-1 min-w-[150px] -translate-x-1/2 -translate-y-full mb-4">
            <div className="font-medium text-primary">{pod.name}</div>
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Namespace:</span>
              <span>{pod.namespace}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Status:</span>
              <span style={{ color: statusColor }}>{pod.status}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Restarts:</span>
              <span>{pod.restarts}</span>
            </div>
          </div>
        </Html>
      )}
    </group>
  );
}

/**
 * ServiceConnections - Visualize service-to-pod connections
 */
function ServiceConnections({
  service,
  pods,
  nodes,
}: {
  service: any;
  pods: KubernetesPod[];
  nodes: KubernetesNode[];
}) {
  // Find pods matching service selector
  const matchingPods = pods.filter((pod) => {
    if (pod.namespace !== service.namespace) return false;
    return Object.entries(service.selector).every(
      ([key, value]) => pod.labels?.[key] === value
    );
  });

  if (matchingPods.length === 0) return null;

  return (
    <group>
      {matchingPods.map((pod) => {
        const node = nodes.find((n) => n.name === pod.nodeName);
        if (!node || !node.position) return null;

        // Service position (center of cluster)
        const servicePos = new Vector3(0, 5, 0);

        // Pod position (orbiting node)
        const time = Date.now() * 0.0005;
        const radius = 3;
        const angle =
          (parseInt(pod.id.slice(-2), 16) / 256) * Math.PI * 2 + time;
        const podPos = new Vector3(
          node.position.x + Math.cos(angle) * radius,
          node.position.y + Math.sin(time) * 0.5,
          node.position.z + Math.sin(angle) * radius
        );

        return (
          <line key={pod.id}>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={
                  new Float32Array([
                    servicePos.x,
                    servicePos.y,
                    servicePos.z,
                    podPos.x,
                    podPos.y,
                    podPos.z,
                  ])
                }
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color="#6366f1" opacity={0.3} transparent />
          </line>
        );
      })}
    </group>
  );
}
