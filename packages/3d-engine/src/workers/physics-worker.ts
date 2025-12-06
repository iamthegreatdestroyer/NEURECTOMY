/**
 * @fileoverview Web Worker for physics calculations
 * @module @neurectomy/3d-engine/workers/physics-worker
 *
 * Offloads expensive force-directed layout and collision detection
 * to a separate thread using transferable ArrayBuffers for zero-copy
 * data transfer between main thread and worker.
 */

/// <reference lib="webworker" />

// Worker-safe message types
interface PhysicsNode {
  id: string;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  mass: number;
  fixed: boolean;
}

interface PhysicsEdge {
  sourceIndex: number;
  targetIndex: number;
  strength: number;
  idealLength: number;
}

interface PhysicsConfig {
  repulsion: number;
  attraction: number;
  damping: number;
  theta: number; // Barnes-Hut approximation threshold
  maxVelocity: number;
  gravityCenterStrength: number;
  collisionRadius: number;
  timestep: number;
}

interface InitMessage {
  type: "init";
  nodeCount: number;
  edgeCount: number;
  config: PhysicsConfig;
}

interface UpdateNodesMessage {
  type: "update-nodes";
  positions: ArrayBuffer; // Float32Array: [x, y, z, vx, vy, vz, mass, fixed] * nodeCount
  nodeCount: number;
}

interface UpdateEdgesMessage {
  type: "update-edges";
  edges: ArrayBuffer; // Uint32Array: [sourceIndex, targetIndex] * edgeCount + Float32Array: [strength, idealLength] * edgeCount
  edgeCount: number;
}

interface StepMessage {
  type: "step";
  iterations: number;
  returnVelocities: boolean;
}

interface ConfigMessage {
  type: "config";
  config: Partial<PhysicsConfig>;
}

interface StopMessage {
  type: "stop";
}

interface PauseMessage {
  type: "pause";
}

interface ResumeMessage {
  type: "resume";
}

type WorkerMessage =
  | InitMessage
  | UpdateNodesMessage
  | UpdateEdgesMessage
  | StepMessage
  | ConfigMessage
  | StopMessage
  | PauseMessage
  | ResumeMessage;

interface StepResult {
  type: "step-result";
  positions: ArrayBuffer;
  velocities?: ArrayBuffer;
  energy: number;
  converged: boolean;
  iterationsRun: number;
}

interface ErrorResult {
  type: "error";
  message: string;
}

interface ReadyResult {
  type: "ready";
}

type WorkerResult = StepResult | ErrorResult | ReadyResult;

// Physics state
let config: PhysicsConfig = {
  repulsion: 10000,
  attraction: 0.001,
  damping: 0.9,
  theta: 0.5,
  maxVelocity: 50,
  gravityCenterStrength: 0.0001,
  collisionRadius: 10,
  timestep: 1,
};

let positions: Float32Array | null = null;
let velocities: Float32Array | null = null;
let masses: Float32Array | null = null;
let fixed: Uint8Array | null = null;
let edgeSourceIndices: Uint32Array | null = null;
let edgeTargetIndices: Uint32Array | null = null;
let edgeStrengths: Float32Array | null = null;
let edgeIdealLengths: Float32Array | null = null;
let nodeCount = 0;
let edgeCount = 0;
let isPaused = false;

// Barnes-Hut octree for O(n log n) force calculation
interface OctreeNode {
  centerOfMass: { x: number; y: number; z: number };
  totalMass: number;
  bounds: {
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
    minZ: number;
    maxZ: number;
  };
  children: (OctreeNode | null)[];
  nodeIndex: number | null; // Leaf node index or null for internal nodes
}

/**
 * Build Barnes-Hut octree for efficient force calculation
 */
function buildOctree(): OctreeNode | null {
  if (!positions || nodeCount === 0) return null;

  // Find bounds
  let minX = Infinity,
    maxX = -Infinity;
  let minY = Infinity,
    maxY = -Infinity;
  let minZ = Infinity,
    maxZ = -Infinity;

  for (let i = 0; i < nodeCount; i++) {
    const idx = i * 3;
    minX = Math.min(minX, positions[idx]!);
    maxX = Math.max(maxX, positions[idx]!);
    minY = Math.min(minY, positions[idx + 1]!);
    maxY = Math.max(maxY, positions[idx + 1]!);
    minZ = Math.min(minZ, positions[idx + 2]!);
    maxZ = Math.max(maxZ, positions[idx + 2]!);
  }

  // Add padding
  const padding = 1;
  minX -= padding;
  maxX += padding;
  minY -= padding;
  maxY += padding;
  minZ -= padding;
  maxZ += padding;

  const root: OctreeNode = {
    centerOfMass: { x: 0, y: 0, z: 0 },
    totalMass: 0,
    bounds: { minX, maxX, minY, maxY, minZ, maxZ },
    children: [null, null, null, null, null, null, null, null],
    nodeIndex: null,
  };

  // Insert all nodes
  for (let i = 0; i < nodeCount; i++) {
    insertIntoOctree(root, i);
  }

  // Calculate centers of mass
  calculateCenterOfMass(root);

  return root;
}

function insertIntoOctree(node: OctreeNode, nodeIndex: number): void {
  if (!positions || !masses) return;

  const { minX, maxX, minY, maxY, minZ, maxZ } = node.bounds;
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;
  const midZ = (minZ + maxZ) / 2;

  const idx = nodeIndex * 3;
  const x = positions[idx]!;
  const y = positions[idx + 1]!;
  const z = positions[idx + 2]!;

  // Check if this is an empty leaf
  if (node.nodeIndex === null && node.children.every((c) => c === null)) {
    node.nodeIndex = nodeIndex;
    return;
  }

  // If this was a leaf with a node, we need to push it down
  if (node.nodeIndex !== null) {
    const existingIndex = node.nodeIndex;
    node.nodeIndex = null;
    insertIntoOctree(node, existingIndex);
  }

  // Determine which octant
  const octant = (x > midX ? 1 : 0) + (y > midY ? 2 : 0) + (z > midZ ? 4 : 0);

  if (node.children[octant] === null) {
    node.children[octant] = {
      centerOfMass: { x: 0, y: 0, z: 0 },
      totalMass: 0,
      bounds: {
        minX: octant & 1 ? midX : minX,
        maxX: octant & 1 ? maxX : midX,
        minY: octant & 2 ? midY : minY,
        maxY: octant & 2 ? maxY : midY,
        minZ: octant & 4 ? midZ : minZ,
        maxZ: octant & 4 ? maxZ : midZ,
      },
      children: [null, null, null, null, null, null, null, null],
      nodeIndex: null,
    };
  }

  insertIntoOctree(node.children[octant]!, nodeIndex);
}

function calculateCenterOfMass(node: OctreeNode): void {
  if (!positions || !masses) return;

  // Leaf node
  if (node.nodeIndex !== null) {
    const idx = node.nodeIndex * 3;
    node.centerOfMass.x = positions[idx]!;
    node.centerOfMass.y = positions[idx + 1]!;
    node.centerOfMass.z = positions[idx + 2]!;
    node.totalMass = masses[node.nodeIndex]!;
    return;
  }

  // Internal node
  let totalMass = 0;
  let cx = 0,
    cy = 0,
    cz = 0;

  for (const child of node.children) {
    if (child) {
      calculateCenterOfMass(child);
      totalMass += child.totalMass;
      cx += child.centerOfMass.x * child.totalMass;
      cy += child.centerOfMass.y * child.totalMass;
      cz += child.centerOfMass.z * child.totalMass;
    }
  }

  if (totalMass > 0) {
    node.centerOfMass.x = cx / totalMass;
    node.centerOfMass.y = cy / totalMass;
    node.centerOfMass.z = cz / totalMass;
  }
  node.totalMass = totalMass;
}

/**
 * Calculate repulsion force using Barnes-Hut approximation
 */
function calculateRepulsionBarnesHut(
  nodeIndex: number,
  octree: OctreeNode,
  fx: Float32Array,
  fy: Float32Array,
  fz: Float32Array
): void {
  if (!positions || !masses) return;

  const idx = nodeIndex * 3;
  const x = positions[idx]!;
  const y = positions[idx + 1]!;
  const z = positions[idx + 2]!;

  const stack: OctreeNode[] = [octree];

  while (stack.length > 0) {
    const node = stack.pop()!;

    if (node.totalMass === 0) continue;

    // Leaf node with this node itself
    if (node.nodeIndex === nodeIndex) continue;

    const dx = node.centerOfMass.x - x;
    const dy = node.centerOfMass.y - y;
    const dz = node.centerOfMass.z - z;
    const distSq = dx * dx + dy * dy + dz * dz + 0.001; // Softening
    const dist = Math.sqrt(distSq);

    // Calculate width of the node
    const width = node.bounds.maxX - node.bounds.minX;

    // Barnes-Hut criterion: if width/dist < theta, use approximation
    if (node.nodeIndex !== null || width / dist < config.theta) {
      // Use center of mass approximation
      const force =
        (-config.repulsion * masses[nodeIndex]! * node.totalMass) / distSq;
      const forceX = (force * dx) / dist;
      const forceY = (force * dy) / dist;
      const forceZ = (force * dz) / dist;

      fx[nodeIndex] = (fx[nodeIndex] ?? 0) + forceX;
      fy[nodeIndex] = (fy[nodeIndex] ?? 0) + forceY;
      fz[nodeIndex] = (fz[nodeIndex] ?? 0) + forceZ;
    } else {
      // Node too close, recurse into children
      for (const child of node.children) {
        if (child) stack.push(child);
      }
    }
  }
}

/**
 * Run physics simulation step
 */
function step(iterations: number): { energy: number; converged: boolean } {
  if (!positions || !velocities || !masses || !fixed) {
    return { energy: 0, converged: true };
  }

  let totalEnergy = 0;

  for (let iter = 0; iter < iterations; iter++) {
    // Allocate force arrays
    const fx = new Float32Array(nodeCount);
    const fy = new Float32Array(nodeCount);
    const fz = new Float32Array(nodeCount);

    // Build octree for Barnes-Hut
    const octree = buildOctree();

    // Calculate repulsion forces using Barnes-Hut
    if (octree) {
      for (let i = 0; i < nodeCount; i++) {
        calculateRepulsionBarnesHut(i, octree, fx, fy, fz);
      }
    }

    // Calculate attraction forces from edges
    if (
      edgeSourceIndices &&
      edgeTargetIndices &&
      edgeStrengths &&
      edgeIdealLengths
    ) {
      for (let e = 0; e < edgeCount; e++) {
        const source = edgeSourceIndices[e]!;
        const target = edgeTargetIndices[e]!;
        const strength = edgeStrengths[e]!;
        const idealLength = edgeIdealLengths[e]!;

        const si = source * 3;
        const ti = target * 3;

        const dx = positions[ti]! - positions[si]!;
        const dy = positions[ti + 1]! - positions[si + 1]!;
        const dz = positions[ti + 2]! - positions[si + 2]!;

        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.001;
        const displacement = dist - idealLength;

        const force = config.attraction * strength * displacement;
        const forceX = (force * dx) / dist;
        const forceY = (force * dy) / dist;
        const forceZ = (force * dz) / dist;

        fx[source] = (fx[source] ?? 0) + forceX;
        fy[source] = (fy[source] ?? 0) + forceY;
        fz[source] = (fz[source] ?? 0) + forceZ;
        fx[target] = (fx[target] ?? 0) - forceX;
        fy[target] = (fy[target] ?? 0) - forceY;
        fz[target] = (fz[target] ?? 0) - forceZ;
      }
    }

    // Apply gravity toward center
    for (let i = 0; i < nodeCount; i++) {
      const idx = i * 3;
      fx[i] =
        (fx[i] ?? 0) -
        positions[idx]! * config.gravityCenterStrength * masses[i]!;
      fy[i] =
        (fy[i] ?? 0) -
        positions[idx + 1]! * config.gravityCenterStrength * masses[i]!;
      fz[i] =
        (fz[i] ?? 0) -
        positions[idx + 2]! * config.gravityCenterStrength * masses[i]!;
    }

    // Update velocities and positions
    totalEnergy = 0;
    for (let i = 0; i < nodeCount; i++) {
      if (fixed[i]) continue;

      const idx = i * 3;
      const mass = masses[i]!;

      // Update velocity
      velocities[idx] =
        (velocities[idx]! + ((fx[i] ?? 0) / mass) * config.timestep) *
        config.damping;
      velocities[idx + 1] =
        (velocities[idx + 1]! + ((fy[i] ?? 0) / mass) * config.timestep) *
        config.damping;
      velocities[idx + 2] =
        (velocities[idx + 2]! + ((fz[i] ?? 0) / mass) * config.timestep) *
        config.damping;

      // Clamp velocity
      const speed = Math.sqrt(
        velocities[idx]! ** 2 +
          velocities[idx + 1]! ** 2 +
          velocities[idx + 2]! ** 2
      );

      if (speed > config.maxVelocity) {
        const scale = config.maxVelocity / speed;
        velocities[idx] = velocities[idx]! * scale;
        velocities[idx + 1] = velocities[idx + 1]! * scale;
        velocities[idx + 2] = velocities[idx + 2]! * scale;
      }

      // Update position
      positions[idx] = positions[idx]! + velocities[idx]! * config.timestep;
      positions[idx + 1] =
        positions[idx + 1]! + velocities[idx + 1]! * config.timestep;
      positions[idx + 2] =
        positions[idx + 2]! + velocities[idx + 2]! * config.timestep;

      // Calculate energy
      totalEnergy +=
        0.5 *
        mass *
        (velocities[idx]! ** 2 +
          velocities[idx + 1]! ** 2 +
          velocities[idx + 2]! ** 2);
    }
  }

  const converged = totalEnergy < 0.001 * nodeCount;

  return { energy: totalEnergy, converged };
}

/**
 * Handle incoming messages
 */
self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;

  try {
    switch (message.type) {
      case "init": {
        nodeCount = message.nodeCount;
        edgeCount = message.edgeCount;
        config = { ...config, ...message.config };

        // Allocate arrays
        positions = new Float32Array(nodeCount * 3);
        velocities = new Float32Array(nodeCount * 3);
        masses = new Float32Array(nodeCount);
        fixed = new Uint8Array(nodeCount);

        if (edgeCount > 0) {
          edgeSourceIndices = new Uint32Array(edgeCount);
          edgeTargetIndices = new Uint32Array(edgeCount);
          edgeStrengths = new Float32Array(edgeCount);
          edgeIdealLengths = new Float32Array(edgeCount);
        }

        // Initialize masses to 1
        masses.fill(1);

        const result: ReadyResult = { type: "ready" };
        self.postMessage(result);
        break;
      }

      case "update-nodes": {
        const data = new Float32Array(message.positions);
        nodeCount = message.nodeCount;

        // Reallocate if needed
        if (!positions || positions.length < nodeCount * 3) {
          positions = new Float32Array(nodeCount * 3);
          velocities = new Float32Array(nodeCount * 3);
          masses = new Float32Array(nodeCount);
          fixed = new Uint8Array(nodeCount);
        }

        // Copy data: [x, y, z, vx, vy, vz, mass, fixed] per node
        for (let i = 0; i < nodeCount; i++) {
          const srcIdx = i * 8;
          const posIdx = i * 3;

          positions![posIdx] = data[srcIdx]!;
          positions![posIdx + 1] = data[srcIdx + 1]!;
          positions![posIdx + 2] = data[srcIdx + 2]!;
          velocities![posIdx] = data[srcIdx + 3]!;
          velocities![posIdx + 1] = data[srcIdx + 4]!;
          velocities![posIdx + 2] = data[srcIdx + 5]!;
          masses![i] = data[srcIdx + 6]!;
          fixed![i] = data[srcIdx + 7] ? 1 : 0;
        }
        break;
      }

      case "update-edges": {
        edgeCount = message.edgeCount;
        const data = new DataView(message.edges);

        // Reallocate if needed
        if (!edgeSourceIndices || edgeSourceIndices.length < edgeCount) {
          edgeSourceIndices = new Uint32Array(edgeCount);
          edgeTargetIndices = new Uint32Array(edgeCount);
          edgeStrengths = new Float32Array(edgeCount);
          edgeIdealLengths = new Float32Array(edgeCount);
        }

        // Copy edge data
        // Format: [sourceIndex, targetIndex] (uint32) + [strength, idealLength] (float32)
        const indexBytes = edgeCount * 2 * 4; // uint32 pairs
        for (let i = 0; i < edgeCount; i++) {
          edgeSourceIndices![i] = data.getUint32(i * 8, true);
          edgeTargetIndices![i] = data.getUint32(i * 8 + 4, true);
        }

        for (let i = 0; i < edgeCount; i++) {
          edgeStrengths![i] = data.getFloat32(indexBytes + i * 8, true);
          edgeIdealLengths![i] = data.getFloat32(indexBytes + i * 8 + 4, true);
        }
        break;
      }

      case "step": {
        if (isPaused) {
          const result: StepResult = {
            type: "step-result",
            positions: new ArrayBuffer(0),
            energy: 0,
            converged: true,
            iterationsRun: 0,
          };
          self.postMessage(result);
          return;
        }

        const { energy, converged } = step(message.iterations);

        // Prepare result with transferable positions
        const positionsCopy = positions!.slice().buffer;
        const result: StepResult = {
          type: "step-result",
          positions: positionsCopy,
          energy,
          converged,
          iterationsRun: message.iterations,
        };

        if (message.returnVelocities && velocities) {
          result.velocities = velocities.slice().buffer;
          self.postMessage(result, [positionsCopy, result.velocities]);
        } else {
          self.postMessage(result, [positionsCopy]);
        }
        break;
      }

      case "config": {
        config = { ...config, ...message.config };
        break;
      }

      case "pause": {
        isPaused = true;
        break;
      }

      case "resume": {
        isPaused = false;
        break;
      }

      case "stop": {
        // Clean up
        positions = null;
        velocities = null;
        masses = null;
        fixed = null;
        edgeSourceIndices = null;
        edgeTargetIndices = null;
        edgeStrengths = null;
        edgeIdealLengths = null;
        nodeCount = 0;
        edgeCount = 0;
        self.close();
        break;
      }
    }
  } catch (error) {
    const result: ErrorResult = {
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    };
    self.postMessage(result);
  }
};

// Export nothing - this is a worker script
export {};
