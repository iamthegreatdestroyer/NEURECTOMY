/**
 * Model Router Cosmos - Forge×Foundry Innovation #3
 *
 * 3D visualization of model routing decisions through ensembles and
 * mixture-of-experts systems. Visualizes how requests flow through
 * multiple models, showing gating decisions, expert activations,
 * and routing confidence in an immersive cosmic space metaphor.
 *
 * @module ModelRouterCosmos
 * @category CrossDomain/Innovation
 */

import { EventEmitter } from "events";

// ============================================================================
// Types & Interfaces
// ============================================================================

/**
 * 3D position vector
 */
export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

/**
 * Model node in the cosmos
 */
export interface ModelNode {
  id: string;
  name: string;
  type: ModelNodeType;
  position: Vector3D;

  // Visual properties
  size: number;
  color: string;
  glowIntensity: number;
  rotationSpeed: number;

  // Model properties
  config: ModelNodeConfig;
  stats: ModelNodeStats;

  // Connections
  incomingRoutes: string[];
  outgoingRoutes: string[];
}

/**
 * Types of model nodes
 */
export type ModelNodeType =
  | "input" // Input gateway
  | "router" // Routing/gating layer
  | "expert" // Expert model in MoE
  | "base_model" // Standard model
  | "aggregator" // Output aggregation
  | "ensemble" // Ensemble combination
  | "output"; // Final output

/**
 * Model node configuration
 */
export interface ModelNodeConfig {
  modelType: string;
  parameters: number;
  specialty?: string;
  capacity?: number;
  temperature?: number;
}

/**
 * Model node statistics
 */
export interface ModelNodeStats {
  activationCount: number;
  averageLatency: number;
  averageConfidence: number;
  throughput: number;
  errorRate: number;
  loadFactor: number;
}

/**
 * Route between models
 */
export interface ModelRoute {
  id: string;
  sourceId: string;
  targetId: string;

  // Visual properties
  color: string;
  thickness: number;
  particleSpeed: number;
  particleDensity: number;

  // Route properties
  routeType: RouteType;
  weight: number;
  confidence: number;

  // Statistics
  stats: RouteStats;
}

/**
 * Route types
 */
export type RouteType =
  | "direct" // Direct connection
  | "gated" // Gated by router
  | "weighted" // Weighted combination
  | "conditional"; // Conditional routing

/**
 * Route statistics
 */
export interface RouteStats {
  requestCount: number;
  averageWeight: number;
  activationRate: number;
  lastActivated: number;
}

/**
 * Request flowing through the cosmos
 */
export interface RoutingRequest {
  id: string;
  input: RequestInput;
  timestamp: number;

  // Current state
  currentNodeId: string;
  path: PathStep[];

  // Routing decisions
  routingDecisions: RoutingDecision[];

  // Visual properties
  position: Vector3D;
  velocity: Vector3D;
  color: string;
  trailLength: number;
}

/**
 * Request input data
 */
export interface RequestInput {
  type: string;
  features: number[];
  metadata: Record<string, unknown>;
}

/**
 * Step in the request path
 */
export interface PathStep {
  nodeId: string;
  entryTime: number;
  exitTime: number;
  routeId: string;
  confidence: number;
}

/**
 * Routing decision made by a router
 */
export interface RoutingDecision {
  routerId: string;
  timestamp: number;

  // Expert scores
  expertScores: Map<string, number>;

  // Selected routes
  selectedRoutes: Array<{
    routeId: string;
    weight: number;
    confidence: number;
  }>;

  // Gating information
  gatingType: GatingType;
  topK?: number;
  threshold?: number;
}

/**
 * Gating mechanism types
 */
export type GatingType =
  | "softmax" // Softmax gating
  | "top_k" // Top-K selection
  | "threshold" // Threshold-based
  | "learned"; // Learned gating

/**
 * Cosmos configuration
 */
export interface CosmosConfig {
  // Layout
  layoutType: "radial" | "hierarchical" | "force_directed" | "custom";
  centerPosition: Vector3D;
  radius: number;
  layerSpacing: number;

  // Visual
  backgroundColor: string;
  starFieldEnabled: boolean;
  nebulaEnabled: boolean;
  connectionParticles: boolean;

  // Animation
  animationSpeed: number;
  requestTrailLength: number;
  glowEnabled: boolean;

  // Physics
  physicsEnabled: boolean;
  repulsionForce: number;
  attractionForce: number;
}

/**
 * Expert utilization analysis
 */
export interface ExpertUtilization {
  expertId: string;
  totalRequests: number;
  averageLoad: number;
  peakLoad: number;

  // Specialty analysis
  dominantInputTypes: string[];
  specializationScore: number;

  // Balance metrics
  utilizationBalance: number;
  overloadEvents: number;
}

/**
 * Routing pattern analysis
 */
export interface RoutingPatternAnalysis {
  // Common paths
  frequentPaths: Array<{
    path: string[];
    frequency: number;
    averageLatency: number;
  }>;

  // Bottlenecks
  bottlenecks: Array<{
    nodeId: string;
    congestionScore: number;
    recommendation: string;
  }>;

  // Dead routes
  deadRoutes: string[];

  // Load imbalance
  loadImbalance: number;
}

/**
 * Camera configuration
 */
export interface CosmosCamera {
  position: Vector3D;
  target: Vector3D;
  fov: number;
  near: number;
  far: number;
}

/**
 * Selection state
 */
export interface SelectionState {
  selectedNodeId: string | null;
  selectedRouteId: string | null;
  selectedRequestId: string | null;
  highlightedPath: string[];
}

// ============================================================================
// Model Router Cosmos Implementation
// ============================================================================

/**
 * Model Router Cosmos
 *
 * Immersive 3D visualization of model routing and mixture-of-experts
 * systems as a cosmic space where requests travel as particles
 * between model "planets".
 */
export class ModelRouterCosmos extends EventEmitter {
  private config: CosmosConfig;
  private nodes: Map<string, ModelNode>;
  private routes: Map<string, ModelRoute>;
  private activeRequests: Map<string, RoutingRequest>;
  private completedRequests: RoutingRequest[];
  private camera: CosmosCamera;
  private selection: SelectionState;
  private isSimulating: boolean;
  private animationFrame: number | null;
  private simulationTime: number;

  // Analysis cache
  private expertUtilization: Map<string, ExpertUtilization>;
  private routingPatterns: RoutingPatternAnalysis | null;

  constructor(config: Partial<CosmosConfig> = {}) {
    super();

    this.config = this.mergeConfig(config);
    this.nodes = new Map();
    this.routes = new Map();
    this.activeRequests = new Map();
    this.completedRequests = [];
    this.camera = this.createDefaultCamera();
    this.selection = this.createDefaultSelection();
    this.isSimulating = false;
    this.animationFrame = null;
    this.simulationTime = 0;
    this.expertUtilization = new Map();
    this.routingPatterns = null;
  }

  /**
   * Merge user config with defaults
   */
  private mergeConfig(config: Partial<CosmosConfig>): CosmosConfig {
    return {
      layoutType: config.layoutType ?? "radial",
      centerPosition: config.centerPosition ?? { x: 0, y: 0, z: 0 },
      radius: config.radius ?? 50,
      layerSpacing: config.layerSpacing ?? 30,
      backgroundColor: config.backgroundColor ?? "#0a0a1e",
      starFieldEnabled: config.starFieldEnabled ?? true,
      nebulaEnabled: config.nebulaEnabled ?? true,
      connectionParticles: config.connectionParticles ?? true,
      animationSpeed: config.animationSpeed ?? 1.0,
      requestTrailLength: config.requestTrailLength ?? 50,
      glowEnabled: config.glowEnabled ?? true,
      physicsEnabled: config.physicsEnabled ?? false,
      repulsionForce: config.repulsionForce ?? 100,
      attractionForce: config.attractionForce ?? 0.1,
    };
  }

  /**
   * Create default camera
   */
  private createDefaultCamera(): CosmosCamera {
    return {
      position: { x: 0, y: 100, z: 150 },
      target: { x: 0, y: 0, z: 0 },
      fov: 60,
      near: 0.1,
      far: 2000,
    };
  }

  /**
   * Create default selection state
   */
  private createDefaultSelection(): SelectionState {
    return {
      selectedNodeId: null,
      selectedRouteId: null,
      selectedRequestId: null,
      highlightedPath: [],
    };
  }

  // ============================================================================
  // Node Management
  // ============================================================================

  /**
   * Add a model node
   */
  addNode(config: Partial<ModelNode>): string {
    const node = this.createNode(config);
    this.nodes.set(node.id, node);

    this.applyLayout();

    this.emit("node:added", node);
    return node.id;
  }

  /**
   * Create node with defaults
   */
  private createNode(config: Partial<ModelNode>): ModelNode {
    const id = config.id || this.generateId("node");
    const type = config.type || "base_model";

    return {
      id,
      name: config.name || `Model ${id}`,
      type,
      position: config.position || { x: 0, y: 0, z: 0 },
      size: config.size ?? this.getDefaultNodeSize(type),
      color: config.color ?? this.getDefaultNodeColor(type),
      glowIntensity: config.glowIntensity ?? 0.5,
      rotationSpeed: config.rotationSpeed ?? 0.01,
      config: config.config || {
        modelType: "transformer",
        parameters: 0,
      },
      stats: config.stats || {
        activationCount: 0,
        averageLatency: 0,
        averageConfidence: 0,
        throughput: 0,
        errorRate: 0,
        loadFactor: 0,
      },
      incomingRoutes: config.incomingRoutes || [],
      outgoingRoutes: config.outgoingRoutes || [],
    };
  }

  /**
   * Get default node size
   */
  private getDefaultNodeSize(type: ModelNodeType): number {
    const sizes: Record<ModelNodeType, number> = {
      input: 8,
      router: 10,
      expert: 12,
      base_model: 15,
      aggregator: 10,
      ensemble: 18,
      output: 8,
    };
    return sizes[type] || 10;
  }

  /**
   * Get default node color
   */
  private getDefaultNodeColor(type: ModelNodeType): string {
    const colors: Record<ModelNodeType, string> = {
      input: "#4fc3f7", // Cyan
      router: "#ffd54f", // Yellow
      expert: "#81c784", // Green
      base_model: "#7986cb", // Blue
      aggregator: "#ba68c8", // Purple
      ensemble: "#ff8a65", // Orange
      output: "#4db6ac", // Teal
    };
    return colors[type] || "#ffffff";
  }

  /**
   * Remove a node
   */
  removeNode(nodeId: string): void {
    const node = this.nodes.get(nodeId);
    if (!node) return;

    // Remove associated routes
    for (const routeId of [...node.incomingRoutes, ...node.outgoingRoutes]) {
      this.removeRoute(routeId);
    }

    this.nodes.delete(nodeId);

    this.emit("node:removed", nodeId);
  }

  /**
   * Update node
   */
  updateNode(nodeId: string, updates: Partial<ModelNode>): void {
    const node = this.nodes.get(nodeId);
    if (!node) return;

    Object.assign(node, updates);

    this.emit("node:updated", node);
  }

  /**
   * Get node
   */
  getNode(nodeId: string): ModelNode | undefined {
    return this.nodes.get(nodeId);
  }

  /**
   * Get all nodes
   */
  getNodes(): ModelNode[] {
    return Array.from(this.nodes.values());
  }

  /**
   * Get nodes by type
   */
  getNodesByType(type: ModelNodeType): ModelNode[] {
    return Array.from(this.nodes.values()).filter((n) => n.type === type);
  }

  // ============================================================================
  // Route Management
  // ============================================================================

  /**
   * Add a route between nodes
   */
  addRoute(
    sourceId: string,
    targetId: string,
    config: Partial<ModelRoute> = {}
  ): string {
    const source = this.nodes.get(sourceId);
    const target = this.nodes.get(targetId);

    if (!source || !target) {
      throw new Error(`Invalid route: source or target node not found`);
    }

    const route = this.createRoute(sourceId, targetId, config);
    this.routes.set(route.id, route);

    // Update node connections
    source.outgoingRoutes.push(route.id);
    target.incomingRoutes.push(route.id);

    this.emit("route:added", route);
    return route.id;
  }

  /**
   * Create route with defaults
   */
  private createRoute(
    sourceId: string,
    targetId: string,
    config: Partial<ModelRoute>
  ): ModelRoute {
    const id = config.id || this.generateId("route");

    return {
      id,
      sourceId,
      targetId,
      color: config.color ?? "#ffffff",
      thickness: config.thickness ?? 2,
      particleSpeed: config.particleSpeed ?? 1,
      particleDensity: config.particleDensity ?? 5,
      routeType: config.routeType ?? "direct",
      weight: config.weight ?? 1,
      confidence: config.confidence ?? 1,
      stats: config.stats || {
        requestCount: 0,
        averageWeight: 0,
        activationRate: 0,
        lastActivated: 0,
      },
    };
  }

  /**
   * Remove a route
   */
  removeRoute(routeId: string): void {
    const route = this.routes.get(routeId);
    if (!route) return;

    // Update node connections
    const source = this.nodes.get(route.sourceId);
    const target = this.nodes.get(route.targetId);

    if (source) {
      source.outgoingRoutes = source.outgoingRoutes.filter(
        (id) => id !== routeId
      );
    }
    if (target) {
      target.incomingRoutes = target.incomingRoutes.filter(
        (id) => id !== routeId
      );
    }

    this.routes.delete(routeId);

    this.emit("route:removed", routeId);
  }

  /**
   * Update route
   */
  updateRoute(routeId: string, updates: Partial<ModelRoute>): void {
    const route = this.routes.get(routeId);
    if (!route) return;

    Object.assign(route, updates);

    this.emit("route:updated", route);
  }

  /**
   * Get route
   */
  getRoute(routeId: string): ModelRoute | undefined {
    return this.routes.get(routeId);
  }

  /**
   * Get all routes
   */
  getRoutes(): ModelRoute[] {
    return Array.from(this.routes.values());
  }

  // ============================================================================
  // Request Flow
  // ============================================================================

  /**
   * Submit a routing request
   */
  submitRequest(input: RequestInput): string {
    const request = this.createRequest(input);
    this.activeRequests.set(request.id, request);

    // Start at input node
    const inputNodes = this.getNodesByType("input");
    if (inputNodes.length > 0) {
      request.currentNodeId = inputNodes[0].id;
      request.position = { ...inputNodes[0].position };
    }

    this.emit("request:submitted", request);
    return request.id;
  }

  /**
   * Create routing request
   */
  private createRequest(input: RequestInput): RoutingRequest {
    const id = this.generateId("req");

    return {
      id,
      input,
      timestamp: Date.now(),
      currentNodeId: "",
      path: [],
      routingDecisions: [],
      position: { x: 0, y: 0, z: 0 },
      velocity: { x: 0, y: 0, z: 0 },
      color: this.getRandomRequestColor(),
      trailLength: this.config.requestTrailLength,
    };
  }

  /**
   * Get random request color
   */
  private getRandomRequestColor(): string {
    const colors = ["#ff6b6b", "#4ecdc4", "#ffe66d", "#c792ea", "#89ddff"];
    return colors[Math.floor(Math.random() * colors.length)];
  }

  /**
   * Route request through a router node
   */
  routeRequest(requestId: string, decision: RoutingDecision): void {
    const request = this.activeRequests.get(requestId);
    if (!request) return;

    request.routingDecisions.push(decision);

    // Update route statistics
    for (const selected of decision.selectedRoutes) {
      const route = this.routes.get(selected.routeId);
      if (route) {
        route.stats.requestCount++;
        route.stats.averageWeight =
          (route.stats.averageWeight * (route.stats.requestCount - 1) +
            selected.weight) /
          route.stats.requestCount;
        route.stats.lastActivated = Date.now();
      }
    }

    this.emit("request:routed", { requestId, decision });
  }

  /**
   * Move request to next node
   */
  advanceRequest(requestId: string, routeId: string): void {
    const request = this.activeRequests.get(requestId);
    const route = this.routes.get(routeId);

    if (!request || !route) return;

    const targetNode = this.nodes.get(route.targetId);
    if (!targetNode) return;

    // Record path step
    const step: PathStep = {
      nodeId: route.sourceId,
      entryTime:
        request.path.length > 0
          ? request.path[request.path.length - 1].exitTime
          : request.timestamp,
      exitTime: Date.now(),
      routeId,
      confidence: route.confidence,
    };
    request.path.push(step);

    // Update position
    request.currentNodeId = targetNode.id;

    // Calculate velocity toward target
    const dx = targetNode.position.x - request.position.x;
    const dy = targetNode.position.y - request.position.y;
    const dz = targetNode.position.z - request.position.z;
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

    if (distance > 0) {
      const speed = 2 * this.config.animationSpeed;
      request.velocity = {
        x: (dx / distance) * speed,
        y: (dy / distance) * speed,
        z: (dz / distance) * speed,
      };
    }

    // Update node stats
    targetNode.stats.activationCount++;

    this.emit("request:advanced", { requestId, routeId });
  }

  /**
   * Complete a request
   */
  completeRequest(requestId: string): void {
    const request = this.activeRequests.get(requestId);
    if (!request) return;

    // Final path step
    const lastStep: PathStep = {
      nodeId: request.currentNodeId,
      entryTime:
        request.path.length > 0
          ? request.path[request.path.length - 1].exitTime
          : request.timestamp,
      exitTime: Date.now(),
      routeId: "",
      confidence: 1,
    };
    request.path.push(lastStep);

    // Move to completed
    this.activeRequests.delete(requestId);
    this.completedRequests.push(request);

    // Limit completed requests history
    if (this.completedRequests.length > 1000) {
      this.completedRequests = this.completedRequests.slice(-500);
    }

    this.emit("request:completed", request);
  }

  /**
   * Get active requests
   */
  getActiveRequests(): RoutingRequest[] {
    return Array.from(this.activeRequests.values());
  }

  /**
   * Get completed requests
   */
  getCompletedRequests(): RoutingRequest[] {
    return [...this.completedRequests];
  }

  // ============================================================================
  // Layout
  // ============================================================================

  /**
   * Apply layout algorithm
   */
  applyLayout(): void {
    switch (this.config.layoutType) {
      case "radial":
        this.applyRadialLayout();
        break;
      case "hierarchical":
        this.applyHierarchicalLayout();
        break;
      case "force_directed":
        this.applyForceDirectedLayout();
        break;
    }

    this.emit("layout:applied");
  }

  /**
   * Apply radial layout
   */
  private applyRadialLayout(): void {
    const center = this.config.centerPosition;
    const radius = this.config.radius;

    // Group nodes by type (layers)
    const layers: ModelNodeType[] = [
      "input",
      "router",
      "expert",
      "base_model",
      "aggregator",
      "ensemble",
      "output",
    ];
    const nodesByLayer = new Map<ModelNodeType, ModelNode[]>();

    for (const type of layers) {
      nodesByLayer.set(type, this.getNodesByType(type));
    }

    // Position each layer in a ring
    let layerIndex = 0;
    for (const [type, nodes] of nodesByLayer) {
      if (nodes.length === 0) continue;

      const layerRadius = radius + layerIndex * this.config.layerSpacing;
      const angleStep = (2 * Math.PI) / nodes.length;

      nodes.forEach((node, i) => {
        const angle = i * angleStep;
        node.position = {
          x: center.x + layerRadius * Math.cos(angle),
          y: center.y + (Math.random() - 0.5) * 10, // Slight Y variation
          z: center.z + layerRadius * Math.sin(angle),
        };
      });

      layerIndex++;
    }
  }

  /**
   * Apply hierarchical layout
   */
  private applyHierarchicalLayout(): void {
    const center = this.config.centerPosition;
    const spacing = this.config.layerSpacing;

    // Compute depth of each node
    const depths = this.computeNodeDepths();
    const maxDepth = Math.max(0, ...depths.values());

    // Group by depth
    const nodesByDepth = new Map<number, ModelNode[]>();
    for (const node of this.nodes.values()) {
      const depth = depths.get(node.id) || 0;
      if (!nodesByDepth.has(depth)) {
        nodesByDepth.set(depth, []);
      }
      nodesByDepth.get(depth)!.push(node);
    }

    // Position nodes
    for (let depth = 0; depth <= maxDepth; depth++) {
      const nodes = nodesByDepth.get(depth) || [];
      const y = center.y + (depth - maxDepth / 2) * spacing;
      const width = nodes.length * 30;

      nodes.forEach((node, i) => {
        node.position = {
          x: center.x + (i - nodes.length / 2) * 30,
          y,
          z: center.z + (Math.random() - 0.5) * 20,
        };
      });
    }
  }

  /**
   * Apply force-directed layout
   */
  private applyForceDirectedLayout(): void {
    const iterations = 100;
    const repulsion = this.config.repulsionForce;
    const attraction = this.config.attractionForce;
    const damping = 0.9;

    // Initialize velocities
    const velocities = new Map<string, Vector3D>();
    for (const node of this.nodes.values()) {
      velocities.set(node.id, { x: 0, y: 0, z: 0 });
    }

    // Iterate
    for (let iter = 0; iter < iterations; iter++) {
      const temperature = 1 - iter / iterations;

      // Repulsion between all nodes
      for (const nodeA of this.nodes.values()) {
        for (const nodeB of this.nodes.values()) {
          if (nodeA.id === nodeB.id) continue;

          const dx = nodeA.position.x - nodeB.position.x;
          const dy = nodeA.position.y - nodeB.position.y;
          const dz = nodeA.position.z - nodeB.position.z;
          const distance = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.1;

          const force = repulsion / (distance * distance);

          const velA = velocities.get(nodeA.id)!;
          velA.x += (dx / distance) * force * temperature;
          velA.y += (dy / distance) * force * temperature;
          velA.z += (dz / distance) * force * temperature;
        }
      }

      // Attraction along routes
      for (const route of this.routes.values()) {
        const source = this.nodes.get(route.sourceId);
        const target = this.nodes.get(route.targetId);
        if (!source || !target) continue;

        const dx = target.position.x - source.position.x;
        const dy = target.position.y - source.position.y;
        const dz = target.position.z - source.position.z;
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

        const force = attraction * distance;

        const velSource = velocities.get(source.id)!;
        const velTarget = velocities.get(target.id)!;

        velSource.x += dx * force * temperature;
        velSource.y += dy * force * temperature;
        velSource.z += dz * force * temperature;

        velTarget.x -= dx * force * temperature;
        velTarget.y -= dy * force * temperature;
        velTarget.z -= dz * force * temperature;
      }

      // Apply velocities
      for (const node of this.nodes.values()) {
        const vel = velocities.get(node.id)!;

        node.position.x += vel.x;
        node.position.y += vel.y;
        node.position.z += vel.z;

        // Damping
        vel.x *= damping;
        vel.y *= damping;
        vel.z *= damping;
      }
    }

    // Center the layout
    this.centerLayout();
  }

  /**
   * Compute node depths
   */
  private computeNodeDepths(): Map<string, number> {
    const depths = new Map<string, number>();
    const visited = new Set<string>();

    // Start from input nodes
    const inputNodes = this.getNodesByType("input");
    const queue: Array<{ nodeId: string; depth: number }> = inputNodes.map(
      (n) => ({ nodeId: n.id, depth: 0 })
    );

    while (queue.length > 0) {
      const { nodeId, depth } = queue.shift()!;

      if (visited.has(nodeId)) {
        depths.set(nodeId, Math.max(depths.get(nodeId) || 0, depth));
        continue;
      }

      visited.add(nodeId);
      depths.set(nodeId, depth);

      const node = this.nodes.get(nodeId);
      if (!node) continue;

      for (const routeId of node.outgoingRoutes) {
        const route = this.routes.get(routeId);
        if (route) {
          queue.push({ nodeId: route.targetId, depth: depth + 1 });
        }
      }
    }

    return depths;
  }

  /**
   * Center the layout
   */
  private centerLayout(): void {
    if (this.nodes.size === 0) return;

    let sumX = 0,
      sumY = 0,
      sumZ = 0;
    for (const node of this.nodes.values()) {
      sumX += node.position.x;
      sumY += node.position.y;
      sumZ += node.position.z;
    }

    const center = this.config.centerPosition;
    const offsetX = center.x - sumX / this.nodes.size;
    const offsetY = center.y - sumY / this.nodes.size;
    const offsetZ = center.z - sumZ / this.nodes.size;

    for (const node of this.nodes.values()) {
      node.position.x += offsetX;
      node.position.y += offsetY;
      node.position.z += offsetZ;
    }
  }

  // ============================================================================
  // Simulation
  // ============================================================================

  /**
   * Start simulation
   */
  startSimulation(): void {
    if (this.isSimulating) return;

    this.isSimulating = true;
    this.startAnimationLoop();

    this.emit("simulation:started");
  }

  /**
   * Stop simulation
   */
  stopSimulation(): void {
    this.isSimulating = false;
    this.stopAnimationLoop();

    this.emit("simulation:stopped");
  }

  /**
   * Start animation loop
   */
  private startAnimationLoop(): void {
    if (this.animationFrame !== null) return;

    const animate = () => {
      if (!this.isSimulating) return;

      this.updateSimulation();
      this.animationFrame = requestAnimationFrame(animate);
    };

    this.animationFrame = requestAnimationFrame(animate);
  }

  /**
   * Stop animation loop
   */
  private stopAnimationLoop(): void {
    if (this.animationFrame !== null) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }

  /**
   * Update simulation frame
   */
  private updateSimulation(): void {
    this.simulationTime += (1 / 60) * this.config.animationSpeed;

    // Update node rotations
    for (const node of this.nodes.values()) {
      // Nodes slowly rotate - visual only
    }

    // Update request positions
    for (const request of this.activeRequests.values()) {
      this.updateRequestPosition(request);
    }

    // Update route particles
    for (const route of this.routes.values()) {
      // Particle animation would be handled by 3D renderer
    }

    this.emit("frame:updated", this.simulationTime);
  }

  /**
   * Update request position
   */
  private updateRequestPosition(request: RoutingRequest): void {
    const targetNode = this.nodes.get(request.currentNodeId);
    if (!targetNode) return;

    // Move toward target
    request.position.x += request.velocity.x;
    request.position.y += request.velocity.y;
    request.position.z += request.velocity.z;

    // Check if reached target
    const dx = targetNode.position.x - request.position.x;
    const dy = targetNode.position.y - request.position.y;
    const dz = targetNode.position.z - request.position.z;
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

    if (distance < 2) {
      // Snap to target
      request.position = { ...targetNode.position };
      request.velocity = { x: 0, y: 0, z: 0 };
    }
  }

  // ============================================================================
  // Analysis
  // ============================================================================

  /**
   * Analyze expert utilization
   */
  analyzeExpertUtilization(): Map<string, ExpertUtilization> {
    this.expertUtilization.clear();

    const expertNodes = this.getNodesByType("expert");

    for (const expert of expertNodes) {
      // Analyze completed requests that went through this expert
      const requestsThrough = this.completedRequests.filter((r) =>
        r.path.some((p) => p.nodeId === expert.id)
      );

      // Calculate utilization metrics
      const totalRequests = requestsThrough.length;
      const averageLoad = expert.stats.loadFactor;

      // Find dominant input types
      const inputTypeCounts = new Map<string, number>();
      for (const req of requestsThrough) {
        const type = req.input.type;
        inputTypeCounts.set(type, (inputTypeCounts.get(type) || 0) + 1);
      }

      const dominantInputTypes = Array.from(inputTypeCounts.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([type]) => type);

      // Specialization score
      const maxTypeCount = Math.max(0, ...inputTypeCounts.values());
      const specializationScore =
        totalRequests > 0 ? maxTypeCount / totalRequests : 0;

      const utilization: ExpertUtilization = {
        expertId: expert.id,
        totalRequests,
        averageLoad,
        peakLoad: averageLoad * 1.5, // Would track actual peak
        dominantInputTypes,
        specializationScore,
        utilizationBalance: 1 - specializationScore,
        overloadEvents: 0,
      };

      this.expertUtilization.set(expert.id, utilization);
    }

    this.emit("analysis:expertUtilization", this.expertUtilization);
    return this.expertUtilization;
  }

  /**
   * Analyze routing patterns
   */
  analyzeRoutingPatterns(): RoutingPatternAnalysis {
    // Find frequent paths
    const pathCounts = new Map<
      string,
      { count: number; latencies: number[] }
    >();

    for (const request of this.completedRequests) {
      const pathKey = request.path.map((p) => p.nodeId).join("->");

      if (!pathCounts.has(pathKey)) {
        pathCounts.set(pathKey, { count: 0, latencies: [] });
      }

      const pathData = pathCounts.get(pathKey)!;
      pathData.count++;

      const latency =
        request.path.length > 0
          ? request.path[request.path.length - 1].exitTime - request.timestamp
          : 0;
      pathData.latencies.push(latency);
    }

    const frequentPaths = Array.from(pathCounts.entries())
      .map(([path, data]) => ({
        path: path.split("->"),
        frequency: data.count,
        averageLatency:
          data.latencies.reduce((s, l) => s + l, 0) / data.latencies.length,
      }))
      .sort((a, b) => b.frequency - a.frequency)
      .slice(0, 10);

    // Find bottlenecks
    const nodeThroughputs = new Map<string, number>();
    for (const request of this.completedRequests) {
      for (const step of request.path) {
        nodeThroughputs.set(
          step.nodeId,
          (nodeThroughputs.get(step.nodeId) || 0) + 1
        );
      }
    }

    const avgThroughput =
      Array.from(nodeThroughputs.values()).reduce((s, t) => s + t, 0) /
      nodeThroughputs.size;

    const bottlenecks = Array.from(nodeThroughputs.entries())
      .filter(([_, throughput]) => throughput > avgThroughput * 1.5)
      .map(([nodeId, throughput]) => ({
        nodeId,
        congestionScore: throughput / avgThroughput,
        recommendation: "Consider adding parallel experts",
      }));

    // Find dead routes
    const deadRoutes = Array.from(this.routes.values())
      .filter((r) => r.stats.requestCount === 0)
      .map((r) => r.id);

    // Calculate load imbalance
    const loads = Array.from(this.nodes.values())
      .filter((n) => n.type === "expert")
      .map((n) => n.stats.activationCount);

    const avgLoad = loads.reduce((s, l) => s + l, 0) / loads.length || 1;
    const loadVariance =
      loads.reduce((s, l) => s + (l - avgLoad) ** 2, 0) / loads.length;
    const loadImbalance = Math.sqrt(loadVariance) / avgLoad;

    this.routingPatterns = {
      frequentPaths,
      bottlenecks,
      deadRoutes,
      loadImbalance,
    };

    this.emit("analysis:routingPatterns", this.routingPatterns);
    return this.routingPatterns;
  }

  // ============================================================================
  // Selection & Interaction
  // ============================================================================

  /**
   * Select a node
   */
  selectNode(nodeId: string | null): void {
    this.selection.selectedNodeId = nodeId;
    this.selection.selectedRouteId = null;
    this.selection.selectedRequestId = null;

    if (nodeId) {
      // Highlight paths through this node
      this.highlightPathsThrough(nodeId);
    } else {
      this.selection.highlightedPath = [];
    }

    this.emit("selection:changed", this.selection);
  }

  /**
   * Select a route
   */
  selectRoute(routeId: string | null): void {
    this.selection.selectedNodeId = null;
    this.selection.selectedRouteId = routeId;
    this.selection.selectedRequestId = null;
    this.selection.highlightedPath = [];

    this.emit("selection:changed", this.selection);
  }

  /**
   * Select a request
   */
  selectRequest(requestId: string | null): void {
    this.selection.selectedNodeId = null;
    this.selection.selectedRouteId = null;
    this.selection.selectedRequestId = requestId;

    if (requestId) {
      const request =
        this.activeRequests.get(requestId) ||
        this.completedRequests.find((r) => r.id === requestId);

      if (request) {
        this.selection.highlightedPath = request.path.map((p) => p.nodeId);
      }
    } else {
      this.selection.highlightedPath = [];
    }

    this.emit("selection:changed", this.selection);
  }

  /**
   * Highlight paths through a node
   */
  private highlightPathsThrough(nodeId: string): void {
    const paths: string[] = [];

    // Find all requests that went through this node
    for (const request of this.completedRequests) {
      if (request.path.some((p) => p.nodeId === nodeId)) {
        paths.push(...request.path.map((p) => p.nodeId));
      }
    }

    this.selection.highlightedPath = [...new Set(paths)];
  }

  /**
   * Get selection state
   */
  getSelection(): SelectionState {
    return { ...this.selection };
  }

  // ============================================================================
  // Camera
  // ============================================================================

  /**
   * Set camera
   */
  setCamera(camera: Partial<CosmosCamera>): void {
    Object.assign(this.camera, camera);
    this.emit("camera:changed", this.camera);
  }

  /**
   * Get camera
   */
  getCamera(): CosmosCamera {
    return { ...this.camera };
  }

  /**
   * Focus camera on node
   */
  focusOnNode(nodeId: string): void {
    const node = this.nodes.get(nodeId);
    if (!node) return;

    const offset = 50;
    this.camera.target = { ...node.position };
    this.camera.position = {
      x: node.position.x + offset,
      y: node.position.y + offset / 2,
      z: node.position.z + offset,
    };

    this.emit("camera:changed", this.camera);
  }

  /**
   * Reset camera to default view
   */
  resetCamera(): void {
    this.camera = this.createDefaultCamera();
    this.emit("camera:changed", this.camera);
  }

  // ============================================================================
  // Presets & Templates
  // ============================================================================

  /**
   * Load MoE (Mixture of Experts) preset
   */
  loadMoEPreset(numExperts: number = 8): void {
    this.clear();

    // Input node
    this.addNode({ type: "input", name: "Input" });

    // Router
    this.addNode({
      type: "router",
      name: "Gating Network",
      config: { modelType: "gating", parameters: 10000 },
    });

    // Experts
    const expertIds: string[] = [];
    for (let i = 0; i < numExperts; i++) {
      const id = this.addNode({
        type: "expert",
        name: `Expert ${i + 1}`,
        config: {
          modelType: "transformer",
          parameters: 1000000,
          specialty: `Specialty ${i + 1}`,
        },
      });
      expertIds.push(id);
    }

    // Aggregator
    this.addNode({ type: "aggregator", name: "Output Aggregator" });

    // Output
    this.addNode({ type: "output", name: "Output" });

    // Connect
    const nodes = this.getNodes();
    const input = nodes.find((n) => n.type === "input")!;
    const router = nodes.find((n) => n.type === "router")!;
    const aggregator = nodes.find((n) => n.type === "aggregator")!;
    const output = nodes.find((n) => n.type === "output")!;

    this.addRoute(input.id, router.id);

    for (const expertId of expertIds) {
      this.addRoute(router.id, expertId, { routeType: "gated" });
      this.addRoute(expertId, aggregator.id);
    }

    this.addRoute(aggregator.id, output.id);

    this.applyLayout();

    this.emit("preset:loaded", "moe");
  }

  /**
   * Load ensemble preset
   */
  loadEnsemblePreset(numModels: number = 5): void {
    this.clear();

    // Input
    this.addNode({ type: "input", name: "Input" });

    // Base models
    const modelIds: string[] = [];
    for (let i = 0; i < numModels; i++) {
      const id = this.addNode({
        type: "base_model",
        name: `Model ${i + 1}`,
        config: { modelType: "transformer", parameters: 500000 },
      });
      modelIds.push(id);
    }

    // Ensemble combiner
    this.addNode({ type: "ensemble", name: "Ensemble Combiner" });

    // Output
    this.addNode({ type: "output", name: "Output" });

    // Connect
    const nodes = this.getNodes();
    const input = nodes.find((n) => n.type === "input")!;
    const ensemble = nodes.find((n) => n.type === "ensemble")!;
    const output = nodes.find((n) => n.type === "output")!;

    for (const modelId of modelIds) {
      this.addRoute(input.id, modelId, { routeType: "direct" });
      this.addRoute(modelId, ensemble.id, { routeType: "weighted" });
    }

    this.addRoute(ensemble.id, output.id);

    this.applyLayout();

    this.emit("preset:loaded", "ensemble");
  }

  /**
   * Clear cosmos
   */
  clear(): void {
    this.nodes.clear();
    this.routes.clear();
    this.activeRequests.clear();
    this.completedRequests = [];
    this.selection = this.createDefaultSelection();

    this.emit("cosmos:cleared");
  }

  // ============================================================================
  // Utilities
  // ============================================================================

  /**
   * Generate unique ID
   */
  private generateId(prefix: string): string {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  /**
   * Get configuration
   */
  getConfig(): CosmosConfig {
    return { ...this.config };
  }

  /**
   * Update configuration
   */
  updateConfig(updates: Partial<CosmosConfig>): void {
    Object.assign(this.config, updates);
    this.emit("config:changed", this.config);
  }

  /**
   * Export cosmos state
   */
  exportState(): string {
    return JSON.stringify(
      {
        nodes: Array.from(this.nodes.values()),
        routes: Array.from(this.routes.values()),
        config: this.config,
        camera: this.camera,
      },
      null,
      2
    );
  }

  /**
   * Import cosmos state
   */
  importState(json: string): void {
    try {
      const state = JSON.parse(json);

      this.clear();

      for (const node of state.nodes) {
        this.nodes.set(node.id, node);
      }

      for (const route of state.routes) {
        this.routes.set(route.id, route);
      }

      if (state.config) {
        Object.assign(this.config, state.config);
      }

      if (state.camera) {
        Object.assign(this.camera, state.camera);
      }

      this.emit("state:imported");
    } catch (error) {
      this.emit("error", {
        type: "import_error",
        message: "Invalid state JSON",
      });
    }
  }

  /**
   * Dispose
   */
  dispose(): void {
    this.stopSimulation();
    this.clear();
    this.removeAllListeners();
  }
}

// ============================================================================
// Factory
// ============================================================================

/**
 * Create Model Router Cosmos
 */
export function createModelRouterCosmos(
  config?: Partial<CosmosConfig>
): ModelRouterCosmos {
  return new ModelRouterCosmos(config);
}
