/**
 * Multi-Fidelity Swarm Twins POC
 *
 * Dynamically allocates computational fidelity across swarm agents based on
 * importance, resource constraints, and predictive accuracy requirements.
 *
 * Key Innovations:
 * - Dynamic fidelity allocation based on importance estimation
 * - Multi-resolution simulation (coarse, medium, fine)
 * - Resource-aware scheduling with priority queues
 * - Adaptive fidelity switching based on prediction error
 * - Swarm coordination with heterogeneous models
 *
 * Research Foundations:
 * - Peherstorfer et al. (2018): Survey of multifidelity methods
 * - Kennedy & O'Hagan (2000): Predicting the output from a complex computer code
 * - Reynolds (1987): Flocks, herds and schools: A distributed behavioral model
 * - Dorigo & Stützle (2004): Ant Colony Optimization
 *
 * @elite-agents @VELOCITY @ARCHITECT @APEX
 */

import { cloneDeep } from "lodash";

// ============================================================================
// Type Definitions
// ============================================================================

type AgentId = string;
type Timestamp = number;

enum FidelityLevel {
  COARSE = "coarse",
  MEDIUM = "medium",
  FINE = "fine",
}

enum AgentRole {
  SCOUT = "scout",
  WORKER = "worker",
  COORDINATOR = "coordinator",
  SENTINEL = "sentinel",
}

interface Vector3D {
  x: number;
  y: number;
  z: number;
}

interface AgentState {
  id: AgentId;
  position: Vector3D;
  velocity: Vector3D;
  role: AgentRole;
  fidelity: FidelityLevel;
  importance: number; // 0-1
  computeCost: number;
  predictionError: number;
  lastUpdate: Timestamp;
}

interface FidelityConfig {
  level: FidelityLevel;
  updateFrequency: number; // Hz
  physicsSteps: number;
  computeCost: number;
  accuracy: number;
}

interface SwarmMetrics {
  totalAgents: number;
  fidelityDistribution: Map<FidelityLevel, number>;
  totalComputeCost: number;
  averageAccuracy: number;
  resourceUtilization: number;
}

interface ImportanceFactors {
  distanceToTarget: number;
  velocityMagnitude: number;
  roleMultiplier: number;
  neighborDensity: number;
  predictionError: number;
}

// ============================================================================
// Fidelity Configurations
// ============================================================================

const FIDELITY_CONFIGS: Record<FidelityLevel, FidelityConfig> = {
  [FidelityLevel.COARSE]: {
    level: FidelityLevel.COARSE,
    updateFrequency: 10, // 10 Hz
    physicsSteps: 1,
    computeCost: 1.0,
    accuracy: 0.6,
  },
  [FidelityLevel.MEDIUM]: {
    level: FidelityLevel.MEDIUM,
    updateFrequency: 30, // 30 Hz
    physicsSteps: 5,
    computeCost: 3.0,
    accuracy: 0.85,
  },
  [FidelityLevel.FINE]: {
    level: FidelityLevel.FINE,
    updateFrequency: 60, // 60 Hz
    physicsSteps: 10,
    computeCost: 10.0,
    accuracy: 0.98,
  },
};

// ============================================================================
// Importance Estimator
// ============================================================================

class ImportanceEstimator {
  private targetPosition: Vector3D;
  private importanceHistory: Map<AgentId, number[]>;

  constructor(targetPosition: Vector3D) {
    this.targetPosition = targetPosition;
    this.importanceHistory = new Map();
  }

  /**
   * Estimate agent importance for fidelity allocation
   */
  estimateImportance(agent: AgentState, neighbors: AgentState[]): number {
    const factors = this.computeFactors(agent, neighbors);

    // Weighted importance score
    const importance =
      factors.distanceToTarget * 0.3 +
      factors.velocityMagnitude * 0.2 +
      factors.roleMultiplier * 0.25 +
      factors.neighborDensity * 0.15 +
      factors.predictionError * 0.1;

    // Track history
    const history = this.importanceHistory.get(agent.id) ?? [];
    history.push(importance);
    if (history.length > 10) history.shift();
    this.importanceHistory.set(agent.id, history);

    return Math.max(0, Math.min(1, importance));
  }

  private computeFactors(
    agent: AgentState,
    neighbors: AgentState[]
  ): ImportanceFactors {
    // Distance to target (closer = more important)
    const distance = this.distance(agent.position, this.targetPosition);
    const maxDistance = 100;
    const distanceToTarget = 1 - Math.min(distance / maxDistance, 1);

    // Velocity magnitude (faster = more important)
    const speed = Math.sqrt(
      agent.velocity.x ** 2 + agent.velocity.y ** 2 + agent.velocity.z ** 2
    );
    const maxSpeed = 10;
    const velocityMagnitude = Math.min(speed / maxSpeed, 1);

    // Role multiplier
    const roleMultipliers: Record<AgentRole, number> = {
      [AgentRole.SCOUT]: 1.0,
      [AgentRole.WORKER]: 0.7,
      [AgentRole.COORDINATOR]: 0.9,
      [AgentRole.SENTINEL]: 0.6,
    };
    const roleMultiplier = roleMultipliers[agent.role];

    // Neighbor density (more neighbors = more important for coordination)
    const nearbyNeighbors = neighbors.filter(
      (n) => this.distance(agent.position, n.position) < 10
    ).length;
    const neighborDensity = Math.min(nearbyNeighbors / 5, 1);

    // Prediction error (higher error = needs more fidelity)
    const predictionError = agent.predictionError;

    return {
      distanceToTarget,
      velocityMagnitude,
      roleMultiplier,
      neighborDensity,
      predictionError,
    };
  }

  private distance(p1: Vector3D, p2: Vector3D): number {
    return Math.sqrt(
      (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2
    );
  }

  setTargetPosition(position: Vector3D): void {
    this.targetPosition = position;
  }

  getImportanceTrend(agentId: AgentId): number {
    const history = this.importanceHistory.get(agentId);
    if (!history || history.length < 2) return 0;

    // Compute trend (increasing or decreasing)
    const recent = history.slice(-3);
    const earlier = history.slice(-6, -3);

    if (earlier.length === 0) return 0;

    const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
    const earlierAvg = earlier.reduce((a, b) => a + b, 0) / earlier.length;

    return recentAvg - earlierAvg;
  }
}

// ============================================================================
// Fidelity Allocator
// ============================================================================

class FidelityAllocator {
  private estimator: ImportanceEstimator;
  private totalComputeBudget: number;
  private fidelityThresholds: {
    fine: number;
    medium: number;
  };

  constructor(
    estimator: ImportanceEstimator,
    totalComputeBudget: number = 100
  ) {
    this.estimator = estimator;
    this.totalComputeBudget = totalComputeBudget;
    this.fidelityThresholds = {
      fine: 0.7,
      medium: 0.4,
    };
  }

  /**
   * Allocate fidelity levels to agents
   */
  allocateFidelity(agents: AgentState[]): Map<AgentId, FidelityLevel> {
    const allocation = new Map<AgentId, FidelityLevel>();

    // Compute importance for all agents
    const importanceScores = agents.map((agent) => ({
      agent,
      importance: this.estimator.estimateImportance(
        agent,
        agents.filter((a) => a.id !== agent.id)
      ),
    }));

    // Sort by importance (descending)
    importanceScores.sort((a, b) => b.importance - a.importance);

    // Allocate greedily with budget constraint
    let remainingBudget = this.totalComputeBudget;

    for (const { agent, importance } of importanceScores) {
      let assignedFidelity = FidelityLevel.COARSE;

      // Try to assign highest fidelity that fits budget
      if (importance >= this.fidelityThresholds.fine) {
        const fineCost = FIDELITY_CONFIGS[FidelityLevel.FINE].computeCost;
        if (remainingBudget >= fineCost) {
          assignedFidelity = FidelityLevel.FINE;
          remainingBudget -= fineCost;
        } else if (
          remainingBudget >= FIDELITY_CONFIGS[FidelityLevel.MEDIUM].computeCost
        ) {
          assignedFidelity = FidelityLevel.MEDIUM;
          remainingBudget -= FIDELITY_CONFIGS[FidelityLevel.MEDIUM].computeCost;
        } else {
          assignedFidelity = FidelityLevel.COARSE;
          remainingBudget -= FIDELITY_CONFIGS[FidelityLevel.COARSE].computeCost;
        }
      } else if (importance >= this.fidelityThresholds.medium) {
        const mediumCost = FIDELITY_CONFIGS[FidelityLevel.MEDIUM].computeCost;
        if (remainingBudget >= mediumCost) {
          assignedFidelity = FidelityLevel.MEDIUM;
          remainingBudget -= mediumCost;
        } else {
          assignedFidelity = FidelityLevel.COARSE;
          remainingBudget -= FIDELITY_CONFIGS[FidelityLevel.COARSE].computeCost;
        }
      } else {
        assignedFidelity = FidelityLevel.COARSE;
        remainingBudget -= FIDELITY_CONFIGS[FidelityLevel.COARSE].computeCost;
      }

      allocation.set(agent.id, assignedFidelity);
    }

    return allocation;
  }

  /**
   * Adjust fidelity based on prediction error
   */
  adjustForError(
    currentAllocation: Map<AgentId, FidelityLevel>,
    agents: AgentState[]
  ): Map<AgentId, FidelityLevel> {
    const adjusted = new Map(currentAllocation);

    for (const agent of agents) {
      const currentFidelity =
        currentAllocation.get(agent.id) ?? FidelityLevel.COARSE;

      // If error is high, try to upgrade fidelity
      if (agent.predictionError > 0.15) {
        if (currentFidelity === FidelityLevel.COARSE) {
          adjusted.set(agent.id, FidelityLevel.MEDIUM);
        } else if (currentFidelity === FidelityLevel.MEDIUM) {
          adjusted.set(agent.id, FidelityLevel.FINE);
        }
      }

      // If error is low, try to downgrade fidelity (save compute)
      if (agent.predictionError < 0.05) {
        if (currentFidelity === FidelityLevel.FINE) {
          adjusted.set(agent.id, FidelityLevel.MEDIUM);
        } else if (currentFidelity === FidelityLevel.MEDIUM) {
          adjusted.set(agent.id, FidelityLevel.COARSE);
        }
      }
    }

    return adjusted;
  }

  setComputeBudget(budget: number): void {
    this.totalComputeBudget = budget;
  }

  setFidelityThresholds(fine: number, medium: number): void {
    this.fidelityThresholds = { fine, medium };
  }
}

// ============================================================================
// Swarm Coordinator
// ============================================================================

class SwarmCoordinator {
  private agents: Map<AgentId, AgentState>;
  private allocator: FidelityAllocator;
  private currentAllocation: Map<AgentId, FidelityLevel>;
  private simulationTime: number;

  constructor(allocator: FidelityAllocator) {
    this.agents = new Map();
    this.allocator = allocator;
    this.currentAllocation = new Map();
    this.simulationTime = 0;
  }

  /**
   * Add agent to swarm
   */
  addAgent(agent: AgentState): void {
    this.agents.set(agent.id, agent);
    this.currentAllocation.set(agent.id, FidelityLevel.COARSE);
  }

  /**
   * Update swarm simulation
   */
  update(dt: number): void {
    const agentList = Array.from(this.agents.values());

    // Allocate fidelity
    const newAllocation = this.allocator.allocateFidelity(agentList);

    // Apply fidelity allocation
    for (const [agentId, fidelity] of newAllocation) {
      const agent = this.agents.get(agentId);
      if (agent) {
        agent.fidelity = fidelity;
        agent.computeCost = FIDELITY_CONFIGS[fidelity].computeCost;
      }
    }

    this.currentAllocation = newAllocation;

    // Simulate agents
    for (const agent of agentList) {
      this.simulateAgent(agent, dt);
    }

    this.simulationTime += dt;
  }

  /**
   * Simulate single agent based on fidelity
   */
  private simulateAgent(agent: AgentState, dt: number): void {
    const config = FIDELITY_CONFIGS[agent.fidelity];

    // Update frequency determines how often we simulate
    const shouldUpdate =
      this.simulationTime % (1 / config.updateFrequency) < dt;

    if (!shouldUpdate) return;

    // Physics simulation (simplified)
    for (let i = 0; i < config.physicsSteps; i++) {
      const subDt = dt / config.physicsSteps;

      // Update position
      agent.position.x += agent.velocity.x * subDt;
      agent.position.y += agent.velocity.y * subDt;
      agent.position.z += agent.velocity.z * subDt;

      // Add noise based on fidelity (lower fidelity = more noise)
      const noise = (1 - config.accuracy) * 0.1;
      agent.position.x += (Math.random() - 0.5) * noise;
      agent.position.y += (Math.random() - 0.5) * noise;
      agent.position.z += (Math.random() - 0.5) * noise;

      // Update prediction error (inversely related to accuracy)
      agent.predictionError = (1 - config.accuracy) * Math.random();
    }

    agent.lastUpdate = Date.now();
  }

  /**
   * Get swarm metrics
   */
  getMetrics(): SwarmMetrics {
    const agentList = Array.from(this.agents.values());

    const fidelityDistribution = new Map<FidelityLevel, number>([
      [FidelityLevel.COARSE, 0],
      [FidelityLevel.MEDIUM, 0],
      [FidelityLevel.FINE, 0],
    ]);

    let totalComputeCost = 0;
    let totalAccuracy = 0;

    for (const agent of agentList) {
      const count = fidelityDistribution.get(agent.fidelity) ?? 0;
      fidelityDistribution.set(agent.fidelity, count + 1);

      totalComputeCost += agent.computeCost;
      totalAccuracy += FIDELITY_CONFIGS[agent.fidelity].accuracy;
    }

    const averageAccuracy = totalAccuracy / agentList.length;
    const maxPossibleCost =
      agentList.length * FIDELITY_CONFIGS[FidelityLevel.FINE].computeCost;
    const resourceUtilization = totalComputeCost / maxPossibleCost;

    return {
      totalAgents: agentList.length,
      fidelityDistribution,
      totalComputeCost,
      averageAccuracy,
      resourceUtilization,
    };
  }

  /**
   * Get agents by fidelity
   */
  getAgentsByFidelity(fidelity: FidelityLevel): AgentState[] {
    return Array.from(this.agents.values()).filter(
      (a) => a.fidelity === fidelity
    );
  }

  getAllAgents(): AgentState[] {
    return Array.from(this.agents.values());
  }
}

// ============================================================================
// Demonstration
// ============================================================================

export async function demonstrateMultiFidelity(): Promise<void> {
  console.log("=".repeat(80));
  console.log("MULTI-FIDELITY SWARM TWINS DEMONSTRATION");
  console.log("=".repeat(80));

  // Demo 1: Fidelity Configurations
  console.log("\n⚙️ Demo 1: Fidelity Level Configurations");
  console.log("-".repeat(80));

  console.log("Available fidelity levels:\n");
  for (const [level, config] of Object.entries(FIDELITY_CONFIGS)) {
    console.log(`${level.toUpperCase()}:`);
    console.log(`  Update frequency: ${config.updateFrequency} Hz`);
    console.log(`  Physics steps: ${config.physicsSteps}`);
    console.log(`  Compute cost: ${config.computeCost}x`);
    console.log(`  Accuracy: ${(config.accuracy * 100).toFixed(1)}%\n`);
  }

  // Demo 2: Importance Estimation
  console.log("📊 Demo 2: Agent Importance Estimation");
  console.log("-".repeat(80));

  const targetPosition: Vector3D = { x: 50, y: 50, z: 0 };
  const estimator = new ImportanceEstimator(targetPosition);

  const testAgent: AgentState = {
    id: "agent-1",
    position: { x: 45, y: 48, z: 0 },
    velocity: { x: 2, y: 1, z: 0 },
    role: AgentRole.SCOUT,
    fidelity: FidelityLevel.COARSE,
    importance: 0,
    computeCost: 1,
    predictionError: 0.05,
    lastUpdate: Date.now(),
  };

  const neighbors: AgentState[] = [
    {
      id: "agent-2",
      position: { x: 47, y: 49, z: 0 },
      velocity: { x: 1, y: 0.5, z: 0 },
      role: AgentRole.WORKER,
      fidelity: FidelityLevel.COARSE,
      importance: 0,
      computeCost: 1,
      predictionError: 0.08,
      lastUpdate: Date.now(),
    },
  ];

  const importance = estimator.estimateImportance(testAgent, neighbors);
  console.log(`Agent: ${testAgent.id} (${testAgent.role})`);
  console.log(`Position: (${testAgent.position.x}, ${testAgent.position.y})`);
  console.log(
    `Distance to target: ${estimator["distance"](testAgent.position, targetPosition).toFixed(2)}`
  );
  console.log(`Importance score: ${importance.toFixed(3)}`);

  // Demo 3: Fidelity Allocation
  console.log("\n🎯 Demo 3: Dynamic Fidelity Allocation");
  console.log("-".repeat(80));

  const allocator = new FidelityAllocator(estimator, 50);

  const swarmAgents: AgentState[] = [];
  for (let i = 0; i < 10; i++) {
    swarmAgents.push({
      id: `agent-${i}`,
      position: {
        x: Math.random() * 100,
        y: Math.random() * 100,
        z: 0,
      },
      velocity: {
        x: (Math.random() - 0.5) * 4,
        y: (Math.random() - 0.5) * 4,
        z: 0,
      },
      role:
        i < 2
          ? AgentRole.SCOUT
          : i < 7
            ? AgentRole.WORKER
            : AgentRole.COORDINATOR,
      fidelity: FidelityLevel.COARSE,
      importance: 0,
      computeCost: 1,
      predictionError: Math.random() * 0.2,
      lastUpdate: Date.now(),
    });
  }

  console.log(`Swarm size: ${swarmAgents.length} agents`);
  console.log(`Compute budget: 50 units`);
  console.log("\nAllocating fidelity...\n");

  const allocation = allocator.allocateFidelity(swarmAgents);

  const fidelityCounts = {
    [FidelityLevel.COARSE]: 0,
    [FidelityLevel.MEDIUM]: 0,
    [FidelityLevel.FINE]: 0,
  };

  for (const [agentId, fidelity] of allocation) {
    fidelityCounts[fidelity]++;
    const agent = swarmAgents.find((a) => a.id === agentId)!;
    const importance = estimator.estimateImportance(
      agent,
      swarmAgents.filter((a) => a.id !== agentId)
    );
    console.log(
      `${agentId}: ${fidelity} (importance: ${importance.toFixed(2)})`
    );
  }

  console.log(`\nFidelity distribution:`);
  console.log(`  Fine: ${fidelityCounts[FidelityLevel.FINE]}`);
  console.log(`  Medium: ${fidelityCounts[FidelityLevel.MEDIUM]}`);
  console.log(`  Coarse: ${fidelityCounts[FidelityLevel.COARSE]}`);

  // Demo 4: Swarm Coordination
  console.log("\n🐝 Demo 4: Swarm Coordination Simulation");
  console.log("-".repeat(80));

  const coordinator = new SwarmCoordinator(allocator);

  for (const agent of swarmAgents) {
    coordinator.addAgent(agent);
  }

  console.log("Running simulation for 5 steps...\n");

  for (let step = 0; step < 5; step++) {
    coordinator.update(0.1);

    if (step % 2 === 0) {
      const metrics = coordinator.getMetrics();
      console.log(`Step ${step + 1}:`);
      console.log(
        `  Total compute cost: ${metrics.totalComputeCost.toFixed(1)}`
      );
      console.log(
        `  Average accuracy: ${(metrics.averageAccuracy * 100).toFixed(1)}%`
      );
      console.log(
        `  Resource utilization: ${(metrics.resourceUtilization * 100).toFixed(1)}%`
      );
    }
  }

  // Demo 5: Adaptive Fidelity
  console.log("\n🔄 Demo 5: Adaptive Fidelity Based on Error");
  console.log("-".repeat(80));

  console.log("Simulating high prediction errors for some agents...\n");

  // Inject high errors
  swarmAgents[0].predictionError = 0.25;
  swarmAgents[1].predictionError = 0.2;
  swarmAgents[5].predictionError = 0.02;

  const currentAlloc = allocator.allocateFidelity(swarmAgents);
  console.log("Current allocation:");
  for (let i = 0; i < 3; i++) {
    console.log(
      `  ${swarmAgents[i].id}: ${currentAlloc.get(swarmAgents[i].id)} (error: ${swarmAgents[i].predictionError.toFixed(2)})`
    );
  }

  const adjusted = allocator.adjustForError(currentAlloc, swarmAgents);
  console.log("\nAfter error-based adjustment:");
  for (let i = 0; i < 3; i++) {
    const before = currentAlloc.get(swarmAgents[i].id);
    const after = adjusted.get(swarmAgents[i].id);
    const changed = before !== after ? "→" : "=";
    console.log(`  ${swarmAgents[i].id}: ${before} ${changed} ${after}`);
  }

  // Demo 6: Budget Scaling
  console.log("\n💰 Demo 6: Compute Budget Scaling");
  console.log("-".repeat(80));

  const budgets = [30, 50, 100];

  console.log("Testing different compute budgets:\n");

  for (const budget of budgets) {
    allocator.setComputeBudget(budget);
    const alloc = allocator.allocateFidelity(swarmAgents);

    const counts = {
      [FidelityLevel.COARSE]: 0,
      [FidelityLevel.MEDIUM]: 0,
      [FidelityLevel.FINE]: 0,
    };

    for (const fidelity of alloc.values()) {
      counts[fidelity]++;
    }

    console.log(`Budget: ${budget} units`);
    console.log(
      `  Fine: ${counts[FidelityLevel.FINE]}, Medium: ${counts[FidelityLevel.MEDIUM]}, Coarse: ${counts[FidelityLevel.COARSE]}`
    );
  }

  console.log("\n✅ Multi-Fidelity Swarm Twins demonstration complete!");
  console.log("=".repeat(80));
}

// Export classes for programmatic use
export {
  ImportanceEstimator,
  FidelityAllocator,
  SwarmCoordinator,
  FidelityLevel,
  AgentRole,
  FIDELITY_CONFIGS,
  type AgentState,
  type FidelityConfig,
  type SwarmMetrics,
  type ImportanceFactors,
  type Vector3D,
};
