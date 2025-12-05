/**
 * NEURECTOMY Swarm Arena
 * @module @neurectomy/experimentation-engine/swarm
 * @agent @OMNISCIENT @NEURAL
 *
 * Multi-agent swarm arena for testing emergent behaviors,
 * collective intelligence, and competition dynamics.
 */

import { EventEmitter } from "eventemitter3";
import { v4 as uuidv4 } from "uuid";
import { z } from "zod";

// ============================================================================
// Arena Configuration Schemas
// ============================================================================

export const AgentTypeSchema = z.enum([
  "explorer", // Searches for resources/solutions
  "exploiter", // Optimizes known solutions
  "communicator", // Shares information between agents
  "predator", // Attacks/disrupts other agents
  "defender", // Protects resources/agents
  "neutral", // Passive observer
  "hybrid", // Multi-role agent
]);

export const ArenaTopologySchema = z.enum([
  "grid", // 2D grid topology
  "torus", // Wraparound grid
  "graph", // Arbitrary graph
  "tree", // Hierarchical tree
  "ring", // Circular topology
  "hypercube", // N-dimensional cube
  "random", // Random connections
  "fully_connected", // All-to-all
]);

export const ResourceTypeSchema = z.enum([
  "energy",
  "information",
  "territory",
  "points",
  "tokens",
  "custom",
]);

export const InteractionModeSchema = z.enum([
  "cooperative", // Agents work together
  "competitive", // Agents compete
  "mixed", // Combination
  "adversarial", // Zero-sum competition
]);

export const ArenaConfigSchema = z.object({
  id: z.string().uuid(),
  name: z.string(),
  description: z.string().optional(),
  topology: ArenaTopologySchema,
  dimensions: z.object({
    width: z.number().int().positive(),
    height: z.number().int().positive(),
    depth: z.number().int().positive().optional(),
  }),
  maxAgents: z.number().int().positive(),
  tickRate: z.number().positive().default(10), // ticks per second
  maxTicks: z.number().int().positive().optional(),
  resources: z.array(
    z.object({
      type: ResourceTypeSchema,
      name: z.string(),
      initialAmount: z.number().nonnegative(),
      regenerationRate: z.number().nonnegative().default(0),
      maxAmount: z.number().positive().optional(),
      distribution: z.enum(["uniform", "clustered", "random", "custom"]),
    })
  ),
  interactionMode: InteractionModeSchema,
  rules: z.object({
    allowCollisions: z.boolean().default(true),
    deathEnabled: z.boolean().default(false),
    reproductionEnabled: z.boolean().default(false),
    communicationRange: z.number().nonnegative().optional(),
    visionRange: z.number().nonnegative().optional(),
    movementCost: z.number().nonnegative().default(0),
  }),
  metrics: z
    .array(z.string())
    .default([
      "total_agents",
      "resources_collected",
      "interactions",
      "entropy",
    ]),
});

export const AgentConfigSchema = z.object({
  id: z.string().uuid(),
  name: z.string(),
  type: AgentTypeSchema,
  team: z.string().optional(),
  position: z.object({
    x: z.number(),
    y: z.number(),
    z: z.number().optional(),
  }),
  attributes: z.object({
    energy: z.number().nonnegative().default(100),
    speed: z.number().positive().default(1),
    visionRange: z.number().nonnegative().default(5),
    strength: z.number().nonnegative().default(1),
    intelligence: z.number().nonnegative().default(1),
    communication: z.number().nonnegative().default(1),
  }),
  behavior: z.discriminatedUnion("type", [
    z.object({
      type: z.literal("scripted"),
      script: z.string(),
    }),
    z.object({
      type: z.literal("rule_based"),
      rules: z.array(
        z.object({
          condition: z.string(),
          action: z.string(),
          priority: z.number().default(0),
        })
      ),
    }),
    z.object({
      type: z.literal("neural"),
      modelId: z.string(),
      weights: z.array(z.number()).optional(),
    }),
    z.object({
      type: z.literal("llm"),
      prompt: z.string(),
      modelId: z.string(),
    }),
    z.object({
      type: z.literal("random"),
      seed: z.number().optional(),
    }),
  ]),
  memory: z.record(z.unknown()).default({}),
});

// ============================================================================
// Types
// ============================================================================

export type AgentType = z.infer<typeof AgentTypeSchema>;
export type ArenaTopology = z.infer<typeof ArenaTopologySchema>;
export type ResourceType = z.infer<typeof ResourceTypeSchema>;
export type InteractionMode = z.infer<typeof InteractionModeSchema>;
export type ArenaConfig = z.infer<typeof ArenaConfigSchema>;
export type AgentConfig = z.infer<typeof AgentConfigSchema>;

export interface Position {
  x: number;
  y: number;
  z?: number;
}

export interface SwarmAgent {
  config: AgentConfig;
  state: AgentState;
  history: AgentAction[];
}

export interface AgentState {
  alive: boolean;
  energy: number;
  resources: Map<string, number>;
  position: Position;
  facing: number; // Direction in radians
  lastAction?: AgentAction;
  ticksAlive: number;
  interactions: number;
  score: number;
}

export interface AgentAction {
  tick: number;
  type:
    | "move"
    | "collect"
    | "attack"
    | "defend"
    | "communicate"
    | "reproduce"
    | "idle";
  target?: Position | string;
  data?: Record<string, unknown>;
  success: boolean;
  energyCost: number;
}

export interface ArenaCell {
  position: Position;
  agents: string[];
  resources: Map<string, number>;
  terrain?: string;
  metadata: Record<string, unknown>;
}

export interface ArenaState {
  tick: number;
  agents: Map<string, SwarmAgent>;
  cells: Map<string, ArenaCell>;
  resources: Map<string, number>;
  messages: Message[];
  events: ArenaEvent[];
  metrics: Map<string, number>;
  paused: boolean;
  completed: boolean;
}

export interface Message {
  id: string;
  tick: number;
  from: string;
  to: string | "broadcast";
  content: unknown;
  range?: number;
}

export interface ArenaEvent {
  tick: number;
  type: string;
  agentId?: string;
  data: Record<string, unknown>;
}

export interface ArenaResults {
  arenaId: string;
  config: ArenaConfig;
  totalTicks: number;
  duration: number;
  finalAgentCount: number;
  survivingAgents: string[];
  rankings: AgentRanking[];
  teamRankings?: TeamRanking[];
  metrics: Record<string, number>;
  timeline: TimelineEntry[];
  emergentBehaviors: EmergentBehavior[];
}

export interface AgentRanking {
  agentId: string;
  name: string;
  team?: string;
  score: number;
  rank: number;
  resourcesCollected: number;
  interactions: number;
  ticksAlive: number;
}

export interface TeamRanking {
  team: string;
  totalScore: number;
  avgScore: number;
  members: number;
  rank: number;
}

export interface TimelineEntry {
  tick: number;
  agentCount: number;
  resourcesAvailable: number;
  metrics: Record<string, number>;
}

export interface EmergentBehavior {
  name: string;
  description: string;
  firstObserved: number;
  frequency: number;
  involvedAgents: string[];
}

export interface SwarmArenaEvents {
  initialized: (config: ArenaConfig) => void;
  started: () => void;
  paused: () => void;
  resumed: () => void;
  tick: (state: ArenaState) => void;
  agentSpawned: (agent: SwarmAgent) => void;
  agentDied: (agentId: string, cause: string) => void;
  agentAction: (agentId: string, action: AgentAction) => void;
  interaction: (agent1: string, agent2: string, type: string) => void;
  message: (message: Message) => void;
  resourceDepleted: (resourceType: string, position: Position) => void;
  emergentBehavior: (behavior: EmergentBehavior) => void;
  completed: (results: ArenaResults) => void;
  error: (error: Error) => void;
}

// ============================================================================
// Swarm Arena Implementation
// ============================================================================

/**
 * SwarmArena - Multi-agent simulation environment
 *
 * Features:
 * - Multiple topology types
 * - Customizable agent behaviors
 * - Resource management
 * - Inter-agent communication
 * - Emergent behavior detection
 * - Team competitions
 */
export class SwarmArena extends EventEmitter<SwarmArenaEvents> {
  private config: ArenaConfig;
  private state: ArenaState;
  private behaviorEngines: Map<string, BehaviorEngine> = new Map();
  private tickInterval?: NodeJS.Timeout;
  private startTime?: Date;

  constructor(config: Omit<ArenaConfig, "id">) {
    super();

    this.config = ArenaConfigSchema.parse({
      ...config,
      id: uuidv4(),
    });

    this.state = this.initializeState();
    this.registerDefaultBehaviors();

    this.emit("initialized", this.config);
  }

  // ============================================================================
  // Initialization
  // ============================================================================

  private initializeState(): ArenaState {
    const cells = new Map<string, ArenaCell>();

    // Initialize grid cells
    for (let x = 0; x < this.config.dimensions.width; x++) {
      for (let y = 0; y < this.config.dimensions.height; y++) {
        const key = this.positionKey({ x, y });
        cells.set(key, {
          position: { x, y },
          agents: [],
          resources: new Map(),
          metadata: {},
        });
      }
    }

    // Distribute initial resources
    this.distributeResources(cells);

    return {
      tick: 0,
      agents: new Map(),
      cells,
      resources: new Map(
        this.config.resources.map((r) => [r.name, r.initialAmount])
      ),
      messages: [],
      events: [],
      metrics: new Map(),
      paused: false,
      completed: false,
    };
  }

  private distributeResources(cells: Map<string, ArenaCell>): void {
    for (const resource of this.config.resources) {
      let remaining = resource.initialAmount;

      switch (resource.distribution) {
        case "uniform": {
          const perCell = remaining / cells.size;
          for (const cell of cells.values()) {
            cell.resources.set(resource.name, perCell);
          }
          break;
        }

        case "random": {
          const cellArray = Array.from(cells.values());
          while (remaining > 0) {
            const randomCell =
              cellArray[Math.floor(Math.random() * cellArray.length)];
            const amount = Math.min(remaining, Math.random() * 10);
            const current = randomCell.resources.get(resource.name) || 0;
            randomCell.resources.set(resource.name, current + amount);
            remaining -= amount;
          }
          break;
        }

        case "clustered": {
          // Create a few clusters
          const clusterCount = Math.ceil(Math.sqrt(cells.size) / 2);
          const cellArray = Array.from(cells.values());

          for (let i = 0; i < clusterCount && remaining > 0; i++) {
            const centerCell =
              cellArray[Math.floor(Math.random() * cellArray.length)];
            const clusterAmount = remaining / (clusterCount - i);

            // Distribute around center
            const nearby = this.getCellsInRange(centerCell.position, 3);
            const perNearby = clusterAmount / nearby.length;

            for (const cell of nearby) {
              const current = cell.resources.get(resource.name) || 0;
              cell.resources.set(resource.name, current + perNearby);
              remaining -= perNearby;
            }
          }
          break;
        }
      }
    }
  }

  private registerDefaultBehaviors(): void {
    this.behaviorEngines.set("random", new RandomBehaviorEngine());
    this.behaviorEngines.set("rule_based", new RuleBasedBehaviorEngine());
    this.behaviorEngines.set("scripted", new ScriptedBehaviorEngine());
  }

  // ============================================================================
  // Agent Management
  // ============================================================================

  /**
   * Spawn a new agent in the arena
   */
  spawnAgent(config: Omit<AgentConfig, "id">): SwarmAgent {
    if (this.state.agents.size >= this.config.maxAgents) {
      throw new Error("Maximum agent count reached");
    }

    const agentConfig = AgentConfigSchema.parse({
      ...config,
      id: uuidv4(),
    });

    // Validate position
    if (!this.isValidPosition(agentConfig.position)) {
      throw new Error("Invalid spawn position");
    }

    const agent: SwarmAgent = {
      config: agentConfig,
      state: {
        alive: true,
        energy: agentConfig.attributes.energy,
        resources: new Map(),
        position: { ...agentConfig.position },
        facing: 0,
        ticksAlive: 0,
        interactions: 0,
        score: 0,
      },
      history: [],
    };

    this.state.agents.set(agentConfig.id, agent);

    // Add to cell
    const cellKey = this.positionKey(agent.state.position);
    const cell = this.state.cells.get(cellKey);
    if (cell) {
      cell.agents.push(agentConfig.id);
    }

    this.emit("agentSpawned", agent);
    return agent;
  }

  /**
   * Remove an agent from the arena
   */
  removeAgent(agentId: string, cause: string = "removed"): boolean {
    const agent = this.state.agents.get(agentId);
    if (!agent) return false;

    // Remove from cell
    const cellKey = this.positionKey(agent.state.position);
    const cell = this.state.cells.get(cellKey);
    if (cell) {
      const idx = cell.agents.indexOf(agentId);
      if (idx !== -1) {
        cell.agents.splice(idx, 1);
      }
    }

    agent.state.alive = false;
    this.emit("agentDied", agentId, cause);

    return true;
  }

  /**
   * Get agent by ID
   */
  getAgent(agentId: string): SwarmAgent | undefined {
    return this.state.agents.get(agentId);
  }

  /**
   * Get all alive agents
   */
  getAliveAgents(): SwarmAgent[] {
    return Array.from(this.state.agents.values()).filter((a) => a.state.alive);
  }

  // ============================================================================
  // Simulation Control
  // ============================================================================

  /**
   * Start the simulation
   */
  start(): void {
    if (this.tickInterval) return;

    this.startTime = new Date();
    this.state.paused = false;

    const tickDelay = 1000 / this.config.tickRate;
    this.tickInterval = setInterval(() => this.executeTick(), tickDelay);

    this.emit("started");
  }

  /**
   * Pause the simulation
   */
  pause(): void {
    this.state.paused = true;
    this.emit("paused");
  }

  /**
   * Resume the simulation
   */
  resume(): void {
    this.state.paused = false;
    this.emit("resumed");
  }

  /**
   * Stop the simulation
   */
  stop(): ArenaResults {
    if (this.tickInterval) {
      clearInterval(this.tickInterval);
      this.tickInterval = undefined;
    }

    this.state.completed = true;
    const results = this.generateResults();
    this.emit("completed", results);

    return results;
  }

  /**
   * Execute a single tick manually
   */
  step(): void {
    this.executeTick();
  }

  // ============================================================================
  // Tick Execution
  // ============================================================================

  private executeTick(): void {
    if (this.state.paused || this.state.completed) return;

    this.state.tick++;

    // Check completion
    if (this.config.maxTicks && this.state.tick >= this.config.maxTicks) {
      this.stop();
      return;
    }

    // Process all agents
    const aliveAgents = this.getAliveAgents();
    for (const agent of aliveAgents) {
      this.processAgent(agent);
    }

    // Regenerate resources
    this.regenerateResources();

    // Process messages
    this.processMessages();

    // Update metrics
    this.updateMetrics();

    // Detect emergent behaviors
    this.detectEmergentBehaviors();

    // Clear old messages
    this.state.messages = this.state.messages.filter(
      (m) => this.state.tick - m.tick < 10
    );

    this.emit("tick", this.state);
  }

  private processAgent(agent: SwarmAgent): void {
    agent.state.ticksAlive++;

    // Get agent's decision
    const action = this.getAgentAction(agent);

    // Execute action
    const result = this.executeAction(agent, action);

    // Record action
    agent.history.push(result);
    agent.state.lastAction = result;

    // Apply energy cost
    agent.state.energy -= result.energyCost;

    // Check death conditions
    if (this.config.rules.deathEnabled && agent.state.energy <= 0) {
      this.removeAgent(agent.config.id, "energy_depleted");
    }

    this.emit("agentAction", agent.config.id, result);
  }

  private getAgentAction(agent: SwarmAgent): AgentAction {
    const behaviorType = agent.config.behavior.type;
    const engine = this.behaviorEngines.get(behaviorType);

    if (!engine) {
      return this.createIdleAction();
    }

    try {
      return engine.decide(agent, this.state, this.config);
    } catch {
      return this.createIdleAction();
    }
  }

  private createIdleAction(): AgentAction {
    return {
      tick: this.state.tick,
      type: "idle",
      success: true,
      energyCost: 0,
    };
  }

  private executeAction(agent: SwarmAgent, action: AgentAction): AgentAction {
    action.tick = this.state.tick;

    switch (action.type) {
      case "move":
        return this.executeMove(agent, action);

      case "collect":
        return this.executeCollect(agent, action);

      case "attack":
        return this.executeAttack(agent, action);

      case "defend":
        return this.executeDefend(agent, action);

      case "communicate":
        return this.executeCommunicate(agent, action);

      case "reproduce":
        return this.executeReproduce(agent, action);

      case "idle":
      default:
        action.success = true;
        action.energyCost = 0;
        return action;
    }
  }

  private executeMove(agent: SwarmAgent, action: AgentAction): AgentAction {
    const target = action.target as Position | undefined;
    if (!target) {
      action.success = false;
      return action;
    }

    // Calculate distance
    const distance = this.calculateDistance(agent.state.position, target);
    const maxMove = agent.config.attributes.speed;

    if (distance > maxMove) {
      // Move towards target
      const dx = target.x - agent.state.position.x;
      const dy = target.y - agent.state.position.y;
      const ratio = maxMove / distance;

      target.x = agent.state.position.x + dx * ratio;
      target.y = agent.state.position.y + dy * ratio;
    }

    // Validate target position
    if (!this.isValidPosition(target)) {
      action.success = false;
      return action;
    }

    // Check collisions
    if (!this.config.rules.allowCollisions) {
      const cellKey = this.positionKey(target);
      const targetCell = this.state.cells.get(cellKey);
      if (targetCell && targetCell.agents.length > 0) {
        action.success = false;
        return action;
      }
    }

    // Update position
    const oldKey = this.positionKey(agent.state.position);
    const newKey = this.positionKey(target);

    const oldCell = this.state.cells.get(oldKey);
    if (oldCell) {
      const idx = oldCell.agents.indexOf(agent.config.id);
      if (idx !== -1) oldCell.agents.splice(idx, 1);
    }

    const newCell = this.state.cells.get(newKey);
    if (newCell) {
      newCell.agents.push(agent.config.id);
    }

    agent.state.position = { ...target };
    agent.state.facing = Math.atan2(
      target.y - agent.state.position.y,
      target.x - agent.state.position.x
    );

    action.success = true;
    action.energyCost = this.config.rules.movementCost * distance;
    return action;
  }

  private executeCollect(agent: SwarmAgent, action: AgentAction): AgentAction {
    const cellKey = this.positionKey(agent.state.position);
    const cell = this.state.cells.get(cellKey);

    if (!cell) {
      action.success = false;
      return action;
    }

    let collected = false;
    for (const [resourceName, amount] of cell.resources) {
      if (amount > 0) {
        const collectAmount = Math.min(
          amount,
          agent.config.attributes.strength
        );
        cell.resources.set(resourceName, amount - collectAmount);

        const current = agent.state.resources.get(resourceName) || 0;
        agent.state.resources.set(resourceName, current + collectAmount);

        agent.state.score += collectAmount;
        collected = true;

        if (amount - collectAmount <= 0) {
          this.emit("resourceDepleted", resourceName, agent.state.position);
        }
      }
    }

    action.success = collected;
    action.energyCost = collected ? 1 : 0;
    return action;
  }

  private executeAttack(agent: SwarmAgent, action: AgentAction): AgentAction {
    const targetId = action.target as string | undefined;
    if (!targetId) {
      action.success = false;
      return action;
    }

    const target = this.state.agents.get(targetId);
    if (!target || !target.state.alive) {
      action.success = false;
      return action;
    }

    // Check range
    const distance = this.calculateDistance(
      agent.state.position,
      target.state.position
    );
    if (distance > agent.config.attributes.visionRange) {
      action.success = false;
      return action;
    }

    // Calculate damage
    const damage = agent.config.attributes.strength;
    target.state.energy -= damage;

    agent.state.interactions++;
    target.state.interactions++;

    this.emit("interaction", agent.config.id, targetId, "attack");

    if (target.state.energy <= 0 && this.config.rules.deathEnabled) {
      this.removeAgent(targetId, "killed");
      agent.state.score += 10;
    }

    action.success = true;
    action.energyCost = 2;
    return action;
  }

  private executeDefend(agent: SwarmAgent, action: AgentAction): AgentAction {
    // Increase defense for this tick
    agent.state.energy += 5; // Recover some energy
    action.success = true;
    action.energyCost = 0;
    return action;
  }

  private executeCommunicate(
    agent: SwarmAgent,
    action: AgentAction
  ): AgentAction {
    const target = action.target as string | undefined;
    const data = action.data;

    if (!data) {
      action.success = false;
      return action;
    }

    const message: Message = {
      id: uuidv4(),
      tick: this.state.tick,
      from: agent.config.id,
      to: target || "broadcast",
      content: data,
      range: this.config.rules.communicationRange,
    };

    this.state.messages.push(message);
    this.emit("message", message);

    action.success = true;
    action.energyCost = 0.5;
    return action;
  }

  private executeReproduce(
    agent: SwarmAgent,
    action: AgentAction
  ): AgentAction {
    if (!this.config.rules.reproductionEnabled) {
      action.success = false;
      return action;
    }

    if (agent.state.energy < 50) {
      action.success = false;
      return action;
    }

    // Find empty adjacent cell
    const neighbors = this.getNeighborPositions(agent.state.position);
    const emptyNeighbor = neighbors.find((pos) => {
      const cell = this.state.cells.get(this.positionKey(pos));
      return cell && cell.agents.length === 0;
    });

    if (!emptyNeighbor) {
      action.success = false;
      return action;
    }

    try {
      // Create offspring
      this.spawnAgent({
        name: `${agent.config.name}_offspring`,
        type: agent.config.type,
        team: agent.config.team,
        position: emptyNeighbor,
        attributes: {
          ...agent.config.attributes,
          energy: 50,
        },
        behavior: agent.config.behavior,
      });

      agent.state.energy -= 30;
      action.success = true;
      action.energyCost = 30;
    } catch {
      action.success = false;
    }

    return action;
  }

  // ============================================================================
  // Resource Management
  // ============================================================================

  private regenerateResources(): void {
    for (const resourceConfig of this.config.resources) {
      if (resourceConfig.regenerationRate === 0) continue;

      for (const cell of this.state.cells.values()) {
        const current = cell.resources.get(resourceConfig.name) || 0;
        const maxAmount = resourceConfig.maxAmount ?? Infinity;

        if (current < maxAmount) {
          const newAmount = Math.min(
            current + resourceConfig.regenerationRate,
            maxAmount
          );
          cell.resources.set(resourceConfig.name, newAmount);
        }
      }
    }
  }

  // ============================================================================
  // Message Processing
  // ============================================================================

  private processMessages(): void {
    for (const message of this.state.messages) {
      if (message.tick !== this.state.tick - 1) continue;

      const sender = this.state.agents.get(message.from);
      if (!sender) continue;

      if (message.to === "broadcast") {
        // Send to all agents in range
        for (const agent of this.state.agents.values()) {
          if (agent.config.id === message.from) continue;
          if (!agent.state.alive) continue;

          const distance = this.calculateDistance(
            sender.state.position,
            agent.state.position
          );

          if (!message.range || distance <= message.range) {
            this.deliverMessage(agent, message);
          }
        }
      } else {
        // Send to specific agent
        const target = this.state.agents.get(message.to);
        if (target && target.state.alive) {
          this.deliverMessage(target, message);
        }
      }
    }
  }

  private deliverMessage(agent: SwarmAgent, message: Message): void {
    // Store in agent's memory
    const messages = (agent.config.memory["messages"] as Message[]) || [];
    messages.push(message);
    agent.config.memory["messages"] = messages.slice(-10); // Keep last 10
  }

  // ============================================================================
  // Metrics & Analysis
  // ============================================================================

  private updateMetrics(): void {
    const aliveAgents = this.getAliveAgents();

    this.state.metrics.set("total_agents", this.state.agents.size);
    this.state.metrics.set("alive_agents", aliveAgents.length);
    this.state.metrics.set("total_resources", this.getTotalResources());
    this.state.metrics.set(
      "total_interactions",
      aliveAgents.reduce((sum, a) => sum + a.state.interactions, 0)
    );
    this.state.metrics.set("entropy", this.calculateEntropy());
    this.state.metrics.set(
      "avg_energy",
      aliveAgents.reduce((sum, a) => sum + a.state.energy, 0) /
        Math.max(aliveAgents.length, 1)
    );

    // Team metrics
    const teams = new Map<string, SwarmAgent[]>();
    for (const agent of aliveAgents) {
      if (agent.config.team) {
        const teamAgents = teams.get(agent.config.team) || [];
        teamAgents.push(agent);
        teams.set(agent.config.team, teamAgents);
      }
    }

    for (const [team, members] of teams) {
      this.state.metrics.set(`team_${team}_count`, members.length);
      this.state.metrics.set(
        `team_${team}_score`,
        members.reduce((sum, a) => sum + a.state.score, 0)
      );
    }
  }

  private calculateEntropy(): number {
    // Calculate spatial entropy based on agent distribution
    const cellCounts: number[] = [];
    for (const cell of this.state.cells.values()) {
      cellCounts.push(cell.agents.length);
    }

    const total = cellCounts.reduce((a, b) => a + b, 0);
    if (total === 0) return 0;

    let entropy = 0;
    for (const count of cellCounts) {
      if (count > 0) {
        const p = count / total;
        entropy -= p * Math.log2(p);
      }
    }

    return entropy;
  }

  private detectEmergentBehaviors(): void {
    // Detect clustering
    const clusters = this.detectClusters();
    if (clusters.length > 1) {
      this.emit("emergentBehavior", {
        name: "clustering",
        description: `${clusters.length} agent clusters detected`,
        firstObserved: this.state.tick,
        frequency: 1,
        involvedAgents: clusters.flat(),
      });
    }

    // Detect cooperation patterns
    const cooperationEvents = this.state.events.filter(
      (e) => e.type === "cooperation" && e.tick > this.state.tick - 10
    );
    if (cooperationEvents.length >= 3) {
      this.emit("emergentBehavior", {
        name: "cooperation_pattern",
        description: "Repeated cooperative behavior detected",
        firstObserved: cooperationEvents[0]?.tick ?? this.state.tick,
        frequency: cooperationEvents.length,
        involvedAgents: cooperationEvents
          .map((e) => e.agentId!)
          .filter(Boolean),
      });
    }
  }

  private detectClusters(): string[][] {
    const clusters: string[][] = [];
    const visited = new Set<string>();

    for (const agent of this.getAliveAgents()) {
      if (visited.has(agent.config.id)) continue;

      const cluster: string[] = [];
      const queue = [agent];

      while (queue.length > 0) {
        const current = queue.shift()!;
        if (visited.has(current.config.id)) continue;

        visited.add(current.config.id);
        cluster.push(current.config.id);

        // Find nearby agents
        for (const other of this.getAliveAgents()) {
          if (visited.has(other.config.id)) continue;

          const distance = this.calculateDistance(
            current.state.position,
            other.state.position
          );

          if (distance < 3) {
            queue.push(other);
          }
        }
      }

      if (cluster.length >= 3) {
        clusters.push(cluster);
      }
    }

    return clusters;
  }

  // ============================================================================
  // Results Generation
  // ============================================================================

  private generateResults(): ArenaResults {
    const aliveAgents = this.getAliveAgents();

    // Calculate rankings
    const rankings: AgentRanking[] = Array.from(this.state.agents.values())
      .map((agent) => ({
        agentId: agent.config.id,
        name: agent.config.name,
        team: agent.config.team,
        score: agent.state.score,
        rank: 0,
        resourcesCollected: Array.from(agent.state.resources.values()).reduce(
          (a, b) => a + b,
          0
        ),
        interactions: agent.state.interactions,
        ticksAlive: agent.state.ticksAlive,
      }))
      .sort((a, b) => b.score - a.score);

    rankings.forEach((r, i) => (r.rank = i + 1));

    // Calculate team rankings
    const teamScores = new Map<string, { total: number; count: number }>();
    for (const ranking of rankings) {
      if (ranking.team) {
        const current = teamScores.get(ranking.team) || { total: 0, count: 0 };
        current.total += ranking.score;
        current.count++;
        teamScores.set(ranking.team, current);
      }
    }

    const teamRankings: TeamRanking[] = Array.from(teamScores.entries())
      .map(([team, data]) => ({
        team,
        totalScore: data.total,
        avgScore: data.total / data.count,
        members: data.count,
        rank: 0,
      }))
      .sort((a, b) => b.totalScore - a.totalScore);

    teamRankings.forEach((t, i) => (t.rank = i + 1));

    // Build timeline
    const timeline: TimelineEntry[] = [];
    // Would populate from recorded tick data

    return {
      arenaId: this.config.id,
      config: this.config,
      totalTicks: this.state.tick,
      duration: this.startTime ? Date.now() - this.startTime.getTime() : 0,
      finalAgentCount: aliveAgents.length,
      survivingAgents: aliveAgents.map((a) => a.config.id),
      rankings,
      teamRankings: teamRankings.length > 0 ? teamRankings : undefined,
      metrics: Object.fromEntries(this.state.metrics),
      timeline,
      emergentBehaviors: [], // Would collect from events
    };
  }

  // ============================================================================
  // Utility Methods
  // ============================================================================

  private positionKey(pos: Position): string {
    return `${Math.floor(pos.x)},${Math.floor(pos.y)}`;
  }

  private isValidPosition(pos: Position): boolean {
    return (
      pos.x >= 0 &&
      pos.x < this.config.dimensions.width &&
      pos.y >= 0 &&
      pos.y < this.config.dimensions.height
    );
  }

  private calculateDistance(a: Position, b: Position): number {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  private getCellsInRange(center: Position, range: number): ArenaCell[] {
    const cells: ArenaCell[] = [];

    for (let dx = -range; dx <= range; dx++) {
      for (let dy = -range; dy <= range; dy++) {
        const pos = { x: center.x + dx, y: center.y + dy };
        if (this.isValidPosition(pos)) {
          const cell = this.state.cells.get(this.positionKey(pos));
          if (cell) cells.push(cell);
        }
      }
    }

    return cells;
  }

  private getNeighborPositions(pos: Position): Position[] {
    const neighbors: Position[] = [];
    const directions = [
      { dx: -1, dy: 0 },
      { dx: 1, dy: 0 },
      { dx: 0, dy: -1 },
      { dx: 0, dy: 1 },
    ];

    for (const { dx, dy } of directions) {
      const newPos = { x: pos.x + dx, y: pos.y + dy };
      if (this.isValidPosition(newPos)) {
        neighbors.push(newPos);
      }
    }

    return neighbors;
  }

  private getTotalResources(): number {
    let total = 0;
    for (const cell of this.state.cells.values()) {
      for (const amount of cell.resources.values()) {
        total += amount;
      }
    }
    return total;
  }

  // ============================================================================
  // Accessors
  // ============================================================================

  getConfig(): ArenaConfig {
    return this.config;
  }

  getState(): ArenaState {
    return this.state;
  }

  getTick(): number {
    return this.state.tick;
  }

  isRunning(): boolean {
    return this.tickInterval !== undefined && !this.state.paused;
  }

  isCompleted(): boolean {
    return this.state.completed;
  }
}

// ============================================================================
// Behavior Engine Interface
// ============================================================================

export interface BehaviorEngine {
  decide(
    agent: SwarmAgent,
    state: ArenaState,
    config: ArenaConfig
  ): AgentAction;
}

/**
 * Random behavior engine
 */
export class RandomBehaviorEngine implements BehaviorEngine {
  decide(
    agent: SwarmAgent,
    state: ArenaState,
    config: ArenaConfig
  ): AgentAction {
    const actions: AgentAction["type"][] = ["move", "collect", "idle"];
    const actionType = actions[Math.floor(Math.random() * actions.length)];

    const action: AgentAction = {
      tick: state.tick,
      type: actionType,
      success: false,
      energyCost: 0,
    };

    if (actionType === "move") {
      action.target = {
        x: agent.state.position.x + (Math.random() - 0.5) * 2,
        y: agent.state.position.y + (Math.random() - 0.5) * 2,
      };
    }

    return action;
  }
}

/**
 * Rule-based behavior engine
 */
export class RuleBasedBehaviorEngine implements BehaviorEngine {
  decide(
    agent: SwarmAgent,
    state: ArenaState,
    _config: ArenaConfig
  ): AgentAction {
    if (agent.config.behavior.type !== "rule_based") {
      return this.defaultAction(state.tick);
    }

    const rules = agent.config.behavior.rules.sort(
      (a, b) => b.priority - a.priority
    );

    for (const rule of rules) {
      if (this.evaluateCondition(rule.condition, agent, state)) {
        return this.parseAction(rule.action, agent, state);
      }
    }

    return this.defaultAction(state.tick);
  }

  private evaluateCondition(
    condition: string,
    agent: SwarmAgent,
    state: ArenaState
  ): boolean {
    // Simple condition evaluation
    if (condition === "low_energy") {
      return agent.state.energy < 30;
    }
    if (condition === "has_resources") {
      return agent.state.resources.size > 0;
    }
    if (condition === "near_resource") {
      const cell = state.cells.get(
        `${Math.floor(agent.state.position.x)},${Math.floor(agent.state.position.y)}`
      );
      return cell
        ? Array.from(cell.resources.values()).some((v) => v > 0)
        : false;
    }
    if (condition === "always") {
      return true;
    }
    return false;
  }

  private parseAction(
    actionStr: string,
    agent: SwarmAgent,
    state: ArenaState
  ): AgentAction {
    const [type, ...params] = actionStr.split(":");

    const action: AgentAction = {
      tick: state.tick,
      type: type as AgentAction["type"],
      success: false,
      energyCost: 0,
    };

    if (type === "move" && params.length >= 2) {
      action.target = {
        x: agent.state.position.x + parseFloat(params[0]!),
        y: agent.state.position.y + parseFloat(params[1]!),
      };
    }

    return action;
  }

  private defaultAction(tick: number): AgentAction {
    return {
      tick,
      type: "idle",
      success: true,
      energyCost: 0,
    };
  }
}

/**
 * Scripted behavior engine
 */
export class ScriptedBehaviorEngine implements BehaviorEngine {
  decide(
    agent: SwarmAgent,
    state: ArenaState,
    _config: ArenaConfig
  ): AgentAction {
    if (agent.config.behavior.type !== "scripted") {
      return {
        tick: state.tick,
        type: "idle",
        success: true,
        energyCost: 0,
      };
    }

    // Execute script (simplified - in production would use a sandbox)
    try {
      // Scripts would be pre-compiled and cached
      const script = agent.config.behavior.script;

      // Very simplified script execution
      if (script.includes("explore")) {
        return {
          tick: state.tick,
          type: "move",
          target: {
            x: agent.state.position.x + (Math.random() - 0.5) * 4,
            y: agent.state.position.y + (Math.random() - 0.5) * 4,
          },
          success: false,
          energyCost: 0,
        };
      }

      if (script.includes("collect")) {
        return {
          tick: state.tick,
          type: "collect",
          success: false,
          energyCost: 0,
        };
      }
    } catch {
      // Script error - idle
    }

    return {
      tick: state.tick,
      type: "idle",
      success: true,
      energyCost: 0,
    };
  }
}
