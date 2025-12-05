/**
 * NEURECTOMY Swarm Agent Spawner
 * @module @neurectomy/experimentation-engine/swarm
 * @agent @OMNISCIENT @NEURAL
 *
 * Agent spawning, templating, and population management
 * for swarm arena simulations.
 */

import { v4 as uuidv4 } from "uuid";
import { z } from "zod";
import type {
  AgentConfig,
  AgentType,
  SwarmAgent,
  SwarmArena,
  Position,
} from "./arena";
import { AgentTypeSchema } from "./arena";

// ============================================================================
// Spawner Configuration Schemas
// ============================================================================

export const AgentTemplateSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string().optional(),
  baseType: AgentTypeSchema,
  attributeRanges: z.object({
    energy: z.object({ min: z.number(), max: z.number() }),
    speed: z.object({ min: z.number(), max: z.number() }),
    visionRange: z.object({ min: z.number(), max: z.number() }),
    strength: z.object({ min: z.number(), max: z.number() }),
    intelligence: z.object({ min: z.number(), max: z.number() }),
    communication: z.object({ min: z.number(), max: z.number() }),
  }),
  behaviors: z.array(z.string()),
  traits: z.array(z.string()).optional(),
  colorScheme: z.string().optional(),
});

export const SpawnPatternSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("random"),
    count: z.number().int().positive(),
  }),
  z.object({
    type: z.literal("cluster"),
    count: z.number().int().positive(),
    center: z.object({ x: z.number(), y: z.number() }),
    radius: z.number().positive(),
  }),
  z.object({
    type: z.literal("grid"),
    rows: z.number().int().positive(),
    cols: z.number().int().positive(),
    spacing: z.number().positive(),
    origin: z.object({ x: z.number(), y: z.number() }),
  }),
  z.object({
    type: z.literal("ring"),
    count: z.number().int().positive(),
    center: z.object({ x: z.number(), y: z.number() }),
    radius: z.number().positive(),
  }),
  z.object({
    type: z.literal("line"),
    count: z.number().int().positive(),
    start: z.object({ x: z.number(), y: z.number() }),
    end: z.object({ x: z.number(), y: z.number() }),
  }),
  z.object({
    type: z.literal("corner"),
    count: z.number().int().positive(),
    corner: z.enum(["top_left", "top_right", "bottom_left", "bottom_right"]),
    spread: z.number().positive(),
  }),
  z.object({
    type: z.literal("positions"),
    positions: z.array(z.object({ x: z.number(), y: z.number() })),
  }),
]);

export const PopulationConfigSchema = z.object({
  templateId: z.string(),
  team: z.string().optional(),
  spawnPattern: SpawnPatternSchema,
  mutationRate: z.number().min(0).max(1).default(0),
  namePrefix: z.string().optional(),
});

export const SpawnerConfigSchema = z.object({
  arena: z.custom<SwarmArena>(),
  templates: z.array(AgentTemplateSchema),
  defaultMutationRate: z.number().min(0).max(1).default(0.1),
  attributeNoise: z.number().min(0).max(1).default(0.1),
  uniqueNames: z.boolean().default(true),
});

// ============================================================================
// Types
// ============================================================================

export type AgentTemplate = z.infer<typeof AgentTemplateSchema>;
export type SpawnPattern = z.infer<typeof SpawnPatternSchema>;
export type PopulationConfig = z.infer<typeof PopulationConfigSchema>;
export type SpawnerConfig = z.infer<typeof SpawnerConfigSchema>;

export interface SpawnResult {
  success: boolean;
  agents: SwarmAgent[];
  failed: number;
  errors: string[];
}

export interface GenerationStats {
  generation: number;
  totalSpawned: number;
  templateDistribution: Record<string, number>;
  teamDistribution: Record<string, number>;
  averageAttributes: Record<string, number>;
}

// ============================================================================
// Agent Spawner Implementation
// ============================================================================

/**
 * AgentSpawner - Create and manage agent populations
 *
 * Features:
 * - Template-based agent creation
 * - Multiple spawn patterns
 * - Attribute variation and mutation
 * - Population management
 * - Team assignment
 */
export class AgentSpawner {
  private arena: SwarmArena;
  private templates: Map<string, AgentTemplate> = new Map();
  private defaultMutationRate: number;
  private attributeNoise: number;
  private uniqueNames: boolean;
  private nameCounter: Map<string, number> = new Map();
  private generation: number = 0;
  private spawnHistory: SpawnResult[] = [];

  constructor(config: SpawnerConfig) {
    this.arena = config.arena;
    this.defaultMutationRate = config.defaultMutationRate;
    this.attributeNoise = config.attributeNoise;
    this.uniqueNames = config.uniqueNames;

    for (const template of config.templates) {
      this.templates.set(template.id, template);
    }
  }

  // ============================================================================
  // Template Management
  // ============================================================================

  /**
   * Register a new agent template
   */
  registerTemplate(template: AgentTemplate): void {
    this.templates.set(template.id, template);
  }

  /**
   * Get a template by ID
   */
  getTemplate(templateId: string): AgentTemplate | undefined {
    return this.templates.get(templateId);
  }

  /**
   * List all templates
   */
  listTemplates(): AgentTemplate[] {
    return Array.from(this.templates.values());
  }

  /**
   * Remove a template
   */
  removeTemplate(templateId: string): boolean {
    return this.templates.delete(templateId);
  }

  // ============================================================================
  // Spawning Methods
  // ============================================================================

  /**
   * Spawn agents based on population config
   */
  spawnPopulation(config: PopulationConfig): SpawnResult {
    const template = this.templates.get(config.templateId);
    if (!template) {
      return {
        success: false,
        agents: [],
        failed: 0,
        errors: [`Template not found: ${config.templateId}`],
      };
    }

    const positions = this.generatePositions(config.spawnPattern);
    const result: SpawnResult = {
      success: true,
      agents: [],
      failed: 0,
      errors: [],
    };

    for (const position of positions) {
      try {
        const agentConfig = this.createAgentConfig(
          template,
          position,
          config.team,
          config.namePrefix,
          config.mutationRate
        );

        const agent = this.arena.spawnAgent(agentConfig);
        result.agents.push(agent);
      } catch (error) {
        result.failed++;
        result.errors.push(
          error instanceof Error ? error.message : String(error)
        );
      }
    }

    result.success = result.failed === 0;
    this.spawnHistory.push(result);
    this.generation++;

    return result;
  }

  /**
   * Spawn a single agent
   */
  spawnSingle(
    templateId: string,
    position: Position,
    team?: string,
    overrides?: Partial<AgentConfig>
  ): SwarmAgent | undefined {
    const template = this.templates.get(templateId);
    if (!template) return undefined;

    try {
      const agentConfig = this.createAgentConfig(
        template,
        position,
        team,
        undefined,
        this.defaultMutationRate
      );

      // Apply overrides
      if (overrides) {
        Object.assign(agentConfig, overrides);
      }

      return this.arena.spawnAgent(agentConfig);
    } catch {
      return undefined;
    }
  }

  /**
   * Spawn agents from multiple populations
   */
  spawnMultiplePopulations(configs: PopulationConfig[]): SpawnResult[] {
    return configs.map((config) => this.spawnPopulation(config));
  }

  // ============================================================================
  // Position Generation
  // ============================================================================

  private generatePositions(pattern: SpawnPattern): Position[] {
    const arenaConfig = this.arena.getConfig();
    const width = arenaConfig.dimensions.width;
    const height = arenaConfig.dimensions.height;

    switch (pattern.type) {
      case "random":
        return this.generateRandomPositions(pattern.count, width, height);

      case "cluster":
        return this.generateClusterPositions(
          pattern.count,
          pattern.center,
          pattern.radius,
          width,
          height
        );

      case "grid":
        return this.generateGridPositions(
          pattern.rows,
          pattern.cols,
          pattern.spacing,
          pattern.origin
        );

      case "ring":
        return this.generateRingPositions(
          pattern.count,
          pattern.center,
          pattern.radius
        );

      case "line":
        return this.generateLinePositions(
          pattern.count,
          pattern.start,
          pattern.end
        );

      case "corner":
        return this.generateCornerPositions(
          pattern.count,
          pattern.corner,
          pattern.spread,
          width,
          height
        );

      case "positions":
        return pattern.positions;
    }
  }

  private generateRandomPositions(
    count: number,
    width: number,
    height: number
  ): Position[] {
    const positions: Position[] = [];
    for (let i = 0; i < count; i++) {
      positions.push({
        x: Math.random() * width,
        y: Math.random() * height,
      });
    }
    return positions;
  }

  private generateClusterPositions(
    count: number,
    center: Position,
    radius: number,
    width: number,
    height: number
  ): Position[] {
    const positions: Position[] = [];
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r = Math.random() * radius;
      const x = Math.max(
        0,
        Math.min(width - 1, center.x + Math.cos(angle) * r)
      );
      const y = Math.max(
        0,
        Math.min(height - 1, center.y + Math.sin(angle) * r)
      );
      positions.push({ x, y });
    }
    return positions;
  }

  private generateGridPositions(
    rows: number,
    cols: number,
    spacing: number,
    origin: Position
  ): Position[] {
    const positions: Position[] = [];
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        positions.push({
          x: origin.x + col * spacing,
          y: origin.y + row * spacing,
        });
      }
    }
    return positions;
  }

  private generateRingPositions(
    count: number,
    center: Position,
    radius: number
  ): Position[] {
    const positions: Position[] = [];
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2;
      positions.push({
        x: center.x + Math.cos(angle) * radius,
        y: center.y + Math.sin(angle) * radius,
      });
    }
    return positions;
  }

  private generateLinePositions(
    count: number,
    start: Position,
    end: Position
  ): Position[] {
    const positions: Position[] = [];
    for (let i = 0; i < count; i++) {
      const t = count > 1 ? i / (count - 1) : 0;
      positions.push({
        x: start.x + (end.x - start.x) * t,
        y: start.y + (end.y - start.y) * t,
      });
    }
    return positions;
  }

  private generateCornerPositions(
    count: number,
    corner: "top_left" | "top_right" | "bottom_left" | "bottom_right",
    spread: number,
    width: number,
    height: number
  ): Position[] {
    let centerX: number, centerY: number;

    switch (corner) {
      case "top_left":
        centerX = spread;
        centerY = spread;
        break;
      case "top_right":
        centerX = width - spread;
        centerY = spread;
        break;
      case "bottom_left":
        centerX = spread;
        centerY = height - spread;
        break;
      case "bottom_right":
        centerX = width - spread;
        centerY = height - spread;
        break;
    }

    return this.generateClusterPositions(
      count,
      { x: centerX, y: centerY },
      spread,
      width,
      height
    );
  }

  // ============================================================================
  // Agent Configuration
  // ============================================================================

  private createAgentConfig(
    template: AgentTemplate,
    position: Position,
    team?: string,
    namePrefix?: string,
    mutationRate?: number
  ): Omit<AgentConfig, "id"> {
    const name = this.generateName(template, namePrefix);
    const attributes = this.generateAttributes(template, mutationRate);
    const behavior = this.selectBehavior(template);

    return {
      name,
      type: this.mutateType(template.baseType, mutationRate),
      team,
      position,
      attributes,
      behavior,
      memory: {},
    };
  }

  private generateName(template: AgentTemplate, prefix?: string): string {
    const baseName = prefix || template.name;

    if (this.uniqueNames) {
      const count = this.nameCounter.get(baseName) || 0;
      this.nameCounter.set(baseName, count + 1);
      return `${baseName}_${count + 1}`;
    }

    return `${baseName}_${uuidv4().slice(0, 8)}`;
  }

  private generateAttributes(
    template: AgentTemplate,
    mutationRate?: number
  ): AgentConfig["attributes"] {
    const rate = mutationRate ?? this.defaultMutationRate;
    const noise = this.attributeNoise;

    const randomInRange = (min: number, max: number): number => {
      const base = min + Math.random() * (max - min);
      const mutation = (Math.random() - 0.5) * 2 * rate * (max - min);
      const variation = (Math.random() - 0.5) * 2 * noise * (max - min);
      return Math.max(min, Math.min(max, base + mutation + variation));
    };

    return {
      energy: randomInRange(
        template.attributeRanges.energy.min,
        template.attributeRanges.energy.max
      ),
      speed: randomInRange(
        template.attributeRanges.speed.min,
        template.attributeRanges.speed.max
      ),
      visionRange: randomInRange(
        template.attributeRanges.visionRange.min,
        template.attributeRanges.visionRange.max
      ),
      strength: randomInRange(
        template.attributeRanges.strength.min,
        template.attributeRanges.strength.max
      ),
      intelligence: randomInRange(
        template.attributeRanges.intelligence.min,
        template.attributeRanges.intelligence.max
      ),
      communication: randomInRange(
        template.attributeRanges.communication.min,
        template.attributeRanges.communication.max
      ),
    };
  }

  private selectBehavior(template: AgentTemplate): AgentConfig["behavior"] {
    const behaviorType =
      template.behaviors[Math.floor(Math.random() * template.behaviors.length)];

    switch (behaviorType) {
      case "random":
        return { type: "random" };

      case "rule_based":
        return {
          type: "rule_based",
          rules: [
            { condition: "near_resource", action: "collect", priority: 10 },
            { condition: "low_energy", action: "defend", priority: 5 },
            { condition: "always", action: "move:1:0", priority: 0 },
          ],
        };

      case "scripted":
        return {
          type: "scripted",
          script: "explore()",
        };

      default:
        return { type: "random" };
    }
  }

  private mutateType(baseType: AgentType, mutationRate?: number): AgentType {
    const rate = mutationRate ?? this.defaultMutationRate;

    if (Math.random() < rate * 0.1) {
      const types: AgentType[] = [
        "explorer",
        "exploiter",
        "communicator",
        "predator",
        "defender",
        "neutral",
        "hybrid",
      ];
      return types[Math.floor(Math.random() * types.length)]!;
    }

    return baseType;
  }

  // ============================================================================
  // Statistics & History
  // ============================================================================

  /**
   * Get current generation number
   */
  getGeneration(): number {
    return this.generation;
  }

  /**
   * Get spawn history
   */
  getSpawnHistory(): SpawnResult[] {
    return this.spawnHistory;
  }

  /**
   * Get generation statistics
   */
  getGenerationStats(): GenerationStats {
    const templateDistribution: Record<string, number> = {};
    const teamDistribution: Record<string, number> = {};
    const attributeSums: Record<string, number> = {
      energy: 0,
      speed: 0,
      visionRange: 0,
      strength: 0,
      intelligence: 0,
      communication: 0,
    };

    let totalSpawned = 0;

    for (const result of this.spawnHistory) {
      for (const agent of result.agents) {
        totalSpawned++;

        // Template distribution (using type as proxy)
        const type = agent.config.type;
        templateDistribution[type] = (templateDistribution[type] || 0) + 1;

        // Team distribution
        if (agent.config.team) {
          teamDistribution[agent.config.team] =
            (teamDistribution[agent.config.team] || 0) + 1;
        }

        // Attribute sums
        for (const [attr, value] of Object.entries(agent.config.attributes)) {
          attributeSums[attr] = (attributeSums[attr] || 0) + value;
        }
      }
    }

    // Calculate averages
    const averageAttributes: Record<string, number> = {};
    if (totalSpawned > 0) {
      for (const [attr, sum] of Object.entries(attributeSums)) {
        averageAttributes[attr] = sum / totalSpawned;
      }
    }

    return {
      generation: this.generation,
      totalSpawned,
      templateDistribution,
      teamDistribution,
      averageAttributes,
    };
  }

  /**
   * Clear spawn history
   */
  clearHistory(): void {
    this.spawnHistory = [];
    this.nameCounter.clear();
  }
}

// ============================================================================
// Predefined Templates
// ============================================================================

export const PredefinedTemplates: AgentTemplate[] = [
  {
    id: "scout",
    name: "Scout",
    description: "Fast explorer with high vision range",
    baseType: "explorer",
    attributeRanges: {
      energy: { min: 80, max: 100 },
      speed: { min: 2, max: 3 },
      visionRange: { min: 8, max: 12 },
      strength: { min: 0.5, max: 1 },
      intelligence: { min: 1, max: 2 },
      communication: { min: 1, max: 2 },
    },
    behaviors: ["random", "rule_based"],
    traits: ["curious", "cautious"],
    colorScheme: "#4CAF50",
  },
  {
    id: "harvester",
    name: "Harvester",
    description: "Efficient resource collector",
    baseType: "exploiter",
    attributeRanges: {
      energy: { min: 100, max: 150 },
      speed: { min: 1, max: 1.5 },
      visionRange: { min: 4, max: 6 },
      strength: { min: 2, max: 3 },
      intelligence: { min: 0.5, max: 1 },
      communication: { min: 0.5, max: 1 },
    },
    behaviors: ["rule_based"],
    traits: ["diligent", "focused"],
    colorScheme: "#FFC107",
  },
  {
    id: "guardian",
    name: "Guardian",
    description: "Defensive protector",
    baseType: "defender",
    attributeRanges: {
      energy: { min: 150, max: 200 },
      speed: { min: 0.5, max: 1 },
      visionRange: { min: 5, max: 7 },
      strength: { min: 3, max: 5 },
      intelligence: { min: 1, max: 1.5 },
      communication: { min: 1, max: 2 },
    },
    behaviors: ["rule_based"],
    traits: ["protective", "vigilant"],
    colorScheme: "#2196F3",
  },
  {
    id: "hunter",
    name: "Hunter",
    description: "Aggressive predator",
    baseType: "predator",
    attributeRanges: {
      energy: { min: 120, max: 160 },
      speed: { min: 2, max: 3 },
      visionRange: { min: 6, max: 10 },
      strength: { min: 4, max: 6 },
      intelligence: { min: 2, max: 3 },
      communication: { min: 0.5, max: 1 },
    },
    behaviors: ["rule_based", "scripted"],
    traits: ["aggressive", "cunning"],
    colorScheme: "#F44336",
  },
  {
    id: "messenger",
    name: "Messenger",
    description: "Information spreader",
    baseType: "communicator",
    attributeRanges: {
      energy: { min: 70, max: 90 },
      speed: { min: 2.5, max: 3.5 },
      visionRange: { min: 4, max: 6 },
      strength: { min: 0.5, max: 1 },
      intelligence: { min: 2, max: 3 },
      communication: { min: 4, max: 6 },
    },
    behaviors: ["rule_based"],
    traits: ["social", "quick"],
    colorScheme: "#9C27B0",
  },
  {
    id: "generalist",
    name: "Generalist",
    description: "Balanced all-rounder",
    baseType: "hybrid",
    attributeRanges: {
      energy: { min: 100, max: 120 },
      speed: { min: 1.5, max: 2 },
      visionRange: { min: 5, max: 7 },
      strength: { min: 1.5, max: 2.5 },
      intelligence: { min: 1.5, max: 2.5 },
      communication: { min: 1.5, max: 2.5 },
    },
    behaviors: ["random", "rule_based"],
    traits: ["adaptable", "versatile"],
    colorScheme: "#607D8B",
  },
];
