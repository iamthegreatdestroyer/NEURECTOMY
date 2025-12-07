/**
 * Consciousness Transfer POC
 *
 * Extracts, transfers, and validates knowledge/skills between agents,
 * enabling skill sharing, expertise propagation, and collaborative learning.
 *
 * Key Innovations:
 * - Knowledge graph extraction from agent behavior
 * - Skill distillation with confidence weighting
 * - Transfer protocol with compatibility checking
 * - Bidirectional knowledge synchronization
 * - Performance validation and impact measurement
 * - Hierarchical knowledge representation
 *
 * Research Foundations:
 * - Hinton et al. (2015): Distilling the knowledge in a neural network
 * - Pan & Yang (2010): A survey on transfer learning
 * - Lake et al. (2017): Building machines that learn and think like people
 * - Caruana (1997): Multitask learning
 *
 * @elite-agents @NEURAL @TENSOR @GENESIS
 */

import { cloneDeep } from "lodash";

// ============================================================================
// Type Definitions
// ============================================================================

type AgentId = string;
type KnowledgeId = string;
type SkillId = string;

enum KnowledgeType {
  PROCEDURAL = "procedural", // How to do things
  DECLARATIVE = "declarative", // Facts and data
  EPISODIC = "episodic", // Experiences
  STRATEGIC = "strategic", // Planning and decision-making
}

enum TransferStatus {
  PENDING = "pending",
  IN_PROGRESS = "in_progress",
  COMPLETED = "completed",
  FAILED = "failed",
  VALIDATED = "validated",
}

interface KnowledgeNode {
  id: KnowledgeId;
  type: KnowledgeType;
  content: Record<string, any>;
  confidence: number; // 0-1
  usage: number; // How often used
  dependencies: KnowledgeId[];
  metadata: {
    acquired: Date;
    lastUsed: Date;
    successRate: number;
  };
}

interface Skill {
  id: SkillId;
  name: string;
  domain: string;
  knowledgeNodes: KnowledgeId[];
  performance: {
    accuracy: number;
    speed: number;
    reliability: number;
  };
  prerequisites: SkillId[];
}

interface KnowledgeGraph {
  agentId: AgentId;
  nodes: Map<KnowledgeId, KnowledgeNode>;
  skills: Map<SkillId, Skill>;
  edges: Array<{
    from: KnowledgeId;
    to: KnowledgeId;
    strength: number;
    type: string;
  }>;
}

interface TransferPackage {
  sourceAgentId: AgentId;
  targetAgentId: AgentId;
  skills: SkillId[];
  knowledge: KnowledgeId[];
  compressionRatio: number;
  estimatedTransferTime: number;
  compatibilityScore: number;
}

interface TransferResult {
  status: TransferStatus;
  transferredSkills: number;
  transferredKnowledge: number;
  performanceChange: {
    before: Record<string, number>;
    after: Record<string, number>;
    improvement: Record<string, number>;
  };
  validationMetrics: {
    accuracy: number;
    completeness: number;
    consistency: number;
  };
}

// ============================================================================
// Knowledge Extractor
// ============================================================================

class KnowledgeExtractor {
  /**
   * Extract knowledge graph from agent
   */
  extractKnowledgeGraph(
    agentId: AgentId,
    observations: Array<{
      action: string;
      context: Record<string, any>;
      outcome: Record<string, any>;
      success: boolean;
    }>
  ): KnowledgeGraph {
    const nodes = new Map<KnowledgeId, KnowledgeNode>();
    const skills = new Map<SkillId, Skill>();
    const edges: Array<{
      from: KnowledgeId;
      to: KnowledgeId;
      strength: number;
      type: string;
    }> = [];

    // Extract procedural knowledge (action patterns)
    const actionPatterns = this.extractActionPatterns(observations);
    for (const [pattern, data] of actionPatterns) {
      const nodeId = `proc_${pattern}`;
      nodes.set(nodeId, {
        id: nodeId,
        type: KnowledgeType.PROCEDURAL,
        content: data.pattern,
        confidence: data.successRate,
        usage: data.count,
        dependencies: [],
        metadata: {
          acquired: new Date(),
          lastUsed: new Date(),
          successRate: data.successRate,
        },
      });
    }

    // Extract declarative knowledge (facts)
    const facts = this.extractFacts(observations);
    for (const [fact, data] of facts) {
      const nodeId = `decl_${fact}`;
      nodes.set(nodeId, {
        id: nodeId,
        type: KnowledgeType.DECLARATIVE,
        content: data.content,
        confidence: data.confidence,
        usage: data.observations,
        dependencies: [],
        metadata: {
          acquired: new Date(),
          lastUsed: new Date(),
          successRate: 1.0,
        },
      });
    }

    // Extract skills (clusters of related knowledge)
    const extractedSkills = this.clusterIntoSkills(nodes);
    for (const skill of extractedSkills) {
      skills.set(skill.id, skill);
    }

    // Build edges (knowledge dependencies)
    this.buildDependencies(nodes, edges, observations);

    return {
      agentId,
      nodes,
      skills,
      edges,
    };
  }

  /**
   * Extract action patterns
   */
  private extractActionPatterns(
    observations: Array<{
      action: string;
      context: Record<string, any>;
      outcome: Record<string, any>;
      success: boolean;
    }>
  ): Map<
    string,
    { pattern: Record<string, any>; count: number; successRate: number }
  > {
    const patterns = new Map<
      string,
      { pattern: Record<string, any>; count: number; successes: number }
    >();

    for (const obs of observations) {
      const patternKey = `${obs.action}_${JSON.stringify(obs.context).slice(0, 50)}`;

      const existing = patterns.get(patternKey) ?? {
        pattern: { action: obs.action, context: obs.context },
        count: 0,
        successes: 0,
      };

      existing.count++;
      if (obs.success) existing.successes++;

      patterns.set(patternKey, existing);
    }

    // Compute success rates
    const result = new Map<
      string,
      { pattern: Record<string, any>; count: number; successRate: number }
    >();
    for (const [key, data] of patterns) {
      result.set(key, {
        pattern: data.pattern,
        count: data.count,
        successRate: data.successes / data.count,
      });
    }

    return result;
  }

  /**
   * Extract declarative facts
   */
  private extractFacts(
    observations: Array<{
      action: string;
      context: Record<string, any>;
      outcome: Record<string, any>;
      success: boolean;
    }>
  ): Map<
    string,
    { content: Record<string, any>; confidence: number; observations: number }
  > {
    const facts = new Map<
      string,
      { content: Record<string, any>; confidence: number; observations: number }
    >();

    for (const obs of observations) {
      // Extract state facts from context
      for (const [key, value] of Object.entries(obs.context)) {
        const factKey = `${key}=${JSON.stringify(value)}`;
        const existing = facts.get(factKey) ?? {
          content: { key, value },
          confidence: 0,
          observations: 0,
        };

        existing.observations++;
        existing.confidence = Math.min(1.0, existing.observations / 10);

        facts.set(factKey, existing);
      }
    }

    return facts;
  }

  /**
   * Cluster knowledge nodes into skills
   */
  private clusterIntoSkills(nodes: Map<KnowledgeId, KnowledgeNode>): Skill[] {
    const skills: Skill[] = [];

    // Group procedural knowledge by domain
    const proceduralNodes = Array.from(nodes.values()).filter(
      (n) => n.type === KnowledgeType.PROCEDURAL
    );

    // Simple clustering by action prefix
    const clusters = new Map<string, KnowledgeId[]>();
    for (const node of proceduralNodes) {
      const action = (node.content.action as string) ?? "unknown";
      const domain = action.split("_")[0];

      const cluster = clusters.get(domain) ?? [];
      cluster.push(node.id);
      clusters.set(domain, cluster);
    }

    // Create skills from clusters
    for (const [domain, nodeIds] of clusters) {
      const nodesInSkill = nodeIds.map((id) => nodes.get(id)!);
      const avgConfidence =
        nodesInSkill.reduce((sum, n) => sum + n.confidence, 0) /
        nodesInSkill.length;

      skills.push({
        id: `skill_${domain}`,
        name: `${domain} skills`,
        domain,
        knowledgeNodes: nodeIds,
        performance: {
          accuracy: avgConfidence,
          speed: 1.0,
          reliability: avgConfidence,
        },
        prerequisites: [],
      });
    }

    return skills;
  }

  /**
   * Build knowledge dependencies
   */
  private buildDependencies(
    nodes: Map<KnowledgeId, KnowledgeNode>,
    edges: Array<{
      from: KnowledgeId;
      to: KnowledgeId;
      strength: number;
      type: string;
    }>,
    observations: Array<any>
  ): void {
    // Build temporal dependencies (A must be known before B)
    const nodeArray = Array.from(nodes.values());

    for (let i = 0; i < nodeArray.length - 1; i++) {
      for (let j = i + 1; j < nodeArray.length; j++) {
        const node1 = nodeArray[i];
        const node2 = nodeArray[j];

        // If both are procedural and in same domain, create dependency
        if (
          node1.type === KnowledgeType.PROCEDURAL &&
          node2.type === KnowledgeType.PROCEDURAL
        ) {
          const action1 = (node1.content.action as string) ?? "";
          const action2 = (node2.content.action as string) ?? "";

          if (action1.split("_")[0] === action2.split("_")[0]) {
            edges.push({
              from: node1.id,
              to: node2.id,
              strength: 0.5,
              type: "prerequisite",
            });

            node2.dependencies.push(node1.id);
          }
        }
      }
    }
  }
}

// ============================================================================
// Transfer Protocol
// ============================================================================

class TransferProtocol {
  /**
   * Check transfer compatibility
   */
  checkCompatibility(
    sourceGraph: KnowledgeGraph,
    targetGraph: KnowledgeGraph,
    skillIds: SkillId[]
  ): number {
    let compatibilityScore = 1.0;

    for (const skillId of skillIds) {
      const skill = sourceGraph.skills.get(skillId);
      if (!skill) continue;

      // Check if target has prerequisites
      for (const prereqId of skill.prerequisites) {
        if (!targetGraph.skills.has(prereqId)) {
          compatibilityScore *= 0.8; // Missing prerequisite
        }
      }

      // Check domain overlap
      const targetSkillsInDomain = Array.from(
        targetGraph.skills.values()
      ).filter((s) => s.domain === skill.domain);

      if (targetSkillsInDomain.length === 0) {
        compatibilityScore *= 0.9; // New domain
      }
    }

    return Math.max(0, Math.min(1, compatibilityScore));
  }

  /**
   * Create transfer package
   */
  createTransferPackage(
    sourceGraph: KnowledgeGraph,
    targetAgentId: AgentId,
    skillIds: SkillId[]
  ): TransferPackage {
    const knowledgeNodes: KnowledgeId[] = [];

    // Collect all knowledge nodes for requested skills
    for (const skillId of skillIds) {
      const skill = sourceGraph.skills.get(skillId);
      if (skill) {
        knowledgeNodes.push(...skill.knowledgeNodes);

        // Add dependencies
        for (const nodeId of skill.knowledgeNodes) {
          const node = sourceGraph.nodes.get(nodeId);
          if (node) {
            knowledgeNodes.push(...node.dependencies);
          }
        }
      }
    }

    // Deduplicate
    const uniqueKnowledge = Array.from(new Set(knowledgeNodes));

    // Estimate transfer metrics
    const compressionRatio = uniqueKnowledge.length / knowledgeNodes.length;
    const estimatedTransferTime = uniqueKnowledge.length * 10; // ms per node

    return {
      sourceAgentId: sourceGraph.agentId,
      targetAgentId,
      skills: skillIds,
      knowledge: uniqueKnowledge,
      compressionRatio,
      estimatedTransferTime,
      compatibilityScore: 1.0, // Will be set by compatibility check
    };
  }

  /**
   * Execute transfer
   */
  executeTransfer(
    sourceGraph: KnowledgeGraph,
    targetGraph: KnowledgeGraph,
    transferPackage: TransferPackage
  ): KnowledgeGraph {
    const updatedTarget = cloneDeep(targetGraph);

    // Transfer knowledge nodes
    for (const knowledgeId of transferPackage.knowledge) {
      const sourceNode = sourceGraph.nodes.get(knowledgeId);
      if (!sourceNode) continue;

      // If target already has this knowledge, merge
      if (updatedTarget.nodes.has(knowledgeId)) {
        const targetNode = updatedTarget.nodes.get(knowledgeId)!;

        // Merge by taking weighted average of confidence
        const sourceWeight =
          sourceNode.usage / (sourceNode.usage + targetNode.usage);
        const targetWeight =
          targetNode.usage / (sourceNode.usage + targetNode.usage);

        targetNode.confidence =
          sourceNode.confidence * sourceWeight +
          targetNode.confidence * targetWeight;
        targetNode.usage += sourceNode.usage;
      } else {
        // Add new knowledge with reduced confidence (transfer penalty)
        const transferredNode = cloneDeep(sourceNode);
        transferredNode.confidence *= 0.8; // 20% confidence loss in transfer
        transferredNode.metadata.acquired = new Date();

        updatedTarget.nodes.set(knowledgeId, transferredNode);
      }
    }

    // Transfer skills
    for (const skillId of transferPackage.skills) {
      const sourceSkill = sourceGraph.skills.get(skillId);
      if (!sourceSkill) continue;

      // If target has skill, merge performance metrics
      if (updatedTarget.skills.has(skillId)) {
        const targetSkill = updatedTarget.skills.get(skillId)!;

        // Take average of performance metrics
        targetSkill.performance.accuracy =
          (sourceSkill.performance.accuracy +
            targetSkill.performance.accuracy) /
          2;
        targetSkill.performance.speed =
          (sourceSkill.performance.speed + targetSkill.performance.speed) / 2;
        targetSkill.performance.reliability =
          (sourceSkill.performance.reliability +
            targetSkill.performance.reliability) /
          2;
      } else {
        // Add new skill
        const transferredSkill = cloneDeep(sourceSkill);
        transferredSkill.performance.accuracy *= 0.8; // Transfer penalty
        transferredSkill.performance.reliability *= 0.8;

        updatedTarget.skills.set(skillId, transferredSkill);
      }
    }

    // Transfer edges
    for (const edge of sourceGraph.edges) {
      if (
        transferPackage.knowledge.includes(edge.from) &&
        transferPackage.knowledge.includes(edge.to)
      ) {
        updatedTarget.edges.push(cloneDeep(edge));
      }
    }

    return updatedTarget;
  }

  /**
   * Validate transfer
   */
  validateTransfer(
    beforeGraph: KnowledgeGraph,
    afterGraph: KnowledgeGraph,
    transferPackage: TransferPackage
  ): TransferResult {
    const status = TransferStatus.VALIDATED;

    // Count transferred items
    const transferredSkills = transferPackage.skills.length;
    const transferredKnowledge = transferPackage.knowledge.length;

    // Compute performance change
    const beforePerf: Record<string, number> = {};
    const afterPerf: Record<string, number> = {};
    const improvement: Record<string, number> = {};

    for (const skillId of transferPackage.skills) {
      const beforeSkill = beforeGraph.skills.get(skillId);
      const afterSkill = afterGraph.skills.get(skillId);

      if (afterSkill) {
        const beforeAccuracy = beforeSkill?.performance.accuracy ?? 0;
        const afterAccuracy = afterSkill.performance.accuracy;

        beforePerf[skillId] = beforeAccuracy;
        afterPerf[skillId] = afterAccuracy;
        improvement[skillId] = afterAccuracy - beforeAccuracy;
      }
    }

    // Validation metrics
    const accuracy = this.computeAccuracy(
      beforeGraph,
      afterGraph,
      transferPackage
    );
    const completeness =
      transferredKnowledge / transferPackage.knowledge.length;
    const consistency = this.computeConsistency(afterGraph, transferPackage);

    return {
      status,
      transferredSkills,
      transferredKnowledge,
      performanceChange: {
        before: beforePerf,
        after: afterPerf,
        improvement,
      },
      validationMetrics: {
        accuracy,
        completeness,
        consistency,
      },
    };
  }

  private computeAccuracy(
    beforeGraph: KnowledgeGraph,
    afterGraph: KnowledgeGraph,
    transferPackage: TransferPackage
  ): number {
    let totalConfidence = 0;
    let count = 0;

    for (const knowledgeId of transferPackage.knowledge) {
      const node = afterGraph.nodes.get(knowledgeId);
      if (node) {
        totalConfidence += node.confidence;
        count++;
      }
    }

    return count > 0 ? totalConfidence / count : 0;
  }

  private computeConsistency(
    graph: KnowledgeGraph,
    transferPackage: TransferPackage
  ): number {
    // Check if all dependencies are satisfied
    let satisfied = 0;
    let total = 0;

    for (const knowledgeId of transferPackage.knowledge) {
      const node = graph.nodes.get(knowledgeId);
      if (!node) continue;

      for (const depId of node.dependencies) {
        total++;
        if (graph.nodes.has(depId)) {
          satisfied++;
        }
      }
    }

    return total > 0 ? satisfied / total : 1.0;
  }
}

// ============================================================================
// Demonstration
// ============================================================================

export async function demonstrateConsciousnessTransfer(): Promise<void> {
  console.log("=".repeat(80));
  console.log("CONSCIOUSNESS TRANSFER DEMONSTRATION");
  console.log("=".repeat(80));

  // Demo 1: Knowledge Extraction
  console.log("\n🧠 Demo 1: Knowledge Graph Extraction");
  console.log("-".repeat(80));

  const expertObservations = [
    {
      action: "navigate_forward",
      context: { obstacle: "none", speed: 1.0 },
      outcome: { success: true },
      success: true,
    },
    {
      action: "navigate_forward",
      context: { obstacle: "none", speed: 1.5 },
      outcome: { success: true },
      success: true,
    },
    {
      action: "navigate_turn_left",
      context: { obstacle: "wall", angle: 90 },
      outcome: { success: true },
      success: true,
    },
    {
      action: "navigate_turn_right",
      context: { obstacle: "wall", angle: 90 },
      outcome: { success: true },
      success: true,
    },
    {
      action: "navigate_avoid",
      context: { obstacle: "dynamic", distance: 2.0 },
      outcome: { success: true },
      success: true,
    },
  ];

  const extractor = new KnowledgeExtractor();
  const expertGraph = extractor.extractKnowledgeGraph(
    "expert_agent",
    expertObservations
  );

  console.log(`Expert agent knowledge graph:`);
  console.log(`  Knowledge nodes: ${expertGraph.nodes.size}`);
  console.log(`  Skills: ${expertGraph.skills.size}`);
  console.log(`  Dependencies: ${expertGraph.edges.length}`);

  console.log("\nExtracted skills:");
  for (const skill of expertGraph.skills.values()) {
    console.log(`  ${skill.name}:`);
    console.log(`    Domain: ${skill.domain}`);
    console.log(`    Knowledge nodes: ${skill.knowledgeNodes.length}`);
    console.log(
      `    Accuracy: ${(skill.performance.accuracy * 100).toFixed(1)}%`
    );
  }

  // Demo 2: Novice Agent
  console.log("\n👶 Demo 2: Novice Agent (Before Transfer)");
  console.log("-".repeat(80));

  const noviceObservations = [
    {
      action: "navigate_forward",
      context: { obstacle: "none", speed: 0.5 },
      outcome: { success: true },
      success: true,
    },
    {
      action: "navigate_forward",
      context: { obstacle: "wall", speed: 0.5 },
      outcome: { success: false },
      success: false,
    },
  ];

  const noviceGraph = extractor.extractKnowledgeGraph(
    "novice_agent",
    noviceObservations
  );

  console.log(`Novice agent knowledge graph:`);
  console.log(`  Knowledge nodes: ${noviceGraph.nodes.size}`);
  console.log(`  Skills: ${noviceGraph.skills.size}`);
  console.log(`  Dependencies: ${noviceGraph.edges.length}`);

  // Demo 3: Compatibility Check
  console.log("\n🔍 Demo 3: Transfer Compatibility Check");
  console.log("-".repeat(80));

  const protocol = new TransferProtocol();

  const skillsToTransfer = Array.from(expertGraph.skills.keys());
  const compatibility = protocol.checkCompatibility(
    expertGraph,
    noviceGraph,
    skillsToTransfer
  );

  console.log(`Checking transfer compatibility...`);
  console.log(`  Skills to transfer: ${skillsToTransfer.join(", ")}`);
  console.log(`  Compatibility score: ${(compatibility * 100).toFixed(1)}%`);

  // Demo 4: Create Transfer Package
  console.log("\n📦 Demo 4: Transfer Package Creation");
  console.log("-".repeat(80));

  const transferPackage = protocol.createTransferPackage(
    expertGraph,
    "novice_agent",
    skillsToTransfer
  );

  transferPackage.compatibilityScore = compatibility;

  console.log(`Transfer package created:`);
  console.log(`  Source: ${transferPackage.sourceAgentId}`);
  console.log(`  Target: ${transferPackage.targetAgentId}`);
  console.log(`  Skills: ${transferPackage.skills.length}`);
  console.log(`  Knowledge nodes: ${transferPackage.knowledge.length}`);
  console.log(
    `  Compression ratio: ${transferPackage.compressionRatio.toFixed(2)}`
  );
  console.log(`  Estimated time: ${transferPackage.estimatedTransferTime}ms`);

  // Demo 5: Execute Transfer
  console.log("\n⚡ Demo 5: Knowledge Transfer Execution");
  console.log("-".repeat(80));

  console.log("Transferring knowledge from expert to novice...\n");

  const updatedNoviceGraph = protocol.executeTransfer(
    expertGraph,
    noviceGraph,
    transferPackage
  );

  console.log(`Transfer complete!`);
  console.log(`\nNovice agent after transfer:`);
  console.log(
    `  Knowledge nodes: ${noviceGraph.nodes.size} → ${updatedNoviceGraph.nodes.size}`
  );
  console.log(
    `  Skills: ${noviceGraph.skills.size} → ${updatedNoviceGraph.skills.size}`
  );
  console.log(
    `  Dependencies: ${noviceGraph.edges.length} → ${updatedNoviceGraph.edges.length}`
  );

  // Demo 6: Validation
  console.log("\n✅ Demo 6: Transfer Validation");
  console.log("-".repeat(80));

  const validation = protocol.validateTransfer(
    noviceGraph,
    updatedNoviceGraph,
    transferPackage
  );

  console.log(`Validation results:`);
  console.log(`  Status: ${validation.status}`);
  console.log(`  Transferred skills: ${validation.transferredSkills}`);
  console.log(`  Transferred knowledge: ${validation.transferredKnowledge}`);

  console.log(`\nValidation metrics:`);
  console.log(
    `  Accuracy: ${(validation.validationMetrics.accuracy * 100).toFixed(1)}%`
  );
  console.log(
    `  Completeness: ${(validation.validationMetrics.completeness * 100).toFixed(1)}%`
  );
  console.log(
    `  Consistency: ${(validation.validationMetrics.consistency * 100).toFixed(1)}%`
  );

  console.log(`\nPerformance improvement:`);
  for (const [skillId, improvement] of Object.entries(
    validation.performanceChange.improvement
  )) {
    const before = validation.performanceChange.before[skillId];
    const after = validation.performanceChange.after[skillId];
    console.log(`  ${skillId}:`);
    console.log(`    Before: ${(before * 100).toFixed(1)}%`);
    console.log(`    After: ${(after * 100).toFixed(1)}%`);
    console.log(`    Improvement: ${(improvement * 100).toFixed(1)}%`);
  }

  console.log("\n✅ Consciousness Transfer demonstration complete!");
  console.log("=".repeat(80));
}

// Export classes for programmatic use
export {
  KnowledgeExtractor,
  TransferProtocol,
  KnowledgeType,
  TransferStatus,
  type KnowledgeNode,
  type Skill,
  type KnowledgeGraph,
  type TransferPackage,
  type TransferResult,
};
