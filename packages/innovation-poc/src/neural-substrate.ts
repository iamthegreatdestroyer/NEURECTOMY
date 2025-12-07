/**
 * Neural Substrate Mapping POC
 *
 * Maps agent architectures to biologically-inspired neural substrates with
 * specialized brain regions, synaptic plasticity, and emotion-driven processing.
 *
 * Key Innovations:
 * - Biologically-inspired brain regions (PFC, hippocampus, amygdala, etc.)
 * - Hebbian learning and synaptic plasticity
 * - Emotion-driven modulation of decision making
 * - Memory formation and consolidation
 * - Regional specialization and hierarchical processing
 *
 * Research Foundations:
 * - Hebb (1949): The Organization of Behavior
 * - LeDoux (1996): The Emotional Brain
 * - Squire & Zola (1996): Structure and function of declarative and nondeclarative memory
 * - Miller & Cohen (2001): An Integrative Theory of Prefrontal Cortex Function
 *
 * @elite-agents @NEURAL @HELIX @CORE
 */

import { cloneDeep } from "lodash";

// ============================================================================
// Type Definitions
// ============================================================================

type NeuronId = string;
type RegionId = string;
type SynapseId = string;

enum BrainRegion {
  PREFRONTAL_CORTEX = "PFC", // Executive function, planning, decision-making
  HIPPOCAMPUS = "HIPP", // Memory formation, spatial navigation
  AMYGDALA = "AMYG", // Emotion processing, threat detection
  SENSORY_CORTEX = "SENS", // Sensory processing
  MOTOR_CORTEX = "MOTOR", // Motor control
  BASAL_GANGLIA = "BG", // Action selection, habit formation
  THALAMUS = "THAL", // Sensory relay
}

enum EmotionType {
  FEAR = "fear",
  REWARD = "reward",
  CURIOSITY = "curiosity",
  FRUSTRATION = "frustration",
  SATISFACTION = "satisfaction",
}

interface NeuronConfig {
  id: NeuronId;
  region: BrainRegion;
  threshold: number;
  baseActivation: number;
  refractoryPeriod: number;
}

interface Synapse {
  id: SynapseId;
  preNeuron: NeuronId;
  postNeuron: NeuronId;
  weight: number;
  plasticityRate: number;
  lastActivation: number;
}

interface EmotionalState {
  valence: number; // -1 (negative) to +1 (positive)
  arousal: number; // 0 (calm) to 1 (excited)
  dominance: number; // 0 (submissive) to 1 (dominant)
  primaryEmotion: EmotionType;
  intensity: number;
}

interface MemoryTrace {
  id: string;
  timestamp: number;
  pattern: Map<NeuronId, number>;
  emotion: EmotionalState;
  consolidationStrength: number;
  retrievalCount: number;
}

interface RegionalActivity {
  region: BrainRegion;
  activationLevel: number;
  neuronCount: number;
  activeNeurons: number;
}

// ============================================================================
// Neural Layer (Base)
// ============================================================================

class NeuralLayer {
  private neurons: Map<NeuronId, Neuron>;
  private synapses: Map<SynapseId, Synapse>;
  private region: BrainRegion;
  private lastUpdateTime: number;

  constructor(region: BrainRegion) {
    this.neurons = new Map();
    this.synapses = new Map();
    this.region = region;
    this.lastUpdateTime = 0;
  }

  addNeuron(config: NeuronConfig): void {
    this.neurons.set(config.id, new Neuron(config));
  }

  addSynapse(synapse: Synapse): void {
    this.synapses.set(synapse.id, synapse);
  }

  /**
   * Forward propagation with activation
   */
  propagate(inputs: Map<NeuronId, number>, dt: number): Map<NeuronId, number> {
    const outputs = new Map<NeuronId, number>();

    // Set external inputs
    for (const [neuronId, value] of inputs) {
      const neuron = this.neurons.get(neuronId);
      if (neuron) {
        neuron.receiveInput(value);
      }
    }

    // Propagate through synapses
    for (const synapse of this.synapses.values()) {
      const preNeuron = this.neurons.get(synapse.preNeuron);
      const postNeuron = this.neurons.get(synapse.postNeuron);

      if (preNeuron && postNeuron) {
        const signal = preNeuron.getActivation() * synapse.weight;
        postNeuron.receiveInput(signal);
      }
    }

    // Update all neurons
    for (const [id, neuron] of this.neurons) {
      neuron.update(dt);
      outputs.set(id, neuron.getActivation());
    }

    this.lastUpdateTime += dt;
    return outputs;
  }

  /**
   * Apply Hebbian learning: "Neurons that fire together, wire together"
   */
  applyHebbianLearning(learningRate: number = 0.01): void {
    for (const synapse of this.synapses.values()) {
      const preNeuron = this.neurons.get(synapse.preNeuron);
      const postNeuron = this.neurons.get(synapse.postNeuron);

      if (preNeuron && postNeuron) {
        const preActivation = preNeuron.getActivation();
        const postActivation = postNeuron.getActivation();

        // Hebbian rule: Δw = η * pre * post
        const deltaWeight =
          learningRate *
          synapse.plasticityRate *
          preActivation *
          postActivation;
        synapse.weight += deltaWeight;

        // Weight bounds
        synapse.weight = Math.max(-2.0, Math.min(2.0, synapse.weight));
      }
    }
  }

  getActivityLevel(): number {
    const activations = Array.from(this.neurons.values()).map((n) =>
      n.getActivation()
    );
    return activations.reduce((sum, a) => sum + a, 0) / activations.length;
  }

  getRegion(): BrainRegion {
    return this.region;
  }

  getNeuronCount(): number {
    return this.neurons.size;
  }

  getActiveNeuronCount(threshold: number = 0.5): number {
    return Array.from(this.neurons.values()).filter(
      (n) => n.getActivation() > threshold
    ).length;
  }
}

// ============================================================================
// Neuron
// ============================================================================

class Neuron {
  private id: NeuronId;
  private region: BrainRegion;
  private threshold: number;
  private activation: number;
  private inputSum: number;
  private refractoryPeriod: number;
  private lastSpikeTime: number;
  private timeSinceSpike: number;

  constructor(config: NeuronConfig) {
    this.id = config.id;
    this.region = config.region;
    this.threshold = config.threshold;
    this.activation = config.baseActivation;
    this.inputSum = 0;
    this.refractoryPeriod = config.refractoryPeriod;
    this.lastSpikeTime = -Infinity;
    this.timeSinceSpike = Infinity;
  }

  receiveInput(value: number): void {
    this.inputSum += value;
  }

  update(dt: number): void {
    this.timeSinceSpike += dt;

    // Check refractory period
    if (this.timeSinceSpike < this.refractoryPeriod) {
      this.activation = 0;
      this.inputSum = 0;
      return;
    }

    // Sigmoid activation function
    this.activation = 1.0 / (1.0 + Math.exp(-(this.inputSum - this.threshold)));

    // Spike if threshold exceeded
    if (this.inputSum > this.threshold) {
      this.lastSpikeTime = Date.now();
      this.timeSinceSpike = 0;
    }

    // Decay input
    this.inputSum *= 0.9;
  }

  getActivation(): number {
    return this.activation;
  }

  getId(): NeuronId {
    return this.id;
  }
}

// ============================================================================
// Hippocampus (Memory System)
// ============================================================================

class Hippocampus extends NeuralLayer {
  private memories: Map<string, MemoryTrace>;
  private consolidationThreshold: number;

  constructor() {
    super(BrainRegion.HIPPOCAMPUS);
    this.memories = new Map();
    this.consolidationThreshold = 0.7;
  }

  /**
   * Encode new memory
   */
  encode(pattern: Map<NeuronId, number>, emotion: EmotionalState): string {
    const memoryId = `mem_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;

    const memory: MemoryTrace = {
      id: memoryId,
      timestamp: Date.now(),
      pattern: cloneDeep(pattern),
      emotion: cloneDeep(emotion),
      consolidationStrength: emotion.intensity, // Emotional events consolidate better
      retrievalCount: 0,
    };

    this.memories.set(memoryId, memory);
    return memoryId;
  }

  /**
   * Retrieve memory by similarity
   */
  retrieve(cue: Map<NeuronId, number>, k: number = 3): MemoryTrace[] {
    const similarities: Array<{ memory: MemoryTrace; similarity: number }> = [];

    for (const memory of this.memories.values()) {
      const similarity = this.computeSimilarity(cue, memory.pattern);
      similarities.push({ memory, similarity });
    }

    // Sort by similarity
    similarities.sort((a, b) => b.similarity - a.similarity);

    // Retrieve top-k
    const retrieved = similarities.slice(0, k).map((s) => s.memory);

    // Update retrieval counts
    for (const memory of retrieved) {
      memory.retrievalCount++;
      memory.consolidationStrength = Math.min(
        1.0,
        memory.consolidationStrength + 0.05
      );
    }

    return retrieved;
  }

  /**
   * Consolidate memories (strengthen frequently retrieved)
   */
  consolidate(): void {
    for (const memory of this.memories.values()) {
      // Decay unused memories
      if (memory.retrievalCount === 0) {
        memory.consolidationStrength *= 0.95;
      }

      // Remove weak memories
      if (memory.consolidationStrength < 0.1) {
        this.memories.delete(memory.id);
      }
    }
  }

  private computeSimilarity(
    pattern1: Map<NeuronId, number>,
    pattern2: Map<NeuronId, number>
  ): number {
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    const allKeys = new Set([...pattern1.keys(), ...pattern2.keys()]);

    for (const key of allKeys) {
      const v1 = pattern1.get(key) ?? 0;
      const v2 = pattern2.get(key) ?? 0;
      dotProduct += v1 * v2;
      norm1 += v1 * v1;
      norm2 += v2 * v2;
    }

    if (norm1 === 0 || norm2 === 0) return 0;
    return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
  }

  getMemoryCount(): number {
    return this.memories.size;
  }

  getStrongestMemories(n: number = 5): MemoryTrace[] {
    const sorted = Array.from(this.memories.values()).sort(
      (a, b) => b.consolidationStrength - a.consolidationStrength
    );
    return sorted.slice(0, n);
  }
}

// ============================================================================
// Amygdala (Emotion Processing)
// ============================================================================

class Amygdala extends NeuralLayer {
  private emotionalState: EmotionalState;
  private threatThreshold: number;
  private rewardThreshold: number;

  constructor() {
    super(BrainRegion.AMYGDALA);
    this.emotionalState = {
      valence: 0,
      arousal: 0,
      dominance: 0.5,
      primaryEmotion: EmotionType.CURIOSITY,
      intensity: 0.3,
    };
    this.threatThreshold = 0.6;
    this.rewardThreshold = 0.7;
  }

  /**
   * Process sensory input and generate emotional response
   */
  processStimulus(
    stimulus: Map<string, number>,
    context: Map<string, number>
  ): EmotionalState {
    const threatLevel = stimulus.get("threat") ?? 0;
    const rewardLevel = stimulus.get("reward") ?? 0;
    const novelty = stimulus.get("novelty") ?? 0;

    // Detect threats
    if (threatLevel > this.threatThreshold) {
      this.emotionalState.valence = -threatLevel;
      this.emotionalState.arousal = threatLevel;
      this.emotionalState.dominance = 0.3;
      this.emotionalState.primaryEmotion = EmotionType.FEAR;
      this.emotionalState.intensity = threatLevel;
    }
    // Detect rewards
    else if (rewardLevel > this.rewardThreshold) {
      this.emotionalState.valence = rewardLevel;
      this.emotionalState.arousal = rewardLevel * 0.8;
      this.emotionalState.dominance = 0.7;
      this.emotionalState.primaryEmotion = EmotionType.REWARD;
      this.emotionalState.intensity = rewardLevel;
    }
    // Novelty drives curiosity
    else if (novelty > 0.5) {
      this.emotionalState.valence = 0.3;
      this.emotionalState.arousal = novelty;
      this.emotionalState.dominance = 0.5;
      this.emotionalState.primaryEmotion = EmotionType.CURIOSITY;
      this.emotionalState.intensity = novelty;
    }
    // Default: mild curiosity
    else {
      this.emotionalState.valence *= 0.9;
      this.emotionalState.arousal *= 0.95;
      this.emotionalState.intensity *= 0.95;
    }

    return cloneDeep(this.emotionalState);
  }

  /**
   * Modulate decision making based on emotion
   */
  modulateDecision(options: Map<string, number>): Map<string, number> {
    const modulated = new Map<string, number>();

    for (const [option, value] of options) {
      let modulatedValue = value;

      // Fear reduces risk-taking
      if (this.emotionalState.primaryEmotion === EmotionType.FEAR) {
        if (option.includes("risky") || option.includes("aggressive")) {
          modulatedValue *= 0.5;
        }
      }

      // Reward increases approach behavior
      if (this.emotionalState.primaryEmotion === EmotionType.REWARD) {
        if (option.includes("approach") || option.includes("engage")) {
          modulatedValue *= 1.5;
        }
      }

      // Curiosity increases exploration
      if (this.emotionalState.primaryEmotion === EmotionType.CURIOSITY) {
        if (option.includes("explore") || option.includes("novel")) {
          modulatedValue *= 1.3;
        }
      }

      modulated.set(option, modulatedValue);
    }

    return modulated;
  }

  getEmotionalState(): EmotionalState {
    return cloneDeep(this.emotionalState);
  }
}

// ============================================================================
// Prefrontal Cortex (Executive Function)
// ============================================================================

class PrefrontalCortex extends NeuralLayer {
  private workingMemory: Map<string, any>;
  private goalStack: Array<string>;
  private inhibitionStrength: number;

  constructor() {
    super(BrainRegion.PREFRONTAL_CORTEX);
    this.workingMemory = new Map();
    this.goalStack = [];
    this.inhibitionStrength = 0.5;
  }

  /**
   * Executive decision making with emotion modulation
   */
  makeDecision(
    options: Map<string, number>,
    emotionalState: EmotionalState,
    memories: MemoryTrace[]
  ): string {
    // Apply emotional modulation
    let modulatedOptions = this.applyEmotionalModulation(
      options,
      emotionalState
    );

    // Apply memory-based adjustments
    modulatedOptions = this.applyMemoryInfluence(modulatedOptions, memories);

    // Apply inhibition (impulse control)
    modulatedOptions = this.applyInhibition(modulatedOptions, emotionalState);

    // Select best option
    let bestOption = "";
    let bestValue = -Infinity;

    for (const [option, value] of modulatedOptions) {
      if (value > bestValue) {
        bestValue = value;
        bestOption = option;
      }
    }

    return bestOption;
  }

  private applyEmotionalModulation(
    options: Map<string, number>,
    emotion: EmotionalState
  ): Map<string, number> {
    const modulated = new Map<string, number>();

    for (const [option, value] of options) {
      let adjusted = value;

      // High arousal reduces deliberation
      adjusted *= 1 - emotion.arousal * 0.2;

      // Negative valence increases caution
      if (emotion.valence < 0) {
        adjusted *= 1 - Math.abs(emotion.valence) * 0.3;
      }

      modulated.set(option, adjusted);
    }

    return modulated;
  }

  private applyMemoryInfluence(
    options: Map<string, number>,
    memories: MemoryTrace[]
  ): Map<string, number> {
    const influenced = new Map<string, number>();

    for (const [option, value] of options) {
      let adjusted = value;

      // Boost options similar to positive memories
      for (const memory of memories) {
        if (memory.emotion.valence > 0 && memory.consolidationStrength > 0.7) {
          adjusted *= 1.1;
        }
      }

      influenced.set(option, adjusted);
    }

    return influenced;
  }

  private applyInhibition(
    options: Map<string, number>,
    emotion: EmotionalState
  ): Map<string, number> {
    const inhibited = new Map<string, number>();

    for (const [option, value] of options) {
      let adjusted = value;

      // Stronger inhibition with high arousal
      if (emotion.arousal > 0.7) {
        adjusted *= this.inhibitionStrength;
      }

      inhibited.set(option, adjusted);
    }

    return inhibited;
  }

  updateGoal(goal: string): void {
    this.goalStack.push(goal);
  }

  getCurrentGoal(): string | undefined {
    return this.goalStack[this.goalStack.length - 1];
  }
}

// ============================================================================
// Neural Substrate (Complete Brain)
// ============================================================================

class NeuralSubstrate {
  private regions: Map<BrainRegion, NeuralLayer>;
  private hippocampus: Hippocampus;
  private amygdala: Amygdala;
  private pfc: PrefrontalCortex;
  private currentTime: number;

  constructor() {
    this.regions = new Map();
    this.hippocampus = new Hippocampus();
    this.amygdala = new Amygdala();
    this.pfc = new PrefrontalCortex();

    this.regions.set(BrainRegion.HIPPOCAMPUS, this.hippocampus);
    this.regions.set(BrainRegion.AMYGDALA, this.amygdala);
    this.regions.set(BrainRegion.PREFRONTAL_CORTEX, this.pfc);

    this.currentTime = 0;
  }

  /**
   * Process stimulus and make decision
   */
  processAndDecide(
    stimulus: Map<string, number>,
    options: Map<string, number>
  ): {
    decision: string;
    emotion: EmotionalState;
    memoryId: string;
  } {
    // 1. Emotional processing (Amygdala)
    const emotion = this.amygdala.processStimulus(stimulus, new Map());

    // 2. Memory retrieval (Hippocampus)
    const cue = new Map<NeuronId, number>();
    for (const [key, value] of stimulus) {
      cue.set(`hipp_${key}`, value);
    }
    const memories = this.hippocampus.retrieve(cue, 3);

    // 3. Executive decision (PFC)
    const decision = this.pfc.makeDecision(options, emotion, memories);

    // 4. Memory encoding
    const memoryPattern = new Map<NeuronId, number>();
    memoryPattern.set("decision", options.get(decision) ?? 0);
    for (const [key, value] of stimulus) {
      memoryPattern.set(key, value);
    }
    const memoryId = this.hippocampus.encode(memoryPattern, emotion);

    return { decision, emotion, memoryId };
  }

  /**
   * Get regional activity
   */
  getRegionalActivity(): RegionalActivity[] {
    const activities: RegionalActivity[] = [];

    for (const [region, layer] of this.regions) {
      activities.push({
        region,
        activationLevel: layer.getActivityLevel(),
        neuronCount: layer.getNeuronCount(),
        activeNeurons: layer.getActiveNeuronCount(),
      });
    }

    return activities;
  }

  /**
   * Consolidate memories
   */
  consolidateMemories(): void {
    this.hippocampus.consolidate();
  }

  getMemoryCount(): number {
    return this.hippocampus.getMemoryCount();
  }
}

// ============================================================================
// Demonstration
// ============================================================================

export async function demonstrateNeuralSubstrate(): Promise<void> {
  console.log("=".repeat(80));
  console.log("NEURAL SUBSTRATE MAPPING DEMONSTRATION");
  console.log("=".repeat(80));

  // Demo 1: Basic Neural Layer
  console.log("\n🧠 Demo 1: Basic Neural Layer with Hebbian Learning");
  console.log("-".repeat(80));

  const layer = new NeuralLayer(BrainRegion.SENSORY_CORTEX);

  layer.addNeuron({
    id: "n1",
    region: BrainRegion.SENSORY_CORTEX,
    threshold: 0.5,
    baseActivation: 0,
    refractoryPeriod: 10,
  });
  layer.addNeuron({
    id: "n2",
    region: BrainRegion.SENSORY_CORTEX,
    threshold: 0.5,
    baseActivation: 0,
    refractoryPeriod: 10,
  });

  layer.addSynapse({
    id: "s1",
    preNeuron: "n1",
    postNeuron: "n2",
    weight: 0.5,
    plasticityRate: 1.0,
    lastActivation: 0,
  });

  console.log(
    "Created layer with 2 neurons and 1 synapse (initial weight: 0.5)"
  );

  const inputs = new Map([["n1", 1.0]]);

  console.log("\nApplying input to n1 and propagating...");
  for (let i = 0; i < 3; i++) {
    const outputs = layer.propagate(inputs, 1.0);
    layer.applyHebbianLearning(0.05);

    console.log(`Step ${i + 1}:`);
    console.log(`  n1 activation: ${outputs.get("n1")?.toFixed(3)}`);
    console.log(`  n2 activation: ${outputs.get("n2")?.toFixed(3)}`);
  }

  console.log("\n✓ Hebbian learning strengthens co-active connections");

  // Demo 2: Memory Formation
  console.log("\n💾 Demo 2: Hippocampal Memory Formation");
  console.log("-".repeat(80));

  const hippocampus = new Hippocampus();

  const emotionalMemory: EmotionalState = {
    valence: 0.8,
    arousal: 0.7,
    dominance: 0.6,
    primaryEmotion: EmotionType.REWARD,
    intensity: 0.9,
  };

  const pattern1 = new Map([
    ["stim_reward", 0.9],
    ["stim_novelty", 0.5],
  ]);

  const memId1 = hippocampus.encode(pattern1, emotionalMemory);
  console.log(`Encoded emotional memory: ${memId1}`);
  console.log(
    `  Emotion: ${emotionalMemory.primaryEmotion} (intensity: ${emotionalMemory.intensity})`
  );

  const neutralMemory: EmotionalState = {
    valence: 0,
    arousal: 0.3,
    dominance: 0.5,
    primaryEmotion: EmotionType.CURIOSITY,
    intensity: 0.3,
  };

  const pattern2 = new Map([
    ["stim_reward", 0.2],
    ["stim_novelty", 0.8],
  ]);

  hippocampus.encode(pattern2, neutralMemory);

  const retrievalCue = new Map([
    ["stim_reward", 0.85],
    ["stim_novelty", 0.6],
  ]);

  console.log("\nRetrieving similar memories...");
  const retrieved = hippocampus.retrieve(retrievalCue, 2);

  for (let i = 0; i < retrieved.length; i++) {
    console.log(`  Memory ${i + 1}:`);
    console.log(`    Emotion: ${retrieved[i].emotion.primaryEmotion}`);
    console.log(
      `    Consolidation: ${(retrieved[i].consolidationStrength * 100).toFixed(1)}%`
    );
  }

  // Demo 3: Emotion-Driven Decision Making
  console.log("\n😨 Demo 3: Emotion-Driven Decision Making");
  console.log("-".repeat(80));

  const amygdala = new Amygdala();

  const threatStimulus = new Map([
    ["threat", 0.8],
    ["reward", 0.2],
    ["novelty", 0.3],
  ]);

  console.log("Processing threat stimulus (threat: 0.8)...");
  const fearEmotion = amygdala.processStimulus(threatStimulus, new Map());

  console.log(`Emotional response:`);
  console.log(`  Primary emotion: ${fearEmotion.primaryEmotion}`);
  console.log(
    `  Valence: ${fearEmotion.valence.toFixed(2)} (${fearEmotion.valence < 0 ? "negative" : "positive"})`
  );
  console.log(`  Arousal: ${fearEmotion.arousal.toFixed(2)}`);

  const decisionOptions = new Map([
    ["approach_risky", 0.6],
    ["avoid_safe", 0.5],
    ["explore_novel", 0.4],
  ]);

  console.log("\nDecision options:");
  for (const [option, value] of decisionOptions) {
    console.log(`  ${option}: ${value.toFixed(2)}`);
  }

  const modulatedOptions = amygdala.modulateDecision(decisionOptions);

  console.log("\nAfter emotional modulation:");
  for (const [option, value] of modulatedOptions) {
    console.log(`  ${option}: ${value.toFixed(2)}`);
  }

  console.log("\n✓ Fear reduces risky choices");

  // Demo 4: Complete Neural Substrate
  console.log("\n🧬 Demo 4: Complete Neural Substrate Processing");
  console.log("-".repeat(80));

  const brain = new NeuralSubstrate();

  const scenario1 = new Map([
    ["threat", 0.3],
    ["reward", 0.8],
    ["novelty", 0.5],
  ]);

  const choices = new Map([
    ["take_reward", 0.7],
    ["explore_area", 0.5],
    ["retreat_safely", 0.3],
  ]);

  console.log("Scenario: High reward, low threat");
  console.log(
    "Options: take_reward (0.7), explore_area (0.5), retreat_safely (0.3)"
  );

  const result1 = brain.processAndDecide(scenario1, choices);

  console.log(`\nDecision: ${result1.decision}`);
  console.log(
    `Emotion: ${result1.emotion.primaryEmotion} (${result1.emotion.valence > 0 ? "positive" : "negative"})`
  );
  console.log(`Memory encoded: ${result1.memoryId}`);

  // Demo 5: Memory Consolidation
  console.log("\n🔄 Demo 5: Memory Consolidation and Evolution");
  console.log("-".repeat(80));

  console.log("Encoding multiple experiences...\n");

  for (let i = 0; i < 5; i++) {
    const randomStimulus = new Map([
      ["threat", Math.random()],
      ["reward", Math.random()],
      ["novelty", Math.random()],
    ]);

    brain.processAndDecide(randomStimulus, choices);
  }

  console.log(`Total memories before consolidation: ${brain.getMemoryCount()}`);

  brain.consolidateMemories();

  console.log(`Total memories after consolidation: ${brain.getMemoryCount()}`);
  console.log("\n✓ Weak memories decay; strong memories persist");

  console.log("\nRegional activity:");
  const activities = brain.getRegionalActivity();
  for (const activity of activities) {
    console.log(
      `  ${activity.region}: ${(activity.activationLevel * 100).toFixed(1)}% active`
    );
  }

  console.log("\n✅ Neural Substrate Mapping demonstration complete!");
  console.log("=".repeat(80));
}

// Export classes for programmatic use
export {
  NeuralLayer,
  Neuron,
  Hippocampus,
  Amygdala,
  PrefrontalCortex,
  NeuralSubstrate,
  BrainRegion,
  EmotionType,
  type NeuronConfig,
  type Synapse,
  type EmotionalState,
  type MemoryTrace,
  type RegionalActivity,
};
