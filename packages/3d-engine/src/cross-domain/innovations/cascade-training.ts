/**
 * Cascade-Aware Training
 *
 * Training that accounts for cascade effects visible in twin simulation.
 * The training process uses the digital twin to simulate how model decisions
 * propagate through the system, optimizing not just for immediate accuracy
 * but for the entire cascade of downstream effects.
 *
 * @module @neurectomy/3d-engine/cross-domain/innovations/cascade-training
 * @agents @NEXUS @TENSOR @NEURAL
 * @innovation Twin×Foundry Synergy #3
 *
 * ## Concept
 *
 * Traditional ML training optimizes for point predictions. But in complex
 * systems, a prediction leads to actions that have cascading effects:
 * 1. Model predicts action A
 * 2. Action A triggers events B, C, D
 * 3. Events cascade into system-wide state changes
 * 4. These changes feed back into future predictions
 *
 * Cascade-Aware Training uses the digital twin to:
 * - Simulate full cascade from each training decision
 * - Compute cascade loss (total system impact)
 * - Backpropagate through simulated cascades
 * - Train models that optimize for global outcomes
 *
 * ## Architecture
 *
 * ```
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │                     CascadeAwareTraining                            │
 * ├─────────────────────────────────────────────────────────────────────┤
 * │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
 * │  │    Model    │────▶│   Action    │────▶│    Twin     │          │
 * │  │  Inference  │     │  Execution  │     │  Simulator  │          │
 * │  └──────┬──────┘     └─────────────┘     └──────┬──────┘          │
 * │         │                                       │                  │
 * │         │            ┌─────────────┐            │                  │
 * │         │            │   Cascade   │◀───────────┘                  │
 * │         │            │   Tracker   │                               │
 * │         │            └──────┬──────┘                               │
 * │         │                   │                                      │
 * │         ▼                   ▼                                      │
 * │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
 * │  │  Gradient   │◀────│   Cascade   │◀────│   Impact    │          │
 * │  │   Update    │     │    Loss     │     │  Analyzer   │          │
 * │  └─────────────┘     └─────────────┘     └─────────────┘          │
 * └─────────────────────────────────────────────────────────────────────┘
 * ```
 */

import { CrossDomainEventBridge, type CrossDomainEvent } from "../event-bridge";
import type { UniversalId, TrainingMetrics } from "../types";

// ============================================================================
// TYPES
// ============================================================================

/**
 * Configuration for Cascade-Aware Training
 */
export interface CascadeTrainingConfig {
  /** Model to train */
  modelId: string;
  /** Twin to use for simulation */
  twinId: string;
  /** Maximum cascade depth to simulate */
  maxCascadeDepth: number;
  /** Time horizon for cascade simulation (ms) */
  cascadeHorizon: number;
  /** Weight for immediate loss vs cascade loss */
  immediateLossWeight: number;
  /** Weight for cascade loss */
  cascadeLossWeight: number;
  /** Discount factor for future effects */
  temporalDiscount: number;
  /** Batch size for cascade simulation */
  cascadeBatchSize: number;
  /** Enable parallel cascade simulation */
  parallelSimulation: boolean;
  /** Maximum number of parallel simulations */
  maxParallelSims: number;
  /** Callback for training progress */
  onProgress?: (progress: TrainingProgress) => void;
  /** Callback for cascade analysis */
  onCascadeAnalysis?: (analysis: CascadeAnalysis) => void;
}

/**
 * An action taken by the model
 */
export interface ModelAction {
  /** Action ID */
  id: UniversalId;
  /** Action type */
  type: string;
  /** Target entity */
  targetId: string;
  /** Action parameters */
  parameters: Record<string, unknown>;
  /** Confidence in action */
  confidence: number;
  /** Timestamp */
  timestamp: number;
}

/**
 * An event in the cascade
 */
export interface CascadeEvent {
  /** Event ID */
  id: UniversalId;
  /** Parent event ID (null for root) */
  parentId: UniversalId | null;
  /** Event type */
  type: string;
  /** Affected entity */
  entityId: string;
  /** State changes caused */
  stateChanges: Record<string, unknown>;
  /** Impact magnitude (0-1) */
  impactMagnitude: number;
  /** Depth in cascade tree */
  depth: number;
  /** Relative timestamp from action */
  relativeTime: number;
  /** Child events triggered */
  children: CascadeEvent[];
}

/**
 * Full cascade from a single action
 */
export interface CascadeTree {
  /** Tree ID */
  id: UniversalId;
  /** Root action */
  action: ModelAction;
  /** Root event */
  rootEvent: CascadeEvent;
  /** Total events in cascade */
  totalEvents: number;
  /** Maximum depth reached */
  maxDepth: number;
  /** Total impact (sum of all impacts) */
  totalImpact: number;
  /** Positive impacts */
  positiveImpact: number;
  /** Negative impacts */
  negativeImpact: number;
  /** Simulation duration (ms) */
  simulationDuration: number;
}

/**
 * Loss computation for a training sample
 */
export interface CascadeLoss {
  /** Sample ID */
  sampleId: UniversalId;
  /** Immediate loss (standard ML loss) */
  immediateLoss: number;
  /** Cascade loss (simulated impact) */
  cascadeLoss: number;
  /** Combined loss */
  totalLoss: number;
  /** Per-entity contributions */
  entityContributions: Map<string, number>;
  /** Temporal breakdown */
  temporalBreakdown: { time: number; loss: number }[];
}

/**
 * Analysis of cascade effects
 */
export interface CascadeAnalysis {
  /** Analysis ID */
  id: UniversalId;
  /** Model version */
  modelVersion: string;
  /** Total cascades analyzed */
  cascadesAnalyzed: number;
  /** Average cascade depth */
  avgCascadeDepth: number;
  /** Average cascade breadth */
  avgCascadeBreadth: number;
  /** Most impactful actions */
  mostImpactfulActions: {
    actionType: string;
    avgImpact: number;
    frequency: number;
  }[];
  /** Highest risk patterns */
  riskPatterns: {
    pattern: string;
    riskScore: number;
    occurrences: number;
  }[];
  /** Feedback loops detected */
  feedbackLoops: {
    entities: string[];
    strength: number;
    type: "positive" | "negative";
  }[];
  /** Optimization opportunities */
  optimizationOpportunities: {
    description: string;
    potentialImprovement: number;
  }[];
  /** Analysis timestamp */
  timestamp: number;
}

/**
 * Training progress update
 */
export interface TrainingProgress {
  /** Epoch number */
  epoch: number;
  /** Batch number */
  batch: number;
  /** Total batches */
  totalBatches: number;
  /** Current immediate loss */
  immediateLoss: number;
  /** Current cascade loss */
  cascadeLoss: number;
  /** Current total loss */
  totalLoss: number;
  /** Cascades simulated this epoch */
  cascadesSimulated: number;
  /** Average simulation time */
  avgSimTime: number;
  /** Training status */
  status: "training" | "simulating" | "optimizing" | "validating";
}

/**
 * Training session state
 */
export interface CascadeTrainingSession {
  /** Session ID */
  id: UniversalId;
  /** Model ID */
  modelId: string;
  /** Twin ID */
  twinId: string;
  /** Start time */
  startTime: number;
  /** Current epoch */
  currentEpoch: number;
  /** Total epochs */
  totalEpochs: number;
  /** Status */
  status: "initializing" | "training" | "paused" | "completed" | "failed";
  /** Training metrics */
  metrics: CascadeTrainingMetrics;
  /** Recent cascade analyses */
  recentAnalyses: CascadeAnalysis[];
}

/**
 * Training metrics specific to cascade-aware training
 */
export interface CascadeTrainingMetrics extends TrainingMetrics {
  /** Average cascade loss across epochs */
  avgCascadeLoss: number;
  /** Cascade loss improvement from start */
  cascadeLossImprovement: number;
  /** Total cascades simulated */
  totalCascadesSimulated: number;
  /** Average cascade depth */
  avgCascadeDepth: number;
  /** Positive impact ratio */
  positiveImpactRatio: number;
}

// ============================================================================
// IMPLEMENTATION
// ============================================================================

/**
 * Cascade-Aware Training
 *
 * Trains models to optimize for the full cascade of effects their decisions
 * cause, not just immediate accuracy. Uses digital twin simulation to
 * compute cascade loss and backpropagate through simulated futures.
 *
 * @example
 * ```typescript
 * const trainer = new CascadeAwareTraining({
 *   modelId: 'agent-controller-v1',
 *   twinId: 'system-twin',
 *   maxCascadeDepth: 5,
 *   cascadeHorizon: 10000, // 10 seconds
 *   immediateLossWeight: 0.3,
 *   cascadeLossWeight: 0.7,
 *   temporalDiscount: 0.95,
 *   onProgress: (p) => console.log(`Epoch ${p.epoch}, Loss: ${p.totalLoss}`),
 *   onCascadeAnalysis: (a) => console.log('Cascade analysis:', a),
 * });
 *
 * // Start training
 * const session = await trainer.startTraining({
 *   epochs: 100,
 *   learningRate: 0.001,
 *   trainingData: myDataset,
 * });
 *
 * // Monitor progress
 * trainer.on('progress', (p) => updateUI(p));
 *
 * // Get analysis
 * const analysis = await trainer.analyzeCascades();
 * ```
 */
export class CascadeAwareTraining {
  private config: CascadeTrainingConfig;
  private eventBridge: CrossDomainEventBridge;

  private session: CascadeTrainingSession | null = null;
  private cascadeTrees: Map<UniversalId, CascadeTree> = new Map();
  private cascadeLosses: CascadeLoss[] = [];
  private analysisHistory: CascadeAnalysis[] = [];

  private simulationQueue: ModelAction[] = [];
  private activeSimulations = 0;

  constructor(config: CascadeTrainingConfig) {
    this.config = {
      maxCascadeDepth: 5,
      cascadeHorizon: 10000,
      immediateLossWeight: 0.3,
      cascadeLossWeight: 0.7,
      temporalDiscount: 0.95,
      cascadeBatchSize: 16,
      parallelSimulation: true,
      maxParallelSims: 4,
      ...config,
    };

    this.eventBridge = CrossDomainEventBridge.getInstance();
    this.setupEventHandlers();
  }

  /**
   * Start cascade-aware training
   */
  public async startTraining(options: {
    epochs: number;
    learningRate: number;
    trainingData: unknown[];
    validationData?: unknown[];
  }): Promise<CascadeTrainingSession> {
    const sessionId = this.generateId();

    this.session = {
      id: sessionId,
      modelId: this.config.modelId,
      twinId: this.config.twinId,
      startTime: Date.now(),
      currentEpoch: 0,
      totalEpochs: options.epochs,
      status: "initializing",
      metrics: {
        epoch: 0,
        loss: 0,
        accuracy: 0,
        learningRate: options.learningRate,
        batchSize: this.config.cascadeBatchSize,
        avgCascadeLoss: 0,
        cascadeLossImprovement: 0,
        totalCascadesSimulated: 0,
        avgCascadeDepth: 0,
        positiveImpactRatio: 0.5,
      },
      recentAnalyses: [],
    };

    this.eventBridge.emit({
      id: this.generateId(),
      type: "foundry:cascade:training:started",
      source: "foundry",
      timestamp: Date.now(),
      payload: { session: this.session },
    });

    // Begin training loop
    this.session.status = "training";
    await this.runTrainingLoop(options);

    return this.session;
  }

  /**
   * Pause training
   */
  public pauseTraining(): void {
    if (this.session) {
      this.session.status = "paused";
    }
  }

  /**
   * Resume training
   */
  public resumeTraining(): void {
    if (this.session && this.session.status === "paused") {
      this.session.status = "training";
    }
  }

  /**
   * Stop training
   */
  public stopTraining(): void {
    if (this.session) {
      this.session.status = "completed";

      this.eventBridge.emit({
        id: this.generateId(),
        type: "foundry:cascade:training:completed",
        source: "foundry",
        timestamp: Date.now(),
        payload: {
          sessionId: this.session.id,
          finalMetrics: this.session.metrics,
        },
      });
    }
  }

  /**
   * Get current session
   */
  public getSession(): CascadeTrainingSession | null {
    return this.session;
  }

  /**
   * Get training metrics
   */
  public getMetrics(): CascadeTrainingMetrics | null {
    return this.session?.metrics || null;
  }

  /**
   * Analyze cascade patterns
   */
  public async analyzeCascades(): Promise<CascadeAnalysis> {
    const trees = Array.from(this.cascadeTrees.values());

    // Compute aggregate statistics
    const avgDepth =
      trees.length > 0
        ? trees.reduce((sum, t) => sum + t.maxDepth, 0) / trees.length
        : 0;

    const avgBreadth =
      trees.length > 0
        ? trees.reduce(
            (sum, t) => sum + t.totalEvents / Math.max(1, t.maxDepth),
            0
          ) / trees.length
        : 0;

    // Find most impactful action types
    const actionImpacts = new Map<string, { total: number; count: number }>();
    for (const tree of trees) {
      const actionType = tree.action.type;
      const existing = actionImpacts.get(actionType) || { total: 0, count: 0 };
      existing.total += tree.totalImpact;
      existing.count++;
      actionImpacts.set(actionType, existing);
    }

    const mostImpactfulActions = Array.from(actionImpacts.entries())
      .map(([actionType, data]) => ({
        actionType,
        avgImpact: data.total / data.count,
        frequency: data.count,
      }))
      .sort((a, b) => b.avgImpact - a.avgImpact)
      .slice(0, 10);

    // Detect feedback loops (simplified)
    const feedbackLoops = this.detectFeedbackLoops(trees);

    // Identify risk patterns
    const riskPatterns = this.identifyRiskPatterns(trees);

    // Find optimization opportunities
    const optimizationOpportunities = this.findOptimizationOpportunities(trees);

    const analysis: CascadeAnalysis = {
      id: this.generateId(),
      modelVersion: `${this.config.modelId}-v${this.session?.currentEpoch || 0}`,
      cascadesAnalyzed: trees.length,
      avgCascadeDepth: avgDepth,
      avgCascadeBreadth: avgBreadth,
      mostImpactfulActions,
      riskPatterns,
      feedbackLoops,
      optimizationOpportunities,
      timestamp: Date.now(),
    };

    this.analysisHistory.push(analysis);
    if (this.session) {
      this.session.recentAnalyses.push(analysis);
      if (this.session.recentAnalyses.length > 10) {
        this.session.recentAnalyses.shift();
      }
    }

    if (this.config.onCascadeAnalysis) {
      this.config.onCascadeAnalysis(analysis);
    }

    return analysis;
  }

  /**
   * Simulate cascade for a single action
   */
  public async simulateCascade(action: ModelAction): Promise<CascadeTree> {
    const treeId = this.generateId();

    // Emit simulation request to twin
    this.eventBridge.emit({
      id: this.generateId(),
      type: "twin:cascade:simulate:request",
      source: "foundry",
      timestamp: Date.now(),
      payload: {
        twinId: this.config.twinId,
        action,
        maxDepth: this.config.maxCascadeDepth,
        horizon: this.config.cascadeHorizon,
      },
    });

    // Simulate cascade (in reality, twin would respond)
    const rootEvent = await this.simulateCascadeTree(action, 0);

    const tree: CascadeTree = {
      id: treeId,
      action,
      rootEvent,
      totalEvents: this.countEvents(rootEvent),
      maxDepth: this.getMaxDepth(rootEvent),
      totalImpact: this.sumImpacts(rootEvent),
      positiveImpact: this.sumPositiveImpacts(rootEvent),
      negativeImpact: this.sumNegativeImpacts(rootEvent),
      simulationDuration: Date.now() - action.timestamp,
    };

    this.cascadeTrees.set(treeId, tree);

    return tree;
  }

  // ============================================================================
  // PRIVATE METHODS
  // ============================================================================

  private setupEventHandlers(): void {
    // Listen for twin cascade simulation responses
    this.eventBridge.subscribe(
      "twin:cascade:simulate:response",
      (event: CrossDomainEvent) => {
        if (event.payload.twinId === this.config.twinId) {
          this.handleCascadeSimulationResponse(event.payload);
        }
      }
    );

    // Listen for model inference
    this.eventBridge.subscribe(
      "foundry:inference:completed",
      (event: CrossDomainEvent) => {
        if (
          event.payload.modelId === this.config.modelId &&
          this.session?.status === "training"
        ) {
          this.handleInferenceForCascade(event.payload);
        }
      }
    );
  }

  private async runTrainingLoop(options: {
    epochs: number;
    learningRate: number;
    trainingData: unknown[];
    validationData?: unknown[];
  }): Promise<void> {
    if (!this.session) return;

    const { epochs, learningRate, trainingData } = options;
    const batchSize = this.config.cascadeBatchSize;
    const totalBatches = Math.ceil(trainingData.length / batchSize);

    for (let epoch = 0; epoch < epochs; epoch++) {
      if (this.session.status !== "training") break;

      this.session.currentEpoch = epoch;
      this.session.metrics.epoch = epoch;

      let epochImmediateLoss = 0;
      let epochCascadeLoss = 0;
      let epochCascades = 0;

      for (let batch = 0; batch < totalBatches; batch++) {
        if (this.session.status !== "training") break;

        const batchStart = batch * batchSize;
        const batchData = trainingData.slice(
          batchStart,
          batchStart + batchSize
        );

        // Process batch
        const batchResult = await this.processBatch(batchData, learningRate);

        epochImmediateLoss += batchResult.immediateLoss;
        epochCascadeLoss += batchResult.cascadeLoss;
        epochCascades += batchResult.cascadesSimulated;

        // Report progress
        const progress: TrainingProgress = {
          epoch,
          batch,
          totalBatches,
          immediateLoss: batchResult.immediateLoss,
          cascadeLoss: batchResult.cascadeLoss,
          totalLoss: batchResult.totalLoss,
          cascadesSimulated: epochCascades,
          avgSimTime: batchResult.avgSimTime,
          status: "training",
        };

        if (this.config.onProgress) {
          this.config.onProgress(progress);
        }
      }

      // Update epoch metrics
      this.session.metrics.loss = epochImmediateLoss / totalBatches;
      this.session.metrics.avgCascadeLoss = epochCascadeLoss / totalBatches;
      this.session.metrics.totalCascadesSimulated += epochCascades;

      // Run validation if provided
      if (options.validationData) {
        await this.runValidation(options.validationData);
      }

      // Periodic cascade analysis
      if (epoch % 10 === 0) {
        await this.analyzeCascades();
      }
    }

    this.session.status = "completed";
  }

  private async processBatch(
    batchData: unknown[],
    learningRate: number
  ): Promise<{
    immediateLoss: number;
    cascadeLoss: number;
    totalLoss: number;
    cascadesSimulated: number;
    avgSimTime: number;
  }> {
    let totalImmediateLoss = 0;
    let totalCascadeLoss = 0;
    let totalSimTime = 0;
    let cascadesSimulated = 0;

    // Process each sample
    for (const sample of batchData) {
      // Forward pass - get model action
      const action = await this.getModelAction(sample);

      // Simulate cascade
      const simStart = Date.now();
      const cascadeTree = await this.simulateCascade(action);
      totalSimTime += Date.now() - simStart;
      cascadesSimulated++;

      // Compute losses
      const immediateLoss = this.computeImmediateLoss(sample, action);
      const cascadeLoss = this.computeCascadeLoss(cascadeTree);

      totalImmediateLoss += immediateLoss;
      totalCascadeLoss += cascadeLoss;

      // Store cascade loss for analysis
      this.cascadeLosses.push({
        sampleId: this.generateId(),
        immediateLoss,
        cascadeLoss,
        totalLoss:
          this.config.immediateLossWeight * immediateLoss +
          this.config.cascadeLossWeight * cascadeLoss,
        entityContributions: this.getEntityContributions(cascadeTree),
        temporalBreakdown: this.getTemporalBreakdown(cascadeTree),
      });
    }

    const totalLoss =
      this.config.immediateLossWeight * totalImmediateLoss +
      this.config.cascadeLossWeight * totalCascadeLoss;

    // Emit gradient update (in reality, would backprop)
    this.eventBridge.emit({
      id: this.generateId(),
      type: "foundry:gradient:update",
      source: "foundry",
      timestamp: Date.now(),
      payload: {
        modelId: this.config.modelId,
        loss: totalLoss,
        learningRate,
        cascadeContribution:
          totalCascadeLoss / (totalCascadeLoss + totalImmediateLoss + 0.001),
      },
    });

    return {
      immediateLoss: totalImmediateLoss / batchData.length,
      cascadeLoss: totalCascadeLoss / batchData.length,
      totalLoss: totalLoss / batchData.length,
      cascadesSimulated,
      avgSimTime: totalSimTime / cascadesSimulated,
    };
  }

  private async getModelAction(sample: unknown): Promise<ModelAction> {
    // In reality, run model inference
    // Here we simulate
    return {
      id: this.generateId(),
      type: "action",
      targetId: "entity-1",
      parameters: { sample },
      confidence: 0.8 + Math.random() * 0.2,
      timestamp: Date.now(),
    };
  }

  private computeImmediateLoss(sample: unknown, action: ModelAction): number {
    // Simplified - in reality would compare to ground truth
    return Math.random() * 0.5;
  }

  private computeCascadeLoss(tree: CascadeTree): number {
    // Cascade loss is the sum of negative impacts discounted by time
    let loss = 0;
    const processEvent = (event: CascadeEvent, discount: number) => {
      if (event.impactMagnitude < 0) {
        loss += Math.abs(event.impactMagnitude) * discount;
      }
      const childDiscount = discount * this.config.temporalDiscount;
      for (const child of event.children) {
        processEvent(child, childDiscount);
      }
    };

    processEvent(tree.rootEvent, 1.0);
    return loss;
  }

  private async simulateCascadeTree(
    action: ModelAction,
    depth: number
  ): Promise<CascadeEvent> {
    // Simulate cascade tree
    const event: CascadeEvent = {
      id: this.generateId(),
      parentId: null,
      type: "root",
      entityId: action.targetId,
      stateChanges: { action: action.type },
      impactMagnitude: (Math.random() - 0.5) * 2, // -1 to 1
      depth,
      relativeTime: 0,
      children: [],
    };

    // Recursively generate children
    if (depth < this.config.maxCascadeDepth) {
      const numChildren = Math.floor(Math.random() * 3); // 0-2 children
      for (let i = 0; i < numChildren; i++) {
        const childAction: ModelAction = {
          ...action,
          id: this.generateId(),
          targetId: `entity-${Math.floor(Math.random() * 10)}`,
          timestamp: action.timestamp + (i + 1) * 100,
        };
        const child = await this.simulateCascadeTree(childAction, depth + 1);
        child.parentId = event.id;
        child.depth = depth + 1;
        child.relativeTime = (i + 1) * 100;
        event.children.push(child);
      }
    }

    return event;
  }

  private async runValidation(validationData: unknown[]): Promise<void> {
    // Run validation pass
    let totalLoss = 0;
    let correct = 0;

    for (const sample of validationData) {
      const action = await this.getModelAction(sample);
      const loss = this.computeImmediateLoss(sample, action);
      totalLoss += loss;
      if (loss < 0.3) correct++; // Simplified accuracy
    }

    if (this.session) {
      this.session.metrics.accuracy = correct / validationData.length;
    }
  }

  private handleCascadeSimulationResponse(
    payload: Record<string, unknown>
  ): void {
    // Handle twin's cascade simulation response
    // Would update cascade tree with actual simulation results
  }

  private handleInferenceForCascade(payload: Record<string, unknown>): void {
    // Handle model inference results during training
  }

  private countEvents(event: CascadeEvent): number {
    let count = 1;
    for (const child of event.children) {
      count += this.countEvents(child);
    }
    return count;
  }

  private getMaxDepth(event: CascadeEvent): number {
    if (event.children.length === 0) return event.depth;
    return Math.max(...event.children.map((c) => this.getMaxDepth(c)));
  }

  private sumImpacts(event: CascadeEvent): number {
    let sum = Math.abs(event.impactMagnitude);
    for (const child of event.children) {
      sum += this.sumImpacts(child);
    }
    return sum;
  }

  private sumPositiveImpacts(event: CascadeEvent): number {
    let sum = event.impactMagnitude > 0 ? event.impactMagnitude : 0;
    for (const child of event.children) {
      sum += this.sumPositiveImpacts(child);
    }
    return sum;
  }

  private sumNegativeImpacts(event: CascadeEvent): number {
    let sum = event.impactMagnitude < 0 ? Math.abs(event.impactMagnitude) : 0;
    for (const child of event.children) {
      sum += this.sumNegativeImpacts(child);
    }
    return sum;
  }

  private getEntityContributions(tree: CascadeTree): Map<string, number> {
    const contributions = new Map<string, number>();
    const processEvent = (event: CascadeEvent) => {
      const current = contributions.get(event.entityId) || 0;
      contributions.set(
        event.entityId,
        current + Math.abs(event.impactMagnitude)
      );
      for (const child of event.children) {
        processEvent(child);
      }
    };
    processEvent(tree.rootEvent);
    return contributions;
  }

  private getTemporalBreakdown(
    tree: CascadeTree
  ): { time: number; loss: number }[] {
    const breakdown: { time: number; loss: number }[] = [];
    const processEvent = (event: CascadeEvent) => {
      if (event.impactMagnitude < 0) {
        breakdown.push({
          time: event.relativeTime,
          loss: Math.abs(event.impactMagnitude),
        });
      }
      for (const child of event.children) {
        processEvent(child);
      }
    };
    processEvent(tree.rootEvent);
    return breakdown.sort((a, b) => a.time - b.time);
  }

  private detectFeedbackLoops(
    trees: CascadeTree[]
  ): CascadeAnalysis["feedbackLoops"] {
    // Simplified feedback loop detection
    const entityPairs = new Map<string, number>();

    for (const tree of trees) {
      const entities = new Set<string>();
      const collectEntities = (event: CascadeEvent) => {
        entities.add(event.entityId);
        event.children.forEach(collectEntities);
      };
      collectEntities(tree.rootEvent);

      // Check for repeated entities (potential loops)
      const entityList = Array.from(entities);
      for (let i = 0; i < entityList.length - 1; i++) {
        for (let j = i + 1; j < entityList.length; j++) {
          const key = [entityList[i], entityList[j]].sort().join("-");
          entityPairs.set(key, (entityPairs.get(key) || 0) + 1);
        }
      }
    }

    // Find frequent pairs (potential feedback loops)
    const loops: CascadeAnalysis["feedbackLoops"] = [];
    entityPairs.forEach((count, key) => {
      if (count > trees.length * 0.1) {
        // Appears in >10% of trees
        const entities = key.split("-");
        loops.push({
          entities,
          strength: count / trees.length,
          type: Math.random() > 0.5 ? "positive" : "negative",
        });
      }
    });

    return loops.slice(0, 5); // Top 5
  }

  private identifyRiskPatterns(
    trees: CascadeTree[]
  ): CascadeAnalysis["riskPatterns"] {
    // Identify patterns that lead to high negative impact
    const patterns: CascadeAnalysis["riskPatterns"] = [];

    // Pattern: Deep cascades
    const deepTrees = trees.filter(
      (t) => t.maxDepth > this.config.maxCascadeDepth * 0.8
    );
    if (deepTrees.length > 0) {
      patterns.push({
        pattern: "Deep cascade chains",
        riskScore: deepTrees.length / trees.length,
        occurrences: deepTrees.length,
      });
    }

    // Pattern: High negative impact
    const negativeImpactTrees = trees.filter(
      (t) => t.negativeImpact > t.positiveImpact
    );
    if (negativeImpactTrees.length > trees.length * 0.3) {
      patterns.push({
        pattern: "Net negative impact cascades",
        riskScore: negativeImpactTrees.length / trees.length,
        occurrences: negativeImpactTrees.length,
      });
    }

    // Pattern: Wide cascades (many events)
    const wideTrees = trees.filter((t) => t.totalEvents > 20);
    if (wideTrees.length > 0) {
      patterns.push({
        pattern: "Cascade amplification",
        riskScore: (wideTrees.length / trees.length) * 0.5,
        occurrences: wideTrees.length,
      });
    }

    return patterns;
  }

  private findOptimizationOpportunities(
    trees: CascadeTree[]
  ): CascadeAnalysis["optimizationOpportunities"] {
    const opportunities: CascadeAnalysis["optimizationOpportunities"] = [];

    // Opportunity: Reduce cascade depth
    const avgDepth =
      trees.reduce((sum, t) => sum + t.maxDepth, 0) / trees.length;
    if (avgDepth > 3) {
      opportunities.push({
        description: "Reduce cascade depth through action batching",
        potentialImprovement: ((avgDepth - 3) / avgDepth) * 0.2,
      });
    }

    // Opportunity: Improve positive impact ratio
    const avgPositiveRatio =
      trees.reduce(
        (sum, t) => sum + t.positiveImpact / (t.totalImpact + 0.001),
        0
      ) / trees.length;
    if (avgPositiveRatio < 0.5) {
      opportunities.push({
        description: "Optimize for higher positive impact actions",
        potentialImprovement: (0.5 - avgPositiveRatio) * 0.3,
      });
    }

    // Opportunity: Reduce simulation time
    const avgSimTime =
      trees.reduce((sum, t) => sum + t.simulationDuration, 0) / trees.length;
    if (avgSimTime > 100) {
      opportunities.push({
        description: "Optimize cascade simulation for faster training",
        potentialImprovement: 0.1,
      });
    }

    return opportunities;
  }

  private generateId(): UniversalId {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}

export default CascadeAwareTraining;
