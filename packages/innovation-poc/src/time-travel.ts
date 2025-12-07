/**
 * Time-Travel Debugging POC
 *
 * Enables deterministic replay of agent execution with temporal breakpoints,
 * counterfactual branching, and bidirectional debugging.
 *
 * Key Innovations:
 * - Complete state recording with minimal overhead
 * - Deterministic replay with exact reproduction
 * - Counterfactual injection ("what if" debugging)
 * - Branching timelines for hypothesis testing
 * - Temporal breakpoints and reverse execution
 * - State diffing across timelines
 *
 * Research Foundations:
 * - Lewis (1979): Counterfactual dependence and time's arrow
 * - Engblom (2012): A review of reverse debugging
 * - Ronsse et al. (2003): RecPlay: A fully integrated practical record/replay system
 * - King et al. (2005): Debugging operating systems with time-traveling virtual machines
 *
 * @elite-agents @ECLIPSE @VELOCITY @APEX
 */

import { cloneDeep } from "lodash";

// ============================================================================
// Type Definitions
// ============================================================================

type Timestamp = number;
type EventId = string;
type TimelineId = string;
type BreakpointId = string;

enum EventType {
  STATE_CHANGE = "state_change",
  ACTION = "action",
  DECISION = "decision",
  COMMUNICATION = "communication",
  EXTERNAL_INPUT = "external_input",
}

interface StateSnapshot {
  timestamp: Timestamp;
  eventId: EventId;
  state: Record<string, any>;
  checksum: string;
}

interface Event {
  id: EventId;
  timestamp: Timestamp;
  type: EventType;
  agentId: string;
  action: string;
  inputs: Record<string, any>;
  outputs: Record<string, any>;
  metadata: Record<string, any>;
}

interface Timeline {
  id: TimelineId;
  parentTimelineId?: TimelineId;
  branchPoint?: Timestamp;
  events: Event[];
  snapshots: StateSnapshot[];
  counterfactuals: Map<Timestamp, Counterfactual>;
}

interface Counterfactual {
  timestamp: Timestamp;
  description: string;
  modifications: Record<string, any>;
  applied: boolean;
}

interface Breakpoint {
  id: BreakpointId;
  condition: (event: Event, state: Record<string, any>) => boolean;
  enabled: boolean;
  hitCount: number;
}

interface ReplayResult {
  finalState: Record<string, any>;
  eventsExecuted: number;
  breakpointsHit: BreakpointId[];
  executionTime: number;
  deviations: Array<{
    timestamp: Timestamp;
    expected: any;
    actual: any;
  }>;
}

// ============================================================================
// Temporal Recorder
// ============================================================================

class TemporalRecorder {
  private timeline: Timeline;
  private currentState: Record<string, any>;
  private recording: boolean;
  private snapshotInterval: number; // milliseconds

  constructor(snapshotInterval: number = 1000) {
    this.timeline = {
      id: this.generateId(),
      events: [],
      snapshots: [],
      counterfactuals: new Map(),
    };
    this.currentState = {};
    this.recording = false;
    this.snapshotInterval = snapshotInterval;
  }

  /**
   * Start recording
   */
  startRecording(initialState: Record<string, any>): void {
    this.recording = true;
    this.currentState = cloneDeep(initialState);

    this.recordSnapshot(Date.now());
  }

  /**
   * Stop recording
   */
  stopRecording(): Timeline {
    this.recording = false;
    return cloneDeep(this.timeline);
  }

  /**
   * Record event
   */
  recordEvent(event: Event): void {
    if (!this.recording) return;

    this.timeline.events.push(cloneDeep(event));

    // Periodic snapshots
    if (this.timeline.events.length % 10 === 0) {
      this.recordSnapshot(event.timestamp);
    }
  }

  /**
   * Record state snapshot
   */
  private recordSnapshot(timestamp: Timestamp): void {
    const snapshot: StateSnapshot = {
      timestamp,
      eventId:
        this.timeline.events[this.timeline.events.length - 1]?.id ?? "initial",
      state: cloneDeep(this.currentState),
      checksum: this.computeChecksum(this.currentState),
    };

    this.timeline.snapshots.push(snapshot);
  }

  /**
   * Update current state
   */
  updateState(updates: Record<string, any>): void {
    Object.assign(this.currentState, updates);
  }

  /**
   * Get timeline
   */
  getTimeline(): Timeline {
    return cloneDeep(this.timeline);
  }

  /**
   * Compute checksum for state
   */
  private computeChecksum(state: Record<string, any>): string {
    const stateString = JSON.stringify(state);
    let hash = 0;
    for (let i = 0; i < stateString.length; i++) {
      const char = stateString.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString(16);
  }

  private generateId(): string {
    return `${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
  }

  isRecording(): boolean {
    return this.recording;
  }
}

// ============================================================================
// Replay Engine
// ============================================================================

class ReplayEngine {
  private breakpoints: Map<BreakpointId, Breakpoint>;
  private pauseOnBreakpoint: boolean;

  constructor() {
    this.breakpoints = new Map();
    this.pauseOnBreakpoint = true;
  }

  /**
   * Replay timeline from start
   */
  replay(
    timeline: Timeline,
    executor: (event: Event, state: Record<string, any>) => Record<string, any>
  ): ReplayResult {
    const startTime = Date.now();
    const breakpointsHit: BreakpointId[] = [];
    const deviations: Array<{
      timestamp: Timestamp;
      expected: any;
      actual: any;
    }> = [];

    // Start from initial snapshot
    let state = timeline.snapshots[0]?.state ?? {};
    let eventsExecuted = 0;

    for (const event of timeline.events) {
      // Check breakpoints
      const hitBreakpoint = this.checkBreakpoints(event, state);
      if (hitBreakpoint) {
        breakpointsHit.push(hitBreakpoint);
        if (this.pauseOnBreakpoint) {
          break; // Pause execution
        }
      }

      // Apply counterfactual if present
      const counterfactual = timeline.counterfactuals.get(event.timestamp);
      if (counterfactual && counterfactual.applied) {
        Object.assign(event.inputs, counterfactual.modifications);
      }

      // Execute event
      const newState = executor(event, state);

      // Check for deviations (if we have snapshots)
      const expectedSnapshot = timeline.snapshots.find(
        (s) => Math.abs(s.timestamp - event.timestamp) < 100
      );
      if (expectedSnapshot) {
        const actualChecksum = this.computeChecksum(newState);
        if (actualChecksum !== expectedSnapshot.checksum) {
          deviations.push({
            timestamp: event.timestamp,
            expected: expectedSnapshot.state,
            actual: newState,
          });
        }
      }

      state = newState;
      eventsExecuted++;
    }

    const executionTime = Date.now() - startTime;

    return {
      finalState: state,
      eventsExecuted,
      breakpointsHit,
      executionTime,
      deviations,
    };
  }

  /**
   * Replay from specific timestamp
   */
  replayFrom(
    timeline: Timeline,
    fromTimestamp: Timestamp,
    executor: (event: Event, state: Record<string, any>) => Record<string, any>
  ): ReplayResult {
    // Find nearest snapshot before timestamp
    const snapshot = this.findNearestSnapshot(timeline, fromTimestamp);

    // Create modified timeline starting from snapshot
    const modifiedTimeline: Timeline = {
      ...timeline,
      events: timeline.events.filter((e) => e.timestamp >= fromTimestamp),
      snapshots: snapshot ? [snapshot] : timeline.snapshots,
    };

    return this.replay(modifiedTimeline, executor);
  }

  /**
   * Reverse replay (backward execution)
   */
  reverseReplay(
    timeline: Timeline,
    fromTimestamp: Timestamp,
    steps: number
  ): StateSnapshot[] {
    const snapshots: StateSnapshot[] = [];

    // Find events before timestamp
    const events = timeline.events
      .filter((e) => e.timestamp <= fromTimestamp)
      .slice(-steps);

    // Find corresponding snapshots
    for (const event of events.reverse()) {
      const snapshot = timeline.snapshots.find(
        (s) => Math.abs(s.timestamp - event.timestamp) < 100
      );
      if (snapshot) {
        snapshots.push(snapshot);
      }
    }

    return snapshots;
  }

  /**
   * Add breakpoint
   */
  addBreakpoint(breakpoint: Breakpoint): void {
    this.breakpoints.set(breakpoint.id, breakpoint);
  }

  /**
   * Remove breakpoint
   */
  removeBreakpoint(breakpointId: BreakpointId): void {
    this.breakpoints.delete(breakpointId);
  }

  /**
   * Check if any breakpoint is hit
   */
  private checkBreakpoints(
    event: Event,
    state: Record<string, any>
  ): BreakpointId | null {
    for (const [id, breakpoint] of this.breakpoints) {
      if (breakpoint.enabled && breakpoint.condition(event, state)) {
        breakpoint.hitCount++;
        return id;
      }
    }
    return null;
  }

  /**
   * Find nearest snapshot before timestamp
   */
  private findNearestSnapshot(
    timeline: Timeline,
    timestamp: Timestamp
  ): StateSnapshot | null {
    const candidates = timeline.snapshots.filter(
      (s) => s.timestamp <= timestamp
    );
    if (candidates.length === 0) return null;

    return candidates.reduce((prev, current) =>
      current.timestamp > prev.timestamp ? current : prev
    );
  }

  private computeChecksum(state: Record<string, any>): string {
    const stateString = JSON.stringify(state);
    let hash = 0;
    for (let i = 0; i < stateString.length; i++) {
      const char = stateString.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return hash.toString(16);
  }

  setPauseOnBreakpoint(pause: boolean): void {
    this.pauseOnBreakpoint = pause;
  }

  getAllBreakpoints(): Breakpoint[] {
    return Array.from(this.breakpoints.values());
  }
}

// ============================================================================
// Counterfactual Injector
// ============================================================================

class CounterfactualInjector {
  /**
   * Inject counterfactual modification
   */
  inject(
    timeline: Timeline,
    timestamp: Timestamp,
    modifications: Record<string, any>,
    description: string
  ): Timeline {
    const modifiedTimeline = cloneDeep(timeline);

    const counterfactual: Counterfactual = {
      timestamp,
      description,
      modifications,
      applied: true,
    };

    modifiedTimeline.counterfactuals.set(timestamp, counterfactual);

    return modifiedTimeline;
  }

  /**
   * Create branching timeline with counterfactual
   */
  createBranch(
    parentTimeline: Timeline,
    branchPoint: Timestamp,
    counterfactuals: Array<{
      timestamp: Timestamp;
      modifications: Record<string, any>;
      description: string;
    }>
  ): Timeline {
    const branchTimeline: Timeline = {
      id: this.generateId(),
      parentTimelineId: parentTimeline.id,
      branchPoint,
      events: parentTimeline.events.filter((e) => e.timestamp >= branchPoint),
      snapshots: parentTimeline.snapshots.filter(
        (s) => s.timestamp >= branchPoint
      ),
      counterfactuals: new Map(),
    };

    // Apply counterfactuals
    for (const cf of counterfactuals) {
      branchTimeline.counterfactuals.set(cf.timestamp, {
        timestamp: cf.timestamp,
        description: cf.description,
        modifications: cf.modifications,
        applied: true,
      });
    }

    return branchTimeline;
  }

  /**
   * Compare timelines
   */
  compareTimelines(
    timeline1: Timeline,
    timeline2: Timeline,
    executor: (event: Event, state: Record<string, any>) => Record<string, any>
  ): {
    timeline1Result: ReplayResult;
    timeline2Result: ReplayResult;
    differences: Array<{
      timestamp: Timestamp;
      field: string;
      timeline1Value: any;
      timeline2Value: any;
    }>;
  } {
    const engine = new ReplayEngine();

    const result1 = engine.replay(timeline1, executor);
    const result2 = engine.replay(timeline2, executor);

    const differences: Array<{
      timestamp: Timestamp;
      field: string;
      timeline1Value: any;
      timeline2Value: any;
    }> = [];

    // Compare final states
    for (const key of Object.keys(result1.finalState)) {
      if (
        JSON.stringify(result1.finalState[key]) !==
        JSON.stringify(result2.finalState[key])
      ) {
        differences.push({
          timestamp: Date.now(),
          field: key,
          timeline1Value: result1.finalState[key],
          timeline2Value: result2.finalState[key],
        });
      }
    }

    return {
      timeline1Result: result1,
      timeline2Result: result2,
      differences,
    };
  }

  private generateId(): string {
    return `timeline_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
  }
}

// ============================================================================
// Demonstration
// ============================================================================

export async function demonstrateTimeTravelDebugging(): Promise<void> {
  console.log("=".repeat(80));
  console.log("TIME-TRAVEL DEBUGGING DEMONSTRATION");
  console.log("=".repeat(80));

  // Demo 1: Record execution
  console.log("\n📹 Demo 1: Recording Agent Execution");
  console.log("-".repeat(80));

  const recorder = new TemporalRecorder(500);

  const initialState = {
    position: { x: 0, y: 0 },
    velocity: { x: 1, y: 1 },
    energy: 100,
    score: 0,
  };

  recorder.startRecording(initialState);
  console.log(
    "Recording started with initial state:",
    JSON.stringify(initialState)
  );

  // Simulate 5 events
  const events: Event[] = [];
  for (let i = 0; i < 5; i++) {
    const event: Event = {
      id: `event_${i}`,
      timestamp: Date.now() + i * 100,
      type: EventType.ACTION,
      agentId: "agent_1",
      action: "move",
      inputs: { direction: i % 2 === 0 ? "right" : "up" },
      outputs: { newPosition: { x: i + 1, y: i } },
      metadata: { step: i },
    };

    recorder.recordEvent(event);
    recorder.updateState({
      position: event.outputs.newPosition,
      energy: initialState.energy - (i + 1) * 10,
      score: i * 5,
    });

    events.push(event);
    console.log(`  Event ${i}: ${event.action} ${event.inputs.direction}`);
  }

  const timeline = recorder.stopRecording();
  console.log(
    `\n✓ Recording complete: ${timeline.events.length} events, ${timeline.snapshots.length} snapshots`
  );

  // Demo 2: Deterministic Replay
  console.log("\n▶️ Demo 2: Deterministic Replay");
  console.log("-".repeat(80));

  const engine = new ReplayEngine();

  const executor = (
    event: Event,
    state: Record<string, any>
  ): Record<string, any> => {
    const newState = cloneDeep(state);

    if (event.action === "move") {
      newState.position = event.outputs.newPosition;
      newState.energy -= 10;
      newState.score += 5;
    }

    return newState;
  };

  const replayResult = engine.replay(timeline, executor);

  console.log(`Replay completed:`);
  console.log(`  Events executed: ${replayResult.eventsExecuted}`);
  console.log(`  Execution time: ${replayResult.executionTime}ms`);
  console.log(
    `  Final state:`,
    JSON.stringify(replayResult.finalState, null, 2)
  );
  console.log(`  Deviations: ${replayResult.deviations.length}`);

  // Demo 3: Breakpoints
  console.log("\n🔴 Demo 3: Temporal Breakpoints");
  console.log("-".repeat(80));

  const breakpoint: Breakpoint = {
    id: "bp_low_energy",
    condition: (event: Event, state: Record<string, any>) => {
      return state.energy < 70;
    },
    enabled: true,
    hitCount: 0,
  };

  engine.addBreakpoint(breakpoint);
  console.log('Added breakpoint: "energy < 70"');

  const replayWithBreakpoint = engine.replay(timeline, executor);

  console.log(`\nReplay with breakpoint:`);
  console.log(
    `  Events executed: ${replayWithBreakpoint.eventsExecuted} (paused at breakpoint)`
  );
  console.log(
    `  Breakpoints hit: ${replayWithBreakpoint.breakpointsHit.join(", ")}`
  );
  console.log(`  Breakpoint hit count: ${breakpoint.hitCount}`);

  // Demo 4: Counterfactual Injection
  console.log('\n🔀 Demo 4: Counterfactual "What If" Analysis');
  console.log("-".repeat(80));

  const injector = new CounterfactualInjector();

  console.log("Original scenario: Agent moves right and up alternately");
  console.log(
    'Counterfactual: "What if agent moved left instead of right at step 2?"\n'
  );

  const counterfactualTimeline = injector.inject(
    timeline,
    events[2].timestamp,
    { direction: "left" },
    "Change direction to left"
  );

  console.log("Counterfactual injected at event 2");
  console.log(
    `Counterfactuals in timeline: ${counterfactualTimeline.counterfactuals.size}`
  );

  // Demo 5: Branching Timelines
  console.log("\n🌳 Demo 5: Branching Timelines");
  console.log("-".repeat(80));

  const branchPoint = events[2].timestamp;
  const branchTimeline = injector.createBranch(timeline, branchPoint, [
    {
      timestamp: events[2].timestamp,
      modifications: { direction: "left", speed: 2 },
      description: "Move left with double speed",
    },
  ]);

  console.log(`Created branch timeline at event 2`);
  console.log(`  Parent timeline: ${timeline.id}`);
  console.log(`  Branch timeline: ${branchTimeline.id}`);
  console.log(`  Branch point: ${branchPoint}`);
  console.log(`  Events in branch: ${branchTimeline.events.length}`);

  // Demo 6: Timeline Comparison
  console.log("\n⚖️ Demo 6: Timeline Comparison");
  console.log("-".repeat(80));

  engine.setPauseOnBreakpoint(false); // Don't pause for comparison

  const comparison = injector.compareTimelines(
    timeline,
    counterfactualTimeline,
    executor
  );

  console.log("Comparing original vs counterfactual timeline:\n");
  console.log(
    `Original final state:`,
    JSON.stringify(comparison.timeline1Result.finalState, null, 2)
  );
  console.log(
    `\nCounterfactual final state:`,
    JSON.stringify(comparison.timeline2Result.finalState, null, 2)
  );

  console.log(`\nDifferences found: ${comparison.differences.length}`);
  for (const diff of comparison.differences) {
    console.log(`  ${diff.field}:`);
    console.log(`    Original: ${JSON.stringify(diff.timeline1Value)}`);
    console.log(`    Counterfactual: ${JSON.stringify(diff.timeline2Value)}`);
  }

  // Demo 7: Reverse Replay
  console.log("\n⏮️ Demo 7: Reverse Replay (Backward Execution)");
  console.log("-".repeat(80));

  const lastEventTime = events[events.length - 1].timestamp;
  const reverseSnapshots = engine.reverseReplay(timeline, lastEventTime, 3);

  console.log(
    `Stepping backward from end (${reverseSnapshots.length} steps):\n`
  );
  for (let i = 0; i < reverseSnapshots.length; i++) {
    console.log(`Step ${i + 1} backward:`);
    console.log(`  State:`, JSON.stringify(reverseSnapshots[i].state, null, 2));
  }

  console.log("\n✅ Time-Travel Debugging demonstration complete!");
  console.log("=".repeat(80));
}

// Export classes for programmatic use
export {
  TemporalRecorder,
  ReplayEngine,
  CounterfactualInjector,
  EventType,
  type Event,
  type Timeline,
  type StateSnapshot,
  type Counterfactual,
  type Breakpoint,
  type ReplayResult,
};
