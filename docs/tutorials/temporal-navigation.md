# Temporal Navigation Guide

This guide covers the 4D temporal navigation system in the Neurectomy 3D Engine, enabling you to visualize, navigate, and analyze data across time.

## Table of Contents

1. [Understanding 4D Visualization](#understanding-4d-visualization)
2. [Timeline Setup](#timeline-setup)
3. [Temporal Navigation Controls](#temporal-navigation-controls)
4. [Time-Series Visualization](#time-series-visualization)
5. [Historical Playback](#historical-playback)
6. [Temporal Queries](#temporal-queries)
7. [Branching Timelines](#branching-timelines)
8. [Performance Optimization](#performance-optimization)

---

## Understanding 4D Visualization

The Neurectomy 3D Engine treats time as a navigable dimension alongside X, Y, and Z. This enables:

- **Historical Analysis**: Review past states of agents and graphs
- **Temporal Patterns**: Identify recurring patterns over time
- **Predictive Visualization**: See predicted future states
- **Branching Exploration**: Explore alternative timeline scenarios

### Time Representation

```typescript
import { TemporalManager, TimelineConfig } from "@neurectomy/3d-engine";

// Time is represented as a continuous coordinate
interface TemporalPosition {
  x: number; // Spatial X
  y: number; // Spatial Y
  z: number; // Spatial Z
  t: number; // Temporal coordinate (milliseconds from epoch)
}

// Or using temporal ranges
interface TemporalRange {
  start: number; // Start time (ms)
  end: number; // End time (ms)
  resolution: "millisecond" | "second" | "minute" | "hour" | "day";
}
```

---

## Timeline Setup

### Basic Timeline Configuration

```typescript
import { TemporalManager } from "@neurectomy/3d-engine";

const temporal = new TemporalManager(scene, {
  // Time range to visualize
  range: {
    start: Date.now() - 24 * 60 * 60 * 1000, // 24 hours ago
    end: Date.now(),
  },

  // Initial time position
  currentTime: Date.now() - 60 * 60 * 1000, // 1 hour ago

  // Visual settings
  visualization: {
    showTimeline: true,
    timelinePosition: "bottom",
    showMarkers: true,
    markerInterval: "hour",
  },

  // Data loading
  dataLoader: {
    chunkSize: 1000, // Records per chunk
    preloadAhead: 60000, // Preload 1 minute ahead
    cacheSize: 100, // Cache 100 chunks
  },
});
```

### Timeline Visualization

```typescript
// Configure timeline appearance
temporal.configureTimeline({
  // Main timeline bar
  bar: {
    height: 60,
    color: "#1f2937",
    hoverColor: "#374151",
    selectedColor: "#3b82f6",
  },

  // Time markers
  markers: {
    primary: {
      interval: "hour",
      color: "#9ca3af",
      height: 20,
      label: true,
      format: "HH:mm",
    },
    secondary: {
      interval: "minute",
      color: "#4b5563",
      height: 10,
      label: false,
    },
  },

  // Current time indicator
  playhead: {
    color: "#ef4444",
    width: 2,
    glow: true,
  },

  // Event markers on timeline
  events: {
    show: true,
    types: {
      error: { color: "#ef4444", icon: "warning" },
      milestone: { color: "#22c55e", icon: "flag" },
      change: { color: "#3b82f6", icon: "edit" },
    },
  },
});
```

---

## Temporal Navigation Controls

### Time Navigation

```typescript
// Jump to specific time
temporal.goToTime(new Date("2024-01-15T14:30:00Z"));

// Jump to relative time
temporal.goToRelativeTime(-3600000); // 1 hour back
temporal.goToRelativeTime(1800000); // 30 minutes forward

// Navigation shortcuts
temporal.goToStart(); // Beginning of range
temporal.goToEnd(); // End of range
temporal.goToNow(); // Current real time

// Step navigation
temporal.stepForward("minute"); // Forward 1 minute
temporal.stepBackward("hour"); // Backward 1 hour
```

### Playback Controls

```typescript
// Start playback
temporal.play({
  speed: 1.0, // Real-time
  direction: "forward",
});

// Playback at different speeds
temporal.play({ speed: 10.0 }); // 10x speed
temporal.play({ speed: 0.5 }); // Half speed
temporal.play({ speed: -1.0 }); // Reverse

// Playback controls
temporal.pause();
temporal.resume();
temporal.stop();

// Loop playback
temporal.setLoop({
  enabled: true,
  start: Date.parse("2024-01-15T12:00:00Z"),
  end: Date.parse("2024-01-15T14:00:00Z"),
  behavior: "loop", // or 'bounce'
});
```

### Keyboard Shortcuts

```typescript
// Enable keyboard navigation
temporal.enableKeyboardControls({
  play: "Space",
  forward: "ArrowRight",
  backward: "ArrowLeft",
  speedUp: "ArrowUp",
  speedDown: "ArrowDown",
  goToStart: "Home",
  goToEnd: "End",
  stepForward: "Shift+ArrowRight",
  stepBackward: "Shift+ArrowLeft",
});
```

---

## Time-Series Visualization

### Binding Data to Time

```typescript
// Register time-varying data source
temporal.registerDataSource("agent-metrics", {
  // Async data fetcher
  fetch: async (timeRange) => {
    const response = await fetch(
      `/api/metrics?start=${timeRange.start}&end=${timeRange.end}`
    );
    return response.json();
  },

  // Transform data for visualization
  transform: (data) =>
    data.map((point) => ({
      time: point.timestamp,
      position: [point.x, point.y, point.z],
      values: {
        cpu: point.cpu_usage,
        memory: point.memory_usage,
        throughput: point.throughput,
      },
    })),

  // Caching strategy
  cache: {
    enabled: true,
    ttl: 300000, // 5 minutes
  },
});
```

### Temporal Graph Data

```typescript
// Load graph history
temporal.loadGraphHistory({
  source: "neo4j",
  query: `
    MATCH (n)-[r]->(m)
    WHERE r.timestamp >= $start AND r.timestamp <= $end
    RETURN n, r, m, r.timestamp as time
  `,

  // How to visualize changes
  visualization: {
    nodeAppear: {
      animation: "fade-in",
      duration: 500,
    },
    nodeDisappear: {
      animation: "fade-out",
      duration: 500,
    },
    edgeAppear: {
      animation: "grow",
      duration: 300,
    },
    propertyChange: {
      animation: "pulse",
      highlight: true,
    },
  },
});
```

### Agent State History

```typescript
// Track agent state over time
temporal.trackAgentHistory("llm-agent-1", {
  properties: ["status", "position", "metrics"],

  // Visual representation
  trail: {
    enabled: true,
    length: 60000, // 1 minute trail
    color: "#3b82f6",
    opacity: 0.5,
    fadeOut: true,
  },

  // State change markers
  stateMarkers: {
    enabled: true,
    showOnTimeline: true,
    clickToJump: true,
  },
});
```

---

## Historical Playback

### Scene Snapshots

```typescript
// Take snapshot at current time
const snapshot = temporal.createSnapshot({
  include: ["agents", "graph", "effects"],
  metadata: {
    description: "Before optimization",
    tags: ["experiment", "baseline"],
  },
});

// Save snapshot
await temporal.saveSnapshot(snapshot, "baseline-2024-01-15.nsnap");

// Load and display snapshot
const loadedSnapshot = await temporal.loadSnapshot("baseline-2024-01-15.nsnap");
temporal.displaySnapshot(loadedSnapshot);

// Compare snapshots
temporal.compareSnapshots(snapshot1, snapshot2, {
  mode: "side-by-side", // or 'overlay', 'diff'
  highlightDifferences: true,
});
```

### Event Recording

```typescript
import { TemporalRecorder } from "@neurectomy/3d-engine";

// Record all scene events
const recorder = new TemporalRecorder(scene, temporal);

recorder.start({
  events: ["agent-state", "graph-change", "user-interaction"],
  compress: true,
});

// ... time passes, events occur ...

recorder.stop();
const recording = await recorder.export();

// Replay recording
const player = temporal.createPlayer(recording);
player.play({ speed: 5.0 });

// Interactive replay
player.onEvent((event) => {
  console.log(`Event at ${event.time}:`, event.type, event.data);
});
```

---

## Temporal Queries

### Querying Time Ranges

```typescript
// Find events in time range
const events = await temporal.queryEvents({
  timeRange: {
    start: Date.parse("2024-01-15T10:00:00Z"),
    end: Date.parse("2024-01-15T12:00:00Z"),
  },
  filters: {
    types: ["error", "warning"],
    agents: ["llm-agent-1", "llm-agent-2"],
    severity: { min: "medium" },
  },
  sort: "time-desc",
  limit: 100,
});

// Aggregate over time
const aggregation = await temporal.aggregate({
  timeRange: lastWeek,
  interval: "hour",
  metrics: ["throughput", "error_count", "latency"],
  groupBy: "agent_type",
});
```

### Temporal Patterns

```typescript
// Find recurring patterns
const patterns = await temporal.findPatterns({
  window: 3600000, // 1 hour window
  minOccurrences: 3,
  patternTypes: ["sequence", "correlation", "anomaly"],
});

// Detect anomalies
const anomalies = await temporal.detectAnomalies({
  metric: "response_time",
  sensitivity: 0.8,
  baseline: "rolling-average",
  baselineWindow: 86400000, // 24 hours
});

// Visualize patterns
patterns.forEach((pattern) => {
  temporal.highlightPattern(pattern, {
    color: "#f97316",
    showOnTimeline: true,
    label: pattern.description,
  });
});
```

---

## Branching Timelines

### Creating Branches

```typescript
import { TimelineBranch } from "@neurectomy/3d-engine";

// Create a branch from current state
const branch = temporal.createBranch({
  name: "What-if Scenario A",
  branchPoint: Date.now(),
  description: "Testing alternative configuration",
});

// Make modifications in branch
temporal.switchToBranch(branch.id);

// Changes here don't affect main timeline
agentVisualizer.updateAgentConfig("llm-agent-1", {
  temperature: 0.5, // Different from main
});

// Compare branches
temporal.compareBranches("main", branch.id, {
  metrics: ["throughput", "accuracy", "latency"],
  visualization: "chart",
});
```

### Branch Visualization

```typescript
// Show branch tree
temporal.showBranchTree({
  layout: "horizontal", // or 'vertical'
  showLabels: true,
  colorByAge: true,
  interactive: true,
});

// Configure branch display
temporal.configureBranchVisualization({
  main: {
    color: "#22c55e",
    width: 4,
    label: "Main Timeline",
  },
  branches: {
    colors: ["#3b82f6", "#f97316", "#a855f7"],
    width: 2,
    showBranchPoints: true,
  },
});

// Show multiple branches simultaneously
temporal.showParallelBranches(["main", "scenario-a", "scenario-b"], {
  arrangement: "stacked", // or 'layered'
  syncPlayback: true,
});
```

### Merging Branches

```typescript
// Merge insights from branch
const mergeResult = await temporal.mergeBranch(branch.id, {
  target: "main",
  strategy: "cherry-pick", // or 'full-merge'
  items: ["config-change-1", "config-change-2"],
});

// Handle conflicts
if (mergeResult.hasConflicts) {
  temporal.showConflicts(mergeResult.conflicts);

  // Resolve manually
  for (const conflict of mergeResult.conflicts) {
    await temporal.resolveConflict(conflict.id, {
      resolution: "use-branch", // or 'use-main', 'custom'
    });
  }
}
```

---

## Performance Optimization

### Data Loading Strategies

```typescript
// Configure progressive loading
temporal.configureDataLoading({
  // Load detail based on zoom level
  levelOfDetail: {
    overview: {
      resolution: "hour",
      maxPoints: 100,
    },
    detail: {
      resolution: "second",
      maxPoints: 10000,
    },
  },

  // Predictive preloading
  preload: {
    enabled: true,
    direction: "forward",
    amount: 300000, // 5 minutes ahead
    priority: "low",
  },

  // Memory management
  cache: {
    maxSize: 100 * 1024 * 1024, // 100MB
    evictionPolicy: "lru",
    persistToDisk: true,
  },
});
```

### Rendering Optimization

```typescript
// Temporal LOD
temporal.enableTemporalLOD({
  // Far past/future: low detail
  far: {
    distance: 86400000, // > 1 day
    nodeDetail: "low",
    edgeDetail: "simplified",
    effects: false,
  },

  // Near present: full detail
  near: {
    distance: 3600000, // < 1 hour
    nodeDetail: "high",
    edgeDetail: "full",
    effects: true,
  },
});

// Batch temporal updates
temporal.enableBatching({
  maxBatchSize: 1000,
  batchInterval: 16, // ~60fps
});

// Use instancing for repeated elements
temporal.enableInstancing({
  minInstances: 10,
  shareGeometry: true,
});
```

### Memory Management

```typescript
// Configure garbage collection
temporal.configureMemory({
  // Automatic cleanup
  autoCleanup: {
    enabled: true,
    interval: 60000, // Check every minute
    threshold: 0.8, // Clean when 80% full
  },

  // Data retention
  retention: {
    snapshots: 50,
    recordings: 10,
    branches: 20,
  },

  // Offload to disk
  offload: {
    enabled: true,
    directory: "./temporal-cache",
    compression: "lz4",
  },
});
```

---

## Best Practices

### Time Zone Handling

```typescript
// Always use UTC internally
temporal.configure({
  internalTimeZone: "UTC",
  displayTimeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
  timeFormat: "ISO8601",
});

// Handle daylight saving
temporal.enableDSTHandling({
  showTransitions: true,
  smoothTransitions: true,
});
```

### Accessibility

```typescript
// Make timeline accessible
temporal.enableAccessibility({
  keyboardNavigation: true,
  screenReaderAnnouncements: true,

  // Announce time changes
  announcements: {
    timeChange: "Moved to {time}",
    eventReached: "Event: {event}",
    playbackStart: "Playback started at {speed}x speed",
    playbackPause: "Playback paused at {time}",
  },

  // High contrast timeline
  highContrast: {
    enabled: false,
    colors: {
      background: "#000000",
      foreground: "#ffffff",
      highlight: "#ffff00",
    },
  },
});
```

---

_For API details, see [Temporal API Reference](../api/3d-engine/temporal.md). For related topics, see [Graph Exploration](./graph-exploration.md) and [Agent Visualization](./agent-visualization.md)._
