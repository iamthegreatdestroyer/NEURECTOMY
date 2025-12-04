# Agent Visualization Guide

This guide covers advanced techniques for visualizing AI agents, their workflows, states, and inter-agent communication in the Neurectomy 3D Engine.

## Table of Contents

1. [Agent Types and Representations](#agent-types-and-representations)
2. [State Visualization](#state-visualization)
3. [Workflow Visualization](#workflow-visualization)
4. [Communication Flows](#communication-flows)
5. [Performance Metrics](#performance-metrics)
6. [Multi-Agent Systems](#multi-agent-systems)
7. [Real-Time Monitoring](#real-time-monitoring)
8. [Advanced Customization](#advanced-customization)

---

## Agent Types and Representations

### Built-in Agent Types

The 3D engine provides distinct visual representations for different agent types:

```typescript
import { AgentVisualizerService, type AgentType } from "@neurectomy/3d-engine";

// Available agent types
type AgentType =
  | "llm" // Large Language Model agents
  | "tool" // Tool-using agents
  | "retriever" // RAG/retrieval agents
  | "router" // Routing/orchestration agents
  | "supervisor" // Supervisory agents
  | "worker" // Worker agents
  | "memory" // Memory management agents
  | "custom"; // Custom agent types

// Create visualizer
const visualizer = new AgentVisualizerService(scene, {
  defaultSize: 1.5,
  animationSpeed: 1.0,
  enableGlow: true,
});
```

### Visual Representation by Type

| Agent Type   | Shape                      | Default Color    | Icon      |
| ------------ | -------------------------- | ---------------- | --------- |
| `llm`        | Sphere with neural pattern | Blue (#3b82f6)   | Brain     |
| `tool`       | Cube with gears            | Orange (#f97316) | Wrench    |
| `retriever`  | Cylinder with search icon  | Green (#22c55e)  | Magnifier |
| `router`     | Diamond                    | Purple (#a855f7) | Fork      |
| `supervisor` | Crown-topped sphere        | Gold (#fbbf24)   | Crown     |
| `worker`     | Octahedron                 | Gray (#6b7280)   | Cog       |
| `memory`     | Torus                      | Cyan (#06b6d4)   | Database  |
| `custom`     | Configurable               | Configurable     | Custom    |

### Creating Agents

```typescript
import type { AgentConfig } from "@neurectomy/3d-engine";

// Basic agent
const basicAgent: AgentConfig = {
  id: "research-agent",
  name: "Research Assistant",
  type: "llm",
  position: [0, 0, 0],
  status: "idle",
};

visualizer.addAgent(basicAgent);

// Agent with custom appearance
const customAgent: AgentConfig = {
  id: "custom-agent",
  name: "Data Processor",
  type: "custom",
  position: [5, 0, 0],
  status: "idle",
  appearance: {
    geometry: "icosahedron",
    color: "#ec4899",
    size: 2.0,
    glow: {
      enabled: true,
      color: "#ec4899",
      intensity: 0.8,
    },
    texture: "path/to/texture.png",
  },
};

visualizer.addAgent(customAgent);
```

---

## State Visualization

Agents have multiple states that are visually represented:

### Agent States

```typescript
import type { AgentStatus } from "@neurectomy/3d-engine";

type AgentStatus =
  | "idle" // Waiting for tasks
  | "running" // Actively processing
  | "thinking" // LLM inference in progress
  | "waiting" // Waiting for external input
  | "error" // Error occurred
  | "completed" // Task completed
  | "paused"; // Temporarily paused

// Update agent state
visualizer.updateAgentStatus("research-agent", "running");

// State with additional context
visualizer.updateAgentStatus("research-agent", "thinking", {
  message: "Analyzing document corpus...",
  progress: 0.45,
});
```

### Visual State Indicators

```typescript
// Configure state visualizations
visualizer.configureStateVisuals({
  idle: {
    color: "#6b7280",
    animation: "subtle-pulse",
    glowIntensity: 0.2,
  },
  running: {
    color: "#22c55e",
    animation: "spin-rings",
    glowIntensity: 0.6,
    particles: {
      enabled: true,
      type: "sparkle",
      count: 50,
    },
  },
  thinking: {
    color: "#3b82f6",
    animation: "neural-pulse",
    glowIntensity: 0.8,
    overlay: "thought-bubbles",
  },
  error: {
    color: "#ef4444",
    animation: "shake",
    glowIntensity: 1.0,
    icon: "warning",
  },
});
```

### Progress Indicators

```typescript
// Show progress bar around agent
visualizer.showProgress("research-agent", {
  type: "ring",
  progress: 0.75,
  color: "#3b82f6",
  showPercentage: true,
});

// Multiple progress indicators
visualizer.showProgress("analysis-agent", {
  type: "multi-ring",
  segments: [
    { label: "Retrieval", progress: 1.0, color: "#22c55e" },
    { label: "Analysis", progress: 0.6, color: "#3b82f6" },
    { label: "Synthesis", progress: 0.0, color: "#6b7280" },
  ],
});
```

---

## Workflow Visualization

### Defining Workflows

```typescript
import {
  WorkflowVisualizer,
  type WorkflowDefinition,
} from "@neurectomy/3d-engine";

const workflow: WorkflowDefinition = {
  id: "research-workflow",
  name: "Research Pipeline",
  steps: [
    {
      id: "step-1",
      name: "Query Processing",
      agentId: "router-agent",
      outputs: ["step-2a", "step-2b"],
    },
    {
      id: "step-2a",
      name: "Web Search",
      agentId: "retriever-agent-1",
      outputs: ["step-3"],
    },
    {
      id: "step-2b",
      name: "Database Query",
      agentId: "retriever-agent-2",
      outputs: ["step-3"],
    },
    {
      id: "step-3",
      name: "Synthesis",
      agentId: "llm-agent",
      inputs: ["step-2a", "step-2b"],
      outputs: ["step-4"],
    },
    {
      id: "step-4",
      name: "Validation",
      agentId: "supervisor-agent",
      outputs: [], // Terminal step
    },
  ],
};

const workflowViz = new WorkflowVisualizer(scene, visualizer);
workflowViz.loadWorkflow(workflow);
```

### Workflow Layout

```typescript
// Apply automatic layout
workflowViz.applyLayout({
  type: "hierarchical",
  direction: "left-to-right", // or 'top-to-bottom'
  spacing: {
    horizontal: 8,
    vertical: 4,
  },
  alignment: "center",
});

// Or use force-directed layout
workflowViz.applyLayout({
  type: "force",
  strength: -200,
  linkDistance: 100,
});
```

### Workflow Execution Visualization

```typescript
// Start workflow execution visualization
const execution = workflowViz.startExecution();

// Highlight current step
execution.activateStep("step-1");

// Show data flowing between steps
execution.flowData({
  from: "step-1",
  to: ["step-2a", "step-2b"],
  data: { query: "research topic" },
  animationDuration: 1500,
});

// Mark step as complete
execution.completeStep("step-2a", {
  result: "success",
  data: { documents: 5 },
});

// Handle step errors
execution.errorStep("step-2b", {
  message: "Database timeout",
  retryable: true,
});
```

---

## Communication Flows

### Direct Communication

```typescript
// Point-to-point message
visualizer.showMessage({
  from: "supervisor-agent",
  to: "worker-agent-1",
  type: "instruction",
  content: "Process batch A",
  visualization: {
    style: "beam",
    color: "#3b82f6",
    duration: 800,
  },
});

// Request-response pattern
visualizer.showRequestResponse({
  requester: "llm-agent",
  responder: "tool-agent",
  request: {
    content: "Execute search",
    visualization: { style: "pulse", color: "#f97316" },
  },
  response: {
    content: "Results: 10 documents",
    visualization: { style: "stream", color: "#22c55e" },
    delay: 500, // Simulated response time
  },
});
```

### Broadcast Communication

```typescript
// One-to-many broadcast
visualizer.showBroadcast({
  from: "supervisor-agent",
  to: ["worker-1", "worker-2", "worker-3", "worker-4"],
  message: "New task available",
  visualization: {
    style: "ripple",
    color: "#a855f7",
    staggerDelay: 100, // Delay between each recipient
  },
});

// Publish-subscribe visualization
visualizer.showPubSub({
  topic: "document-updates",
  publisher: "retriever-agent",
  subscribers: ["analysis-agent", "summary-agent", "archive-agent"],
  event: { type: "new-document", id: "doc-123" },
});
```

### Data Flow Visualization

```typescript
// Stream visualization for continuous data
const stream = visualizer.createDataStream({
  source: "data-source",
  target: "processing-agent",
  style: "particles",
  particleCount: 100,
  speed: 2.0,
  color: "#06b6d4",
});

// Update stream intensity based on throughput
stream.setIntensity(0.8); // 80% capacity

// Show data volume
stream.showMetrics({
  throughput: "1.2K msg/s",
  latency: "45ms",
});

// Cleanup
stream.dispose();
```

---

## Performance Metrics

### Metric Display

```typescript
// Show metrics panel for agent
visualizer.showMetrics("llm-agent", {
  position: "above", // above, below, left, right
  metrics: [
    { label: "Tokens/s", value: 150, format: "number" },
    { label: "Latency", value: 245, format: "ms" },
    { label: "Memory", value: 2.4, format: "gb" },
    { label: "Accuracy", value: 0.94, format: "percentage" },
  ],
  style: {
    background: "rgba(0, 0, 0, 0.7)",
    textColor: "#ffffff",
    accentColor: "#3b82f6",
  },
});
```

### Real-Time Metric Updates

```typescript
// Create metric tracker
const tracker = visualizer.createMetricTracker("llm-agent");

// Update metrics in real-time
setInterval(() => {
  tracker.update({
    tokensPerSecond: getTokenRate(),
    latency: getLatency(),
    queueDepth: getQueueDepth(),
  });
}, 1000);

// Show metric history graph
tracker.showHistory({
  metric: "tokensPerSecond",
  duration: 60000, // Last 60 seconds
  style: "line",
});
```

### Aggregate Metrics

```typescript
// System-wide metrics
visualizer.showSystemMetrics({
  position: "top-right",
  metrics: [
    { label: "Active Agents", value: 12 },
    { label: "Tasks/min", value: 450 },
    { label: "Avg Latency", value: "125ms" },
    { label: "Error Rate", value: "0.02%" },
  ],
});
```

---

## Multi-Agent Systems

### Agent Groups

```typescript
// Create logical groups
visualizer.createGroup("research-team", {
  agents: ["researcher-1", "researcher-2", "researcher-3"],
  visualization: {
    boundary: "sphere",
    color: "#3b82f6",
    opacity: 0.2,
    label: "Research Team",
  },
});

// Nest groups
visualizer.createGroup("department", {
  agents: [],
  subgroups: ["research-team", "analysis-team"],
  visualization: {
    boundary: "box",
    color: "#22c55e",
  },
});
```

### Hierarchical Visualization

```typescript
// Define hierarchy
const hierarchy = {
  root: "master-supervisor",
  children: [
    {
      id: "team-lead-a",
      children: ["worker-1", "worker-2"],
    },
    {
      id: "team-lead-b",
      children: ["worker-3", "worker-4", "worker-5"],
    },
  ],
};

visualizer.showHierarchy(hierarchy, {
  layout: "tree",
  direction: "top-down",
  spacing: 5,
  showConnections: true,
});
```

### Swarm Visualization

```typescript
// For large numbers of similar agents
visualizer.showSwarm({
  template: {
    type: "worker",
    size: 0.5,
  },
  count: 100,
  behavior: "flock",
  center: [0, 0, 0],
  radius: 20,
  colorBy: "status", // Color based on agent status
});

// Update swarm states
visualizer.updateSwarmStatus({
  idle: 20,
  running: 65,
  waiting: 10,
  error: 5,
});
```

---

## Real-Time Monitoring

### Event Stream Integration

```typescript
import { AgentEventStream } from "@neurectomy/3d-engine";

// Connect to event stream
const eventStream = new AgentEventStream({
  url: "ws://localhost:8080/events",
  reconnect: true,
});

// Bind to visualizer
eventStream.on("agent-status-change", (event) => {
  visualizer.updateAgentStatus(event.agentId, event.newStatus);
});

eventStream.on("message-sent", (event) => {
  visualizer.showMessage({
    from: event.senderId,
    to: event.receiverId,
    type: event.messageType,
    content: event.preview,
  });
});

eventStream.on("workflow-step-complete", (event) => {
  workflowViz.completeStep(event.stepId, event.result);
});

eventStream.connect();
```

### Replay Mode

```typescript
import { EventRecorder, EventPlayer } from "@neurectomy/3d-engine";

// Record events
const recorder = new EventRecorder();
recorder.start();

// ... events occur ...

recorder.stop();
const recording = recorder.getRecording();

// Save recording
await recorder.save("session-2024-01-15.nrec");

// Load and replay
const player = new EventPlayer(visualizer);
await player.load("session-2024-01-15.nrec");

player.play({
  speed: 2.0, // 2x speed
  onEvent: (event) => console.log("Replaying:", event),
});

player.pause();
player.seek(30000); // Jump to 30 seconds
player.resume();
```

---

## Advanced Customization

### Custom Agent Shaders

```typescript
import { ShaderMaterial } from "three";
import { AgentShaderLibrary } from "@neurectomy/3d-engine";

// Use built-in shader
const neuralShader = AgentShaderLibrary.get("neural-network");

// Or create custom shader
const customShader = new ShaderMaterial({
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform float time;
    uniform vec3 color;
    varying vec2 vUv;
    
    void main() {
      float pulse = sin(time * 2.0) * 0.5 + 0.5;
      gl_FragColor = vec4(color * pulse, 1.0);
    }
  `,
  uniforms: {
    time: { value: 0 },
    color: { value: new THREE.Color("#3b82f6") },
  },
});

// Apply to agent
visualizer.setAgentMaterial("custom-agent", customShader);
```

### Custom Icons and Labels

```typescript
// Custom SVG icon
visualizer.setAgentIcon("tool-agent", {
  type: "svg",
  svg: `<svg>...</svg>`,
  size: 64,
  offset: [0, 2, 0],
});

// Custom label renderer
visualizer.setLabelRenderer("llm-agent", {
  render: (agent, camera) => {
    const distance = camera.position.distanceTo(agent.position);
    return {
      text: agent.name,
      fontSize: Math.max(12, 24 - distance * 0.5),
      background: agent.status === "error" ? "#ef4444" : "#1f2937",
    };
  },
});
```

### Custom Animations

```typescript
import { AnimationMixer, AnimationClip } from "three";

// Create custom animation
const animation = visualizer.createAnimation("success-burst", {
  duration: 1000,
  keyframes: [
    { time: 0, scale: 1, opacity: 1 },
    { time: 500, scale: 1.5, opacity: 0.8 },
    { time: 1000, scale: 1, opacity: 1 },
  ],
  easing: "easeOutElastic",
});

// Trigger animation on event
visualizer.on("task-complete", (agentId) => {
  visualizer.playAnimation(agentId, "success-burst");
});
```

---

## Best Practices

### Performance Optimization

```typescript
// Use LOD for large agent counts
visualizer.enableLOD({
  levels: [
    { distance: 10, detail: "high" },
    { distance: 50, detail: "medium" },
    { distance: 100, detail: "low" },
    { distance: 200, detail: "billboard" },
  ],
});

// Batch updates
visualizer.batchUpdate(() => {
  for (const agent of agents) {
    visualizer.updateAgentStatus(agent.id, agent.status);
  }
});

// Use object pooling
visualizer.enablePooling({
  maxAgents: 1000,
  maxMessages: 500,
});
```

### Accessibility

```typescript
// Ensure all agents are keyboard navigable
visualizer.enableAccessibility({
  keyboardNavigation: true,
  screenReaderDescriptions: true,
  highContrastMode: false,
});

// Provide descriptions
visualizer.setAccessibilityDescription("llm-agent", {
  role: "article",
  label: "Research Assistant Agent",
  description: "Large language model agent currently processing 3 tasks",
  state: "running, 75% complete",
});
```

---

_For more details, see the [API Reference](../api/3d-engine/agent-visualizer.md) and [Getting Started Guide](./getting-started-3d.md)._
