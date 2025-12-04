# Getting Started with the 3D Engine

This guide will walk you through the basics of using the Neurectomy 3D Engine for visualizing agent workflows and knowledge graphs.

## Table of Contents

1. [Installation](#installation)
2. [Basic Setup](#basic-setup)
3. [Creating Your First Scene](#creating-your-first-scene)
4. [Working with Graphs](#working-with-graphs)
5. [Agent Visualization](#agent-visualization)
6. [Accessibility Features](#accessibility-features)
7. [Internationalization](#internationalization)
8. [Next Steps](#next-steps)

---

## Installation

The 3D engine is part of the `@neurectomy/3d-engine` package:

```bash
# Using pnpm (recommended)
pnpm add @neurectomy/3d-engine

# Using npm
npm install @neurectomy/3d-engine
```

### Peer Dependencies

Make sure you have Three.js installed:

```bash
pnpm add three @types/three
```

---

## Basic Setup

### Importing the Engine

```typescript
import {
  // Core rendering
  createScene,
  createRenderer,
  createCamera,

  // Graph visualization
  GraphRenderer,
  ForceLayoutEngine,

  // Agent visualization
  AgentVisualizerService,

  // Effects
  ParticleSystemManager,
  EffectsManager,
} from "@neurectomy/3d-engine";
```

### Creating a Basic Scene

```typescript
import {
  createScene,
  createRenderer,
  createCamera,
} from "@neurectomy/3d-engine";

// Get your canvas element
const canvas = document.getElementById("canvas") as HTMLCanvasElement;

// Create the Three.js scene
const scene = createScene({
  background: 0x1a1a2e, // Dark blue background
  fog: {
    color: 0x1a1a2e,
    near: 10,
    far: 100,
  },
});

// Create renderer
const renderer = createRenderer(canvas, {
  antialias: true,
  alpha: false,
  powerPreference: "high-performance",
});

// Create camera
const camera = createCamera({
  type: "perspective",
  fov: 75,
  near: 0.1,
  far: 1000,
  position: [0, 5, 10],
});

// Animation loop
function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}

animate();
```

---

## Creating Your First Scene

### Adding Lighting

```typescript
import { createLights } from "@neurectomy/3d-engine";

const lights = createLights({
  ambient: {
    color: 0xffffff,
    intensity: 0.4,
  },
  directional: {
    color: 0xffffff,
    intensity: 0.8,
    position: [5, 10, 5],
    castShadow: true,
  },
});

scene.add(...lights);
```

### Adding Camera Controls

```typescript
import { OrbitControls } from "@neurectomy/3d-engine";

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.enableZoom = true;
controls.enablePan = true;

// Update in animation loop
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
```

---

## Working with Graphs

The 3D engine includes powerful graph visualization capabilities for displaying knowledge graphs and agent relationships.

### Creating a Graph

```typescript
import {
  GraphRenderer,
  ForceLayoutEngine,
  type GraphNode,
  type GraphEdge,
} from "@neurectomy/3d-engine";

// Define your graph data
const nodes: GraphNode[] = [
  {
    id: "agent-1",
    label: "Research Agent",
    type: "agent",
    position: [0, 0, 0],
  },
  {
    id: "agent-2",
    label: "Analysis Agent",
    type: "agent",
    position: [5, 0, 0],
  },
  { id: "data-1", label: "Dataset", type: "data", position: [2.5, 3, 0] },
];

const edges: GraphEdge[] = [
  { id: "e1", source: "agent-1", target: "data-1", type: "dataFlow" },
  { id: "e2", source: "data-1", target: "agent-2", type: "dataFlow" },
  { id: "e3", source: "agent-1", target: "agent-2", type: "communication" },
];

// Create graph renderer
const graphRenderer = new GraphRenderer(scene, {
  nodeSize: 1,
  edgeWidth: 0.1,
  enableLabels: true,
  labelScale: 0.5,
});

// Add nodes and edges
nodes.forEach((node) => graphRenderer.addNode(node));
edges.forEach((edge) => graphRenderer.addEdge(edge));
```

### Force-Directed Layout

```typescript
import { ForceLayoutEngine } from "@neurectomy/3d-engine";

// Create force layout
const layout = new ForceLayoutEngine({
  strength: -100, // Node repulsion
  linkDistance: 50, // Ideal edge length
  linkStrength: 0.5, // Edge spring strength
  gravity: 0.1, // Pull toward center
  friction: 0.9, // Velocity damping
});

// Apply layout
layout.setGraph(nodes, edges);

// Update in animation loop
function animate() {
  requestAnimationFrame(animate);

  // Step the layout simulation
  const positions = layout.step();

  // Update node positions in renderer
  positions.forEach((pos, nodeId) => {
    graphRenderer.updateNodePosition(nodeId, pos);
  });

  controls.update();
  renderer.render(scene, camera);
}
```

### Interactive Selection

```typescript
import { SelectionManager } from "@neurectomy/3d-engine";

const selectionManager = new SelectionManager(
  camera,
  renderer.domElement,
  graphRenderer.getSelectableObjects()
);

// Handle selection events
selectionManager.onSelect((object) => {
  console.log("Selected:", object.userData.id);
  graphRenderer.highlightNode(object.userData.id);
});

selectionManager.onDeselect(() => {
  graphRenderer.clearHighlights();
});
```

---

## Agent Visualization

Visualize AI agents with their states, connections, and activity.

### Basic Agent Display

```typescript
import { AgentVisualizerService, type AgentState } from "@neurectomy/3d-engine";

// Create visualizer
const agentVisualizer = new AgentVisualizerService(scene);

// Define agent state
const agent: AgentState = {
  id: "llm-agent-1",
  name: "Research Assistant",
  type: "llm",
  status: "running",
  position: [0, 0, 0],
  metrics: {
    tokensUsed: 1500,
    requestCount: 10,
    avgResponseTime: 250,
  },
};

// Add agent to scene
agentVisualizer.addAgent(agent);

// Update agent status
agentVisualizer.updateAgentStatus("llm-agent-1", "idle");

// Show agent activity
agentVisualizer.showActivity("llm-agent-1", {
  type: "processing",
  progress: 0.75,
  message: "Analyzing document...",
});
```

### Agent Communication Visualization

```typescript
// Show communication between agents
agentVisualizer.showCommunication({
  from: "llm-agent-1",
  to: "analysis-agent-1",
  type: "data",
  payload: { documents: 5 },
  duration: 1000, // Animation duration in ms
});

// Show broadcast communication
agentVisualizer.showBroadcast({
  from: "supervisor-1",
  to: ["worker-1", "worker-2", "worker-3"],
  type: "command",
  message: "Begin processing",
});
```

---

## Accessibility Features

The 3D engine includes comprehensive accessibility support.

### Screen Reader Support

```typescript
import {
  initializeAccessibility,
  ScreenReaderManager,
  AriaDescriptionGenerator,
} from "@neurectomy/3d-engine/accessibility";

// Initialize accessibility
const { screenReader, descriptionGenerator, keyboardNavigation } =
  initializeAccessibility(canvas, {
    verbosityLevel: "detailed",
    screenReaderMode: true,
    keyboardNavigation: true,
  });

// Announce scene changes
screenReader.announce("Scene loaded with 5 agents and 8 connections");

// Generate descriptions for elements
const description = descriptionGenerator.describe({
  type: "agent",
  id: "agent-1",
  label: "Research Assistant",
  position: { x: 0, y: 0, z: 0 },
});
```

### Keyboard Navigation

```typescript
import { KeyboardNavigationManager } from "@neurectomy/3d-engine/accessibility";

const keyboardNav = new KeyboardNavigationManager({
  rovingTabindex: true,
  announceOnFocus: true,
});

// Register navigable elements
graphRenderer.getNodes().forEach((node) => {
  keyboardNav.registerElement({
    id: node.id,
    element: node.mesh,
    label: node.label,
    description: `${node.type} node`,
    position: node.position,
    onActivate: () => selectionManager.select(node.id),
  });
});

// Initialize with container
keyboardNav.initialize(canvas);
```

### Color Blindness Support

```typescript
import {
  getPaletteForType,
  getHighContrastTheme,
} from "@neurectomy/3d-engine/accessibility";

// Get palette for specific color blindness type
const palette = getPaletteForType("deuteranopia"); // Green-blind friendly

// Apply to graph renderer
graphRenderer.setColorPalette(palette);

// Or use high contrast mode
const highContrastPalette = getHighContrastTheme("dark");
graphRenderer.setColorPalette(highContrastPalette);
```

---

## Internationalization

Support multiple languages in your 3D visualizations.

### Basic Setup

```typescript
import {
  initializeI18n,
  t,
  setLanguage,
  getLanguageSelectorOptions,
} from "@neurectomy/3d-engine/i18n";

// Initialize i18n
const { manager } = initializeI18n({
  defaultLanguage: "en",
  detectBrowserLanguage: true,
  persistLanguage: true,
});

// Use translations
const nodeLabel = t("graph.nodes.types.agent"); // "Agent Node"
const statusText = t("agent.statuses.running"); // "Running"

// With interpolation
const nodeCount = t("graph.nodes.count", { count: 5 }); // "5 nodes"

// Change language
await setLanguage("ja"); // Switch to Japanese
```

### RTL Language Support

```typescript
import { applyRTLStyles, isRTL } from "@neurectomy/3d-engine/i18n";

// Check if current language is RTL
if (isRTL(manager.getLanguage())) {
  applyRTLStyles(containerElement, manager.getLanguage());
}

// Listen for language changes
manager.onLanguageChange((event) => {
  if (isRTL(event.newLanguage)) {
    applyRTLStyles(containerElement, event.newLanguage);
  }
});
```

### Language Selector

```typescript
// Get options for language selector UI
const languageOptions = getLanguageSelectorOptions();

// Returns:
// [
//   { code: 'en', nativeName: 'English', englishName: 'English', direction: 'ltr' },
//   { code: 'ja', nativeName: '日本語', englishName: 'Japanese', direction: 'ltr' },
//   { code: 'ar', nativeName: 'العربية', englishName: 'Arabic', direction: 'rtl' },
//   ...
// ]
```

---

## Next Steps

Now that you understand the basics, explore these advanced topics:

- **[Agent Visualization](./agent-visualization.md)** - Deep dive into agent workflows
- **[Temporal Navigation](./temporal-navigation.md)** - Navigate through time in 4D
- **[Graph Exploration](./graph-exploration.md)** - Advanced Neo4j graph visualization
- **[Digital Twins](./digital-twins.md)** - Create and synchronize digital twins
- **[Custom Effects](./custom-effects.md)** - Particle systems and shaders

---

## Troubleshooting

### Common Issues

**Canvas not rendering:**

```typescript
// Ensure canvas has dimensions
canvas.style.width = "100%";
canvas.style.height = "100%";

// Handle resize
window.addEventListener("resize", () => {
  camera.aspect = canvas.clientWidth / canvas.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);
});
```

**Performance issues:**

```typescript
// Enable level of detail
graphRenderer.enableLOD({
  highDistance: 10,
  mediumDistance: 50,
  lowDistance: 100,
});

// Use instancing for many similar objects
graphRenderer.enableInstancing(true);

// Reduce particle counts
particleSystem.setMaxParticles(1000);
```

**WebGL errors:**

```typescript
// Check WebGL support
if (!renderer.capabilities.isWebGL2) {
  console.warn("WebGL 2 not supported, falling back to WebGL 1");
}

// Handle context loss
canvas.addEventListener("webglcontextlost", (event) => {
  event.preventDefault();
  // Pause animation loop
});

canvas.addEventListener("webglcontextrestored", () => {
  // Reinitialize renderer
  // Resume animation loop
});
```

---

_This guide is part of the Neurectomy documentation. For API reference, see [API Reference](../api/3d-engine/README.md)._
