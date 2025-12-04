# Tutorials

Welcome to the Neurectomy 3D Engine tutorials. These guides will help you understand and use the various features of the engine.

## Getting Started

| Tutorial                                           | Description                                          | Difficulty   |
| -------------------------------------------------- | ---------------------------------------------------- | ------------ |
| [Getting Started with 3D](./getting-started-3d.md) | Basic setup, scene creation, and first visualization | Beginner     |
| [Agent Visualization](./agent-visualization.md)    | Visualize AI agents, workflows, and communication    | Intermediate |
| [Temporal Navigation](./temporal-navigation.md)    | Navigate through time in 4D visualizations           | Intermediate |
| [Graph Exploration](./graph-exploration.md)        | Work with Neo4j knowledge graphs                     | Intermediate |

## Recommended Learning Path

### For Beginners

1. **[Getting Started with 3D](./getting-started-3d.md)** - Start here to understand the basics
   - Installing the engine
   - Creating a scene
   - Basic rendering
   - Adding interactivity

2. **[Graph Exploration](./graph-exploration.md)** - Learn graph visualization
   - Connecting to Neo4j
   - Rendering nodes and edges
   - Basic interactions

### For Intermediate Users

3. **[Agent Visualization](./agent-visualization.md)** - Visualize AI systems
   - Agent types and states
   - Workflow visualization
   - Real-time monitoring

4. **[Temporal Navigation](./temporal-navigation.md)** - Work with time
   - Timeline controls
   - Historical playback
   - Branching timelines

### For Advanced Users

5. **[Digital Twins](./digital-twins.md)** _(Coming Soon)_
   - Creating digital twins
   - Real-time synchronization
   - Predictive analysis

6. **[Custom Effects](./custom-effects.md)** _(Coming Soon)_
   - Particle systems
   - Custom shaders
   - Post-processing

7. **[Performance Optimization](./performance-optimization.md)** _(Coming Soon)_
   - Large-scale graphs
   - Memory management
   - WebWorkers

## Quick Reference

### Common Import Patterns

```typescript
// Core visualization
import {
  createScene,
  createRenderer,
  createCamera,
  GraphRenderer,
  AgentVisualizerService,
} from "@neurectomy/3d-engine";

// Layouts
import {
  ForceLayoutEngine,
  HierarchicalLayout3D,
  ClusteredLayout3D,
} from "@neurectomy/3d-engine/layouts";

// Effects
import {
  ParticleSystemManager,
  EffectsManager,
} from "@neurectomy/3d-engine/effects";

// Accessibility
import {
  initializeAccessibility,
  ScreenReaderManager,
  KeyboardNavigationManager,
} from "@neurectomy/3d-engine/accessibility";

// Internationalization
import { initializeI18n, t, setLanguage } from "@neurectomy/3d-engine/i18n";

// Temporal
import {
  TemporalManager,
  TimelineBranch,
} from "@neurectomy/3d-engine/temporal";
```

### Essential Setup Code

```typescript
// Minimal setup
import {
  createScene,
  createRenderer,
  createCamera,
} from "@neurectomy/3d-engine";

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const scene = createScene({ background: 0x1a1a2e });
const renderer = createRenderer(canvas, { antialias: true });
const camera = createCamera({ type: "perspective", fov: 75 });

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();
```

## Troubleshooting

### Common Issues

| Issue                | Solution                                |
| -------------------- | --------------------------------------- |
| Canvas not rendering | Ensure canvas has explicit width/height |
| WebGL errors         | Check browser WebGL support             |
| Performance issues   | Enable LOD and viewport culling         |
| Memory leaks         | Call dispose() on unused resources      |

### Getting Help

- Check the [API Reference](../api/README.md)
- Review [Architecture Decisions](../architecture/adr/README.md)
- Open an issue on GitHub

---

_Part of the Neurectomy Documentation_
