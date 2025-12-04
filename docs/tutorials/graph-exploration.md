# Graph Exploration Guide

This guide covers advanced techniques for visualizing and exploring Neo4j knowledge graphs in the Neurectomy 3D Engine.

## Table of Contents

1. [Connecting to Neo4j](#connecting-to-neo4j)
2. [Graph Data Model](#graph-data-model)
3. [Visual Mapping](#visual-mapping)
4. [Layout Algorithms](#layout-algorithms)
5. [Interactive Exploration](#interactive-exploration)
6. [Filtering and Search](#filtering-and-search)
7. [Graph Analytics Integration](#graph-analytics-integration)
8. [Performance at Scale](#performance-at-scale)

---

## Connecting to Neo4j

### Basic Connection

```typescript
import { Neo4jConnector, GraphDataSource } from "@neurectomy/3d-engine";

// Create connection
const neo4j = new Neo4jConnector({
  uri: "neo4j://localhost:7687",
  username: "neo4j",
  password: process.env.NEO4J_PASSWORD,
  database: "neurectomy",

  // Connection pool
  pool: {
    maxSize: 10,
    acquisitionTimeout: 30000,
  },
});

// Test connection
const connected = await neo4j.verify();
if (!connected) {
  throw new Error("Failed to connect to Neo4j");
}
```

### Graph Data Source

```typescript
// Create data source for visualization
const graphSource = new GraphDataSource(neo4j, {
  // Auto-refresh configuration
  refresh: {
    enabled: true,
    interval: 5000, // 5 seconds
    strategy: "incremental", // or 'full'
  },

  // Query optimization
  optimization: {
    useProjections: true,
    batchSize: 1000,
    parallelism: 4,
  },
});
```

---

## Graph Data Model

### Node and Edge Types

```typescript
import type {
  Neo4jNode,
  Neo4jEdge,
  NodeStyle,
  EdgeStyle,
} from "@neurectomy/3d-engine";

// Define node types in your graph
interface AgentNode extends Neo4jNode {
  labels: ["Agent"];
  properties: {
    id: string;
    name: string;
    type: "llm" | "tool" | "retriever";
    status: string;
    createdAt: Date;
  };
}

interface ConceptNode extends Neo4jNode {
  labels: ["Concept"];
  properties: {
    id: string;
    name: string;
    category: string;
    embedding: number[];
  };
}

// Define relationship types
interface InteractsRelationship extends Neo4jEdge {
  type: "INTERACTS_WITH";
  properties: {
    messageCount: number;
    lastInteraction: Date;
  };
}

interface KnowsRelationship extends Neo4jEdge {
  type: "KNOWS";
  properties: {
    confidence: number;
    source: string;
  };
}
```

### Loading Graph Data

```typescript
// Load initial graph
const graph = await graphSource.loadGraph({
  // Cypher query
  query: `
    MATCH (a:Agent)-[r]->(b)
    WHERE a.status = 'active'
    RETURN a, r, b
    LIMIT 1000
  `,

  // Or use pattern matching
  patterns: [
    { nodeLabels: ["Agent"], limit: 100 },
    { nodeLabels: ["Concept"], limit: 500 },
    { relationshipTypes: ["KNOWS", "INTERACTS_WITH"] },
  ],
});

// Add to scene
graphRenderer.setGraph(graph);
```

---

## Visual Mapping

### Node Styling

```typescript
import { StyleMapper } from "@neurectomy/3d-engine";

// Create style mapper
const styleMapper = new StyleMapper();

// Map node labels to styles
styleMapper.mapNodeLabels({
  Agent: {
    shape: "sphere",
    size: 2.0,
    color: "#3b82f6",
    glow: {
      enabled: true,
      color: "#3b82f6",
      intensity: 0.5,
    },
  },
  Concept: {
    shape: "octahedron",
    size: 1.5,
    color: "#22c55e",
  },
  Document: {
    shape: "box",
    size: 1.0,
    color: "#f97316",
  },
});

// Map properties to visual attributes
styleMapper.mapNodeProperties({
  status: {
    property: "status",
    mapping: {
      active: { color: "#22c55e", glow: { intensity: 0.8 } },
      inactive: { color: "#6b7280", glow: { intensity: 0.2 } },
      error: { color: "#ef4444", glow: { intensity: 1.0 } },
    },
  },
  importance: {
    property: "pageRank",
    scale: "linear",
    range: {
      size: [1.0, 3.0],
      glowIntensity: [0.2, 1.0],
    },
  },
});

// Apply mapper to renderer
graphRenderer.setStyleMapper(styleMapper);
```

### Edge Styling

```typescript
// Map relationship types to styles
styleMapper.mapEdgeTypes({
  KNOWS: {
    color: "#3b82f6",
    width: 0.1,
    style: "solid",
    arrow: "end",
  },
  INTERACTS_WITH: {
    color: "#22c55e",
    width: 0.15,
    style: "dashed",
    arrow: "both",
    animated: true,
  },
  DEPENDS_ON: {
    color: "#ef4444",
    width: 0.2,
    style: "solid",
    arrow: "end",
  },
});

// Map edge properties
styleMapper.mapEdgeProperties({
  strength: {
    property: "confidence",
    scale: "linear",
    range: {
      width: [0.05, 0.3],
      opacity: [0.3, 1.0],
    },
  },
  activity: {
    property: "messageCount",
    scale: "log",
    range: {
      width: [0.1, 0.5],
    },
    pulseOnHigh: true,
  },
});
```

### Custom Visual Functions

```typescript
// Advanced custom styling
styleMapper.setNodeStyleFunction((node) => {
  const labels = node.labels;
  const props = node.properties;

  // Complex logic for style
  let style: NodeStyle = {
    shape: "sphere",
    size: 1.0,
    color: "#6b7280",
  };

  // Multi-label handling
  if (labels.includes("Agent") && labels.includes("Supervisor")) {
    style = {
      shape: "crown",
      size: 2.5,
      color: "#fbbf24",
      glow: { enabled: true, color: "#fbbf24", intensity: 1.0 },
    };
  }

  // Property-based adjustments
  if (props.embedding) {
    // Color based on embedding cluster
    style.color = getClusterColor(props.embedding);
  }

  return style;
});
```

---

## Layout Algorithms

### Force-Directed Layout

```typescript
import { ForceLayout3D } from "@neurectomy/3d-engine";

const layout = new ForceLayout3D({
  // Force parameters
  forces: {
    charge: {
      strength: -200,
      distanceMin: 1,
      distanceMax: 100,
    },
    link: {
      distance: 30,
      strength: 0.5,
    },
    center: {
      strength: 0.1,
    },
    collision: {
      radius: (node) => node.style.size * 1.5,
      strength: 0.8,
    },
  },

  // Simulation settings
  simulation: {
    alphaMin: 0.001,
    alphaDecay: 0.02,
    velocityDecay: 0.4,
  },
});

// Apply layout
graphRenderer.setLayout(layout);

// Control simulation
layout.start();
layout.stop();
layout.reheat(0.5); // Restart with alpha=0.5
```

### Hierarchical Layout

```typescript
import { HierarchicalLayout3D } from "@neurectomy/3d-engine";

const hierarchicalLayout = new HierarchicalLayout3D({
  // Hierarchy detection
  hierarchy: {
    rootProperty: "isRoot", // or detect from edges
    direction: "top-down", // or 'bottom-up', 'left-right', 'right-left'
  },

  // Spacing
  spacing: {
    levelHeight: 10,
    siblingDistance: 5,
    subtreeDistance: 10,
  },

  // Tree arrangement
  arrangement: {
    algorithm: "reingold-tilford", // or 'buchheim', 'walker'
    balanced: true,
  },
});

graphRenderer.setLayout(hierarchicalLayout);
```

### Clustered Layout

```typescript
import { ClusteredLayout3D } from "@neurectomy/3d-engine";

const clusteredLayout = new ClusteredLayout3D({
  // Clustering method
  clustering: {
    method: "louvain", // Community detection
    resolution: 1.0,
    property: null, // Or use existing property
  },

  // Cluster arrangement
  arrangement: {
    method: "sphere-packing", // or 'grid', 'force'
    clusterSpacing: 20,
    intraClusterForce: {
      charge: -50,
      link: { distance: 10 },
    },
  },

  // Visual grouping
  visualization: {
    showBoundaries: true,
    boundaryStyle: "sphere",
    boundaryOpacity: 0.1,
    colorByCommunity: true,
  },
});
```

### Custom Layout

```typescript
// Implement custom layout algorithm
class EmbeddingLayout implements Layout3D {
  constructor(
    private options: {
      embeddingProperty: string;
      dimensionReduction: "tsne" | "umap" | "pca";
      perplexity?: number;
    }
  ) {}

  async calculate(graph: Graph): Promise<Map<string, Vector3>> {
    const positions = new Map<string, Vector3>();

    // Extract embeddings
    const embeddings = graph.nodes.map(
      (n) => n.properties[this.options.embeddingProperty]
    );

    // Apply dimension reduction
    const reduced = await this.reduce(embeddings);

    // Map to 3D positions
    graph.nodes.forEach((node, i) => {
      positions.set(node.id, {
        x: reduced[i][0] * 50,
        y: reduced[i][1] * 50,
        z: reduced[i][2] * 50,
      });
    });

    return positions;
  }

  private async reduce(embeddings: number[][]): Promise<number[][]> {
    // Implement t-SNE, UMAP, or PCA
    // ...
  }
}

// Use custom layout
const embeddingLayout = new EmbeddingLayout({
  embeddingProperty: "embedding",
  dimensionReduction: "umap",
});

graphRenderer.setLayout(embeddingLayout);
```

---

## Interactive Exploration

### Node Selection

```typescript
import { GraphInteractionManager } from "@neurectomy/3d-engine";

const interaction = new GraphInteractionManager(graphRenderer, camera);

// Single selection
interaction.onNodeClick((node, event) => {
  // Show node details
  showNodePanel(node);

  // Highlight connected nodes
  graphRenderer.highlightNeighbors(node.id, {
    depth: 1,
    highlightEdges: true,
  });
});

// Multi-selection
interaction.enableMultiSelect({
  key: "Shift",
  mode: "additive",
});

interaction.onSelectionChange((selected) => {
  // Show selection summary
  updateSelectionPanel(selected);
});

// Hover effects
interaction.onNodeHover((node) => {
  graphRenderer.setNodeHover(node.id, true);
  showTooltip(node);
});

interaction.onNodeHoverEnd(() => {
  graphRenderer.clearHover();
  hideTooltip();
});
```

### Graph Expansion

```typescript
// Expand from selected node
interaction.onNodeDoubleClick(async (node) => {
  // Load more connected nodes
  const expansion = await graphSource.expand(node.id, {
    depth: 1,
    limit: 50,
    filter: {
      labels: ["Concept", "Document"],
      excludeExisting: true,
    },
  });

  // Add to graph with animation
  await graphRenderer.addNodes(expansion.nodes, {
    animation: "grow-from-center",
    duration: 500,
    center: node.position,
  });

  await graphRenderer.addEdges(expansion.edges, {
    animation: "grow",
    duration: 300,
  });
});

// Collapse sub-graph
interaction.onNodeRightClick((node, event) => {
  showContextMenu(event, [
    {
      label: "Collapse Neighbors",
      action: () => graphRenderer.collapseNeighbors(node.id),
    },
    {
      label: "Hide Node",
      action: () => graphRenderer.hideNode(node.id),
    },
    {
      label: "Focus on Subgraph",
      action: () => graphRenderer.focusOnSubgraph(node.id, { depth: 2 }),
    },
  ]);
});
```

### Path Finding

```typescript
// Interactive path finding
interaction.enablePathFinding({
  startKey: "p",
  endKey: "p",
});

interaction.onPathRequest(async (startNode, endNode) => {
  // Find shortest path
  const path = await graphSource.findPath({
    start: startNode.id,
    end: endNode.id,
    algorithm: "dijkstra", // or 'a-star', 'bellman-ford'
    weightProperty: "distance",
  });

  // Visualize path
  graphRenderer.highlightPath(path, {
    color: "#f97316",
    width: 0.4,
    animated: true,
    animationSpeed: 2.0,
  });

  // Show path details
  showPathPanel(path);
});

// Find all paths
const allPaths = await graphSource.findAllPaths({
  start: nodeA.id,
  end: nodeB.id,
  maxLength: 5,
  limit: 10,
});

graphRenderer.showMultiplePaths(allPaths, {
  colorScheme: "gradient",
});
```

---

## Filtering and Search

### Visual Filters

```typescript
import { GraphFilter } from "@neurectomy/3d-engine";

const filter = new GraphFilter(graphRenderer);

// Filter by label
filter.setLabelFilter({
  include: ["Agent", "Concept"],
  exclude: ["Temp"],
});

// Filter by property
filter.setPropertyFilter({
  "Agent.status": ["active", "pending"],
  "Concept.confidence": { min: 0.5 },
  createdAt: { after: "2024-01-01" },
});

// Filter by degree
filter.setDegreeFilter({
  min: 2,
  max: 100,
});

// Combine filters
filter.apply();

// Show hidden count
const hidden = filter.getHiddenCount();
console.log(`Hiding ${hidden.nodes} nodes and ${hidden.edges} edges`);
```

### Search and Highlight

```typescript
import { GraphSearch } from "@neurectomy/3d-engine";

const search = new GraphSearch(graphSource, graphRenderer);

// Text search
const results = await search.find("machine learning", {
  properties: ["name", "description"],
  fuzzy: true,
  limit: 20,
});

// Highlight results
search.highlight(results, {
  style: "outline",
  color: "#fbbf24",
  pulseOnFind: true,
});

// Navigate through results
search.focusResult(0); // Focus first result
search.nextResult(); // Next result
search.previousResult(); // Previous result

// Clear search
search.clear();
```

### Cypher Query Interface

```typescript
// Enable query mode
const queryMode = graphRenderer.enableQueryMode();

// Execute and visualize query
const results = await queryMode.execute(`
  MATCH path = (a:Agent)-[:KNOWS*1..3]-(b:Agent)
  WHERE a.name = 'Research Agent'
  RETURN path
`);

// Visualize query results
queryMode.visualize(results, {
  highlightMatches: true,
  showVariableBindings: true,
  animateMatches: true,
});

// Save query for reuse
queryMode.saveQuery("find-knowledge-path", query);
```

---

## Graph Analytics Integration

### Running Graph Algorithms

```typescript
import { GraphAnalytics } from "@neurectomy/3d-engine";

const analytics = new GraphAnalytics(graphSource);

// PageRank
const pageRank = await analytics.run("pageRank", {
  maxIterations: 20,
  dampingFactor: 0.85,
});

// Map to node sizes
graphRenderer.mapPropertyToSize("pageRank", {
  scale: "sqrt",
  range: [1.0, 5.0],
});

// Community Detection
const communities = await analytics.run("louvain", {
  resolution: 1.0,
});

// Color by community
graphRenderer.mapPropertyToColor("community", {
  palette: "categorical",
});

// Centrality
const betweenness = await analytics.run("betweenness", {
  normalized: true,
});

// Show centrality as glow intensity
graphRenderer.mapPropertyToGlow("betweenness", {
  range: [0.1, 1.0],
});
```

### Real-Time Analytics

```typescript
// Stream analytics updates
const stream = analytics.createStream({
  algorithm: "pageRank",
  updateInterval: 5000,
  incrementalUpdate: true,
});

stream.on("update", (results) => {
  graphRenderer.updateNodeProperties(results);
});

stream.start();
```

### Custom Analytics

```typescript
// Run custom GDS projection
const projection = await analytics.createProjection("myGraph", {
  nodeLabels: ["Agent", "Concept"],
  relationshipTypes: {
    KNOWS: { orientation: "UNDIRECTED", properties: ["confidence"] },
  },
});

// Run algorithm on projection
const result = await analytics.runOnProjection(projection, "nodeSimilarity", {
  topK: 10,
  similarityMetric: "cosine",
});

// Visualize similarity as edges
graphRenderer.addSimilarityEdges(result, {
  minSimilarity: 0.5,
  style: {
    color: "#a855f7",
    style: "dotted",
    opacity: 0.5,
  },
});
```

---

## Performance at Scale

### Level of Detail

```typescript
// Configure LOD for large graphs
graphRenderer.enableLOD({
  levels: [
    {
      distance: 20,
      nodeDetail: "high",
      edgeDetail: "full",
      labels: "all",
      effects: true,
    },
    {
      distance: 50,
      nodeDetail: "medium",
      edgeDetail: "simplified",
      labels: "selected",
      effects: false,
    },
    {
      distance: 100,
      nodeDetail: "low",
      edgeDetail: "bundled",
      labels: "none",
      effects: false,
    },
    {
      distance: 200,
      nodeDetail: "point",
      edgeDetail: "none",
      labels: "none",
      effects: false,
    },
  ],
});
```

### Edge Bundling

```typescript
// Bundle edges for cleaner visualization
graphRenderer.enableEdgeBundling({
  method: "force-directed", // or 'hierarchical', 'radial'
  strength: 0.5,
  compatibility: 0.6,
  subdivisions: 6,
  cycles: 6,
});
```

### Viewport Culling

```typescript
// Only render visible elements
graphRenderer.enableViewportCulling({
  margin: 1.2, // 20% margin outside viewport
  updateFrequency: 100, // Check every 100ms
});

// Progressive loading based on viewport
graphRenderer.enableProgressiveLoading({
  loadRadius: 50,
  unloadRadius: 100,
  batchSize: 100,
});
```

### WebWorker Processing

```typescript
// Offload computation to workers
graphRenderer.enableWorkerProcessing({
  workers: 4,
  tasks: ["layout", "bundling", "clustering"],
});

// Custom worker for analytics
const analyticsWorker = graphRenderer.createWorker("./analytics-worker.js");

analyticsWorker.postMessage({
  type: "calculate",
  algorithm: "pageRank",
  graph: graphData,
});

analyticsWorker.onMessage((result) => {
  graphRenderer.applyAnalyticsResult(result);
});
```

---

## Best Practices

### Graph Loading Strategy

```typescript
// Load graph progressively
async function loadLargeGraph() {
  // 1. Load high-importance nodes first
  const coreNodes = await graphSource.loadGraph({
    query: `
      MATCH (n)
      WITH n, n.pageRank as importance
      ORDER BY importance DESC
      LIMIT 100
      RETURN n
    `,
  });

  graphRenderer.setGraph(coreNodes);

  // 2. Load connections between core nodes
  const coreEdges = await graphSource.loadEdgesBetween(coreNodes);
  graphRenderer.addEdges(coreEdges);

  // 3. Incrementally load more based on user interaction
  interaction.onNodeExpand(async (node) => {
    const neighbors = await graphSource.expand(node.id, { depth: 1 });
    graphRenderer.addSubgraph(neighbors);
  });
}
```

### Memory Management

```typescript
// Monitor memory usage
graphRenderer.on("memoryWarning", (usage) => {
  console.warn(`High memory usage: ${usage.percentage}%`);

  // Reduce graph
  graphRenderer.reduceGraph({
    keepHighImportance: 0.2, // Keep top 20%
    aggregateClusters: true,
  });
});

// Explicit cleanup
graphRenderer.dispose();
neo4j.close();
```

---

_For API reference, see [Graph API Documentation](../api/3d-engine/graph.md). Related guides: [Getting Started](./getting-started-3d.md) and [Agent Visualization](./agent-visualization.md)._
