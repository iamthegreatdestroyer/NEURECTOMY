# NEURECTOMY User Guide

> Complete guide to using NEURECTOMY for AI-powered codebase intelligence

## Table of Contents

1. [Getting Started](#getting-started)
2. [Core Concepts](#core-concepts)
3. [Discovery Engine](#discovery-engine)
4. [Continuous Intelligence](#continuous-intelligence)
5. [3D Visualization](#3d-visualization)
6. [Container Command](#container-command)
7. [Deployment Orchestrator](#deployment-orchestrator)
8. [Best Practices](#best-practices)

---

## Getting Started

### Prerequisites

- Node.js 18+ (LTS recommended)
- pnpm 8+
- Docker (for containerized services)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/neurectomy.git
cd neurectomy

# Install dependencies
pnpm install

# Build all packages
pnpm build

# Run development mode
pnpm dev
```

### Quick Start

```typescript
import { createDiscoveryEngine } from "@neurectomy/discovery-engine";
import { createContinuousIntelligence } from "@neurectomy/continuous-intelligence";

// Initialize the discovery engine
const discovery = createDiscoveryEngine({
  github: {
    token: process.env.GITHUB_TOKEN!,
    rateLimit: { maxRequestsPerHour: 5000 },
  },
  analysis: {
    maxDepth: 10,
    excludePatterns: ["node_modules", ".git", "dist"],
  },
});

// Scan a repository
const analysis = await discovery.scanRepository("owner", "repo");
console.log("Dependencies:", analysis.dependencies);
console.log("Recommendations:", analysis.recommendations);
```

---

## Core Concepts

### Architecture Overview

NEURECTOMY is built as a modular monorepo with specialized packages:

```
┌─────────────────────────────────────────────────────────────────┐
│                        NEURECTOMY                                │
├─────────────────────────────────────────────────────────────────┤
│  DISCOVERY          │  INTELLIGENCE       │  VISUALIZATION      │
│  ────────────────   │  ──────────────     │  ──────────────     │
│  Repository Scanner │  Self-Improvement   │  3D Graph Engine    │
│  Dependency Analyzer│  Predictive Maint.  │  Temporal Navigation│
│  Recommendations    │  Auto-Optimization  │  Agent Visualization│
├─────────────────────────────────────────────────────────────────┤
│  OPERATIONS         │  GOVERNANCE         │  ENTERPRISE         │
│  ────────────────   │  ──────────────     │  ──────────────     │
│  Container Command  │  Policy Engine      │  Legal Fortress     │
│  Deployment Orch.   │  Compliance Check   │  Performance Engine │
│  Experimentation    │  Audit Logging      │  Scalability        │
└─────────────────────────────────────────────────────────────────┘
```

### Event-Driven Design

All NEURECTOMY packages use an event-driven architecture:

```typescript
// All engines emit events you can subscribe to
engine.on("analysis:complete", (result) => {
  console.log("Analysis finished:", result);
});

engine.on("warning", (warning) => {
  console.warn("Warning detected:", warning);
});

engine.on("error", (error) => {
  console.error("Error occurred:", error);
});
```

### Configuration Pattern

Each package follows a consistent configuration pattern:

```typescript
interface EngineConfig {
  // Common options
  logging?: {
    level: "debug" | "info" | "warn" | "error";
    format: "json" | "pretty";
  };

  // Performance options
  performance?: {
    maxConcurrency: number;
    timeout: number;
    retryAttempts: number;
  };

  // Feature-specific options
  features?: Record<string, boolean>;
}
```

---

## Discovery Engine

### Overview

The Discovery Engine scans repositories, analyzes dependencies, and provides intelligent recommendations.

### Basic Usage

```typescript
import { createDiscoveryEngine } from "@neurectomy/discovery-engine";

const discovery = createDiscoveryEngine({
  github: {
    token: process.env.GITHUB_TOKEN!,
  },
});

// Start discovery
await discovery.start();

// Scan a repository
const result = await discovery.scanRepository("facebook", "react");

// Get dependency analysis
const deps = await discovery.analyzeDependencies(result.files);

// Get recommendations
const recommendations = await discovery.getRecommendations(result);
```

### Repository Scanning

```typescript
// Full repository scan
const fullScan = await discovery.scanRepository("owner", "repo", {
  branch: "main",
  depth: "full",
  includeMetadata: true,
  includeContributors: true,
});

// Partial scan with filters
const partialScan = await discovery.scanRepository("owner", "repo", {
  paths: ["src/**/*.ts", "lib/**/*.ts"],
  excludePaths: ["**/*.test.ts", "**/*.spec.ts"],
});
```

### Dependency Analysis

```typescript
// Analyze all dependencies
const analysis = await discovery.analyzeDependencies(files, {
  includeDevDependencies: true,
  checkVulnerabilities: true,
  analyzeLicenses: true,
});

// Check for circular dependencies
const circular = analysis.circularDependencies;

// Get outdated packages
const outdated = analysis.outdatedPackages;

// Get security vulnerabilities
const vulnerabilities = analysis.vulnerabilities;
```

### Getting Recommendations

```typescript
const recommendations = await discovery.getRecommendations(analysis, {
  categories: ["security", "performance", "maintainability"],
  minConfidence: 0.7,
  maxResults: 20,
});

for (const rec of recommendations) {
  console.log(`[${rec.severity}] ${rec.title}`);
  console.log(`  Impact: ${rec.impact}`);
  console.log(`  Effort: ${rec.effort}`);
  console.log(`  Action: ${rec.suggestedAction}`);
}
```

---

## Continuous Intelligence

### Overview

The Continuous Intelligence package provides self-improving AI capabilities, predictive maintenance, and ML-based auto-optimization.

### Self-Improvement Engine

```typescript
import {
  createContinuousIntelligence,
  createSelfImprovementEngine,
} from "@neurectomy/continuous-intelligence";

// Create the engine
const selfImprove = createSelfImprovementEngine({
  patternDetection: { enabled: true, minConfidence: 0.7 },
  modelManagement: { maxModels: 10 },
  learningRate: 0.01,
});

// Start the engine
await selfImprove.start();

// Record operations for pattern detection
selfImprove.recordOperation({
  type: "query",
  input: { query: "SELECT * FROM users" },
  output: { rows: 100 },
  duration: 150,
  success: true,
});

// Get detected patterns
const patterns = selfImprove.getDetectedPatterns();

// Get improvement strategies
const strategies = await selfImprove.generateStrategies();
```

### Pattern Detection

The self-improvement engine detects several pattern types:

```typescript
// Temporal patterns - operations that occur at specific times
const temporalPatterns = patterns.filter((p) => p.type === "temporal");

// Sequential patterns - operations that follow each other
const sequentialPatterns = patterns.filter((p) => p.type === "sequential");

// Resource spike patterns - unusual resource usage
const resourcePatterns = patterns.filter((p) => p.type === "resource-spike");

// Error cluster patterns - related errors
const errorPatterns = patterns.filter((p) => p.type === "error-cluster");
```

### Predictive Maintenance

```typescript
import { createPredictiveMaintenance } from "@neurectomy/continuous-intelligence";

const predictive = createPredictiveMaintenance({
  thresholds: {
    cpu: 80,
    memory: 85,
    disk: 90,
    errorRate: 0.05,
  },
  predictionHorizon: 3600000, // 1 hour ahead
});

await predictive.start();

// Record metrics
predictive.recordMetric("cpu", 65);
predictive.recordMetric("memory", 72);
predictive.recordMetric("responseTime", 150);

// Get health score
const health = predictive.getHealthScore();
console.log(`System health: ${health.overall}%`);

// Predict failures
const predictions = await predictive.predictFailures();
for (const prediction of predictions) {
  console.log(
    `Warning: ${prediction.type} likely in ${prediction.timeToFailure}ms`
  );
  console.log(`  Confidence: ${prediction.confidence}%`);
  console.log(`  Recommendation: ${prediction.recommendation}`);
}
```

### Auto-Optimization

```typescript
import { createAutoOptimizer } from "@neurectomy/continuous-intelligence";

const optimizer = createAutoOptimizer({
  strategy: "bayesian", // or 'genetic', 'simulated-annealing'
  objective: "minimize", // or 'maximize'
  maxIterations: 100,
});

await optimizer.start();

// Define parameter space
const parameterSpace = {
  learningRate: { min: 0.001, max: 0.1, type: "continuous" },
  batchSize: { values: [16, 32, 64, 128], type: "categorical" },
  layers: { min: 1, max: 10, type: "integer" },
};

// Run optimization
const result = await optimizer.optimize(parameterSpace, async (params) => {
  // Your objective function - returns a score
  const model = trainModel(params);
  return evaluateModel(model);
});

console.log("Best parameters:", result.bestParameters);
console.log("Best score:", result.bestScore);
```

### Unified Continuous Intelligence

```typescript
import { createContinuousIntelligence } from "@neurectomy/continuous-intelligence";

// Create unified engine with all capabilities
const ci = createContinuousIntelligence({
  selfImprovement: { enabled: true },
  predictiveMaintenance: { enabled: true },
  autoOptimization: { enabled: true, strategy: "bayesian" },
});

await ci.start();

// Use all capabilities through unified API
ci.recordOperation({ type: "query", duration: 100 });
ci.recordMetric("cpu", 65);

const insights = await ci.getInsights();
const predictions = await ci.getPredictions();
const optimizations = await ci.getOptimizations();

// Get comprehensive system report
const report = await ci.generateReport();
```

---

## 3D Visualization

### Overview

NEURECTOMY provides immersive 3D visualization for exploring code relationships, dependencies, and temporal changes.

### Basic Setup

```typescript
import { create3DEngine } from "@neurectomy/3d-engine";

const engine = create3DEngine({
  renderer: "webgl2",
  quality: "high",
  interaction: {
    enableZoom: true,
    enablePan: true,
    enableRotate: true,
  },
});

// Mount to DOM
engine.mount(document.getElementById("visualization-container"));

// Load graph data
await engine.loadGraph({
  nodes: [
    { id: "a", label: "Module A", type: "module" },
    { id: "b", label: "Module B", type: "module" },
    { id: "c", label: "Module C", type: "module" },
  ],
  edges: [
    { source: "a", target: "b", type: "dependency" },
    { source: "b", target: "c", type: "dependency" },
  ],
});
```

### Graph Exploration

See [graph-exploration.md](./graph-exploration.md) for detailed graph exploration features.

### Temporal Navigation

See [temporal-navigation.md](./temporal-navigation.md) for time-based code evolution features.

### Agent Visualization

See [agent-visualization.md](./agent-visualization.md) for AI agent activity visualization.

---

## Container Command

### Overview

Container Command provides powerful Docker and Kubernetes orchestration capabilities.

### Docker Operations

```typescript
import { createContainerCommand } from "@neurectomy/container-command";

const docker = createContainerCommand({
  type: "docker",
  socketPath: "/var/run/docker.sock",
});

// List containers
const containers = await docker.listContainers();

// Create and start a container
const container = await docker.createContainer({
  image: "nginx:latest",
  name: "my-nginx",
  ports: [{ host: 8080, container: 80 }],
  env: { NODE_ENV: "production" },
});

await docker.startContainer(container.id);

// Get container logs
const logs = await docker.getLogs(container.id, {
  tail: 100,
  follow: false,
});

// Execute command in container
const result = await docker.exec(container.id, ["ls", "-la"]);
```

### Kubernetes Operations

```typescript
import { createContainerCommand } from "@neurectomy/container-command";

const k8s = createContainerCommand({
  type: "kubernetes",
  kubeconfig: "~/.kube/config",
  context: "my-cluster",
});

// List pods
const pods = await k8s.listPods({ namespace: "default" });

// Deploy application
await k8s.apply({
  apiVersion: "apps/v1",
  kind: "Deployment",
  metadata: { name: "my-app" },
  spec: {
    replicas: 3,
    selector: { matchLabels: { app: "my-app" } },
    template: {
      metadata: { labels: { app: "my-app" } },
      spec: {
        containers: [
          {
            name: "app",
            image: "my-app:latest",
            ports: [{ containerPort: 3000 }],
          },
        ],
      },
    },
  },
});

// Scale deployment
await k8s.scale("deployment", "my-app", 5);
```

---

## Deployment Orchestrator

### Overview

The Deployment Orchestrator manages complex deployment pipelines with rollback capabilities.

### Basic Deployment

```typescript
import { createDeploymentOrchestrator } from "@neurectomy/deployment-orchestrator";

const orchestrator = createDeploymentOrchestrator({
  environments: ["development", "staging", "production"],
  strategy: "blue-green",
  rollback: { enabled: true, keepVersions: 3 },
});

// Create deployment pipeline
const pipeline = orchestrator.createPipeline({
  name: "my-app-deploy",
  stages: [
    {
      name: "build",
      actions: [
        { type: "command", command: "pnpm build" },
        { type: "docker-build", image: "my-app", tag: "${VERSION}" },
      ],
    },
    {
      name: "test",
      actions: [
        { type: "command", command: "pnpm test" },
        { type: "command", command: "pnpm test:e2e" },
      ],
    },
    {
      name: "deploy",
      actions: [{ type: "k8s-deploy", manifest: "k8s/deployment.yaml" }],
    },
  ],
});

// Execute pipeline
const result = await orchestrator.execute(pipeline, {
  environment: "staging",
  variables: { VERSION: "1.0.0" },
});
```

### Rollback

```typescript
// Automatic rollback on failure
orchestrator.on("stage:failed", async (event) => {
  console.log(`Stage ${event.stage} failed, initiating rollback...`);
  await orchestrator.rollback(event.deploymentId);
});

// Manual rollback
await orchestrator.rollback(deploymentId, {
  targetVersion: "v1.0.0",
  strategy: "immediate", // or 'gradual'
});
```

---

## Best Practices

### 1. Error Handling

Always wrap operations in try-catch and handle errors gracefully:

```typescript
try {
  const result = await discovery.scanRepository("owner", "repo");
} catch (error) {
  if (error instanceof RateLimitError) {
    // Wait and retry
    await sleep(error.retryAfter);
    return retry();
  }
  if (error instanceof AuthenticationError) {
    // Re-authenticate
    await refreshToken();
    return retry();
  }
  // Log and re-throw unknown errors
  logger.error("Unexpected error:", error);
  throw error;
}
```

### 2. Resource Management

Always clean up resources:

```typescript
const engine = createDiscoveryEngine(config);

try {
  await engine.start();
  // ... use the engine
} finally {
  await engine.stop(); // Always stop the engine
}
```

### 3. Event Subscription

Subscribe to events for monitoring and debugging:

```typescript
engine.on("progress", (progress) => {
  console.log(`Progress: ${progress.percent}% - ${progress.message}`);
});

engine.on("warning", (warning) => {
  logger.warn(warning);
});

engine.on("error", (error) => {
  logger.error(error);
  notifyOnCall(error);
});
```

### 4. Configuration Validation

Always validate configuration before use:

```typescript
import { validateConfig } from "@neurectomy/core";

const config = loadConfig("./config.yaml");

const validation = validateConfig(config);
if (!validation.valid) {
  console.error("Invalid configuration:", validation.errors);
  process.exit(1);
}
```

### 5. Logging

Use structured logging for better observability:

```typescript
import { createLogger } from "@neurectomy/core";

const logger = createLogger({
  service: "my-service",
  level: process.env.LOG_LEVEL || "info",
  format: process.env.NODE_ENV === "production" ? "json" : "pretty",
});

logger.info("Operation completed", {
  operation: "scan",
  duration: 1500,
  itemsProcessed: 100,
});
```

---

## Next Steps

- [Troubleshooting Guide](./troubleshooting.md) - Common issues and solutions
- [Migration Guide](./migration.md) - Upgrading between versions
- [API Reference](../api/README.md) - Complete API documentation
- [Architecture Decisions](../architecture/adr/) - Understanding design decisions

---

_Last updated: 2025_
