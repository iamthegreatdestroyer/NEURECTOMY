# NEURECTOMY Troubleshooting Guide

> Solutions to common issues and debugging techniques

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Installation Issues](#installation-issues)
3. [Discovery Engine Issues](#discovery-engine-issues)
4. [Continuous Intelligence Issues](#continuous-intelligence-issues)
5. [3D Visualization Issues](#3d-visualization-issues)
6. [Container Command Issues](#container-command-issues)
7. [Performance Issues](#performance-issues)
8. [Error Reference](#error-reference)

---

## Quick Diagnostics

### Health Check Script

Run this script to diagnose common issues:

```bash
# Run the NEURECTOMY health check
pnpm health-check

# Or manually check each component
pnpm run check:dependencies
pnpm run check:services
pnpm run check:config
```

### Log Analysis

```bash
# View logs with filtering
pnpm logs --level error --service discovery-engine

# Export logs for analysis
pnpm logs --export --since "2 hours ago" > debug-logs.json
```

### System Information

```typescript
import { getSystemInfo } from "@neurectomy/core";

const info = await getSystemInfo();
console.log("System Info:", JSON.stringify(info, null, 2));

// Output includes:
// - Node.js version
// - pnpm version
// - OS information
// - Available memory
// - Package versions
// - Configuration status
```

---

## Installation Issues

### Issue: `pnpm install` fails with peer dependency errors

**Symptoms:**

```
ERR_PNPM_PEER_DEP_ISSUES Unmet peer dependencies
```

**Solution:**

```bash
# Use the --shamefully-hoist flag for compatibility
pnpm install --shamefully-hoist

# Or add to .npmrc
echo "shamefully-hoist=true" >> .npmrc
pnpm install
```

### Issue: TypeScript build errors after install

**Symptoms:**

```
error TS2307: Cannot find module '@neurectomy/types'
```

**Solution:**

```bash
# Ensure all packages are built in correct order
pnpm build:types
pnpm build:core
pnpm build

# If still failing, clean and rebuild
pnpm clean
pnpm install
pnpm build
```

### Issue: Node.js version mismatch

**Symptoms:**

```
error: The engine "node" is incompatible with this module
```

**Solution:**

```bash
# Check your Node.js version
node --version

# NEURECTOMY requires Node.js 18+
# Use nvm to switch versions
nvm install 18
nvm use 18

# Or with Volta
volta install node@18
```

### Issue: pnpm workspace not recognized

**Symptoms:**

```
ERR_PNPM_NO_MATCHING_VERSION  No matching version found for @neurectomy/core
```

**Solution:**

```bash
# Ensure you're in the root directory
cd /path/to/neurectomy

# Verify pnpm-workspace.yaml exists
cat pnpm-workspace.yaml

# Reinstall from clean state
rm -rf node_modules
rm pnpm-lock.yaml
pnpm install
```

---

## Discovery Engine Issues

### Issue: GitHub API rate limit exceeded

**Symptoms:**

```
RateLimitError: GitHub API rate limit exceeded. Retry after: 1200s
```

**Solution:**

```typescript
// 1. Use authenticated requests (increases limit from 60 to 5000/hour)
const discovery = createDiscoveryEngine({
  github: {
    token: process.env.GITHUB_TOKEN, // Required!
  },
});

// 2. Implement exponential backoff
const discovery = createDiscoveryEngine({
  github: {
    token: process.env.GITHUB_TOKEN,
    rateLimit: {
      maxRequestsPerHour: 4000, // Leave buffer
      retryOnRateLimit: true,
      backoffMultiplier: 2,
    },
  },
});

// 3. Cache responses
const discovery = createDiscoveryEngine({
  github: { token: process.env.GITHUB_TOKEN },
  cache: {
    enabled: true,
    ttl: 3600000, // 1 hour
    storage: "redis", // or 'memory', 'filesystem'
  },
});
```

### Issue: Repository scan times out

**Symptoms:**

```
TimeoutError: Repository scan exceeded timeout of 30000ms
```

**Solution:**

```typescript
// Increase timeout
const discovery = createDiscoveryEngine({
  performance: {
    timeout: 120000, // 2 minutes
    maxConcurrency: 5,
  },
});

// Or scan incrementally
const result = await discovery.scanRepository("owner", "repo", {
  incremental: true, // Only scan changes since last scan
  depth: "shallow", // Don't recurse deeply
});
```

### Issue: Authentication failed

**Symptoms:**

```
AuthenticationError: Bad credentials
```

**Solution:**

```bash
# 1. Verify your token is valid
curl -H "Authorization: token YOUR_TOKEN" https://api.github.com/user

# 2. Check token permissions (needs repo, read:org scopes)

# 3. Regenerate token if expired
# Go to GitHub Settings > Developer Settings > Personal Access Tokens

# 4. Ensure token is properly set
export GITHUB_TOKEN="your-token-here"
echo $GITHUB_TOKEN # Verify it's set
```

### Issue: Memory exhaustion during large repository scan

**Symptoms:**

```
FATAL ERROR: CALL_AND_RETRY_LAST Allocation failed - JavaScript heap out of memory
```

**Solution:**

```bash
# Increase Node.js memory limit
NODE_OPTIONS="--max-old-space-size=8192" pnpm start

# Or in your script
export NODE_OPTIONS="--max-old-space-size=8192"
```

```typescript
// Use streaming mode for large repos
const discovery = createDiscoveryEngine({
  streaming: {
    enabled: true,
    chunkSize: 1000, // Process 1000 files at a time
  },
});

// Process results as they arrive
discovery.on("chunk", (chunk) => {
  processChunk(chunk);
  // Chunk is garbage collected after processing
});

await discovery.scanRepository("owner", "large-repo");
```

---

## Continuous Intelligence Issues

### Issue: Pattern detection not finding patterns

**Symptoms:**

- `getDetectedPatterns()` returns empty array
- No patterns despite many operations recorded

**Solution:**

```typescript
// 1. Ensure enough data is recorded
const engine = createSelfImprovementEngine({
  patternDetection: {
    minSamples: 50, // Reduce minimum if needed
    minConfidence: 0.5, // Lower confidence threshold
  },
});

// 2. Check operation recording
console.log("Operations recorded:", engine.getOperationCount());

// 3. Manually trigger pattern analysis
await engine.analyzePatterns({ force: true });

// 4. Enable debug logging
const engine = createSelfImprovementEngine({
  logging: { level: "debug" },
});
```

### Issue: Predictive maintenance false positives

**Symptoms:**

- Constant failure predictions that don't materialize
- Alert fatigue from too many warnings

**Solution:**

```typescript
// Adjust thresholds based on your system
const predictive = createPredictiveMaintenance({
  thresholds: {
    cpu: 90, // Increase from default 80
    memory: 95, // Increase from default 85
    errorRate: 0.1, // Increase from default 0.05
  },
  predictionHorizon: 7200000, // 2 hours instead of 1
  minConfidence: 0.8, // Only alert on high confidence
});

// Use rolling averages to smooth spikes
const predictive = createPredictiveMaintenance({
  smoothing: {
    enabled: true,
    windowSize: 60, // 60-sample rolling average
    outlierRemoval: true,
  },
});
```

### Issue: Auto-optimizer stuck in local minima

**Symptoms:**

- Optimization returns same results repeatedly
- Score not improving after many iterations

**Solution:**

```typescript
// 1. Try different optimization strategies
const optimizer = createAutoOptimizer({
  strategy: "simulated-annealing", // Better for escaping local minima
  temperature: 1.0,
  coolingRate: 0.95,
});

// 2. Use ensemble of optimizers
const optimizer = createAutoOptimizer({
  strategy: "ensemble",
  strategies: ["bayesian", "genetic", "simulated-annealing"],
  votingMethod: "best",
});

// 3. Increase exploration
const optimizer = createAutoOptimizer({
  strategy: "genetic",
  mutationRate: 0.3, // Higher mutation for more exploration
  populationSize: 100,
  elitism: 0.1,
});

// 4. Add random restarts
const optimizer = createAutoOptimizer({
  strategy: "bayesian",
  randomRestarts: 5,
  acquisitionFunction: "ei-with-exploration",
});
```

### Issue: Models not persisting across restarts

**Symptoms:**

- Learning lost after application restart
- Pattern detection starts from scratch each time

**Solution:**

```typescript
// Enable persistence
const engine = createSelfImprovementEngine({
  persistence: {
    enabled: true,
    storage: "filesystem", // or 'redis', 'postgres'
    path: "./data/models",
    autoSave: true,
    saveInterval: 300000, // 5 minutes
  },
});

// Or save/load manually
await engine.saveState("./data/engine-state.json");
// Later...
await engine.loadState("./data/engine-state.json");
```

---

## 3D Visualization Issues

### Issue: WebGL not available

**Symptoms:**

```
Error: WebGL is not supported in this browser
```

**Solution:**

```typescript
// 1. Check WebGL availability
import { checkWebGLSupport } from "@neurectomy/3d-engine";

const support = checkWebGLSupport();
if (!support.webgl2) {
  if (support.webgl1) {
    // Fall back to WebGL 1
    const engine = create3DEngine({ renderer: "webgl1" });
  } else {
    // Fall back to 2D visualization
    const engine = create2DEngine();
  }
}

// 2. Enable hardware acceleration in browser
// Chrome: chrome://flags/#ignore-gpu-blocklist
// Firefox: about:config -> webgl.force-enabled = true
```

### Issue: Performance degradation with large graphs

**Symptoms:**

- FPS drops below 30
- UI becomes unresponsive
- Browser crashes

**Solution:**

```typescript
// 1. Enable level-of-detail rendering
const engine = create3DEngine({
  optimization: {
    levelOfDetail: true,
    lodDistances: [100, 500, 1000],
    frustumCulling: true,
    occlusionCulling: true,
  },
});

// 2. Use clustering for large datasets
const engine = create3DEngine({
  clustering: {
    enabled: true,
    threshold: 1000, // Cluster when > 1000 nodes
    algorithm: "k-means",
    expandOnClick: true,
  },
});

// 3. Enable progressive loading
await engine.loadGraph(graphData, {
  progressive: true,
  batchSize: 500,
  renderDuringLoad: true,
});

// 4. Reduce visual quality
const engine = create3DEngine({
  quality: "low",
  shadows: false,
  antialiasing: false,
  particles: false,
});
```

### Issue: Graph layout is chaotic

**Symptoms:**

- Nodes overlap excessively
- Hard to see relationships
- Layout keeps changing

**Solution:**

```typescript
// 1. Use appropriate layout algorithm
const engine = create3DEngine({
  layout: {
    algorithm: "force-directed", // Best for general graphs
    // Or 'hierarchical' for trees/DAGs
    // Or 'radial' for star patterns
    iterations: 500, // More iterations = better layout
    cooling: 0.99,
  },
});

// 2. Pin important nodes
engine.pinNode("root", { x: 0, y: 0, z: 0 });

// 3. Use constraints
engine.addConstraint({
  type: "alignment",
  axis: "y",
  nodes: ["a", "b", "c"],
});

// 4. Manual layout for critical sections
engine.setNodePosition("important-node", { x: 0, y: 100, z: 0 });
```

---

## Container Command Issues

### Issue: Docker socket connection refused

**Symptoms:**

```
Error: connect ENOENT /var/run/docker.sock
```

**Solution:**

```bash
# 1. Check if Docker is running
docker ps

# 2. Start Docker daemon
sudo systemctl start docker

# 3. Check socket permissions
ls -la /var/run/docker.sock
# Should be owned by docker group

# 4. Add user to docker group
sudo usermod -aG docker $USER
# Then log out and back in

# 5. For rootless Docker
export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock
```

```typescript
// Configure correct socket path
const docker = createContainerCommand({
  type: "docker",
  socketPath: process.env.DOCKER_HOST || "/var/run/docker.sock",
});
```

### Issue: Kubernetes context not found

**Symptoms:**

```
Error: Context "my-cluster" not found in kubeconfig
```

**Solution:**

```bash
# 1. List available contexts
kubectl config get-contexts

# 2. Use the correct context
kubectl config use-context correct-context-name

# 3. Verify kubeconfig path
echo $KUBECONFIG
# Or check default location
cat ~/.kube/config
```

```typescript
// Specify correct kubeconfig and context
const k8s = createContainerCommand({
  type: "kubernetes",
  kubeconfig: process.env.KUBECONFIG || "~/.kube/config",
  context: "my-correct-context",
});

// Or let it use current context
const k8s = createContainerCommand({
  type: "kubernetes",
  useCurrentContext: true,
});
```

### Issue: Container exec hangs

**Symptoms:**

- `exec()` call never returns
- Terminal output not captured

**Solution:**

```typescript
// 1. Add timeout
const result = await docker.exec(containerId, ["ls", "-la"], {
  timeout: 30000, // 30 seconds
  abortOnTimeout: true,
});

// 2. Use non-interactive mode
const result = await docker.exec(containerId, ["ls", "-la"], {
  interactive: false,
  tty: false,
});

// 3. Check container state first
const container = await docker.inspect(containerId);
if (container.State.Status !== "running") {
  throw new Error("Container not running");
}
```

---

## Performance Issues

### Issue: Slow startup time

**Symptoms:**

- Application takes > 10 seconds to start
- Long delay before first response

**Solution:**

```typescript
// 1. Use lazy loading
const discovery = createDiscoveryEngine({
  lazyInit: true, // Don't connect to GitHub until needed
});

// 2. Defer non-critical initialization
async function start() {
  // Start critical services first
  await startCriticalServices();

  // Defer secondary services
  setImmediate(async () => {
    await startSecondaryServices();
  });
}

// 3. Use caching for configuration
const discovery = createDiscoveryEngine({
  cache: {
    configCache: true,
    warmUp: true,
  },
});
```

### Issue: High memory usage

**Symptoms:**

- Memory grows continuously
- Eventually crashes with OOM

**Solution:**

```typescript
// 1. Enable garbage collection hints
const engine = createContinuousIntelligence({
  memory: {
    maxOperationHistory: 10000, // Limit stored operations
    pruneInterval: 60000, // Prune every minute
    maxMemoryMB: 1024, // Set memory budget
  },
});

// 2. Use streaming instead of batch
discovery.on("file", (file) => {
  processFile(file);
  // File can be garbage collected immediately
});

// 3. Clear caches periodically
setInterval(() => {
  engine.clearCache();
  global.gc?.(); // Force GC if available
}, 300000);

// 4. Monitor memory usage
engine.on("memory:warning", (usage) => {
  console.warn(`Memory usage: ${usage.heapUsed}MB / ${usage.heapTotal}MB`);
});
```

### Issue: CPU spikes during analysis

**Symptoms:**

- CPU hits 100% during operations
- System becomes unresponsive

**Solution:**

```typescript
// 1. Use worker threads for heavy computation
const engine = createContinuousIntelligence({
  workers: {
    enabled: true,
    count: 4, // Number of worker threads
    maxLoad: 0.8, // Leave 20% CPU headroom
  },
});

// 2. Add delays between operations
const engine = createDiscoveryEngine({
  throttle: {
    operationsPerSecond: 10,
    burstLimit: 20,
  },
});

// 3. Use priority queuing
engine.queueOperation({
  type: "scan",
  priority: "low", // Won't block high-priority operations
  data: largeRepo,
});
```

---

## Error Reference

### Common Error Codes

| Error Code       | Description             | Solution                                |
| ---------------- | ----------------------- | --------------------------------------- |
| `ENOENT`         | File or path not found  | Check file paths and permissions        |
| `ECONNREFUSED`   | Connection refused      | Check if service is running             |
| `ETIMEDOUT`      | Operation timed out     | Increase timeout or check network       |
| `ENOMEM`         | Out of memory           | Increase memory limit or optimize usage |
| `EACCES`         | Permission denied       | Check file/socket permissions           |
| `RATE_LIMIT`     | API rate limit exceeded | Add authentication or implement backoff |
| `AUTH_FAILED`    | Authentication failed   | Check credentials                       |
| `CONFIG_INVALID` | Invalid configuration   | Validate config against schema          |

### Error Classes

```typescript
import {
  NeurectomyError,
  ConfigurationError,
  AuthenticationError,
  RateLimitError,
  TimeoutError,
  ValidationError,
  ResourceNotFoundError,
  ConnectionError,
} from "@neurectomy/core";

try {
  await engine.someOperation();
} catch (error) {
  if (error instanceof RateLimitError) {
    // Wait and retry
    await sleep(error.retryAfter);
  } else if (error instanceof AuthenticationError) {
    // Re-authenticate
    await refreshCredentials();
  } else if (error instanceof ConfigurationError) {
    // Log config issue
    console.error("Config issue:", error.configPath, error.expected);
  } else if (error instanceof NeurectomyError) {
    // General NEURECTOMY error
    console.error("Error:", error.code, error.message);
  } else {
    // Unknown error
    throw error;
  }
}
```

---

## Getting Help

### Debug Mode

```bash
# Enable debug logging
DEBUG=neurectomy:* pnpm start

# Or for specific packages
DEBUG=neurectomy:discovery,neurectomy:intelligence pnpm start
```

### Collecting Diagnostic Information

```typescript
import { collectDiagnostics } from "@neurectomy/core";

const diagnostics = await collectDiagnostics();
// Save to file for support
await fs.writeFile("diagnostics.json", JSON.stringify(diagnostics, null, 2));
```

### Community Resources

- **GitHub Issues**: [github.com/your-org/neurectomy/issues](https://github.com/your-org/neurectomy/issues)
- **Discord**: [discord.gg/neurectomy](https://discord.gg/neurectomy)
- **Documentation**: [neurectomy.dev/docs](https://neurectomy.dev/docs)

---

_Last updated: 2025_
