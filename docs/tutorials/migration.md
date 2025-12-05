# NEURECTOMY Migration Guide

> Upgrading between NEURECTOMY versions

## Table of Contents

1. [Version Compatibility](#version-compatibility)
2. [Migration Overview](#migration-overview)
3. [Migrating to v2.0](#migrating-to-v20)
4. [Migrating to v1.5](#migrating-to-v15)
5. [Migrating to v1.0](#migrating-to-v10)
6. [Breaking Changes Reference](#breaking-changes-reference)
7. [Migration Tools](#migration-tools)
8. [Rollback Procedures](#rollback-procedures)

---

## Version Compatibility

### Compatibility Matrix

| NEURECTOMY Version | Node.js | pnpm | TypeScript | Breaking Changes  |
| ------------------ | ------- | ---- | ---------- | ----------------- |
| 2.0.x              | 20+     | 9+   | 5.3+       | Major API changes |
| 1.5.x              | 18+     | 8+   | 5.0+       | Minor API changes |
| 1.0.x              | 18+     | 8+   | 4.9+       | Initial release   |

### Package Compatibility

| Package                             | v1.0 | v1.5 | v2.0             |
| ----------------------------------- | ---- | ---- | ---------------- |
| @neurectomy/core                    | ✅   | ✅   | ✅               |
| @neurectomy/discovery-engine        | ✅   | ✅   | ✅ (API changes) |
| @neurectomy/continuous-intelligence | ❌   | ✅   | ✅ (API changes) |
| @neurectomy/3d-engine               | ✅   | ✅   | ✅               |
| @neurectomy/container-command       | ✅   | ✅   | ✅               |
| @neurectomy/enterprise              | ❌   | ❌   | ✅ (new)         |

---

## Migration Overview

### Pre-Migration Checklist

Before migrating, ensure you have:

- [ ] Backed up your configuration files
- [ ] Backed up any persisted data (models, patterns, etc.)
- [ ] Reviewed the changelog for your target version
- [ ] Tested the migration in a development environment
- [ ] Verified all dependencies are compatible
- [ ] Updated your CI/CD pipelines if necessary

### General Migration Steps

```bash
# 1. Create backup
cp -r ./config ./config.backup
cp -r ./data ./data.backup

# 2. Update packages
pnpm update @neurectomy/core@^2.0.0
pnpm update @neurectomy/discovery-engine@^2.0.0
# ... update all packages

# 3. Run migration script (if available)
pnpm neurectomy migrate --from 1.5 --to 2.0

# 4. Update configuration
pnpm neurectomy config upgrade

# 5. Rebuild
pnpm clean
pnpm install
pnpm build

# 6. Test
pnpm test
pnpm test:e2e
```

---

## Migrating to v2.0

### Overview

Version 2.0 introduces significant improvements including:

- New unified API for Continuous Intelligence
- Improved TypeScript types with strict mode
- Enhanced performance through streaming APIs
- New governance engine package

### Step 1: Update Package Versions

```json
// package.json
{
  "dependencies": {
    "@neurectomy/core": "^2.0.0",
    "@neurectomy/discovery-engine": "^2.0.0",
    "@neurectomy/continuous-intelligence": "^2.0.0",
    "@neurectomy/3d-engine": "^2.0.0",
    "@neurectomy/container-command": "^2.0.0",
    "@neurectomy/types": "^2.0.0"
  }
}
```

### Step 2: Update Configuration Files

**Before (v1.5):**

```typescript
// config.ts
export const config = {
  discovery: {
    githubToken: process.env.GITHUB_TOKEN,
    maxRequests: 5000,
  },
  intelligence: {
    patternDetection: true,
    autoOptimize: true,
  },
};
```

**After (v2.0):**

```typescript
// config.ts
import { defineConfig } from "@neurectomy/core";

export const config = defineConfig({
  discovery: {
    github: {
      token: process.env.GITHUB_TOKEN!,
      rateLimit: {
        maxRequestsPerHour: 5000,
        retryOnRateLimit: true,
      },
    },
    analysis: {
      maxDepth: 10,
      excludePatterns: ["node_modules", ".git"],
    },
  },
  intelligence: {
    selfImprovement: {
      enabled: true,
      patternDetection: {
        enabled: true,
        minConfidence: 0.7,
      },
    },
    autoOptimization: {
      enabled: true,
      strategy: "bayesian",
    },
  },
});
```

### Step 3: Update API Usage

**Discovery Engine Changes:**

```typescript
// Before (v1.5)
import { DiscoveryEngine } from "@neurectomy/discovery-engine";

const discovery = new DiscoveryEngine(config);
await discovery.init();
const result = discovery.scan("owner", "repo");

// After (v2.0)
import { createDiscoveryEngine } from "@neurectomy/discovery-engine";

const discovery = createDiscoveryEngine(config);
await discovery.start();
const result = await discovery.scanRepository("owner", "repo");
```

**Continuous Intelligence Changes:**

```typescript
// Before (v1.5)
import { PatternEngine, Optimizer } from "@neurectomy/continuous-intelligence";

const patterns = new PatternEngine(config);
const optimizer = new Optimizer(config);

patterns.record(operation);
const detected = patterns.detect();
const optimized = optimizer.optimize(params);

// After (v2.0)
import { createContinuousIntelligence } from "@neurectomy/continuous-intelligence";

const ci = createContinuousIntelligence(config);
await ci.start();

ci.recordOperation(operation);
const detected = ci.getDetectedPatterns();
const optimized = await ci.optimize(params);
```

### Step 4: Update Event Handlers

```typescript
// Before (v1.5)
discovery.on('scan-complete', (data) => { ... });
discovery.on('scan-error', (err) => { ... });

// After (v2.0)
discovery.on('analysis:complete', (data) => { ... });
discovery.on('error', (err) => { ... });
```

### Step 5: Update Type Imports

```typescript
// Before (v1.5)
import { ScanResult, AnalysisConfig } from "@neurectomy/discovery-engine";
import {
  Pattern,
  OptimizationResult,
} from "@neurectomy/continuous-intelligence";

// After (v2.0)
import type {
  RepositoryScanResult,
  DiscoveryEngineConfig,
} from "@neurectomy/discovery-engine";
import type {
  DetectedPattern,
  OptimizationResult,
  ContinuousIntelligenceConfig,
} from "@neurectomy/continuous-intelligence";
```

### Step 6: Migrate Persisted Data

```typescript
import { migrateData } from "@neurectomy/core/migration";

// Migrate pattern data
await migrateData({
  type: "patterns",
  from: "./data/patterns-v1.5.json",
  to: "./data/patterns-v2.0.json",
  fromVersion: "1.5",
  toVersion: "2.0",
});

// Migrate model data
await migrateData({
  type: "models",
  from: "./data/models/",
  to: "./data/models/",
  fromVersion: "1.5",
  toVersion: "2.0",
});
```

---

## Migrating to v1.5

### Overview

Version 1.5 added:

- Continuous Intelligence package
- Improved caching
- Better error handling

### Key Changes

**New Package Addition:**

```bash
pnpm add @neurectomy/continuous-intelligence@^1.5.0
```

**Configuration Updates:**

```typescript
// Add intelligence configuration
export const config = {
  // Existing config...
  intelligence: {
    patternDetection: true,
    predictionHorizon: 3600000,
    optimizationStrategy: "bayesian",
  },
};
```

**New Event Types:**

```typescript
// New events added in 1.5
engine.on('pattern:detected', (pattern) => { ... });
engine.on('prediction:generated', (prediction) => { ... });
engine.on('optimization:complete', (result) => { ... });
```

---

## Migrating to v1.0

### From Pre-Release

If migrating from pre-release versions:

```bash
# Remove old packages
pnpm remove neurectomy-alpha neurectomy-beta

# Install v1.0
pnpm add @neurectomy/core@^1.0.0
pnpm add @neurectomy/discovery-engine@^1.0.0
pnpm add @neurectomy/3d-engine@^1.0.0
```

### Fresh Installation

For new projects, start with:

```bash
# Create new project
mkdir my-project
cd my-project

# Initialize
pnpm init

# Install NEURECTOMY
pnpm add @neurectomy/core @neurectomy/discovery-engine

# Copy example configuration
cp node_modules/@neurectomy/core/examples/config.example.ts ./config.ts
```

---

## Breaking Changes Reference

### v2.0 Breaking Changes

| Change               | Before               | After                  | Migration            |
| -------------------- | -------------------- | ---------------------- | -------------------- |
| Factory functions    | `new Engine(config)` | `createEngine(config)` | Replace constructors |
| Method names         | `scan()`             | `scanRepository()`     | Rename calls         |
| Event names          | `scan-complete`      | `analysis:complete`    | Update listeners     |
| Config structure     | Flat                 | Nested                 | Restructure config   |
| Async initialization | `init()`             | `start()`              | Replace method       |
| Pattern types        | String               | Enum                   | Use new types        |

### v1.5 Breaking Changes

| Change       | Before          | After                      | Migration           |
| ------------ | --------------- | -------------------------- | ------------------- |
| Error types  | Generic `Error` | Specific classes           | Update catch blocks |
| Cache config | `cache: true`   | `cache: { enabled: true }` | Update config       |

---

## Migration Tools

### Automatic Migration

NEURECTOMY provides migration tools to help automate updates:

```bash
# Run the migration assistant
pnpm neurectomy migrate --interactive

# Or specify versions directly
pnpm neurectomy migrate --from 1.5.0 --to 2.0.0

# Dry run (show changes without applying)
pnpm neurectomy migrate --from 1.5.0 --to 2.0.0 --dry-run
```

### Code Codemods

```bash
# Run codemods to update your code
pnpm neurectomy codemod --transform factory-functions ./src
pnpm neurectomy codemod --transform event-names ./src
pnpm neurectomy codemod --transform async-methods ./src

# Run all codemods
pnpm neurectomy codemod --transform all ./src
```

### Configuration Upgrader

```bash
# Upgrade configuration file
pnpm neurectomy config upgrade ./config.ts

# Validate configuration against schema
pnpm neurectomy config validate ./config.ts
```

### Type Checker

```bash
# Check for deprecated type usage
pnpm neurectomy types check ./src

# Generate type migration report
pnpm neurectomy types report ./src > migration-report.md
```

---

## Rollback Procedures

### Quick Rollback

If you need to rollback after a failed migration:

```bash
# 1. Restore package versions
git checkout package.json pnpm-lock.yaml
pnpm install

# 2. Restore configuration
cp ./config.backup/* ./config/

# 3. Restore data
cp -r ./data.backup/* ./data/

# 4. Rebuild
pnpm build
```

### Gradual Rollback

For production systems, use gradual rollback:

```typescript
import { createDiscoveryEngine } from "@neurectomy/discovery-engine";
import { DiscoveryEngine as LegacyEngine } from "@neurectomy/discovery-engine/legacy";

// Use feature flag for gradual rollout
const useNewVersion = getFeatureFlag("neurectomy-v2");

const engine = useNewVersion
  ? createDiscoveryEngine(v2Config)
  : new LegacyEngine(v1Config);
```

### Compatibility Layer

For complex migrations, use the compatibility layer:

```typescript
import { createCompatibilityLayer } from "@neurectomy/core/compat";

// Wraps v2 API with v1 interface
const compat = createCompatibilityLayer({
  version: "1.5",
  engine: createDiscoveryEngine(v2Config),
});

// Use v1 API (will be translated to v2 calls)
compat.scan("owner", "repo"); // Calls scanRepository internally
```

---

## Best Practices

### 1. Test in Isolation

```bash
# Create a test branch
git checkout -b test/migration-v2

# Perform migration
pnpm neurectomy migrate --from 1.5 --to 2.0

# Run full test suite
pnpm test
pnpm test:e2e
pnpm test:performance
```

### 2. Monitor After Migration

```typescript
// Add extra logging during migration period
const engine = createDiscoveryEngine({
  ...config,
  logging: {
    level: "debug",
    includeMetrics: true,
  },
});

// Monitor for issues
engine.on("warning", (warning) => {
  alertTeam(warning);
});
```

### 3. Keep Old Data

```bash
# Archive old data format
tar -czvf data-v1.5-backup.tar.gz ./data

# Store version info with data
echo "1.5.0" > ./data/VERSION
```

### 4. Update Documentation

After migration, update your internal documentation:

- API usage examples
- Configuration templates
- Deployment scripts
- Monitoring dashboards

---

## Getting Help

### Migration Support

If you encounter issues during migration:

1. Check the [Troubleshooting Guide](./troubleshooting.md)
2. Search [GitHub Issues](https://github.com/your-org/neurectomy/issues)
3. Ask in [Discord #migration](https://discord.gg/neurectomy)

### Reporting Migration Bugs

When reporting migration issues, include:

```bash
# Generate migration debug report
pnpm neurectomy migrate --debug-report > migration-debug.txt
```

Include:

- Source version
- Target version
- Error messages
- Configuration (sanitized)
- Steps to reproduce

---

_Last updated: 2025_
