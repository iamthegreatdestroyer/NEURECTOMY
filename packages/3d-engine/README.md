# @neurectomy/3d-engine

**Status**: ✅ Phase 1 Complete  
**Version**: 1.0.0-alpha  
**Last Updated**: January 2025

## Overview

The 3D Engine package provides a comprehensive cross-domain innovation system that bridges Forge (3D visualization), Twin (digital twin), and Foundry (ML training) domains. It enables real-time collaboration, evolutionary architecture exploration, and intelligent training workflows.

## What's Included

### 🚀 P0 Breakthrough Innovations (4)
- **Quantum Architecture Search**: Superposition-based architecture exploration
- **Living Architecture Laboratory**: Self-evolving neural architectures
- **Morphogenic Model Evolution**: Biological growth patterns for networks
- **Causal Training Debugger**: Counterfactual analysis and what-if scenarios

### 🔥 Forge × Twin Innovations (3)
- **Predictive Visualization Cascade**: Real-time insight generation
- **Interactive Training Theater**: Live visualization during training
- **Consciousness-Driven Heatmap**: Attention-based visual feedback

### 🌊 Twin × Foundry Innovations (3)
- **Twin-Guided Architecture Search**: Visual feedback for discovery
- **Model-in-the-Loop Sync**: Real-time coordination
- **Cascade Training Pipeline**: Automated multi-stage workflows

### ⚡ Forge × Foundry Innovations (3)
- **3D Neural Playground**: Interactive network manipulation
- **Training 4D Journey**: Temporal training visualization
- **Model Router Cosmos**: Intelligent model routing

### 🏗️ Infrastructure (2)
- **Cross-Domain Event Bridge**: Real-time event propagation
- **Cross-Domain Orchestrator**: Multi-domain workflow coordination

## Quick Start

```typescript
import {
  // Infrastructure
  CrossDomainEventBridge,
  CrossDomainOrchestrator,
  
  // P0 Innovations
  QuantumArchitectureSearch,
  LivingArchitectureLaboratory,
  
  // Domain Combinations
  PredictiveVisualizationCascade,
  TwinGuidedArchitectureSearch,
  Neural3DPlayground,
  
  // Factory functions
  createQuantumSearch,
  // ... more
} from '@neurectomy/3d-engine';

// Get infrastructure instances
const eventBridge = CrossDomainEventBridge.getInstance();
const orchestrator = CrossDomainOrchestrator.getInstance();

// Create an innovation
const search = new QuantumArchitectureSearch({
  searchSpace: myArchitectures,
  evaluator: myEvaluator,
  superpositionSize: 10
});

// Use it
const candidates = await search.exploreArchitectureSpace();
const ranked = search.rankCandidates();
const best = search.collapseToArchitecture(ranked[0].id);
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Cross-Domain System                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐         ┌──────────┐        ┌──────────┐    │
│  │  FORGE   │ ◄────► │   TWIN   │ ◄────► │ FOUNDRY  │    │
│  │          │         │          │        │          │    │
│  │ 3D Viz   │         │ Digital  │        │   ML     │    │
│  │ Engine   │         │  Twin    │        │ Training │    │
│  └──────────┘         └──────────┘        └──────────┘    │
│       │                    │                     │         │
│       └────────────────────┼─────────────────────┘         │
│                            │                               │
│                   ┌────────▼────────┐                      │
│                   │  Event Bridge   │                      │
│                   │  Orchestrator   │                      │
│                   └─────────────────┘                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Event-Driven Communication

All domains communicate through the event bridge:

```typescript
// Subscribe to events
eventBridge.subscribe<TrainingProgressEvent>(
  'training:progress',
  (event) => {
    // Update visualization
    updateForgeVisualization(event.payload);
  }
);

// Publish events
eventBridge.publish({
  type: 'component:updated',
  domain: 'forge',
  payload: { changes }
});
```

### Orchestrated Workflows

The orchestrator coordinates complex multi-domain workflows:

```typescript
const orchestrator = CrossDomainOrchestrator.getInstance();

// Automatic cross-domain synchronization
orchestrator.startAutoSync();

// Events flow automatically between domains
```

## Features

### ✨ Real-Time Collaboration
- Events propagate across domains instantly
- Automatic synchronization
- Conflict resolution

### 🧬 Evolutionary Architecture
- Quantum-inspired exploration
- Biological growth patterns
- Self-optimization

### 🎨 Interactive Visualization
- 3D network manipulation
- Real-time training feedback
- Temporal journey views

### 🔍 Causal Analysis
- Counterfactual debugging
- What-if scenarios
- Training intervention simulation

### 🚀 Production Ready
- Full TypeScript support
- Comprehensive testing
- Clean, maintainable code

## Testing

```bash
# Run all tests
pnpm test

# Run integration tests
pnpm vitest run src/cross-domain/innovations/__tests__/integration.test.ts

# Run with coverage
pnpm test -- --coverage
```

### Test Status
- **Total Tests**: 50 integration tests
- **Coverage**: All 15 innovations + infrastructure
- **Framework**: Vitest with async support

## Project Status

**Phase 1**: ✅ COMPLETE (97%)
- All 15 innovations implemented
- Event-driven infrastructure operational
- Comprehensive test suite created
- Full TypeScript compilation
- Production-ready core

See [PHASE1_COMPLETION_REPORT.md](./PHASE1_COMPLETION_REPORT.md) for detailed status.

## Development

### Build
```bash
pnpm build
```

### Lint
```bash
pnpm lint
```

### Type Check
```bash
pnpm type-check
```

## File Structure

```
src/cross-domain/
├── event-bridge.ts              # Event infrastructure
├── orchestrator.ts              # Workflow coordination
└── innovations/
    ├── index.ts                 # Central exports
    ├── breakthroughs/           # P0 innovations
    │   ├── quantum-architecture-search.ts
    │   ├── living-architecture-laboratory.ts
    │   ├── morphogenic-model-evolution.ts
    │   └── causal-training-debugger.ts
    ├── forge-twin/              # Forge × Twin
    │   ├── predictive-visualization-cascade.ts
    │   ├── interactive-training-theater.ts
    │   └── consciousness-driven-heatmap.ts
    ├── twin-foundry/            # Twin × Foundry
    │   ├── twin-guided-architecture-search.ts
    │   ├── model-in-loop-sync.ts
    │   └── cascade-training-pipeline.ts
    ├── forge-foundry/           # Forge × Foundry
    │   ├── 3d-neural-playground.ts
    │   ├── training-4d-journey.ts
    │   └── model-router-cosmos.ts
    └── __tests__/
        └── integration.test.ts  # 50 integration tests
```

## API Documentation

### Infrastructure Classes

#### CrossDomainEventBridge
Manages event propagation across domains.

```typescript
class CrossDomainEventBridge {
  static getInstance(): CrossDomainEventBridge;
  
  publish<T extends CrossDomainEvent>(event: T): void;
  
  subscribe<T extends CrossDomainEvent>(
    eventType: string,
    handler: (event: T) => void
  ): string;
  
  unsubscribe(subscriptionId: string): void;
}
```

#### CrossDomainOrchestrator
Coordinates multi-domain workflows.

```typescript
class CrossDomainOrchestrator {
  static getInstance(): CrossDomainOrchestrator;
  
  startAutoSync(): void;
  stopAutoSync(): void;
  getMetrics(): OrchestratorMetrics;
}
```

### Innovation Classes

All innovation classes follow consistent patterns:

```typescript
// Constructor with configuration
new InnovationClass(config: InnovationConfig)

// Core methods
async start(): Promise<void>
async stop(): Promise<void>
getState(): State

// Specific functionality methods vary by innovation
```

See individual innovation files for detailed API documentation.

## Contributing

This package is part of the Neurectomy platform. For contribution guidelines, see the main repository README.

## License

Proprietary - All Rights Reserved

## Support

For issues, questions, or feature requests, please refer to the main Neurectomy repository.

---

**Built with ❤️ for the future of AI development**
