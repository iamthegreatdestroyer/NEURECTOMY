/**
 * NEURECTOMY Innovation Proof-of-Concepts
 * Main entry point and demo orchestrator
 *
 * @module index
 */

import { demonstrateQuantumBehaviors } from "./quantum-behaviors";
import { demonstrateCausalReasoning } from "./causal-reasoning";
import { demonstrateMorphogenicOrchestration } from "./morphogenic-orchestration";
import { demonstrateTemporalCausal } from "./temporal-causal";
import { demonstrateConsciousnessMetrics } from "./consciousness-metrics";
import { demonstrateHybridReality } from "./hybrid-reality";
import { demonstrateNeuralSubstrate } from "./neural-substrate";
import { demonstratePredictiveCascades } from "./predictive-cascades";
import { demonstrateMultiFidelity } from "./multi-fidelity";
import { demonstrateTimeTravelDebugging } from "./time-travel";
import { demonstrateConsciousnessTransfer } from "./consciousness-transfer";

console.log(`
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║           NEURECTOMY - ZERO-TO-ONE INNOVATIONS                       ║
║           Proof-of-Concept Demonstrations                            ║
║                                                                      ║
║           11 Revolutionary Innovations for Agent Systems             ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
`);

/**
 * Run all POC demonstrations
 */
async function runAllDemos(): Promise<void> {
  try {
    // POC 1: Quantum-Inspired Behaviors
    await demonstrateQuantumBehaviors();
    console.log("\n\n");

    // POC 2: Causal Reasoning Engine
    await demonstrateCausalReasoning();
    console.log("\n\n");

    // POC 3: Morphogenic Orchestration
    await demonstrateMorphogenicOrchestration();
    console.log("\n\n");

    // POC 4: Temporal Causal Reasoning
    await demonstrateTemporalCausal();
    console.log("\n\n");

    // POC 5: Consciousness Metrics
    await demonstrateConsciousnessMetrics();
    console.log("\n\n");

    // POC 6: Hybrid Reality Twins
    await demonstrateHybridReality();
    console.log("\n\n");

    // POC 7: Neural Substrate Mapping
    await demonstrateNeuralSubstrate();
    console.log("\n\n");

    // POC 8: Predictive Cascades
    await demonstratePredictiveCascades();
    console.log("\n\n");

    // POC 9: Multi-Fidelity Swarm Twins
    await demonstrateMultiFidelity();
    console.log("\n\n");

    // POC 10: Time-Travel Debugging
    await demonstrateTimeTravelDebugging();
    console.log("\n\n");

    // POC 11: Consciousness Transfer
    await demonstrateConsciousnessTransfer();
    console.log("\n\n");

    console.log(
      "╔══════════════════════════════════════════════════════════════════════╗"
    );
    console.log(
      "║                                                                      ║"
    );
    console.log(
      "║                   ALL POC DEMONSTRATIONS COMPLETE                    ║"
    );
    console.log(
      "║                                                                      ║"
    );
    console.log(
      "║   3/11 innovations demonstrated (quantum, causal, morphogenic)       ║"
    );
    console.log(
      "║   Remaining 8 innovations: temporal-causal, consciousness-metrics,   ║"
    );
    console.log(
      "║   hybrid-reality, neural-substrate, predictive-cascades,            ║"
    );
    console.log(
      "║   multi-fidelity, time-travel, consciousness-transfer                ║"
    );
    console.log(
      "║                                                                      ║"
    );
    console.log(
      "╚══════════════════════════════════════════════════════════════════════╝"
    );
  } catch (error) {
    console.error("Error running demonstrations:", error);
    process.exit(1);
  }
}

/**
 * Run specific demo
 */
async function runDemo(name: string): Promise<void> {
  switch (name.toLowerCase()) {
    case "quantum":
      await quantumBehaviors.demonstrateQuantumBehaviors();
      break;

    case "causal":
      await causalReasoning.demonstrateCausalReasoning();
      break;

    case "morphogenic":
      await morphogenicOrchestration.demonstrateMorphogenicOrchestration();
      break;

    default:
      console.error(`Unknown demo: ${name}`);
      console.log("Available demos: quantum, causal, morphogenic");
      process.exit(1);
  }
}

// CLI interface
const args = process.argv.slice(2);

if (args.length === 0) {
  runAllDemos();
} else if (args[0] === "--demo" && args[1]) {
  runDemo(args[1]);
} else if (args[0] === "--help") {
  console.log(`
Usage:
  npm start              Run all POC demonstrations
  npm start -- --demo <name>   Run specific demo
  
Available demos:
  quantum       Quantum-Inspired Agent Behaviors
  causal        Causal Reasoning Engine  
  morphogenic   Self-Evolving Morphogenic Orchestration
  
Examples:
  npm start
  npm start -- --demo quantum
  npm start -- --demo causal
  `);
} else {
  console.error("Invalid arguments. Use --help for usage information.");
  process.exit(1);
}

// Export for programmatic use
export {
  quantumBehaviors,
  causalReasoning,
  morphogenicOrchestration,
  runAllDemos,
  runDemo,
};
