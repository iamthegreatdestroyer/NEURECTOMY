/**
 * NEURECTOMY Chaos Fault Types
 * @module @neurectomy/experimentation-engine/chaos
 * @agent @ECLIPSE @CORE
 *
 * Advanced fault injection types including network partitions,
 * resource exhaustion, and custom failure scenarios.
 */

import { v4 as uuidv4 } from "uuid";
import { z } from "zod";
import type {
  FaultInjector,
  FaultConfig,
  AffectedTarget,
  FaultType,
} from "./simulator";

// ============================================================================
// Network Fault Schemas
// ============================================================================

export const NetworkPartitionConfigSchema = z.object({
  partitionType: z.enum(["full", "partial", "one_way"]),
  sourceSelector: z.record(z.string()),
  destinationSelector: z.record(z.string()),
  duration: z.number().positive(),
});

export const PacketLossConfigSchema = z.object({
  percentage: z.number().min(0).max(100),
  correlation: z.number().min(0).max(100).default(0),
  distribution: z.enum(["uniform", "normal", "pareto"]).default("uniform"),
});

export const BandwidthLimitConfigSchema = z.object({
  rate: z.string(), // e.g., "1mbit", "100kbit"
  burst: z.string().optional(),
  latency: z.number().optional(), // additional latency in ms
});

export const DNSFailureConfigSchema = z.object({
  failureType: z.enum(["nxdomain", "timeout", "servfail", "refuse"]),
  domains: z.array(z.string()).optional(), // specific domains to fail, or all if empty
  duration: z.number().positive(),
});

// ============================================================================
// Resource Fault Schemas
// ============================================================================

export const CPUStressConfigSchema = z.object({
  workers: z.number().int().positive().default(1),
  load: z.number().min(0).max(100).default(80),
  duration: z.number().positive(),
  method: z.enum(["cpu", "cpu-cycles", "matrix"]).default("cpu"),
});

export const MemoryStressConfigSchema = z.object({
  bytes: z.string(), // e.g., "256M", "1G"
  percentage: z.number().min(0).max(100).optional(),
  workers: z.number().int().positive().default(1),
  duration: z.number().positive(),
  oomScoreAdj: z.number().int().min(-1000).max(1000).optional(),
});

export const DiskStressConfigSchema = z.object({
  workers: z.number().int().positive().default(1),
  size: z.string(), // e.g., "1G"
  path: z.string().default("/tmp"),
  method: z.enum(["write", "read", "mixed"]).default("write"),
  duration: z.number().positive(),
});

export const IOStressConfigSchema = z.object({
  workers: z.number().int().positive().default(1),
  mode: z.enum(["read", "write", "mixed"]).default("mixed"),
  size: z.string(),
  rate: z.string().optional(), // e.g., "10M/s"
  duration: z.number().positive(),
});

// ============================================================================
// Process/Container Fault Schemas
// ============================================================================

export const ProcessKillConfigSchema = z.object({
  signal: z
    .enum(["SIGTERM", "SIGKILL", "SIGSTOP", "SIGCONT"])
    .default("SIGKILL"),
  processName: z.string().optional(),
  pid: z.number().int().positive().optional(),
  interval: z.number().positive().optional(), // repeat kill every N ms
  count: z.number().int().positive().default(1),
});

export const ContainerActionConfigSchema = z.object({
  action: z.enum(["stop", "pause", "restart", "remove"]),
  gracePeriod: z.number().nonnegative().default(10000),
  restart: z.boolean().default(false),
  restartDelay: z.number().nonnegative().default(5000),
});

export const NodeDrainConfigSchema = z.object({
  gracePeriod: z.number().nonnegative().default(300000), // 5 minutes
  force: z.boolean().default(false),
  ignoreDaemonSets: z.boolean().default(true),
  deleteLocalData: z.boolean().default(false),
});

// ============================================================================
// Types
// ============================================================================

export type NetworkPartitionConfig = z.infer<
  typeof NetworkPartitionConfigSchema
>;
export type PacketLossConfig = z.infer<typeof PacketLossConfigSchema>;
export type BandwidthLimitConfig = z.infer<typeof BandwidthLimitConfigSchema>;
export type DNSFailureConfig = z.infer<typeof DNSFailureConfigSchema>;
export type CPUStressConfig = z.infer<typeof CPUStressConfigSchema>;
export type MemoryStressConfig = z.infer<typeof MemoryStressConfigSchema>;
export type DiskStressConfig = z.infer<typeof DiskStressConfigSchema>;
export type IOStressConfig = z.infer<typeof IOStressConfigSchema>;
export type ProcessKillConfig = z.infer<typeof ProcessKillConfigSchema>;
export type ContainerActionConfig = z.infer<typeof ContainerActionConfigSchema>;
export type NodeDrainConfig = z.infer<typeof NodeDrainConfigSchema>;

// ============================================================================
// Network Fault Injectors
// ============================================================================

/**
 * Network partition injector
 * Simulates network splits between services/pods
 */
export class NetworkPartitionInjector implements FaultInjector {
  type: FaultType = "network_partition";
  private activeFaults: Map<string, NetworkPartitionState> = new Map();

  async inject(target: AffectedTarget, config: FaultConfig): Promise<string> {
    const faultId = uuidv4();
    const partitionConfig = NetworkPartitionConfigSchema.parse(
      config.parameters || {}
    );

    console.log(
      `Creating ${partitionConfig.partitionType} network partition for target ${target.id}`
    );

    // In production, this would use:
    // - iptables rules for container networking
    // - Network policies for Kubernetes
    // - Traffic control (tc) for partial failures

    const state: NetworkPartitionState = {
      faultId,
      targetId: target.id,
      config: partitionConfig,
      startedAt: new Date(),
      rules: [],
    };

    // Simulate creating iptables rules
    if (partitionConfig.partitionType === "full") {
      state.rules.push(
        `iptables -A OUTPUT -d ${JSON.stringify(partitionConfig.destinationSelector)} -j DROP`
      );
      state.rules.push(
        `iptables -A INPUT -s ${JSON.stringify(partitionConfig.sourceSelector)} -j DROP`
      );
    } else if (partitionConfig.partitionType === "one_way") {
      state.rules.push(
        `iptables -A OUTPUT -d ${JSON.stringify(partitionConfig.destinationSelector)} -j DROP`
      );
    }

    this.activeFaults.set(faultId, state);
    return faultId;
  }

  async rollback(faultId: string): Promise<void> {
    const state = this.activeFaults.get(faultId);
    if (!state) return;

    console.log(`Removing network partition ${faultId}`);

    // Remove iptables rules (in reverse order)
    for (const rule of state.rules.reverse()) {
      console.log(`Removing rule: ${rule.replace("-A", "-D")}`);
    }

    this.activeFaults.delete(faultId);
  }

  async verify(faultId: string): Promise<boolean> {
    return this.activeFaults.has(faultId);
  }
}

interface NetworkPartitionState {
  faultId: string;
  targetId: string;
  config: NetworkPartitionConfig;
  startedAt: Date;
  rules: string[];
}

/**
 * Packet loss injector
 * Introduces packet loss using traffic control
 */
export class PacketLossInjector implements FaultInjector {
  type: FaultType = "packet_loss";
  private activeFaults: Map<string, PacketLossState> = new Map();

  async inject(target: AffectedTarget, config: FaultConfig): Promise<string> {
    const faultId = uuidv4();
    const lossConfig = PacketLossConfigSchema.parse(config.parameters || {});

    console.log(
      `Injecting ${lossConfig.percentage}% packet loss to target ${target.id}`
    );

    // In production: tc qdisc add dev eth0 root netem loss 10%
    const command = `tc qdisc add dev eth0 root netem loss ${lossConfig.percentage}% ${lossConfig.correlation}%`;

    const state: PacketLossState = {
      faultId,
      targetId: target.id,
      config: lossConfig,
      startedAt: new Date(),
      command,
    };

    this.activeFaults.set(faultId, state);
    return faultId;
  }

  async rollback(faultId: string): Promise<void> {
    const state = this.activeFaults.get(faultId);
    if (!state) return;

    console.log(`Removing packet loss ${faultId}`);
    // tc qdisc del dev eth0 root
    this.activeFaults.delete(faultId);
  }

  async verify(faultId: string): Promise<boolean> {
    return this.activeFaults.has(faultId);
  }
}

interface PacketLossState {
  faultId: string;
  targetId: string;
  config: PacketLossConfig;
  startedAt: Date;
  command: string;
}

/**
 * Bandwidth limit injector
 * Throttles network bandwidth
 */
export class BandwidthLimitInjector implements FaultInjector {
  type: FaultType = "bandwidth_limit";
  private activeFaults: Map<string, BandwidthLimitState> = new Map();

  async inject(target: AffectedTarget, config: FaultConfig): Promise<string> {
    const faultId = uuidv4();
    const bwConfig = BandwidthLimitConfigSchema.parse(config.parameters || {});

    console.log(
      `Limiting bandwidth to ${bwConfig.rate} for target ${target.id}`
    );

    // tc qdisc add dev eth0 root tbf rate 1mbit burst 32kbit latency 400ms
    const command =
      `tc qdisc add dev eth0 root tbf rate ${bwConfig.rate}` +
      (bwConfig.burst ? ` burst ${bwConfig.burst}` : "") +
      (bwConfig.latency ? ` latency ${bwConfig.latency}ms` : "");

    const state: BandwidthLimitState = {
      faultId,
      targetId: target.id,
      config: bwConfig,
      startedAt: new Date(),
      command,
    };

    this.activeFaults.set(faultId, state);
    return faultId;
  }

  async rollback(faultId: string): Promise<void> {
    const state = this.activeFaults.get(faultId);
    if (!state) return;

    console.log(`Removing bandwidth limit ${faultId}`);
    this.activeFaults.delete(faultId);
  }

  async verify(faultId: string): Promise<boolean> {
    return this.activeFaults.has(faultId);
  }
}

interface BandwidthLimitState {
  faultId: string;
  targetId: string;
  config: BandwidthLimitConfig;
  startedAt: Date;
  command: string;
}

/**
 * DNS failure injector
 * Causes DNS resolution failures
 */
export class DNSFailureInjector implements FaultInjector {
  type: FaultType = "dns_failure";
  private activeFaults: Map<string, DNSFailureState> = new Map();

  async inject(target: AffectedTarget, config: FaultConfig): Promise<string> {
    const faultId = uuidv4();
    const dnsConfig = DNSFailureConfigSchema.parse(config.parameters || {});

    console.log(
      `Injecting DNS ${dnsConfig.failureType} failure for target ${target.id}`
    );

    // In production, this would modify /etc/hosts or configure a DNS proxy
    const state: DNSFailureState = {
      faultId,
      targetId: target.id,
      config: dnsConfig,
      startedAt: new Date(),
      originalResolv: "",
    };

    this.activeFaults.set(faultId, state);
    return faultId;
  }

  async rollback(faultId: string): Promise<void> {
    const state = this.activeFaults.get(faultId);
    if (!state) return;

    console.log(`Removing DNS failure ${faultId}`);
    // Restore original DNS configuration
    this.activeFaults.delete(faultId);
  }

  async verify(faultId: string): Promise<boolean> {
    return this.activeFaults.has(faultId);
  }
}

interface DNSFailureState {
  faultId: string;
  targetId: string;
  config: DNSFailureConfig;
  startedAt: Date;
  originalResolv: string;
}

// ============================================================================
// Resource Fault Injectors
// ============================================================================

/**
 * CPU stress injector
 * Generates CPU load using stress-ng or similar tools
 */
export class CPUStressInjector implements FaultInjector {
  type: FaultType = "cpu_stress";
  private activeFaults: Map<string, CPUStressState> = new Map();

  async inject(target: AffectedTarget, config: FaultConfig): Promise<string> {
    const faultId = uuidv4();
    const cpuConfig = CPUStressConfigSchema.parse(config.parameters || {});

    console.log(
      `Injecting ${cpuConfig.load}% CPU stress with ${cpuConfig.workers} workers to target ${target.id}`
    );

    // stress-ng --cpu N --cpu-load P --timeout T
    const command = `stress-ng --${cpuConfig.method} ${cpuConfig.workers} --cpu-load ${cpuConfig.load} --timeout ${cpuConfig.duration}ms`;

    const state: CPUStressState = {
      faultId,
      targetId: target.id,
      config: cpuConfig,
      startedAt: new Date(),
      command,
      pid: undefined,
    };

    // In production, would spawn the process and track PID
    this.activeFaults.set(faultId, state);
    return faultId;
  }

  async rollback(faultId: string): Promise<void> {
    const state = this.activeFaults.get(faultId);
    if (!state) return;

    console.log(`Stopping CPU stress ${faultId}`);

    if (state.pid) {
      // Kill the stress process
      console.log(`Killing process ${state.pid}`);
    }

    this.activeFaults.delete(faultId);
  }

  async verify(faultId: string): Promise<boolean> {
    const state = this.activeFaults.get(faultId);
    if (!state) return false;
    // Check if process is still running
    return state.pid !== undefined;
  }
}

interface CPUStressState {
  faultId: string;
  targetId: string;
  config: CPUStressConfig;
  startedAt: Date;
  command: string;
  pid?: number;
}

/**
 * Memory stress injector
 * Consumes memory to stress the system
 */
export class MemoryStressInjector implements FaultInjector {
  type: FaultType = "memory_stress";
  private activeFaults: Map<string, MemoryStressState> = new Map();

  async inject(target: AffectedTarget, config: FaultConfig): Promise<string> {
    const faultId = uuidv4();
    const memConfig = MemoryStressConfigSchema.parse(config.parameters || {});

    console.log(
      `Injecting ${memConfig.bytes} memory stress to target ${target.id}`
    );

    // stress-ng --vm N --vm-bytes B --timeout T
    const command = `stress-ng --vm ${memConfig.workers} --vm-bytes ${memConfig.bytes} --timeout ${memConfig.duration}ms`;

    const state: MemoryStressState = {
      faultId,
      targetId: target.id,
      config: memConfig,
      startedAt: new Date(),
      command,
      pid: undefined,
    };

    this.activeFaults.set(faultId, state);
    return faultId;
  }

  async rollback(faultId: string): Promise<void> {
    const state = this.activeFaults.get(faultId);
    if (!state) return;

    console.log(`Stopping memory stress ${faultId}`);

    if (state.pid) {
      console.log(`Killing process ${state.pid}`);
    }

    this.activeFaults.delete(faultId);
  }

  async verify(faultId: string): Promise<boolean> {
    const state = this.activeFaults.get(faultId);
    if (!state) return false;
    return state.pid !== undefined;
  }
}

interface MemoryStressState {
  faultId: string;
  targetId: string;
  config: MemoryStressConfig;
  startedAt: Date;
  command: string;
  pid?: number;
}

/**
 * Disk stress injector
 * Fills disk space or stresses disk I/O
 */
export class DiskStressInjector implements FaultInjector {
  type: FaultType = "disk_stress";
  private activeFaults: Map<string, DiskStressState> = new Map();

  async inject(target: AffectedTarget, config: FaultConfig): Promise<string> {
    const faultId = uuidv4();
    const diskConfig = DiskStressConfigSchema.parse(config.parameters || {});

    console.log(
      `Injecting ${diskConfig.size} disk stress (${diskConfig.method}) to target ${target.id}`
    );

    // dd if=/dev/zero of=/tmp/stress bs=1M count=1024
    // or stress-ng --hdd N --hdd-bytes B
    const command = `stress-ng --hdd ${diskConfig.workers} --hdd-bytes ${diskConfig.size} --timeout ${diskConfig.duration}ms`;

    const state: DiskStressState = {
      faultId,
      targetId: target.id,
      config: diskConfig,
      startedAt: new Date(),
      command,
      tempFiles: [],
    };

    this.activeFaults.set(faultId, state);
    return faultId;
  }

  async rollback(faultId: string): Promise<void> {
    const state = this.activeFaults.get(faultId);
    if (!state) return;

    console.log(`Stopping disk stress ${faultId}`);

    // Clean up any temp files created
    for (const file of state.tempFiles) {
      console.log(`Removing temp file: ${file}`);
    }

    this.activeFaults.delete(faultId);
  }

  async verify(faultId: string): Promise<boolean> {
    return this.activeFaults.has(faultId);
  }
}

interface DiskStressState {
  faultId: string;
  targetId: string;
  config: DiskStressConfig;
  startedAt: Date;
  command: string;
  tempFiles: string[];
}

// ============================================================================
// Process/Container Fault Injectors
// ============================================================================

/**
 * Process kill injector
 * Kills processes to test recovery
 */
export class ProcessKillInjector implements FaultInjector {
  type: FaultType = "process_kill";
  private activeFaults: Map<string, ProcessKillState> = new Map();

  async inject(target: AffectedTarget, config: FaultConfig): Promise<string> {
    const faultId = uuidv4();
    const killConfig = ProcessKillConfigSchema.parse(config.parameters || {});

    console.log(
      `Killing process with ${killConfig.signal} for target ${target.id}`
    );

    const state: ProcessKillState = {
      faultId,
      targetId: target.id,
      config: killConfig,
      startedAt: new Date(),
      killCount: 0,
      intervalHandle: undefined,
    };

    // Execute kill(s)
    for (let i = 0; i < killConfig.count; i++) {
      if (killConfig.pid) {
        console.log(`kill -${killConfig.signal} ${killConfig.pid}`);
      } else if (killConfig.processName) {
        console.log(`pkill -${killConfig.signal} ${killConfig.processName}`);
      }
      state.killCount++;

      if (killConfig.interval && i < killConfig.count - 1) {
        await new Promise((resolve) =>
          setTimeout(resolve, killConfig.interval)
        );
      }
    }

    this.activeFaults.set(faultId, state);
    return faultId;
  }

  async rollback(faultId: string): Promise<void> {
    const state = this.activeFaults.get(faultId);
    if (!state) return;

    console.log(`Process kill fault ${faultId} - no rollback action needed`);

    if (state.intervalHandle) {
      clearInterval(state.intervalHandle);
    }

    this.activeFaults.delete(faultId);
  }

  async verify(faultId: string): Promise<boolean> {
    return this.activeFaults.has(faultId);
  }
}

interface ProcessKillState {
  faultId: string;
  targetId: string;
  config: ProcessKillConfig;
  startedAt: Date;
  killCount: number;
  intervalHandle?: NodeJS.Timeout;
}

/**
 * Container action injector
 * Performs actions on containers (stop, pause, restart)
 */
export class ContainerActionInjector implements FaultInjector {
  type: FaultType = "container_stop";
  private activeFaults: Map<string, ContainerActionState> = new Map();

  async inject(target: AffectedTarget, config: FaultConfig): Promise<string> {
    const faultId = uuidv4();
    const actionConfig = ContainerActionConfigSchema.parse(
      config.parameters || {}
    );

    console.log(
      `Performing ${actionConfig.action} on container for target ${target.id}`
    );

    const containerId =
      target.selector["containerId"] || target.selector["name"];

    // docker stop/pause/restart <container>
    const command = `docker ${actionConfig.action} ${containerId}`;

    const state: ContainerActionState = {
      faultId,
      targetId: target.id,
      config: actionConfig,
      containerId: containerId as string,
      startedAt: new Date(),
      command,
    };

    this.activeFaults.set(faultId, state);
    return faultId;
  }

  async rollback(faultId: string): Promise<void> {
    const state = this.activeFaults.get(faultId);
    if (!state) return;

    console.log(`Rolling back container action ${faultId}`);

    // Perform inverse action
    if (state.config.action === "stop" || state.config.action === "pause") {
      if (state.config.restart) {
        console.log(`docker start ${state.containerId}`);
      }
    } else if (state.config.action === "pause") {
      console.log(`docker unpause ${state.containerId}`);
    }

    this.activeFaults.delete(faultId);
  }

  async verify(faultId: string): Promise<boolean> {
    return this.activeFaults.has(faultId);
  }
}

interface ContainerActionState {
  faultId: string;
  targetId: string;
  config: ContainerActionConfig;
  containerId: string;
  startedAt: Date;
  command: string;
}

/**
 * Node drain injector
 * Drains a Kubernetes node
 */
export class NodeDrainInjector implements FaultInjector {
  type: FaultType = "node_drain";
  private activeFaults: Map<string, NodeDrainState> = new Map();

  async inject(target: AffectedTarget, config: FaultConfig): Promise<string> {
    const faultId = uuidv4();
    const drainConfig = NodeDrainConfigSchema.parse(config.parameters || {});

    const nodeName = target.selector["nodeName"] || target.selector["node"];

    console.log(`Draining node ${nodeName} for target ${target.id}`);

    // kubectl drain <node> --grace-period=N --force --ignore-daemonsets
    const command =
      `kubectl drain ${nodeName}` +
      ` --grace-period=${Math.floor(drainConfig.gracePeriod / 1000)}` +
      (drainConfig.force ? " --force" : "") +
      (drainConfig.ignoreDaemonSets ? " --ignore-daemonsets" : "") +
      (drainConfig.deleteLocalData ? " --delete-local-data" : "");

    const state: NodeDrainState = {
      faultId,
      targetId: target.id,
      config: drainConfig,
      nodeName: nodeName as string,
      startedAt: new Date(),
      command,
    };

    this.activeFaults.set(faultId, state);
    return faultId;
  }

  async rollback(faultId: string): Promise<void> {
    const state = this.activeFaults.get(faultId);
    if (!state) return;

    console.log(`Uncordoning node ${state.nodeName}`);
    // kubectl uncordon <node>
    console.log(`kubectl uncordon ${state.nodeName}`);

    this.activeFaults.delete(faultId);
  }

  async verify(faultId: string): Promise<boolean> {
    return this.activeFaults.has(faultId);
  }
}

interface NodeDrainState {
  faultId: string;
  targetId: string;
  config: NodeDrainConfig;
  nodeName: string;
  startedAt: Date;
  command: string;
}

// ============================================================================
// Fault Registry
// ============================================================================

/**
 * Registry of all available fault injectors
 */
export class FaultRegistry {
  private injectors: Map<FaultType, FaultInjector> = new Map();

  constructor() {
    this.registerDefaults();
  }

  private registerDefaults(): void {
    // Network faults
    this.register(new NetworkPartitionInjector());
    this.register(new PacketLossInjector());
    this.register(new BandwidthLimitInjector());
    this.register(new DNSFailureInjector());

    // Resource faults
    this.register(new CPUStressInjector());
    this.register(new MemoryStressInjector());
    this.register(new DiskStressInjector());

    // Process/Container faults
    this.register(new ProcessKillInjector());
    this.register(new ContainerActionInjector());
    this.register(new NodeDrainInjector());
  }

  register(injector: FaultInjector): void {
    this.injectors.set(injector.type, injector);
  }

  get(type: FaultType): FaultInjector | undefined {
    return this.injectors.get(type);
  }

  list(): FaultType[] {
    return Array.from(this.injectors.keys());
  }

  has(type: FaultType): boolean {
    return this.injectors.has(type);
  }
}

// ============================================================================
// Composite Fault Scenarios
// ============================================================================

/**
 * Pre-built chaos scenarios combining multiple faults
 */
export const ChaosScenarios = {
  /**
   * Simulates a network partition between services
   */
  networkPartition: (
    source: Record<string, string>,
    destination: Record<string, string>,
    duration: number
  ): FaultConfig[] => [
    {
      type: "network_partition",
      name: "Network Partition",
      description: "Full network partition between services",
      severity: "high",
      parameters: {
        partitionType: "full",
        sourceSelector: source,
        destinationSelector: destination,
        duration,
      },
    },
  ],

  /**
   * Simulates a slow network
   */
  slowNetwork: (
    latencyMs: number,
    packetLoss: number,
    duration: number
  ): FaultConfig[] => [
    {
      type: "latency",
      name: "High Latency",
      severity: "medium",
      parameters: { latencyMs },
      duration,
    },
    {
      type: "packet_loss",
      name: "Packet Loss",
      severity: "medium",
      parameters: { percentage: packetLoss },
      duration,
    },
  ],

  /**
   * Simulates resource exhaustion
   */
  resourceExhaustion: (
    cpuLoad: number,
    memoryBytes: string,
    duration: number
  ): FaultConfig[] => [
    {
      type: "cpu_stress",
      name: "CPU Stress",
      severity: "high",
      parameters: { load: cpuLoad, workers: 2 },
      duration,
    },
    {
      type: "memory_stress",
      name: "Memory Stress",
      severity: "high",
      parameters: { bytes: memoryBytes, workers: 1 },
      duration,
    },
  ],

  /**
   * Simulates cascading failure
   */
  cascadingFailure: (services: string[], intervalMs: number): FaultConfig[] =>
    services.map((service, index) => ({
      type: "process_kill" as FaultType,
      name: `Kill ${service}`,
      severity: "critical" as const,
      parameters: {
        processName: service,
        signal: "SIGKILL",
      },
      // Stagger the kills
      duration: intervalMs * (index + 1),
    })),

  /**
   * Simulates zone failure
   */
  zoneFailure: (zoneName: string, duration: number): FaultConfig[] => [
    {
      type: "node_drain",
      name: `Drain Zone ${zoneName}`,
      severity: "critical",
      parameters: {
        gracePeriod: 30000,
        force: true,
        ignoreDaemonSets: true,
      },
      duration,
    },
  ],
} as const;
