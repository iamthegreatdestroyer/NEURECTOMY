/**
 * NEURECTOMY Firecracker MicroVM Manager
 *
 * @CORE @VELOCITY - Low-Level Systems + Performance
 *
 * Secure, high-performance microVM management using Firecracker
 * for lightweight agent isolation with sub-second startup times.
 */

import { EventEmitter } from "eventemitter3";
import * as fs from "fs/promises";
import * as path from "path";
import * as net from "net";
import * as http from "http";
import pino from "pino";
import { v4 as uuidv4 } from "uuid";
import { z } from "zod";

// =============================================================================
// Types
// =============================================================================

export interface FirecrackerConfig {
  socketPath?: string;
  kernelImagePath: string;
  rootfsPath: string;
  binaryPath?: string;
}

/** MicroVM configuration type - will be validated against MicroVMConfigSchema */
export interface MicroVMConfig {
  id?: string;
  vcpuCount: number;
  memSizeMib: number;
  kernelImagePath: string;
  kernelBootArgs?: string;
  rootDrive: {
    driveId: string;
    pathOnHost: string;
    isRootDevice: boolean;
    isReadOnly: boolean;
    cacheType?: "Unsafe" | "Writeback";
    rateLimit?: {
      bandwidth?: {
        size: number;
        oneTimeBurst?: number;
        refillTime: number;
      };
      ops?: {
        size: number;
        oneTimeBurst?: number;
        refillTime: number;
      };
    };
  };
  additionalDrives?: Array<{
    driveId: string;
    pathOnHost: string;
    isRootDevice: boolean;
    isReadOnly: boolean;
    cacheType?: "Unsafe" | "Writeback";
  }>;
  networkInterfaces?: Array<{
    ifaceId: string;
    hostDevName: string;
    guestMac?: string;
    rateLimit?: {
      bandwidth?: {
        size: number;
        oneTimeBurst?: number;
        refillTime: number;
      };
    };
  }>;
  vsock?: {
    guestCid: number;
    udsPath: string;
  };
  metadata?: Record<string, string>;
  mmdsVersion?: "V1" | "V2";
  enableHyperthreading?: boolean;
  cpuTemplate?: "T2" | "T2S" | "T2CL" | "T2A" | "V1N1" | "None";
  trackDirtyPages?: boolean;
}

/** Drive configuration for MicroVMs */
export type MicroVMDrive = MicroVMConfig["rootDrive"];

/** Network interface configuration for MicroVMs */
export type MicroVMNetworkInterface = NonNullable<
  MicroVMConfig["networkInterfaces"]
>[number];

export interface VMInstance {
  id: string;
  socketPath: string;
  pid?: number;
  state: VMState;
  config: MicroVMConfig;
  createdAt: Date;
  startedAt?: Date;
  stoppedAt?: Date;
  metrics?: VMMetrics;
}

export type VMState =
  | "created"
  | "booting"
  | "running"
  | "paused"
  | "stopped"
  | "error";

export interface VMMetrics {
  cpuUsagePercent: number;
  memoryUsedBytes: number;
  memoryTotalBytes: number;
  networkRxBytes: number;
  networkTxBytes: number;
  blockReadBytes: number;
  blockWriteBytes: number;
  uptimeSeconds: number;
}

export interface SnapshotConfig {
  snapshotPath: string;
  memFilePath: string;
  snapshotType?: "Full" | "Diff";
}

export interface FirecrackerEvents {
  "vm:created": (vm: VMInstance) => void;
  "vm:booting": (vm: VMInstance) => void;
  "vm:running": (vm: VMInstance) => void;
  "vm:paused": (vm: VMInstance) => void;
  "vm:stopped": (vm: VMInstance) => void;
  "vm:error": (vm: VMInstance, error: Error) => void;
  "vm:metrics": (vm: VMInstance, metrics: VMMetrics) => void;
  "snapshot:created": (vmId: string, path: string) => void;
  "snapshot:restored": (vmId: string) => void;
}

// =============================================================================
// Validation Schemas
// =============================================================================

const MicroVMConfigSchema = z.object({
  id: z.string().optional(),
  vcpuCount: z.number().int().min(1).max(32),
  memSizeMib: z.number().int().min(128).max(32768),
  kernelImagePath: z.string(),
  kernelBootArgs: z.string().optional(),
  rootDrive: z.object({
    driveId: z.string(),
    pathOnHost: z.string(),
    isRootDevice: z.boolean(),
    isReadOnly: z.boolean(),
    cacheType: z.enum(["Unsafe", "Writeback"]).optional(),
    rateLimit: z
      .object({
        bandwidth: z
          .object({
            size: z.number(),
            oneTimeBurst: z.number().optional(),
            refillTime: z.number(),
          })
          .optional(),
        ops: z
          .object({
            size: z.number(),
            oneTimeBurst: z.number().optional(),
            refillTime: z.number(),
          })
          .optional(),
      })
      .optional(),
  }),
  additionalDrives: z
    .array(
      z.object({
        driveId: z.string(),
        pathOnHost: z.string(),
        isRootDevice: z.boolean(),
        isReadOnly: z.boolean(),
        cacheType: z.enum(["Unsafe", "Writeback"]).optional(),
      })
    )
    .optional(),
  networkInterfaces: z
    .array(
      z.object({
        ifaceId: z.string(),
        hostDevName: z.string(),
        guestMac: z.string().optional(),
        rateLimit: z
          .object({
            bandwidth: z
              .object({
                size: z.number(),
                oneTimeBurst: z.number().optional(),
                refillTime: z.number(),
              })
              .optional(),
          })
          .optional(),
      })
    )
    .optional(),
  vsock: z
    .object({
      guestCid: z.number().int().min(3),
      udsPath: z.string(),
    })
    .optional(),
  metadata: z.record(z.string()).optional(),
  mmdsVersion: z.enum(["V1", "V2"]).optional(),
  enableHyperthreading: z.boolean().optional(),
  cpuTemplate: z.enum(["T2", "T2S", "T2CL", "T2A", "V1N1", "None"]).optional(),
  trackDirtyPages: z.boolean().optional(),
});

// =============================================================================
// Firecracker API Client
// =============================================================================

class FirecrackerApiClient {
  private socketPath: string;

  constructor(socketPath: string) {
    this.socketPath = socketPath;
  }

  async request<T>(method: string, path: string, body?: object): Promise<T> {
    return new Promise((resolve, reject) => {
      const options: http.RequestOptions = {
        socketPath: this.socketPath,
        path,
        method,
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
      };

      const req = http.request(options, (res) => {
        let data = "";
        res.on("data", (chunk) => (data += chunk));
        res.on("end", () => {
          if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
            try {
              resolve(data ? JSON.parse(data) : ({} as T));
            } catch {
              resolve({} as T);
            }
          } else {
            reject(
              new FirecrackerApiError(
                `API request failed: ${res.statusCode}`,
                res.statusCode || 500,
                data
              )
            );
          }
        });
      });

      req.on("error", reject);

      if (body) {
        req.write(JSON.stringify(body));
      }

      req.end();
    });
  }

  // Machine Configuration
  async setMachineConfig(config: {
    vcpu_count: number;
    mem_size_mib: number;
    ht_enabled?: boolean;
    cpu_template?: string;
    track_dirty_pages?: boolean;
  }): Promise<void> {
    await this.request("PUT", "/machine-config", config);
  }

  // Boot Source
  async setBootSource(config: {
    kernel_image_path: string;
    boot_args?: string;
    initrd_path?: string;
  }): Promise<void> {
    await this.request("PUT", "/boot-source", config);
  }

  // Drives
  async setDrive(drive: {
    drive_id: string;
    path_on_host: string;
    is_root_device: boolean;
    is_read_only: boolean;
    cache_type?: "Unsafe" | "Writeback";
    rate_limiter?: object;
  }): Promise<void> {
    await this.request("PUT", `/drives/${drive.drive_id}`, drive);
  }

  async patchDrive(
    driveId: string,
    patch: { path_on_host?: string }
  ): Promise<void> {
    await this.request("PATCH", `/drives/${driveId}`, patch);
  }

  // Network Interfaces
  async setNetworkInterface(iface: {
    iface_id: string;
    host_dev_name: string;
    guest_mac?: string;
    rx_rate_limiter?: object;
    tx_rate_limiter?: object;
  }): Promise<void> {
    await this.request("PUT", `/network-interfaces/${iface.iface_id}`, iface);
  }

  // VSOCK
  async setVsock(vsock: {
    guest_cid: number;
    uds_path: string;
  }): Promise<void> {
    await this.request("PUT", "/vsock", vsock);
  }

  // Instance Actions
  async startInstance(): Promise<void> {
    await this.request("PUT", "/actions", { action_type: "InstanceStart" });
  }

  async pauseInstance(): Promise<void> {
    await this.request("PATCH", "/vm", { state: "Paused" });
  }

  async resumeInstance(): Promise<void> {
    await this.request("PATCH", "/vm", { state: "Resumed" });
  }

  async sendCtrlAltDel(): Promise<void> {
    await this.request("PUT", "/actions", { action_type: "SendCtrlAltDel" });
  }

  // MMDS (Microvm Metadata Service)
  async setMmdsConfig(config: {
    version?: "V1" | "V2";
    network_interfaces?: string[];
  }): Promise<void> {
    await this.request("PUT", "/mmds/config", config);
  }

  async setMmdsData(data: object): Promise<void> {
    await this.request("PUT", "/mmds", data);
  }

  async patchMmdsData(data: object): Promise<void> {
    await this.request("PATCH", "/mmds", data);
  }

  async getMmdsData(): Promise<object> {
    return this.request("GET", "/mmds");
  }

  // Snapshots
  async createSnapshot(config: {
    snapshot_type: "Full" | "Diff";
    snapshot_path: string;
    mem_file_path: string;
  }): Promise<void> {
    await this.request("PUT", "/snapshot/create", config);
  }

  async loadSnapshot(config: {
    snapshot_path: string;
    mem_backend?: {
      backend_type: "File" | "Uffd";
      backend_path: string;
    };
    enable_diff_snapshots?: boolean;
    resume_vm?: boolean;
  }): Promise<void> {
    await this.request("PUT", "/snapshot/load", config);
  }

  // Metrics
  async getMetrics(): Promise<{
    utc_timestamp_us: number;
    api_server: object;
    block: object;
    net: object;
    vcpu: object;
    vmm: object;
  }> {
    return this.request("GET", "/metrics");
  }

  async setMetricsConfig(config: { metrics_path: string }): Promise<void> {
    await this.request("PUT", "/metrics", config);
  }

  // Instance Info
  async getInstanceInfo(): Promise<{
    app_name: string;
    id: string;
    state: string;
    vmm_version: string;
    started: boolean;
  }> {
    return this.request("GET", "/");
  }

  async getVmConfig(): Promise<object> {
    return this.request("GET", "/vm/config");
  }
}

// =============================================================================
// Firecracker MicroVM Manager
// =============================================================================

export class FirecrackerManager extends EventEmitter<FirecrackerEvents> {
  private config: FirecrackerConfig;
  private logger: pino.Logger;
  private instances: Map<string, VMInstance> = new Map();
  private apiClients: Map<string, FirecrackerApiClient> = new Map();
  private processes: Map<string, { kill: () => void }> = new Map();
  private socketDir: string;
  private metricsInterval?: NodeJS.Timeout;

  constructor(config: FirecrackerConfig) {
    super();

    this.config = config;
    this.socketDir = config.socketPath || "/tmp/firecracker";

    this.logger = pino({
      name: "firecracker-manager",
      level: process.env.LOG_LEVEL || "info",
    });
  }

  // ===========================================================================
  // Lifecycle Management
  // ===========================================================================

  async initialize(): Promise<void> {
    // Ensure socket directory exists
    await fs.mkdir(this.socketDir, { recursive: true });

    // Verify Firecracker binary is available
    if (this.config.binaryPath) {
      try {
        await fs.access(this.config.binaryPath);
      } catch {
        throw new Error(
          `Firecracker binary not found at ${this.config.binaryPath}`
        );
      }
    }

    // Start metrics collection
    this.metricsInterval = setInterval(() => this.collectAllMetrics(), 5000);

    this.logger.info("Firecracker manager initialized");
  }

  async shutdown(): Promise<void> {
    if (this.metricsInterval) {
      clearInterval(this.metricsInterval);
    }

    // Stop all running VMs
    for (const [id] of this.instances) {
      try {
        await this.stopVM(id);
      } catch (error) {
        this.logger.warn(
          { error, vmId: id },
          "Error stopping VM during shutdown"
        );
      }
    }

    this.logger.info("Firecracker manager shut down");
  }

  // ===========================================================================
  // VM Operations
  // ===========================================================================

  async createVM(config: MicroVMConfig): Promise<VMInstance> {
    const validated = MicroVMConfigSchema.parse(config);
    const id = validated.id || uuidv4();
    const socketPath = path.join(this.socketDir, `${id}.sock`);

    this.logger.info({ vmId: id }, "Creating MicroVM");

    // Clean up any existing socket
    try {
      await fs.unlink(socketPath);
    } catch {
      // Socket doesn't exist, which is fine
    }

    const vm: VMInstance = {
      id,
      socketPath,
      state: "created",
      config: validated,
      createdAt: new Date(),
    };

    this.instances.set(id, vm);
    this.emit("vm:created", vm);

    return vm;
  }

  async configureVM(vmId: string): Promise<void> {
    const vm = this.getVM(vmId);
    const config = vm.config;
    const api = this.getApiClient(vmId);

    this.logger.info({ vmId }, "Configuring MicroVM");

    // Set machine configuration
    await api.setMachineConfig({
      vcpu_count: config.vcpuCount,
      mem_size_mib: config.memSizeMib,
      ht_enabled: config.enableHyperthreading,
      cpu_template: config.cpuTemplate,
      track_dirty_pages: config.trackDirtyPages,
    });

    // Set boot source
    await api.setBootSource({
      kernel_image_path: config.kernelImagePath,
      boot_args: config.kernelBootArgs,
    });

    // Set root drive
    await api.setDrive({
      drive_id: config.rootDrive.driveId,
      path_on_host: config.rootDrive.pathOnHost,
      is_root_device: config.rootDrive.isRootDevice,
      is_read_only: config.rootDrive.isReadOnly,
      cache_type: config.rootDrive.cacheType,
      rate_limiter: config.rootDrive.rateLimit,
    });

    // Set additional drives
    if (config.additionalDrives) {
      for (const drive of config.additionalDrives) {
        await api.setDrive({
          drive_id: drive.driveId,
          path_on_host: drive.pathOnHost,
          is_root_device: drive.isRootDevice,
          is_read_only: drive.isReadOnly,
          cache_type: drive.cacheType,
        });
      }
    }

    // Set network interfaces
    if (config.networkInterfaces) {
      for (const iface of config.networkInterfaces) {
        await api.setNetworkInterface({
          iface_id: iface.ifaceId,
          host_dev_name: iface.hostDevName,
          guest_mac: iface.guestMac,
          rx_rate_limiter: iface.rateLimit?.bandwidth
            ? { bandwidth: iface.rateLimit.bandwidth }
            : undefined,
          tx_rate_limiter: iface.rateLimit?.bandwidth
            ? { bandwidth: iface.rateLimit.bandwidth }
            : undefined,
        });
      }
    }

    // Set vsock
    if (config.vsock) {
      await api.setVsock({
        guest_cid: config.vsock.guestCid,
        uds_path: config.vsock.udsPath,
      });
    }

    // Set MMDS if metadata provided
    if (config.metadata) {
      await api.setMmdsConfig({
        version: config.mmdsVersion || "V2",
        network_interfaces: config.networkInterfaces?.map((i) => i.ifaceId),
      });
      await api.setMmdsData(config.metadata);
    }

    this.logger.info({ vmId }, "MicroVM configured");
  }

  async startVM(vmId: string): Promise<void> {
    const vm = this.getVM(vmId);

    if (vm.state === "running") {
      throw new Error(`VM ${vmId} is already running`);
    }

    this.logger.info({ vmId }, "Starting MicroVM");

    vm.state = "booting";
    this.emit("vm:booting", vm);

    try {
      // Start Firecracker process
      await this.startFirecrackerProcess(vm);

      // Wait for socket to be available
      await this.waitForSocket(vm.socketPath);

      // Create API client
      const api = new FirecrackerApiClient(vm.socketPath);
      this.apiClients.set(vmId, api);

      // Configure VM
      await this.configureVM(vmId);

      // Start the instance
      await api.startInstance();

      vm.state = "running";
      vm.startedAt = new Date();
      this.emit("vm:running", vm);

      this.logger.info({ vmId }, "MicroVM started successfully");
    } catch (error) {
      vm.state = "error";
      this.emit("vm:error", vm, error as Error);
      throw error;
    }
  }

  async stopVM(vmId: string): Promise<void> {
    const vm = this.getVM(vmId);

    if (vm.state === "stopped") {
      return;
    }

    this.logger.info({ vmId }, "Stopping MicroVM");

    try {
      const api = this.apiClients.get(vmId);
      if (api) {
        // Send graceful shutdown signal
        await api.sendCtrlAltDel();

        // Wait briefly for graceful shutdown
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
    } catch {
      // VM might already be stopped
    }

    // Kill the Firecracker process
    const process = this.processes.get(vmId);
    if (process) {
      process.kill();
      this.processes.delete(vmId);
    }

    // Clean up socket
    try {
      await fs.unlink(vm.socketPath);
    } catch {
      // Socket might already be cleaned up
    }

    vm.state = "stopped";
    vm.stoppedAt = new Date();
    this.apiClients.delete(vmId);
    this.emit("vm:stopped", vm);

    this.logger.info({ vmId }, "MicroVM stopped");
  }

  async pauseVM(vmId: string): Promise<void> {
    const vm = this.getVM(vmId);

    if (vm.state !== "running") {
      throw new Error(`VM ${vmId} is not running`);
    }

    const api = this.getApiClient(vmId);
    await api.pauseInstance();

    vm.state = "paused";
    this.emit("vm:paused", vm);

    this.logger.info({ vmId }, "MicroVM paused");
  }

  async resumeVM(vmId: string): Promise<void> {
    const vm = this.getVM(vmId);

    if (vm.state !== "paused") {
      throw new Error(`VM ${vmId} is not paused`);
    }

    const api = this.getApiClient(vmId);
    await api.resumeInstance();

    vm.state = "running";
    this.emit("vm:running", vm);

    this.logger.info({ vmId }, "MicroVM resumed");
  }

  async destroyVM(vmId: string): Promise<void> {
    await this.stopVM(vmId);
    this.instances.delete(vmId);
    this.logger.info({ vmId }, "MicroVM destroyed");
  }

  // ===========================================================================
  // Snapshot Operations
  // ===========================================================================

  async createSnapshot(vmId: string, config: SnapshotConfig): Promise<void> {
    const vm = this.getVM(vmId);

    if (vm.state !== "paused") {
      throw new Error("VM must be paused to create snapshot");
    }

    const api = this.getApiClient(vmId);

    this.logger.info(
      { vmId, snapshotPath: config.snapshotPath },
      "Creating snapshot"
    );

    await api.createSnapshot({
      snapshot_type: config.snapshotType || "Full",
      snapshot_path: config.snapshotPath,
      mem_file_path: config.memFilePath,
    });

    this.emit("snapshot:created", vmId, config.snapshotPath);
    this.logger.info({ vmId }, "Snapshot created");
  }

  async restoreSnapshot(
    vmId: string,
    snapshotPath: string,
    memFilePath: string,
    resume: boolean = true
  ): Promise<void> {
    const vm = this.getVM(vmId);
    const api = this.getApiClient(vmId);

    this.logger.info({ vmId, snapshotPath }, "Restoring snapshot");

    await api.loadSnapshot({
      snapshot_path: snapshotPath,
      mem_backend: {
        backend_type: "File",
        backend_path: memFilePath,
      },
      resume_vm: resume,
    });

    if (resume) {
      vm.state = "running";
      this.emit("vm:running", vm);
    }

    this.emit("snapshot:restored", vmId);
    this.logger.info({ vmId }, "Snapshot restored");
  }

  // ===========================================================================
  // Metadata Service
  // ===========================================================================

  async setMetadata(vmId: string, data: object): Promise<void> {
    const api = this.getApiClient(vmId);
    await api.setMmdsData(data);
    this.logger.debug({ vmId }, "Metadata set");
  }

  async updateMetadata(vmId: string, data: object): Promise<void> {
    const api = this.getApiClient(vmId);
    await api.patchMmdsData(data);
    this.logger.debug({ vmId }, "Metadata updated");
  }

  async getMetadata(vmId: string): Promise<object> {
    const api = this.getApiClient(vmId);
    return api.getMmdsData();
  }

  // ===========================================================================
  // Metrics
  // ===========================================================================

  async getVMMetrics(vmId: string): Promise<VMMetrics> {
    const vm = this.getVM(vmId);
    const api = this.getApiClient(vmId);

    const rawMetrics = await api.getMetrics();

    const vcpu = rawMetrics.vcpu as Record<
      string,
      { user_time_us?: number; system_time_us?: number }
    >;
    const net = rawMetrics.net as Record<
      string,
      { rx_bytes_count?: number; tx_bytes_count?: number }
    >;
    const block = rawMetrics.block as Record<
      string,
      { read_bytes?: number; write_bytes?: number }
    >;
    const vmm = rawMetrics.vmm as { memory_usage_rss_kb?: number };

    const metrics: VMMetrics = {
      cpuUsagePercent: this.calculateCpuUsage(vcpu),
      memoryUsedBytes: (vmm?.memory_usage_rss_kb || 0) * 1024,
      memoryTotalBytes: vm.config.memSizeMib * 1024 * 1024,
      networkRxBytes: this.sumMetricValues(net, "rx_bytes_count"),
      networkTxBytes: this.sumMetricValues(net, "tx_bytes_count"),
      blockReadBytes: this.sumMetricValues(block, "read_bytes"),
      blockWriteBytes: this.sumMetricValues(block, "write_bytes"),
      uptimeSeconds: vm.startedAt
        ? (Date.now() - vm.startedAt.getTime()) / 1000
        : 0,
    };

    vm.metrics = metrics;
    return metrics;
  }

  private calculateCpuUsage(
    vcpu: Record<string, { user_time_us?: number; system_time_us?: number }>
  ): number {
    let totalUserTime = 0;
    let totalSystemTime = 0;

    for (const [key, value] of Object.entries(vcpu)) {
      if (key.startsWith("vcpu")) {
        totalUserTime += value.user_time_us || 0;
        totalSystemTime += value.system_time_us || 0;
      }
    }

    // This is a simplified calculation
    const totalTime = totalUserTime + totalSystemTime;
    return Math.min(100, totalTime / 10000); // Normalize
  }

  private sumMetricValues(
    metrics: Record<string, Record<string, number>>,
    field: string
  ): number {
    let sum = 0;
    for (const value of Object.values(metrics)) {
      if (value && typeof value[field] === "number") {
        sum += value[field];
      }
    }
    return sum;
  }

  private async collectAllMetrics(): Promise<void> {
    for (const [vmId, vm] of this.instances) {
      if (vm.state === "running") {
        try {
          const metrics = await this.getVMMetrics(vmId);
          this.emit("vm:metrics", vm, metrics);
        } catch (error) {
          this.logger.debug({ error, vmId }, "Failed to collect metrics");
        }
      }
    }
  }

  // ===========================================================================
  // Listing & Querying
  // ===========================================================================

  listVMs(): VMInstance[] {
    return Array.from(this.instances.values());
  }

  getVMById(vmId: string): VMInstance | undefined {
    return this.instances.get(vmId);
  }

  listRunningVMs(): VMInstance[] {
    return this.listVMs().filter((vm) => vm.state === "running");
  }

  // ===========================================================================
  // Drive Operations
  // ===========================================================================

  async attachDrive(vmId: string, drive: MicroVMDrive): Promise<void> {
    const api = this.getApiClient(vmId);

    await api.setDrive({
      drive_id: drive.driveId,
      path_on_host: drive.pathOnHost,
      is_root_device: drive.isRootDevice,
      is_read_only: drive.isReadOnly,
      cache_type: drive.cacheType,
    });

    this.logger.info({ vmId, driveId: drive.driveId }, "Drive attached");
  }

  async updateDrivePath(
    vmId: string,
    driveId: string,
    newPath: string
  ): Promise<void> {
    const api = this.getApiClient(vmId);
    await api.patchDrive(driveId, { path_on_host: newPath });
    this.logger.info({ vmId, driveId, newPath }, "Drive path updated");
  }

  // ===========================================================================
  // Private Methods
  // ===========================================================================

  private getVM(vmId: string): VMInstance {
    const vm = this.instances.get(vmId);
    if (!vm) {
      throw new Error(`VM ${vmId} not found`);
    }
    return vm;
  }

  private getApiClient(vmId: string): FirecrackerApiClient {
    const client = this.apiClients.get(vmId);
    if (!client) {
      throw new Error(`API client for VM ${vmId} not available`);
    }
    return client;
  }

  private async startFirecrackerProcess(vm: VMInstance): Promise<void> {
    const { spawn } = await import("child_process");
    const binary = this.config.binaryPath || "firecracker";

    const args = ["--api-sock", vm.socketPath, "--id", vm.id];

    this.logger.debug({ binary, args }, "Starting Firecracker process");

    const proc = spawn(binary, args, {
      stdio: ["ignore", "pipe", "pipe"],
      detached: true,
    });

    proc.stdout?.on("data", (data: Buffer) => {
      this.logger.debug(
        { vmId: vm.id, stdout: data.toString() },
        "Firecracker stdout"
      );
    });

    proc.stderr?.on("data", (data: Buffer) => {
      this.logger.warn(
        { vmId: vm.id, stderr: data.toString() },
        "Firecracker stderr"
      );
    });

    proc.on("error", (error) => {
      this.logger.error({ error, vmId: vm.id }, "Firecracker process error");
      vm.state = "error";
      this.emit("vm:error", vm, error);
    });

    proc.on("exit", (code, signal) => {
      this.logger.info(
        { vmId: vm.id, code, signal },
        "Firecracker process exited"
      );
      if (vm.state !== "stopped") {
        vm.state = "stopped";
        vm.stoppedAt = new Date();
        this.emit("vm:stopped", vm);
      }
    });

    vm.pid = proc.pid;
    this.processes.set(vm.id, {
      kill: () => proc.kill("SIGTERM"),
    });
  }

  private async waitForSocket(
    socketPath: string,
    timeout: number = 10000
  ): Promise<void> {
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      try {
        await fs.access(socketPath);
        // Try to connect to verify socket is ready
        await new Promise<void>((resolve, reject) => {
          const socket = net.createConnection(socketPath);
          socket.on("connect", () => {
            socket.destroy();
            resolve();
          });
          socket.on("error", reject);
        });
        return;
      } catch {
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
    }

    throw new Error(`Timeout waiting for socket: ${socketPath}`);
  }
}

// =============================================================================
// Custom Errors
// =============================================================================

export class FirecrackerApiError extends Error {
  constructor(
    message: string,
    public readonly statusCode: number,
    public readonly body: string
  ) {
    super(message);
    this.name = "FirecrackerApiError";
  }
}

// =============================================================================
// Convenience Export
// =============================================================================

export function createFirecrackerManager(
  config: FirecrackerConfig
): FirecrackerManager {
  return new FirecrackerManager(config);
}
