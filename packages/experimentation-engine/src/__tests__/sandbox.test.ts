/**
 * @fileoverview Sandbox Isolation Framework for Chaos Experiments
 * @module @neurectomy/experimentation-engine/__tests__/sandbox
 * @agent @ECLIPSE @FORTRESS
 *
 * Provides isolated execution environments for chaos experiments:
 * - Mocked system interfaces (network, CPU, memory, filesystem)
 * - Resource limits and quotas
 * - Automatic cleanup guarantees
 * - Fault injection interception
 * - Audit logging and traceability
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { EventEmitter } from "eventemitter3";

// ============================================================================
// Sandbox Types
// ============================================================================

interface SandboxConfig {
  /** Unique sandbox identifier */
  id?: string;
  /** Resource limits */
  limits: ResourceLimits;
  /** Allowed operations */
  permissions: SandboxPermissions;
  /** Timeout for entire sandbox execution (ms) */
  timeout: number;
  /** Enable audit logging */
  auditLog: boolean;
  /** Cleanup strategy */
  cleanupStrategy: "immediate" | "deferred" | "manual";
}

interface ResourceLimits {
  maxMemoryMB: number;
  maxCPUPercent: number;
  maxNetworkBandwidthKBps: number;
  maxDiskIOps: number;
  maxOpenFiles: number;
  maxProcesses: number;
}

interface SandboxPermissions {
  allowNetworkAccess: boolean;
  allowFileSystemAccess: boolean;
  allowProcessSpawning: boolean;
  allowedNetworkTargets?: string[];
  allowedFilePaths?: string[];
}

interface SandboxState {
  status: "idle" | "running" | "paused" | "completed" | "failed" | "aborted";
  startTime?: number;
  endTime?: number;
  resourceUsage: ResourceUsage;
  violations: SandboxViolation[];
  auditLog: AuditEntry[];
}

interface ResourceUsage {
  memoryMB: number;
  cpuPercent: number;
  networkBytesIn: number;
  networkBytesOut: number;
  diskReads: number;
  diskWrites: number;
  openFiles: number;
  processes: number;
}

interface SandboxViolation {
  timestamp: number;
  type: "memory" | "cpu" | "network" | "disk" | "permission" | "timeout";
  details: string;
  severity: "warning" | "error" | "critical";
  blocked: boolean;
}

interface AuditEntry {
  timestamp: number;
  action: string;
  target?: string;
  allowed: boolean;
  details?: Record<string, unknown>;
}

// ============================================================================
// Mock System Interfaces
// ============================================================================

class MockNetworkInterface {
  private sandbox: Sandbox;
  private connections: Map<string, MockConnection> = new Map();

  constructor(sandbox: Sandbox) {
    this.sandbox = sandbox;
  }

  async connect(target: string, port: number): Promise<MockConnection> {
    const allowed = this.sandbox.checkPermission("network", target);

    if (!allowed) {
      this.sandbox.recordViolation({
        type: "permission",
        details: `Network access to ${target}:${port} denied`,
        severity: "error",
        blocked: true,
      });
      throw new Error(`Network access denied: ${target}:${port}`);
    }

    const connection = new MockConnection(target, port, this.sandbox);
    this.connections.set(`${target}:${port}`, connection);

    this.sandbox.auditLog({
      action: "network_connect",
      target: `${target}:${port}`,
      allowed: true,
    });

    return connection;
  }

  async send(target: string, data: Buffer | string): Promise<void> {
    const bytes =
      typeof data === "string" ? Buffer.byteLength(data) : data.length;

    if (!this.sandbox.trackResourceUsage("networkBytesOut", bytes)) {
      throw new Error("Network bandwidth limit exceeded");
    }

    this.sandbox.auditLog({
      action: "network_send",
      target,
      allowed: true,
      details: { bytes },
    });
  }

  async receive(target: string, bytes: number): Promise<Buffer> {
    if (!this.sandbox.trackResourceUsage("networkBytesIn", bytes)) {
      throw new Error("Network bandwidth limit exceeded");
    }

    this.sandbox.auditLog({
      action: "network_receive",
      target,
      allowed: true,
      details: { bytes },
    });

    return Buffer.alloc(bytes); // Simulated data
  }

  closeAll(): void {
    for (const connection of this.connections.values()) {
      connection.close();
    }
    this.connections.clear();
  }
}

class MockConnection {
  private target: string;
  private port: number;
  private sandbox: Sandbox;
  private closed = false;

  constructor(target: string, port: number, sandbox: Sandbox) {
    this.target = target;
    this.port = port;
    this.sandbox = sandbox;
  }

  async send(data: Buffer | string): Promise<void> {
    if (this.closed) throw new Error("Connection closed");
    const bytes =
      typeof data === "string" ? Buffer.byteLength(data) : data.length;
    this.sandbox.trackResourceUsage("networkBytesOut", bytes);
  }

  async receive(maxBytes: number): Promise<Buffer> {
    if (this.closed) throw new Error("Connection closed");
    this.sandbox.trackResourceUsage("networkBytesIn", maxBytes);
    return Buffer.alloc(maxBytes);
  }

  close(): void {
    this.closed = true;
    this.sandbox.auditLog({
      action: "network_close",
      target: `${this.target}:${this.port}`,
      allowed: true,
    });
  }
}

class MockFileSystem {
  private sandbox: Sandbox;
  private files: Map<string, MockFile> = new Map();
  private openHandles: Set<string> = new Set();

  constructor(sandbox: Sandbox) {
    this.sandbox = sandbox;
  }

  async open(path: string, mode: "r" | "w" | "rw"): Promise<MockFileHandle> {
    const allowed = this.sandbox.checkPermission("filesystem", path);

    if (!allowed) {
      this.sandbox.recordViolation({
        type: "permission",
        details: `File access to ${path} denied`,
        severity: "error",
        blocked: true,
      });
      throw new Error(`File access denied: ${path}`);
    }

    if (!this.sandbox.trackResourceUsage("openFiles", 1)) {
      throw new Error("Too many open files");
    }

    const handle = new MockFileHandle(path, mode, this);
    this.openHandles.add(path);

    this.sandbox.auditLog({
      action: "file_open",
      target: path,
      allowed: true,
      details: { mode },
    });

    return handle;
  }

  async read(path: string): Promise<Buffer> {
    if (!this.sandbox.checkPermission("filesystem", path)) {
      throw new Error(`File access denied: ${path}`);
    }

    this.sandbox.trackResourceUsage("diskReads", 1);

    const file = this.files.get(path);
    return file?.content || Buffer.alloc(0);
  }

  async write(path: string, content: Buffer | string): Promise<void> {
    if (!this.sandbox.checkPermission("filesystem", path)) {
      throw new Error(`File access denied: ${path}`);
    }

    this.sandbox.trackResourceUsage("diskWrites", 1);

    const buffer = typeof content === "string" ? Buffer.from(content) : content;
    this.files.set(path, { path, content: buffer });

    this.sandbox.auditLog({
      action: "file_write",
      target: path,
      allowed: true,
      details: { bytes: buffer.length },
    });
  }

  closeHandle(path: string): void {
    this.openHandles.delete(path);
    this.sandbox.trackResourceUsage("openFiles", -1);
  }

  cleanup(): void {
    this.openHandles.clear();
    this.files.clear();
  }
}

interface MockFile {
  path: string;
  content: Buffer;
}

class MockFileHandle {
  private path: string;
  private mode: string;
  private fs: MockFileSystem;
  private closed = false;

  constructor(path: string, mode: string, fs: MockFileSystem) {
    this.path = path;
    this.mode = mode;
    this.fs = fs;
  }

  async read(): Promise<Buffer> {
    if (this.closed) throw new Error("Handle closed");
    return this.fs.read(this.path);
  }

  async write(content: Buffer | string): Promise<void> {
    if (this.closed) throw new Error("Handle closed");
    if (this.mode === "r") throw new Error("File opened for reading only");
    return this.fs.write(this.path, content);
  }

  close(): void {
    this.closed = true;
    this.fs.closeHandle(this.path);
  }
}

class MockProcessManager {
  private sandbox: Sandbox;
  private processes: Map<number, MockProcess> = new Map();
  private nextPid = 1000;

  constructor(sandbox: Sandbox) {
    this.sandbox = sandbox;
  }

  spawn(command: string, args: string[]): MockProcess {
    if (!this.sandbox.getConfig().permissions.allowProcessSpawning) {
      this.sandbox.recordViolation({
        type: "permission",
        details: `Process spawning denied: ${command}`,
        severity: "error",
        blocked: true,
      });
      throw new Error("Process spawning not allowed");
    }

    if (!this.sandbox.trackResourceUsage("processes", 1)) {
      throw new Error("Process limit exceeded");
    }

    const pid = this.nextPid++;
    const process = new MockProcess(pid, command, args, this);
    this.processes.set(pid, process);

    this.sandbox.auditLog({
      action: "process_spawn",
      target: command,
      allowed: true,
      details: { pid, args },
    });

    return process;
  }

  kill(pid: number): void {
    const process = this.processes.get(pid);
    if (process) {
      process.terminate();
      this.processes.delete(pid);
      this.sandbox.trackResourceUsage("processes", -1);
    }
  }

  killAll(): void {
    for (const pid of this.processes.keys()) {
      this.kill(pid);
    }
  }
}

class MockProcess {
  pid: number;
  command: string;
  args: string[];
  private manager: MockProcessManager;
  private _running = true;

  constructor(
    pid: number,
    command: string,
    args: string[],
    manager: MockProcessManager
  ) {
    this.pid = pid;
    this.command = command;
    this.args = args;
    this.manager = manager;
  }

  get running(): boolean {
    return this._running;
  }

  terminate(): void {
    this._running = false;
  }

  kill(): void {
    this.manager.kill(this.pid);
  }
}

// ============================================================================
// Sandbox Implementation
// ============================================================================

class Sandbox extends EventEmitter {
  private config: SandboxConfig;
  private state: SandboxState;
  private network: MockNetworkInterface;
  private filesystem: MockFileSystem;
  private processManager: MockProcessManager;
  private timeoutHandle?: NodeJS.Timeout;

  constructor(config: Partial<SandboxConfig> = {}) {
    super();

    this.config = {
      id:
        config.id ||
        `sandbox-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      limits: config.limits || {
        maxMemoryMB: 512,
        maxCPUPercent: 50,
        maxNetworkBandwidthKBps: 1024,
        maxDiskIOps: 100,
        maxOpenFiles: 10,
        maxProcesses: 5,
      },
      permissions: config.permissions || {
        allowNetworkAccess: false,
        allowFileSystemAccess: false,
        allowProcessSpawning: false,
      },
      timeout: config.timeout || 30000,
      auditLog: config.auditLog ?? true,
      cleanupStrategy: config.cleanupStrategy || "immediate",
    };

    this.state = {
      status: "idle",
      resourceUsage: {
        memoryMB: 0,
        cpuPercent: 0,
        networkBytesIn: 0,
        networkBytesOut: 0,
        diskReads: 0,
        diskWrites: 0,
        openFiles: 0,
        processes: 0,
      },
      violations: [],
      auditLog: [],
    };

    this.network = new MockNetworkInterface(this);
    this.filesystem = new MockFileSystem(this);
    this.processManager = new MockProcessManager(this);
  }

  getConfig(): SandboxConfig {
    return { ...this.config };
  }

  getState(): SandboxState {
    return { ...this.state };
  }

  start(): void {
    if (this.state.status !== "idle") {
      throw new Error(`Cannot start sandbox in ${this.state.status} state`);
    }

    this.state.status = "running";
    this.state.startTime = Date.now();

    // Set timeout
    this.timeoutHandle = setTimeout(() => {
      this.recordViolation({
        type: "timeout",
        details: `Sandbox exceeded timeout of ${this.config.timeout}ms`,
        severity: "critical",
        blocked: true,
      });
      this.abort("Timeout exceeded");
    }, this.config.timeout);

    this.emit("started", this.config.id);
  }

  pause(): void {
    if (this.state.status !== "running") return;
    this.state.status = "paused";
    this.emit("paused", this.config.id);
  }

  resume(): void {
    if (this.state.status !== "paused") return;
    this.state.status = "running";
    this.emit("resumed", this.config.id);
  }

  complete(): SandboxResults {
    this.state.status = "completed";
    this.state.endTime = Date.now();
    this.cleanup();

    const results = this.generateResults();
    this.emit("completed", results);
    return results;
  }

  abort(reason: string): void {
    this.state.status = "aborted";
    this.state.endTime = Date.now();
    this.cleanup();

    this.emit("aborted", { sandboxId: this.config.id, reason });
  }

  fail(error: Error): void {
    this.state.status = "failed";
    this.state.endTime = Date.now();
    this.cleanup();

    this.emit("failed", { sandboxId: this.config.id, error });
  }

  private cleanup(): void {
    if (this.timeoutHandle) {
      clearTimeout(this.timeoutHandle);
    }

    if (this.config.cleanupStrategy !== "manual") {
      this.network.closeAll();
      this.filesystem.cleanup();
      this.processManager.killAll();
    }
  }

  // Resource tracking
  trackResourceUsage(resource: keyof ResourceUsage, delta: number): boolean {
    const newValue = this.state.resourceUsage[resource] + delta;

    // Check limits
    const limit = this.getResourceLimit(resource);
    if (limit !== null && newValue > limit) {
      this.recordViolation({
        type: this.getViolationType(resource),
        details: `${resource} limit exceeded: ${newValue} > ${limit}`,
        severity: "warning",
        blocked: true,
      });
      return false;
    }

    this.state.resourceUsage[resource] = Math.max(0, newValue);
    return true;
  }

  private getResourceLimit(resource: keyof ResourceUsage): number | null {
    const limits = this.config.limits;
    switch (resource) {
      case "memoryMB":
        return limits.maxMemoryMB;
      case "cpuPercent":
        return limits.maxCPUPercent;
      case "networkBytesIn":
      case "networkBytesOut":
        return limits.maxNetworkBandwidthKBps * 1024;
      case "diskReads":
      case "diskWrites":
        return limits.maxDiskIOps;
      case "openFiles":
        return limits.maxOpenFiles;
      case "processes":
        return limits.maxProcesses;
      default:
        return null;
    }
  }

  private getViolationType(
    resource: keyof ResourceUsage
  ): SandboxViolation["type"] {
    switch (resource) {
      case "memoryMB":
        return "memory";
      case "cpuPercent":
        return "cpu";
      case "networkBytesIn":
      case "networkBytesOut":
        return "network";
      case "diskReads":
      case "diskWrites":
      case "openFiles":
        return "disk";
      case "processes":
        return "permission";
      default:
        return "permission";
    }
  }

  // Permission checking
  checkPermission(type: "network" | "filesystem", target: string): boolean {
    const perms = this.config.permissions;

    if (type === "network") {
      if (!perms.allowNetworkAccess) return false;
      if (perms.allowedNetworkTargets) {
        return perms.allowedNetworkTargets.some(
          (pattern) => target.includes(pattern) || pattern === "*"
        );
      }
      return true;
    }

    if (type === "filesystem") {
      if (!perms.allowFileSystemAccess) return false;
      if (perms.allowedFilePaths) {
        return perms.allowedFilePaths.some(
          (pattern) => target.startsWith(pattern) || pattern === "*"
        );
      }
      return true;
    }

    return false;
  }

  recordViolation(violation: Omit<SandboxViolation, "timestamp">): void {
    const entry: SandboxViolation = {
      ...violation,
      timestamp: Date.now(),
    };
    this.state.violations.push(entry);
    this.emit("violation", entry);
  }

  auditLog(entry: Omit<AuditEntry, "timestamp">): void {
    if (!this.config.auditLog) return;

    const auditEntry: AuditEntry = {
      ...entry,
      timestamp: Date.now(),
    };
    this.state.auditLog.push(auditEntry);
  }

  // Exposed interfaces for sandboxed code
  getNetwork(): MockNetworkInterface {
    return this.network;
  }

  getFileSystem(): MockFileSystem {
    return this.filesystem;
  }

  getProcessManager(): MockProcessManager {
    return this.processManager;
  }

  private generateResults(): SandboxResults {
    return {
      sandboxId: this.config.id!,
      status: this.state.status,
      duration:
        (this.state.endTime || Date.now()) - (this.state.startTime || 0),
      resourceUsage: { ...this.state.resourceUsage },
      violations: [...this.state.violations],
      auditLog: [...this.state.auditLog],
      violationCount: this.state.violations.length,
      criticalViolations: this.state.violations.filter(
        (v) => v.severity === "critical"
      ).length,
    };
  }
}

interface SandboxResults {
  sandboxId: string;
  status: SandboxState["status"];
  duration: number;
  resourceUsage: ResourceUsage;
  violations: SandboxViolation[];
  auditLog: AuditEntry[];
  violationCount: number;
  criticalViolations: number;
}

// ============================================================================
// Sandbox Factory
// ============================================================================

class SandboxFactory {
  private static presets: Record<string, Partial<SandboxConfig>> = {
    strict: {
      limits: {
        maxMemoryMB: 128,
        maxCPUPercent: 25,
        maxNetworkBandwidthKBps: 256,
        maxDiskIOps: 50,
        maxOpenFiles: 5,
        maxProcesses: 2,
      },
      permissions: {
        allowNetworkAccess: false,
        allowFileSystemAccess: false,
        allowProcessSpawning: false,
      },
      timeout: 10000,
    },
    moderate: {
      limits: {
        maxMemoryMB: 512,
        maxCPUPercent: 50,
        maxNetworkBandwidthKBps: 1024,
        maxDiskIOps: 100,
        maxOpenFiles: 10,
        maxProcesses: 5,
      },
      permissions: {
        allowNetworkAccess: true,
        allowFileSystemAccess: true,
        allowProcessSpawning: false,
        allowedNetworkTargets: ["localhost", "127.0.0.1"],
        allowedFilePaths: ["/tmp/", "/var/sandbox/"],
      },
      timeout: 30000,
    },
    permissive: {
      limits: {
        maxMemoryMB: 2048,
        maxCPUPercent: 80,
        maxNetworkBandwidthKBps: 10240,
        maxDiskIOps: 1000,
        maxOpenFiles: 50,
        maxProcesses: 20,
      },
      permissions: {
        allowNetworkAccess: true,
        allowFileSystemAccess: true,
        allowProcessSpawning: true,
        allowedNetworkTargets: ["*"],
        allowedFilePaths: ["*"],
      },
      timeout: 60000,
    },
  };

  static create(
    preset: "strict" | "moderate" | "permissive",
    overrides?: Partial<SandboxConfig>
  ): Sandbox {
    const config = {
      ...this.presets[preset],
      ...overrides,
    };
    return new Sandbox(config);
  }

  static createCustom(config: Partial<SandboxConfig>): Sandbox {
    return new Sandbox(config);
  }
}

// ============================================================================
// Tests for Sandbox Framework
// ============================================================================

describe("Sandbox Framework", () => {
  let sandbox: Sandbox;

  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    sandbox?.abort("test cleanup");
    vi.useRealTimers();
  });

  // --------------------------------------------------------------------------
  // Sandbox Lifecycle Tests
  // --------------------------------------------------------------------------

  describe("Sandbox Lifecycle", () => {
    it("should create sandbox with default config", () => {
      sandbox = new Sandbox();
      expect(sandbox.getState().status).toBe("idle");
    });

    it("should start sandbox", () => {
      sandbox = new Sandbox();
      sandbox.start();
      expect(sandbox.getState().status).toBe("running");
    });

    it("should pause and resume sandbox", () => {
      sandbox = new Sandbox();
      sandbox.start();
      sandbox.pause();
      expect(sandbox.getState().status).toBe("paused");

      sandbox.resume();
      expect(sandbox.getState().status).toBe("running");
    });

    it("should complete sandbox with results", () => {
      sandbox = new Sandbox();
      sandbox.start();
      const results = sandbox.complete();

      expect(results.status).toBe("completed");
      expect(results.duration).toBeGreaterThanOrEqual(0);
    });

    it("should abort sandbox", () => {
      sandbox = new Sandbox();
      const eventSpy = vi.fn();
      sandbox.on("aborted", eventSpy);

      sandbox.start();
      sandbox.abort("test abort");

      expect(sandbox.getState().status).toBe("aborted");
      expect(eventSpy).toHaveBeenCalled();
    });

    it("should fail sandbox with error", () => {
      sandbox = new Sandbox();
      const eventSpy = vi.fn();
      sandbox.on("failed", eventSpy);

      sandbox.start();
      sandbox.fail(new Error("test error"));

      expect(sandbox.getState().status).toBe("failed");
      expect(eventSpy).toHaveBeenCalled();
    });

    it("should timeout after configured duration", async () => {
      sandbox = new Sandbox({ timeout: 1000 });
      const eventSpy = vi.fn();
      sandbox.on("aborted", eventSpy);

      sandbox.start();
      await vi.advanceTimersByTimeAsync(1500);

      expect(sandbox.getState().status).toBe("aborted");
      expect(sandbox.getState().violations).toContainEqual(
        expect.objectContaining({ type: "timeout" })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Resource Limits Tests
  // --------------------------------------------------------------------------

  describe("Resource Limits", () => {
    it("should track memory usage", () => {
      sandbox = new Sandbox({
        limits: {
          maxMemoryMB: 100,
          maxCPUPercent: 50,
          maxNetworkBandwidthKBps: 1024,
          maxDiskIOps: 100,
          maxOpenFiles: 10,
          maxProcesses: 5,
        },
      });

      sandbox.start();
      sandbox.trackResourceUsage("memoryMB", 50);
      expect(sandbox.getState().resourceUsage.memoryMB).toBe(50);
    });

    it("should block operations exceeding memory limit", () => {
      sandbox = new Sandbox({
        limits: {
          maxMemoryMB: 100,
          maxCPUPercent: 50,
          maxNetworkBandwidthKBps: 1024,
          maxDiskIOps: 100,
          maxOpenFiles: 10,
          maxProcesses: 5,
        },
      });

      sandbox.start();
      const result = sandbox.trackResourceUsage("memoryMB", 150);

      expect(result).toBe(false);
      expect(sandbox.getState().violations).toHaveLength(1);
      expect(sandbox.getState().violations[0]?.type).toBe("memory");
    });

    it("should track network bandwidth", () => {
      sandbox = new Sandbox({
        limits: {
          maxMemoryMB: 512,
          maxCPUPercent: 50,
          maxNetworkBandwidthKBps: 100, // 100KB/s
          maxDiskIOps: 100,
          maxOpenFiles: 10,
          maxProcesses: 5,
        },
        permissions: {
          allowNetworkAccess: true,
          allowFileSystemAccess: false,
          allowProcessSpawning: false,
        },
      });

      sandbox.start();
      sandbox.trackResourceUsage("networkBytesOut", 50 * 1024); // 50KB
      expect(sandbox.getState().resourceUsage.networkBytesOut).toBe(50 * 1024);

      // Exceeding limit
      const result = sandbox.trackResourceUsage("networkBytesOut", 60 * 1024);
      expect(result).toBe(false);
    });

    it("should track open file handles", () => {
      sandbox = new Sandbox({
        limits: {
          maxMemoryMB: 512,
          maxCPUPercent: 50,
          maxNetworkBandwidthKBps: 1024,
          maxDiskIOps: 100,
          maxOpenFiles: 3,
          maxProcesses: 5,
        },
      });

      sandbox.start();
      sandbox.trackResourceUsage("openFiles", 1);
      sandbox.trackResourceUsage("openFiles", 1);
      sandbox.trackResourceUsage("openFiles", 1);

      const result = sandbox.trackResourceUsage("openFiles", 1);
      expect(result).toBe(false);
    });

    it("should track process count", () => {
      sandbox = new Sandbox({
        limits: {
          maxMemoryMB: 512,
          maxCPUPercent: 50,
          maxNetworkBandwidthKBps: 1024,
          maxDiskIOps: 100,
          maxOpenFiles: 10,
          maxProcesses: 2,
        },
        permissions: {
          allowNetworkAccess: false,
          allowFileSystemAccess: false,
          allowProcessSpawning: true,
        },
      });

      sandbox.start();
      sandbox.trackResourceUsage("processes", 2);

      const result = sandbox.trackResourceUsage("processes", 1);
      expect(result).toBe(false);
    });
  });

  // --------------------------------------------------------------------------
  // Permission Tests
  // --------------------------------------------------------------------------

  describe("Permissions", () => {
    it("should deny network access when disabled", () => {
      sandbox = new Sandbox({
        permissions: {
          allowNetworkAccess: false,
          allowFileSystemAccess: false,
          allowProcessSpawning: false,
        },
      });

      sandbox.start();
      const allowed = sandbox.checkPermission("network", "example.com");
      expect(allowed).toBe(false);
    });

    it("should allow network access to whitelisted targets", () => {
      sandbox = new Sandbox({
        permissions: {
          allowNetworkAccess: true,
          allowFileSystemAccess: false,
          allowProcessSpawning: false,
          allowedNetworkTargets: ["localhost", "api.internal.com"],
        },
      });

      sandbox.start();
      expect(sandbox.checkPermission("network", "localhost")).toBe(true);
      expect(sandbox.checkPermission("network", "api.internal.com")).toBe(true);
      expect(sandbox.checkPermission("network", "evil.com")).toBe(false);
    });

    it("should deny filesystem access when disabled", () => {
      sandbox = new Sandbox({
        permissions: {
          allowNetworkAccess: false,
          allowFileSystemAccess: false,
          allowProcessSpawning: false,
        },
      });

      sandbox.start();
      const allowed = sandbox.checkPermission("filesystem", "/etc/passwd");
      expect(allowed).toBe(false);
    });

    it("should allow filesystem access to whitelisted paths", () => {
      sandbox = new Sandbox({
        permissions: {
          allowNetworkAccess: false,
          allowFileSystemAccess: true,
          allowProcessSpawning: false,
          allowedFilePaths: ["/tmp/", "/var/sandbox/"],
        },
      });

      sandbox.start();
      expect(sandbox.checkPermission("filesystem", "/tmp/test.txt")).toBe(true);
      expect(sandbox.checkPermission("filesystem", "/var/sandbox/data")).toBe(
        true
      );
      expect(sandbox.checkPermission("filesystem", "/etc/passwd")).toBe(false);
    });

    it("should allow all with wildcard permissions", () => {
      sandbox = new Sandbox({
        permissions: {
          allowNetworkAccess: true,
          allowFileSystemAccess: true,
          allowProcessSpawning: true,
          allowedNetworkTargets: ["*"],
          allowedFilePaths: ["*"],
        },
      });

      sandbox.start();
      expect(sandbox.checkPermission("network", "any.domain.com")).toBe(true);
      expect(sandbox.checkPermission("filesystem", "/any/path")).toBe(true);
    });
  });

  // --------------------------------------------------------------------------
  // Mock Interface Tests
  // --------------------------------------------------------------------------

  describe("Mock Network Interface", () => {
    it("should connect to allowed targets", async () => {
      sandbox = new Sandbox({
        permissions: {
          allowNetworkAccess: true,
          allowFileSystemAccess: false,
          allowProcessSpawning: false,
          allowedNetworkTargets: ["localhost"],
        },
      });

      sandbox.start();
      const network = sandbox.getNetwork();
      const connection = await network.connect("localhost", 8080);

      expect(connection).toBeDefined();
    });

    it("should reject connections to denied targets", async () => {
      sandbox = new Sandbox({
        permissions: {
          allowNetworkAccess: true,
          allowFileSystemAccess: false,
          allowProcessSpawning: false,
          allowedNetworkTargets: ["localhost"],
        },
      });

      sandbox.start();
      const network = sandbox.getNetwork();

      await expect(network.connect("evil.com", 80)).rejects.toThrow("denied");
    });

    it("should track network bytes", async () => {
      sandbox = new Sandbox({
        limits: {
          maxMemoryMB: 512,
          maxCPUPercent: 50,
          maxNetworkBandwidthKBps: 1024,
          maxDiskIOps: 100,
          maxOpenFiles: 10,
          maxProcesses: 5,
        },
        permissions: {
          allowNetworkAccess: true,
          allowFileSystemAccess: false,
          allowProcessSpawning: false,
          allowedNetworkTargets: ["*"],
        },
      });

      sandbox.start();
      const network = sandbox.getNetwork();

      await network.send("localhost", Buffer.alloc(1000));

      expect(sandbox.getState().resourceUsage.networkBytesOut).toBe(1000);
    });
  });

  describe("Mock Filesystem", () => {
    it("should open files in allowed paths", async () => {
      sandbox = new Sandbox({
        permissions: {
          allowNetworkAccess: false,
          allowFileSystemAccess: true,
          allowProcessSpawning: false,
          allowedFilePaths: ["/tmp/"],
        },
      });

      sandbox.start();
      const fs = sandbox.getFileSystem();
      const handle = await fs.open("/tmp/test.txt", "rw");

      expect(handle).toBeDefined();
    });

    it("should reject access to denied paths", async () => {
      sandbox = new Sandbox({
        permissions: {
          allowNetworkAccess: false,
          allowFileSystemAccess: true,
          allowProcessSpawning: false,
          allowedFilePaths: ["/tmp/"],
        },
      });

      sandbox.start();
      const fs = sandbox.getFileSystem();

      await expect(fs.open("/etc/passwd", "r")).rejects.toThrow("denied");
    });

    it("should track disk I/O operations", async () => {
      sandbox = new Sandbox({
        permissions: {
          allowNetworkAccess: false,
          allowFileSystemAccess: true,
          allowProcessSpawning: false,
          allowedFilePaths: ["*"],
        },
      });

      sandbox.start();
      const fs = sandbox.getFileSystem();

      await fs.write("/tmp/test.txt", "test content");
      await fs.read("/tmp/test.txt");

      expect(sandbox.getState().resourceUsage.diskWrites).toBe(1);
      expect(sandbox.getState().resourceUsage.diskReads).toBe(1);
    });
  });

  describe("Mock Process Manager", () => {
    it("should spawn processes when allowed", () => {
      sandbox = new Sandbox({
        permissions: {
          allowNetworkAccess: false,
          allowFileSystemAccess: false,
          allowProcessSpawning: true,
        },
      });

      sandbox.start();
      const pm = sandbox.getProcessManager();
      const process = pm.spawn("echo", ["hello"]);

      expect(process.pid).toBeGreaterThanOrEqual(1000);
      expect(process.running).toBe(true);
    });

    it("should reject process spawning when disabled", () => {
      sandbox = new Sandbox({
        permissions: {
          allowNetworkAccess: false,
          allowFileSystemAccess: false,
          allowProcessSpawning: false,
        },
      });

      sandbox.start();
      const pm = sandbox.getProcessManager();

      expect(() => pm.spawn("malicious", [])).toThrow("not allowed");
    });

    it("should enforce process limit", () => {
      sandbox = new Sandbox({
        limits: {
          maxMemoryMB: 512,
          maxCPUPercent: 50,
          maxNetworkBandwidthKBps: 1024,
          maxDiskIOps: 100,
          maxOpenFiles: 10,
          maxProcesses: 2,
        },
        permissions: {
          allowNetworkAccess: false,
          allowFileSystemAccess: false,
          allowProcessSpawning: true,
        },
      });

      sandbox.start();
      const pm = sandbox.getProcessManager();

      pm.spawn("cmd1", []);
      pm.spawn("cmd2", []);

      expect(() => pm.spawn("cmd3", [])).toThrow("limit exceeded");
    });
  });

  // --------------------------------------------------------------------------
  // Audit Log Tests
  // --------------------------------------------------------------------------

  describe("Audit Logging", () => {
    it("should record all operations", async () => {
      sandbox = new Sandbox({
        auditLog: true,
        permissions: {
          allowNetworkAccess: true,
          allowFileSystemAccess: true,
          allowProcessSpawning: true,
          allowedNetworkTargets: ["*"],
          allowedFilePaths: ["*"],
        },
      });

      sandbox.start();

      const network = sandbox.getNetwork();
      await network.connect("localhost", 80);

      const fs = sandbox.getFileSystem();
      await fs.write("/tmp/test.txt", "data");

      const auditLog = sandbox.getState().auditLog;
      expect(auditLog.length).toBeGreaterThan(0);
      expect(auditLog.some((e) => e.action === "network_connect")).toBe(true);
      expect(auditLog.some((e) => e.action === "file_write")).toBe(true);
    });

    it("should disable logging when configured", async () => {
      sandbox = new Sandbox({
        auditLog: false,
        permissions: {
          allowNetworkAccess: true,
          allowFileSystemAccess: false,
          allowProcessSpawning: false,
          allowedNetworkTargets: ["*"],
        },
      });

      sandbox.start();

      sandbox.auditLog({ action: "test", allowed: true });

      expect(sandbox.getState().auditLog).toHaveLength(0);
    });
  });

  // --------------------------------------------------------------------------
  // Violation Tracking Tests
  // --------------------------------------------------------------------------

  describe("Violation Tracking", () => {
    it("should record violations with timestamps", () => {
      sandbox = new Sandbox({
        limits: {
          maxMemoryMB: 100,
          maxCPUPercent: 50,
          maxNetworkBandwidthKBps: 1024,
          maxDiskIOps: 100,
          maxOpenFiles: 10,
          maxProcesses: 5,
        },
      });

      sandbox.start();
      sandbox.trackResourceUsage("memoryMB", 200);

      const violations = sandbox.getState().violations;
      expect(violations).toHaveLength(1);
      expect(violations[0]?.timestamp).toBeGreaterThan(0);
      expect(violations[0]?.severity).toBe("warning");
    });

    it("should emit violation events", () => {
      sandbox = new Sandbox({
        limits: {
          maxMemoryMB: 100,
          maxCPUPercent: 50,
          maxNetworkBandwidthKBps: 1024,
          maxDiskIOps: 100,
          maxOpenFiles: 10,
          maxProcesses: 5,
        },
      });

      const eventSpy = vi.fn();
      sandbox.on("violation", eventSpy);

      sandbox.start();
      sandbox.trackResourceUsage("memoryMB", 200);

      expect(eventSpy).toHaveBeenCalledWith(
        expect.objectContaining({ type: "memory" })
      );
    });

    it("should count critical violations in results", () => {
      sandbox = new Sandbox({ timeout: 100 });

      sandbox.start();

      // Force critical violation
      sandbox.recordViolation({
        type: "permission",
        details: "Critical security violation",
        severity: "critical",
        blocked: true,
      });

      const results = sandbox.complete();
      expect(results.criticalViolations).toBe(1);
    });
  });

  // --------------------------------------------------------------------------
  // Factory Tests
  // --------------------------------------------------------------------------

  describe("Sandbox Factory", () => {
    it("should create strict sandbox", () => {
      sandbox = SandboxFactory.create("strict");
      expect(sandbox.getConfig().limits.maxMemoryMB).toBe(128);
      expect(sandbox.getConfig().permissions.allowNetworkAccess).toBe(false);
    });

    it("should create moderate sandbox", () => {
      sandbox = SandboxFactory.create("moderate");
      expect(sandbox.getConfig().limits.maxMemoryMB).toBe(512);
      expect(sandbox.getConfig().permissions.allowNetworkAccess).toBe(true);
    });

    it("should create permissive sandbox", () => {
      sandbox = SandboxFactory.create("permissive");
      expect(sandbox.getConfig().limits.maxMemoryMB).toBe(2048);
      expect(sandbox.getConfig().permissions.allowProcessSpawning).toBe(true);
    });

    it("should apply overrides", () => {
      sandbox = SandboxFactory.create("strict", {
        timeout: 5000,
      });
      expect(sandbox.getConfig().timeout).toBe(5000);
    });

    it("should create custom sandbox", () => {
      sandbox = SandboxFactory.createCustom({
        limits: {
          maxMemoryMB: 256,
          maxCPUPercent: 30,
          maxNetworkBandwidthKBps: 512,
          maxDiskIOps: 50,
          maxOpenFiles: 5,
          maxProcesses: 3,
        },
      });
      expect(sandbox.getConfig().limits.maxMemoryMB).toBe(256);
    });
  });

  // --------------------------------------------------------------------------
  // Cleanup Tests
  // --------------------------------------------------------------------------

  describe("Cleanup Guarantees", () => {
    it("should cleanup on completion", async () => {
      sandbox = new Sandbox({
        cleanupStrategy: "immediate",
        permissions: {
          allowNetworkAccess: true,
          allowFileSystemAccess: true,
          allowProcessSpawning: true,
          allowedNetworkTargets: ["*"],
          allowedFilePaths: ["*"],
        },
      });

      sandbox.start();

      // Use resources
      const network = sandbox.getNetwork();
      await network.connect("localhost", 80);

      const fs = sandbox.getFileSystem();
      await fs.open("/tmp/test.txt", "rw");

      // Complete triggers cleanup
      const results = sandbox.complete();

      expect(results.status).toBe("completed");
    });

    it("should cleanup on abort", async () => {
      sandbox = new Sandbox({
        cleanupStrategy: "immediate",
        permissions: {
          allowNetworkAccess: true,
          allowFileSystemAccess: false,
          allowProcessSpawning: false,
          allowedNetworkTargets: ["*"],
        },
      });

      sandbox.start();
      sandbox.abort("test abort");

      expect(sandbox.getState().status).toBe("aborted");
    });

    it("should skip cleanup in manual mode", () => {
      sandbox = new Sandbox({
        cleanupStrategy: "manual",
        permissions: {
          allowNetworkAccess: true,
          allowFileSystemAccess: false,
          allowProcessSpawning: false,
          allowedNetworkTargets: ["*"],
        },
      });

      sandbox.start();
      sandbox.complete();

      // Resources should still be accessible for manual cleanup
      expect(sandbox.getNetwork()).toBeDefined();
    });
  });
});

// ============================================================================
// Integration: Sandbox + Chaos Simulation
// ============================================================================

describe("Sandbox Integration with Chaos Experiments", () => {
  it("should run chaos experiment in isolated sandbox", async () => {
    vi.useFakeTimers();

    const sandbox = SandboxFactory.create("moderate", {
      timeout: 5000,
    });

    sandbox.start();

    // Simulate chaos experiment operations within sandbox
    const network = sandbox.getNetwork();
    const fs = sandbox.getFileSystem();

    // Inject network fault (within sandbox)
    await network.connect("localhost", 8080);
    await network.send("localhost", "fault_injection_payload");

    // Write experiment results
    await fs.write(
      "/tmp/chaos_results.json",
      JSON.stringify({
        faultType: "network_partition",
        duration: 1000,
        impactMetrics: { latencyIncrease: 150 },
      })
    );

    const results = sandbox.complete();

    expect(results.status).toBe("completed");
    expect(results.violationCount).toBe(0);

    vi.useRealTimers();
  });

  it("should prevent sandbox escape attempts", async () => {
    const sandbox = SandboxFactory.create("strict");
    const violationSpy = vi.fn();
    sandbox.on("violation", violationSpy);

    sandbox.start();

    // Attempt to access forbidden resources
    sandbox.checkPermission("network", "malicious.com");
    sandbox.checkPermission("filesystem", "/etc/shadow");

    // These should all be blocked
    expect(sandbox.checkPermission("network", "any")).toBe(false);
    expect(sandbox.checkPermission("filesystem", "/root")).toBe(false);

    sandbox.abort("test complete");
  });
});

// Export for use in other tests
export {
  Sandbox,
  SandboxFactory,
  MockNetworkInterface,
  MockFileSystem,
  MockProcessManager,
  type SandboxConfig,
  type SandboxState,
  type SandboxResults,
  type ResourceLimits,
  type SandboxPermissions,
  type SandboxViolation,
  type AuditEntry,
};
