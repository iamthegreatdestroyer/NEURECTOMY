/**
 * NEURECTOMY Wasmtime Runtime Manager
 *
 * @CORE @APEX - Low-Level Systems + Implementation
 *
 * WebAssembly runtime for lightweight agent execution with WASI support,
 * hot-loading capabilities, and secure sandboxing.
 */

import { EventEmitter } from "eventemitter3";
import * as fs from "fs/promises";
import * as path from "path";
import * as crypto from "crypto";
import pino from "pino";
import { v4 as uuidv4 } from "uuid";
import { z } from "zod";

// =============================================================================
// Types
// =============================================================================

export interface WasmModuleConfig {
  id?: string;
  modulePath: string;
  name?: string;

  // WASI configuration
  wasi?: {
    args?: string[];
    env?: Record<string, string>;
    preopenDirs?: Record<string, string>;
    stdin?: string;
    stdout?: "inherit" | "capture" | "null";
    stderr?: "inherit" | "capture" | "null";
  };

  // Resource limits
  limits?: {
    maxMemoryPages?: number;
    maxTableElements?: number;
    maxInstances?: number;
    fuelLimit?: number;
    epochDeadline?: number;
  };

  // Module caching
  cacheCompiled?: boolean;

  // Imports/Exports
  imports?: Record<string, WasmImport>;
  allowedExports?: string[];
}

export interface WasmImport {
  type: "function" | "memory" | "table" | "global";
  handler?: (...args: unknown[]) => unknown;
  value?: unknown;
}

export interface WasmInstance {
  id: string;
  moduleId: string;
  state: WasmInstanceState;
  createdAt: Date;
  startedAt?: Date;
  finishedAt?: Date;
  exitCode?: number;
  output?: {
    stdout: string;
    stderr: string;
  };
  metrics?: WasmMetrics;
}

export type WasmInstanceState =
  | "created"
  | "running"
  | "suspended"
  | "completed"
  | "error"
  | "trapped";

export interface WasmModule {
  id: string;
  path: string;
  name: string;
  hash: string;
  size: number;
  compiledAt?: Date;
  exports: WasmExportInfo[];
  imports: WasmImportInfo[];
  config: WasmModuleConfig;
}

export interface WasmExportInfo {
  name: string;
  kind: "function" | "memory" | "table" | "global";
  signature?: string;
}

export interface WasmImportInfo {
  module: string;
  name: string;
  kind: "function" | "memory" | "table" | "global";
  signature?: string;
}

export interface WasmMetrics {
  fuelConsumed: number;
  memoryBytesUsed: number;
  executionTimeMs: number;
  instructionCount: number;
}

export interface ComponentModelConfig {
  witPath?: string;
  worldName?: string;
  imports?: Record<string, ComponentImport>;
}

export interface ComponentImport {
  interfaceName: string;
  functions: Record<string, (...args: unknown[]) => unknown>;
}

export interface WasmtimeEvents {
  "module:loaded": (module: WasmModule) => void;
  "module:compiled": (module: WasmModule) => void;
  "module:unloaded": (moduleId: string) => void;
  "instance:created": (instance: WasmInstance) => void;
  "instance:started": (instance: WasmInstance) => void;
  "instance:completed": (instance: WasmInstance) => void;
  "instance:error": (instance: WasmInstance, error: Error) => void;
  "instance:trapped": (instance: WasmInstance, trap: string) => void;
  "fuel:exhausted": (instanceId: string) => void;
  "epoch:deadline": (instanceId: string) => void;
}

// =============================================================================
// Validation Schemas
// =============================================================================

const WasmModuleConfigSchema = z.object({
  id: z.string().optional(),
  modulePath: z.string().min(1),
  name: z.string().optional(),
  wasi: z
    .object({
      args: z.array(z.string()).optional(),
      env: z.record(z.string()).optional(),
      preopenDirs: z.record(z.string()).optional(),
      stdin: z.string().optional(),
      stdout: z.enum(["inherit", "capture", "null"]).optional(),
      stderr: z.enum(["inherit", "capture", "null"]).optional(),
    })
    .optional(),
  limits: z
    .object({
      maxMemoryPages: z.number().int().positive().optional(),
      maxTableElements: z.number().int().positive().optional(),
      maxInstances: z.number().int().positive().optional(),
      fuelLimit: z.number().int().positive().optional(),
      epochDeadline: z.number().int().positive().optional(),
    })
    .optional(),
  cacheCompiled: z.boolean().optional(),
  allowedExports: z.array(z.string()).optional(),
});

// =============================================================================
// Wasmtime Manager Implementation
// =============================================================================

export class WasmtimeManager extends EventEmitter<WasmtimeEvents> {
  private logger: pino.Logger;
  private modules: Map<string, WasmModule> = new Map();
  private instances: Map<string, WasmInstance> = new Map();
  private cacheDir: string;
  private defaultLimits: Required<WasmModuleConfig["limits"]>;

  constructor(
    options: {
      cacheDir?: string;
      defaultLimits?: WasmModuleConfig["limits"];
    } = {}
  ) {
    super();

    this.cacheDir = options.cacheDir || "/tmp/neurectomy-wasm-cache";
    this.defaultLimits = {
      maxMemoryPages: 256, // 16 MB
      maxTableElements: 10000,
      maxInstances: 100,
      fuelLimit: 1000000000, // 1 billion instructions
      epochDeadline: 30000, // 30 seconds
      ...options.defaultLimits,
    };

    this.logger = pino({
      name: "wasmtime-manager",
      level: process.env.LOG_LEVEL || "info",
    });
  }

  // ===========================================================================
  // Module Management
  // ===========================================================================

  async loadModule(config: WasmModuleConfig): Promise<WasmModule> {
    const validated = WasmModuleConfigSchema.parse(config);
    const id = validated.id || uuidv4();

    this.logger.info(
      { moduleId: id, path: validated.modulePath },
      "Loading WASM module"
    );

    // Read and hash the module
    const moduleBytes = await fs.readFile(validated.modulePath);
    const hash = crypto.createHash("sha256").update(moduleBytes).digest("hex");

    // Parse module to extract exports/imports
    const { exports, imports } = await this.parseModule(moduleBytes);

    const module: WasmModule = {
      id,
      path: validated.modulePath,
      name: validated.name || path.basename(validated.modulePath, ".wasm"),
      hash,
      size: moduleBytes.length,
      exports,
      imports,
      config: validated,
    };

    // Check cache for pre-compiled module
    if (validated.cacheCompiled) {
      const cached = await this.loadCachedCompilation(hash);
      if (cached) {
        module.compiledAt = cached.compiledAt;
        this.logger.debug({ moduleId: id }, "Using cached compilation");
      } else {
        // Compile and cache
        await this.compileAndCache(moduleBytes, hash);
        module.compiledAt = new Date();
      }
    }

    this.modules.set(id, module);
    this.emit("module:loaded", module);

    this.logger.info(
      { moduleId: id, exports: exports.length, imports: imports.length },
      "WASM module loaded"
    );

    return module;
  }

  async unloadModule(moduleId: string): Promise<void> {
    const module = this.modules.get(moduleId);
    if (!module) {
      throw new Error(`Module ${moduleId} not found`);
    }

    // Stop all instances using this module
    for (const [instanceId, instance] of this.instances) {
      if (instance.moduleId === moduleId) {
        await this.terminateInstance(instanceId);
      }
    }

    this.modules.delete(moduleId);
    this.emit("module:unloaded", moduleId);

    this.logger.info({ moduleId }, "WASM module unloaded");
  }

  async reloadModule(moduleId: string): Promise<WasmModule> {
    const existing = this.modules.get(moduleId);
    if (!existing) {
      throw new Error(`Module ${moduleId} not found`);
    }

    this.logger.info({ moduleId }, "Hot-reloading WASM module");

    // Load new version
    const newModule = await this.loadModule({
      ...existing.config,
      id: moduleId,
    });

    // Optionally restart instances (depending on use case)
    return newModule;
  }

  getModule(moduleId: string): WasmModule | undefined {
    return this.modules.get(moduleId);
  }

  listModules(): WasmModule[] {
    return Array.from(this.modules.values());
  }

  // ===========================================================================
  // Instance Management
  // ===========================================================================

  async createInstance(moduleId: string): Promise<WasmInstance> {
    const module = this.modules.get(moduleId);
    if (!module) {
      throw new Error(`Module ${moduleId} not found`);
    }

    const instanceId = uuidv4();

    const instance: WasmInstance = {
      id: instanceId,
      moduleId,
      state: "created",
      createdAt: new Date(),
    };

    this.instances.set(instanceId, instance);
    this.emit("instance:created", instance);

    this.logger.info({ instanceId, moduleId }, "WASM instance created");

    return instance;
  }

  async runInstance(
    instanceId: string,
    entrypoint: string = "_start",
    args: unknown[] = []
  ): Promise<unknown> {
    const instance = this.getInstance(instanceId);
    const module = this.modules.get(instance.moduleId);
    if (!module) {
      throw new Error(`Module ${instance.moduleId} not found`);
    }

    this.logger.info({ instanceId, entrypoint }, "Running WASM instance");

    instance.state = "running";
    instance.startedAt = new Date();
    this.emit("instance:started", instance);

    const startTime = Date.now();

    try {
      // This is a simulated execution - in a real implementation,
      // you would use the actual wasmtime Node.js bindings
      const result = await this.executeWasm(module, instance, entrypoint, args);

      instance.state = "completed";
      instance.finishedAt = new Date();
      instance.exitCode = 0;
      instance.metrics = {
        fuelConsumed: 0, // Would be tracked by actual runtime
        memoryBytesUsed: 0,
        executionTimeMs: Date.now() - startTime,
        instructionCount: 0,
      };

      this.emit("instance:completed", instance);

      this.logger.info(
        { instanceId, executionTimeMs: instance.metrics.executionTimeMs },
        "WASM instance completed"
      );

      return result;
    } catch (error) {
      const errorObj = error as Error;

      if (errorObj.message.includes("trap")) {
        instance.state = "trapped";
        this.emit("instance:trapped", instance, errorObj.message);
      } else {
        instance.state = "error";
        this.emit("instance:error", instance, errorObj);
      }

      instance.finishedAt = new Date();

      this.logger.error({ error, instanceId }, "WASM instance error");
      throw error;
    }
  }

  async callExport(
    instanceId: string,
    exportName: string,
    args: unknown[] = []
  ): Promise<unknown> {
    const instance = this.getInstance(instanceId);
    const module = this.modules.get(instance.moduleId);
    if (!module) {
      throw new Error(`Module ${instance.moduleId} not found`);
    }

    // Verify export exists
    const exportInfo = module.exports.find((e) => e.name === exportName);
    if (!exportInfo) {
      throw new Error(`Export ${exportName} not found in module`);
    }

    if (exportInfo.kind !== "function") {
      throw new Error(`Export ${exportName} is not a function`);
    }

    return this.executeWasm(module, instance, exportName, args);
  }

  async suspendInstance(instanceId: string): Promise<void> {
    const instance = this.getInstance(instanceId);

    if (instance.state !== "running") {
      throw new Error(`Instance ${instanceId} is not running`);
    }

    // In real implementation, this would use async stack switching
    // or fuel-based preemption
    instance.state = "suspended";

    this.logger.info({ instanceId }, "WASM instance suspended");
  }

  async resumeInstance(instanceId: string): Promise<void> {
    const instance = this.getInstance(instanceId);

    if (instance.state !== "suspended") {
      throw new Error(`Instance ${instanceId} is not suspended`);
    }

    instance.state = "running";

    this.logger.info({ instanceId }, "WASM instance resumed");
  }

  async terminateInstance(instanceId: string): Promise<void> {
    const instance = this.instances.get(instanceId);
    if (!instance) {
      return;
    }

    instance.state = "completed";
    instance.finishedAt = new Date();

    this.logger.info({ instanceId }, "WASM instance terminated");
  }

  getInstance(instanceId: string): WasmInstance {
    const instance = this.instances.get(instanceId);
    if (!instance) {
      throw new Error(`Instance ${instanceId} not found`);
    }
    return instance;
  }

  listInstances(): WasmInstance[] {
    return Array.from(this.instances.values());
  }

  // ===========================================================================
  // Component Model Support
  // ===========================================================================

  async loadComponent(
    componentPath: string,
    config?: ComponentModelConfig
  ): Promise<WasmModule> {
    this.logger.info({ componentPath }, "Loading WASM component");

    // Components use the component model which allows for richer
    // interfaces and better composability
    const moduleConfig: WasmModuleConfig = {
      modulePath: componentPath,
      name: path.basename(componentPath, ".wasm"),
    };

    const module = await this.loadModule(moduleConfig);

    // Store component-specific config
    // In real implementation, this would parse WIT and set up interfaces

    return module;
  }

  async composeComponents(
    components: string[],
    outputPath: string
  ): Promise<void> {
    this.logger.info(
      { components: components.length, outputPath },
      "Composing WASM components"
    );

    // In real implementation, this would use wasm-tools to compose
    // multiple components into a single module

    throw new Error("Component composition requires wasm-tools CLI");
  }

  // ===========================================================================
  // Security Sandbox
  // ===========================================================================

  async createSandbox(config: {
    moduleId: string;
    allowedImports?: string[];
    allowedExports?: string[];
    memoryLimit?: number;
    cpuLimit?: number;
    networkAccess?: boolean;
    fileSystemAccess?: string[];
  }): Promise<WasmInstance> {
    const module = this.modules.get(config.moduleId);
    if (!module) {
      throw new Error(`Module ${config.moduleId} not found`);
    }

    this.logger.info(
      { moduleId: config.moduleId },
      "Creating sandboxed instance"
    );

    // Verify module only uses allowed imports
    if (config.allowedImports) {
      for (const imp of module.imports) {
        const importKey = `${imp.module}:${imp.name}`;
        if (!config.allowedImports.includes(importKey)) {
          throw new Error(`Module uses disallowed import: ${importKey}`);
        }
      }
    }

    // Create instance with restricted capabilities
    const instance = await this.createInstance(config.moduleId);

    // In real implementation, set up capability-based security

    return instance;
  }

  validateModuleSecurity(moduleId: string): {
    safe: boolean;
    warnings: string[];
    errors: string[];
  } {
    const module = this.modules.get(moduleId);
    if (!module) {
      throw new Error(`Module ${moduleId} not found`);
    }

    const warnings: string[] = [];
    const errors: string[] = [];

    // Check for dangerous imports
    const dangerousImports = [
      "wasi_snapshot_preview1:sock_accept",
      "wasi_snapshot_preview1:sock_recv",
      "wasi_snapshot_preview1:sock_send",
    ];

    for (const imp of module.imports) {
      const importKey = `${imp.module}:${imp.name}`;
      if (dangerousImports.includes(importKey)) {
        warnings.push(
          `Module imports potentially dangerous function: ${importKey}`
        );
      }
    }

    // Check for memory exports
    const memoryExport = module.exports.find((e) => e.kind === "memory");
    if (memoryExport) {
      warnings.push("Module exports memory - ensure proper sandboxing");
    }

    return {
      safe: errors.length === 0,
      warnings,
      errors,
    };
  }

  // ===========================================================================
  // Container Bridge
  // ===========================================================================

  async bridgeToContainer(config: {
    instanceId: string;
    containerEndpoint: string;
    bridgeFunctions?: string[];
  }): Promise<void> {
    this.logger.info(
      { instanceId: config.instanceId, endpoint: config.containerEndpoint },
      "Creating WASM-to-container bridge"
    );

    // This would set up host functions that proxy calls to a container
    // allowing WASM modules to interact with containerized services

    // Example: HTTP calls, gRPC, or custom protocols
  }

  // ===========================================================================
  // Private Methods
  // ===========================================================================

  private async parseModule(moduleBytes: Buffer): Promise<{
    exports: WasmExportInfo[];
    imports: WasmImportInfo[];
  }> {
    // Simplified WASM parsing - in production use @aspect-build/wasm-tools
    // or similar library
    const exports: WasmExportInfo[] = [];
    const imports: WasmImportInfo[] = [];

    // Read WASM header
    const magic = moduleBytes.readUInt32LE(0);
    const version = moduleBytes.readUInt32LE(4);

    if (magic !== 0x6d736100) {
      throw new Error("Invalid WASM magic number");
    }

    if (version !== 1) {
      this.logger.warn({ version }, "Unexpected WASM version");
    }

    // In production, fully parse the module to extract:
    // - Type section (function signatures)
    // - Import section (required imports)
    // - Export section (available exports)
    // - Custom sections (name section, etc.)

    // For now, return placeholder data
    exports.push(
      { name: "_start", kind: "function" },
      { name: "memory", kind: "memory" }
    );

    imports.push(
      { module: "wasi_snapshot_preview1", name: "fd_write", kind: "function" },
      { module: "wasi_snapshot_preview1", name: "proc_exit", kind: "function" }
    );

    return { exports, imports };
  }

  private async loadCachedCompilation(
    hash: string
  ): Promise<{ compiledAt: Date } | null> {
    const cachePath = path.join(this.cacheDir, `${hash}.compiled`);

    try {
      const stat = await fs.stat(cachePath);
      return { compiledAt: stat.mtime };
    } catch {
      return null;
    }
  }

  private async compileAndCache(
    moduleBytes: Buffer,
    hash: string
  ): Promise<void> {
    await fs.mkdir(this.cacheDir, { recursive: true });

    const cachePath = path.join(this.cacheDir, `${hash}.compiled`);

    // In production, this would use wasmtime's AOT compilation
    // For now, just store the raw bytes as placeholder
    await fs.writeFile(cachePath, moduleBytes);

    this.logger.debug({ hash, cachePath }, "Compiled module cached");
  }

  private async executeWasm(
    module: WasmModule,
    instance: WasmInstance,
    entrypoint: string,
    args: unknown[]
  ): Promise<unknown> {
    // This is a placeholder implementation
    // In production, use actual wasmtime Node.js bindings:
    //
    // const { Engine, Store, Module, Instance, WASI } = require('@aspect-build/wasmtime');
    //
    // const engine = new Engine();
    // const store = new Store(engine);
    // const wasmModule = Module.fromFile(engine, module.path);
    //
    // if (module.config.wasi) {
    //   const wasi = new WASI({
    //     version: 'preview1',
    //     args: module.config.wasi.args,
    //     env: module.config.wasi.env,
    //     preopens: module.config.wasi.preopenDirs,
    //   });
    //   const wasmInstance = wasi.instantiate(store, wasmModule, {});
    //   return wasmInstance.exports[entrypoint](...args);
    // }

    this.logger.debug(
      { moduleId: module.id, entrypoint, argCount: args.length },
      "Executing WASM (simulated)"
    );

    // Simulate execution delay based on module size
    const delay = Math.min(100, module.size / 10000);
    await new Promise((resolve) => setTimeout(resolve, delay));

    // Return simulated result
    return { success: true, entrypoint, moduleId: module.id };
  }
}

// =============================================================================
// WASI Builder
// =============================================================================

export class WasiBuilder {
  private args: string[] = [];
  private env: Record<string, string> = {};
  private preopens: Record<string, string> = {};
  private stdin: string | undefined;
  private stdout: "inherit" | "capture" | "null" = "capture";
  private stderr: "inherit" | "capture" | "null" = "capture";

  withArgs(...args: string[]): WasiBuilder {
    this.args.push(...args);
    return this;
  }

  withEnv(key: string, value: string): WasiBuilder {
    this.env[key] = value;
    return this;
  }

  withEnvs(env: Record<string, string>): WasiBuilder {
    Object.assign(this.env, env);
    return this;
  }

  withPreopen(guestPath: string, hostPath: string): WasiBuilder {
    this.preopens[guestPath] = hostPath;
    return this;
  }

  withStdin(input: string): WasiBuilder {
    this.stdin = input;
    return this;
  }

  withStdout(mode: "inherit" | "capture" | "null"): WasiBuilder {
    this.stdout = mode;
    return this;
  }

  withStderr(mode: "inherit" | "capture" | "null"): WasiBuilder {
    this.stderr = mode;
    return this;
  }

  build(): WasmModuleConfig["wasi"] {
    return {
      args: this.args,
      env: this.env,
      preopenDirs: this.preopens,
      stdin: this.stdin,
      stdout: this.stdout,
      stderr: this.stderr,
    };
  }
}

// =============================================================================
// Convenience Exports
// =============================================================================

export function createWasmtimeManager(options?: {
  cacheDir?: string;
  defaultLimits?: WasmModuleConfig["limits"];
}): WasmtimeManager {
  return new WasmtimeManager(options);
}

export function wasi(): WasiBuilder {
  return new WasiBuilder();
}
