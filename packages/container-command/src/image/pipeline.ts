/**
 * NEURECTOMY Image Pipeline
 *
 * @FLUX @FORGE - DevOps + Build Systems
 *
 * Multi-stage image build pipeline with layer caching,
 * security scanning, and registry management.
 */

import Docker from "dockerode";
import { EventEmitter } from "eventemitter3";
import * as fs from "fs/promises";
import * as path from "path";
import * as crypto from "crypto";
import pino from "pino";
import tar from "tar-stream";
import { z } from "zod";

// =============================================================================
// Types
// =============================================================================

export interface BuildConfig {
  context: string;
  dockerfile?: string;
  tags: string[];
  buildArgs?: Record<string, string>;
  target?: string;
  platform?: string;
  labels?: Record<string, string>;
  cacheFrom?: string[];
  cacheTo?: string;
  secrets?: Array<{
    id: string;
    source: string;
  }>;
  ssh?: Array<{
    id: string;
    source?: string;
  }>;
  network?: string;
  squash?: boolean;
  pull?: boolean;
  noCache?: boolean;
}

export interface BuildStage {
  name: string;
  baseImage: string;
  commands: string[];
  startLine: number;
  endLine: number;
}

export interface LayerInfo {
  id: string;
  size: number;
  created: Date;
  createdBy: string;
  cached: boolean;
}

export interface BuildProgress {
  stage: string;
  step: number;
  totalSteps: number;
  message: string;
  cached: boolean;
}

export interface ScanResult {
  vulnerabilities: Vulnerability[];
  summary: VulnerabilitySummary;
  scannedAt: Date;
  scanner: string;
}

export interface Vulnerability {
  id: string;
  package: string;
  version: string;
  fixedVersion?: string;
  severity: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN";
  title: string;
  description: string;
  references: string[];
}

export interface VulnerabilitySummary {
  critical: number;
  high: number;
  medium: number;
  low: number;
  unknown: number;
  total: number;
}

export interface RegistryConfig {
  url: string;
  username?: string;
  password?: string;
  email?: string;
  serveraddress?: string;
}

export interface PushProgress {
  status: string;
  progress?: string;
  progressDetail?: {
    current: number;
    total: number;
  };
  id?: string;
}

export interface ImagePipelineEvents {
  "build:start": (config: BuildConfig) => void;
  "build:progress": (progress: BuildProgress) => void;
  "build:stage": (stage: BuildStage) => void;
  "build:complete": (imageId: string, tags: string[]) => void;
  "build:error": (error: Error) => void;
  "scan:start": (imageId: string) => void;
  "scan:progress": (message: string) => void;
  "scan:complete": (result: ScanResult) => void;
  "scan:error": (error: Error) => void;
  "push:start": (tag: string) => void;
  "push:progress": (progress: PushProgress) => void;
  "push:complete": (tag: string, digest: string) => void;
  "push:error": (error: Error) => void;
  "cache:hit": (layer: string) => void;
  "cache:miss": (layer: string) => void;
}

// =============================================================================
// Validation Schemas
// =============================================================================

const BuildConfigSchema = z.object({
  context: z.string().min(1),
  dockerfile: z.string().optional(),
  tags: z.array(z.string().min(1)).min(1),
  buildArgs: z.record(z.string()).optional(),
  target: z.string().optional(),
  platform: z.string().optional(),
  labels: z.record(z.string()).optional(),
  cacheFrom: z.array(z.string()).optional(),
  cacheTo: z.string().optional(),
  secrets: z
    .array(
      z.object({
        id: z.string(),
        source: z.string(),
      })
    )
    .optional(),
  ssh: z
    .array(
      z.object({
        id: z.string(),
        source: z.string().optional(),
      })
    )
    .optional(),
  network: z.string().optional(),
  squash: z.boolean().optional(),
  pull: z.boolean().optional(),
  noCache: z.boolean().optional(),
});

// =============================================================================
// Image Pipeline Implementation
// =============================================================================

export class ImagePipeline extends EventEmitter<ImagePipelineEvents> {
  private docker: Docker;
  private logger: pino.Logger;
  private cacheDir: string;
  private scannerPath?: string;

  constructor(
    options: {
      docker?: Docker;
      cacheDir?: string;
      scannerPath?: string;
    } = {}
  ) {
    super();

    this.docker = options.docker || new Docker();
    this.cacheDir = options.cacheDir || "/tmp/neurectomy-image-cache";
    this.scannerPath = options.scannerPath;

    this.logger = pino({
      name: "image-pipeline",
      level: process.env.LOG_LEVEL || "info",
    });
  }

  // ===========================================================================
  // Build Operations
  // ===========================================================================

  async build(config: BuildConfig): Promise<{
    imageId: string;
    tags: string[];
    stages: BuildStage[];
    layers: LayerInfo[];
    duration: number;
  }> {
    // Validate config
    const validated = BuildConfigSchema.parse(config);

    this.logger.info({ tags: validated.tags }, "Starting image build");
    this.emit("build:start", validated);

    const startTime = Date.now();

    try {
      // Parse Dockerfile to extract stages
      const dockerfilePath = path.join(
        validated.context,
        validated.dockerfile || "Dockerfile"
      );
      const dockerfileContent = await fs.readFile(dockerfilePath, "utf-8");
      const stages = this.parseDockerfile(dockerfileContent);

      // Create build context tar
      const contextTar = await this.createBuildContext(validated);

      // Build options
      const buildOptions: Docker.ImageBuildOptions = {
        t: validated.tags[0],
        dockerfile: validated.dockerfile,
        buildargs: validated.buildArgs,
        target: validated.target,
        platform: validated.platform,
        labels: {
          ...validated.labels,
          "neurectomy.built-at": new Date().toISOString(),
          "neurectomy.builder": "image-pipeline",
        },
        cachefrom: validated.cacheFrom
          ? JSON.stringify(validated.cacheFrom)
          : undefined,
        networkmode: validated.network,
        squash: validated.squash,
        pull: validated.pull,
        nocache: validated.noCache,
      };

      // Perform build
      const stream = await this.docker.buildImage(contextTar, buildOptions);

      // Process build output
      const { imageId, layers } = await this.processBuildStream(stream, stages);

      // Tag additional tags
      const image = this.docker.getImage(imageId);
      for (const tag of validated.tags.slice(1)) {
        const [repo, tagPart] = this.parseImageTag(tag);
        await image.tag({ repo, tag: tagPart });
      }

      const duration = Date.now() - startTime;

      this.logger.info(
        { imageId, tags: validated.tags, duration },
        "Build completed successfully"
      );
      this.emit("build:complete", imageId, validated.tags);

      return {
        imageId,
        tags: validated.tags,
        stages,
        layers,
        duration,
      };
    } catch (error) {
      this.logger.error({ error }, "Build failed");
      this.emit("build:error", error as Error);
      throw error;
    }
  }

  async buildMultiPlatform(
    config: BuildConfig,
    platforms: string[]
  ): Promise<Map<string, string>> {
    this.logger.info(
      { tags: config.tags, platforms },
      "Starting multi-platform build"
    );

    const results = new Map<string, string>();

    for (const platform of platforms) {
      const platformTag = `${config.tags[0]}-${platform.replace("/", "-")}`;
      const platformConfig = {
        ...config,
        platform,
        tags: [platformTag],
      };

      const result = await this.build(platformConfig);
      results.set(platform, result.imageId);
    }

    return results;
  }

  // ===========================================================================
  // Security Scanning
  // ===========================================================================

  async scan(imageRef: string): Promise<ScanResult> {
    this.logger.info({ imageRef }, "Starting security scan");
    this.emit("scan:start", imageRef);

    try {
      // Try using Trivy if available
      const result = await this.runTrivyScan(imageRef);

      this.emit("scan:complete", result);
      return result;
    } catch (error) {
      // Fall back to basic scan using Docker inspect
      this.logger.warn(
        { error },
        "Trivy not available, using basic inspection"
      );

      const result = await this.runBasicScan(imageRef);
      this.emit("scan:complete", result);
      return result;
    }
  }

  private async runTrivyScan(imageRef: string): Promise<ScanResult> {
    const { exec } = await import("child_process");
    const { promisify } = await import("util");
    const execAsync = promisify(exec);

    const trivyCmd = this.scannerPath || "trivy";

    const { stdout } = await execAsync(
      `${trivyCmd} image --format json ${imageRef}`,
      { maxBuffer: 50 * 1024 * 1024 }
    );

    const trivyResult = JSON.parse(stdout);

    const vulnerabilities: Vulnerability[] = [];
    const summary: VulnerabilitySummary = {
      critical: 0,
      high: 0,
      medium: 0,
      low: 0,
      unknown: 0,
      total: 0,
    };

    for (const result of trivyResult.Results || []) {
      for (const vuln of result.Vulnerabilities || []) {
        const severity = vuln.Severity?.toUpperCase() || "UNKNOWN";

        vulnerabilities.push({
          id: vuln.VulnerabilityID,
          package: vuln.PkgName,
          version: vuln.InstalledVersion,
          fixedVersion: vuln.FixedVersion,
          severity: severity as Vulnerability["severity"],
          title: vuln.Title || vuln.VulnerabilityID,
          description: vuln.Description || "",
          references: vuln.References || [],
        });

        switch (severity) {
          case "CRITICAL":
            summary.critical++;
            break;
          case "HIGH":
            summary.high++;
            break;
          case "MEDIUM":
            summary.medium++;
            break;
          case "LOW":
            summary.low++;
            break;
          default:
            summary.unknown++;
        }
        summary.total++;
      }
    }

    return {
      vulnerabilities,
      summary,
      scannedAt: new Date(),
      scanner: "trivy",
    };
  }

  private async runBasicScan(imageRef: string): Promise<ScanResult> {
    // Basic scan using Docker inspect - limited vulnerability detection
    const image = this.docker.getImage(imageRef);
    const inspection = await image.inspect();

    const vulnerabilities: Vulnerability[] = [];
    const summary: VulnerabilitySummary = {
      critical: 0,
      high: 0,
      medium: 0,
      low: 0,
      unknown: 0,
      total: 0,
    };

    // Check for known insecure configurations
    const config = inspection.Config;

    // Check if running as root
    if (!config.User || config.User === "" || config.User === "root") {
      vulnerabilities.push({
        id: "CONFIG-001",
        package: "container",
        version: "N/A",
        severity: "MEDIUM",
        title: "Container runs as root",
        description:
          "The container is configured to run as root user, which is a security risk.",
        references: [
          "https://docs.docker.com/develop/develop-images/dockerfile_best-practices/",
        ],
      });
      summary.medium++;
      summary.total++;
    }

    // Check for exposed ports
    if (inspection.Config.ExposedPorts) {
      const exposedPorts = Object.keys(inspection.Config.ExposedPorts);
      const dangerousPorts = ["22/tcp", "23/tcp", "3389/tcp"];

      for (const port of exposedPorts) {
        if (dangerousPorts.includes(port)) {
          vulnerabilities.push({
            id: `CONFIG-PORT-${port.split("/")[0]}`,
            package: "network",
            version: "N/A",
            severity: "HIGH",
            title: `Dangerous port exposed: ${port}`,
            description: `Port ${port} is commonly targeted and should not be exposed.`,
            references: [],
          });
          summary.high++;
          summary.total++;
        }
      }
    }

    return {
      vulnerabilities,
      summary,
      scannedAt: new Date(),
      scanner: "basic-inspect",
    };
  }

  // ===========================================================================
  // Push Operations
  // ===========================================================================

  async push(
    imageRef: string,
    registry?: RegistryConfig
  ): Promise<{ digest: string }> {
    this.logger.info({ imageRef }, "Pushing image");
    this.emit("push:start", imageRef);

    try {
      const image = this.docker.getImage(imageRef);

      const authConfig = registry
        ? {
            username: registry.username,
            password: registry.password,
            email: registry.email,
            serveraddress: registry.serveraddress || registry.url,
          }
        : undefined;

      const stream = await image.push({ authconfig: authConfig });

      const digest = await this.processPushStream(stream, imageRef);

      this.logger.info({ imageRef, digest }, "Push completed");
      this.emit("push:complete", imageRef, digest);

      return { digest };
    } catch (error) {
      this.logger.error({ error, imageRef }, "Push failed");
      this.emit("push:error", error as Error);
      throw error;
    }
  }

  async pushAll(
    tags: string[],
    registry?: RegistryConfig
  ): Promise<Map<string, string>> {
    const results = new Map<string, string>();

    for (const tag of tags) {
      const { digest } = await this.push(tag, registry);
      results.set(tag, digest);
    }

    return results;
  }

  // ===========================================================================
  // Registry Operations
  // ===========================================================================

  async login(registry: RegistryConfig): Promise<void> {
    this.logger.info({ url: registry.url }, "Logging into registry");

    await this.docker.checkAuth({
      username: registry.username!,
      password: registry.password!,
      email: registry.email,
      serveraddress: registry.serveraddress || registry.url,
    });

    this.logger.info({ url: registry.url }, "Login successful");
  }

  async listTags(repository: string): Promise<string[]> {
    // Note: This requires registry API access
    const image = this.docker.getImage(repository);
    const inspection = await image.inspect();
    return inspection.RepoTags || [];
  }

  // ===========================================================================
  // Cache Management
  // ===========================================================================

  async warmCache(images: string[]): Promise<void> {
    this.logger.info({ count: images.length }, "Warming cache");

    for (const image of images) {
      try {
        this.logger.debug({ image }, "Pulling image for cache");
        await this.pullImage(image);
        this.emit("cache:hit", image);
      } catch (error) {
        this.logger.warn({ error, image }, "Failed to pull image for cache");
        this.emit("cache:miss", image);
      }
    }
  }

  async pruneCache(
    options: {
      olderThan?: Date;
      keepLast?: number;
      dangling?: boolean;
    } = {}
  ): Promise<{
    spaceReclaimed: number;
    imagesRemoved: number;
  }> {
    this.logger.info({ options }, "Pruning image cache");

    let spaceReclaimed = 0;
    let imagesRemoved = 0;

    const images = await this.docker.listImages({ all: true });

    for (const image of images) {
      const shouldPrune =
        (options.dangling && image.RepoTags?.includes("<none>:<none>")) ||
        (options.olderThan &&
          new Date(image.Created * 1000) < options.olderThan);

      if (shouldPrune) {
        try {
          const img = this.docker.getImage(image.Id);
          await img.remove({ force: false });
          spaceReclaimed += image.Size;
          imagesRemoved++;
        } catch (error) {
          // Image might be in use
          this.logger.debug(
            { error, image: image.Id },
            "Could not remove image"
          );
        }
      }
    }

    this.logger.info(
      { spaceReclaimed, imagesRemoved },
      "Cache pruning completed"
    );

    return { spaceReclaimed, imagesRemoved };
  }

  async exportCache(outputPath: string): Promise<void> {
    await fs.mkdir(this.cacheDir, { recursive: true });

    const cacheManifest = {
      version: 1,
      exportedAt: new Date().toISOString(),
      layers: [] as string[],
    };

    // Get all cached layers
    const images = await this.docker.listImages();
    for (const image of images) {
      if (image.Labels?.["neurectomy.builder"]) {
        cacheManifest.layers.push(image.Id);
      }
    }

    await fs.writeFile(outputPath, JSON.stringify(cacheManifest, null, 2));
    this.logger.info(
      { outputPath, layers: cacheManifest.layers.length },
      "Cache exported"
    );
  }

  // ===========================================================================
  // Utility Methods
  // ===========================================================================

  private async pullImage(imageRef: string): Promise<void> {
    return new Promise((resolve, reject) => {
      this.docker.pull(
        imageRef,
        (err: Error | null, stream: NodeJS.ReadableStream) => {
          if (err) {
            reject(err);
            return;
          }

          this.docker.modem.followProgress(stream, (err: Error | null) => {
            if (err) reject(err);
            else resolve();
          });
        }
      );
    });
  }

  private async createBuildContext(
    config: BuildConfig
  ): Promise<NodeJS.ReadableStream> {
    const pack = tar.pack();

    const addFile = async (
      filePath: string,
      basePath: string
    ): Promise<void> => {
      const relativePath = path.relative(basePath, filePath);
      const stat = await fs.stat(filePath);

      if (stat.isDirectory()) {
        const entries = await fs.readdir(filePath);
        for (const entry of entries) {
          // Skip .git and node_modules by default
          if (entry === ".git" || entry === "node_modules") continue;
          await addFile(path.join(filePath, entry), basePath);
        }
      } else {
        const content = await fs.readFile(filePath);
        pack.entry({ name: relativePath, size: content.length }, content);
      }
    };

    await addFile(config.context, config.context);
    pack.finalize();

    return pack;
  }

  private parseDockerfile(content: string): BuildStage[] {
    const stages: BuildStage[] = [];
    const lines = content.split("\n");

    let currentStage: BuildStage | null = null;
    let lineNumber = 0;

    for (const line of lines) {
      lineNumber++;
      const trimmed = line.trim();

      // Skip empty lines and comments
      if (!trimmed || trimmed.startsWith("#")) continue;

      const fromMatch = trimmed.match(/^FROM\s+(\S+)(?:\s+[Aa][Ss]\s+(\S+))?/i);
      if (fromMatch) {
        if (currentStage) {
          currentStage.endLine = lineNumber - 1;
          stages.push(currentStage);
        }

        currentStage = {
          name: fromMatch[2] || `stage-${stages.length}`,
          baseImage: fromMatch[1],
          commands: [trimmed],
          startLine: lineNumber,
          endLine: lineNumber,
        };
      } else if (currentStage) {
        currentStage.commands.push(trimmed);
      }
    }

    if (currentStage) {
      currentStage.endLine = lineNumber;
      stages.push(currentStage);
    }

    return stages;
  }

  private async processBuildStream(
    stream: NodeJS.ReadableStream,
    stages: BuildStage[]
  ): Promise<{ imageId: string; layers: LayerInfo[] }> {
    return new Promise((resolve, reject) => {
      let imageId = "";
      const layers: LayerInfo[] = [];
      let currentStep = 0;
      let currentStageIndex = 0;

      this.docker.modem.followProgress(
        stream,
        (
          err: Error | null,
          output: Array<{ stream?: string; aux?: { ID: string } }>
        ) => {
          if (err) {
            reject(err);
            return;
          }

          // Extract image ID from output
          for (const item of output) {
            if (item.aux?.ID) {
              imageId = item.aux.ID;
            }
          }

          if (!imageId) {
            reject(new Error("Build completed but no image ID found"));
            return;
          }

          resolve({ imageId, layers });
        },
        (event: {
          stream?: string;
          status?: string;
          id?: string;
          aux?: { ID: string };
        }) => {
          if (event.stream) {
            const stepMatch = event.stream.match(/Step (\d+)\/(\d+)/);
            if (stepMatch) {
              currentStep = parseInt(stepMatch[1]);
              const totalSteps = parseInt(stepMatch[2]);

              // Determine current stage
              while (
                currentStageIndex < stages.length - 1 &&
                currentStep > stages[currentStageIndex].commands.length
              ) {
                currentStageIndex++;
              }

              const cached = event.stream.includes("---> Using cache");

              this.emit("build:progress", {
                stage: stages[currentStageIndex]?.name || "unknown",
                step: currentStep,
                totalSteps,
                message: event.stream.trim(),
                cached,
              });

              if (cached) {
                this.emit("cache:hit", `step-${currentStep}`);
              } else {
                this.emit("cache:miss", `step-${currentStep}`);
              }
            }
          }

          // Track layer creation
          if (event.aux?.ID) {
            layers.push({
              id: event.aux.ID,
              size: 0,
              created: new Date(),
              createdBy: "",
              cached: false,
            });
          }
        }
      );
    });
  }

  private async processPushStream(
    stream: NodeJS.ReadableStream,
    tag: string
  ): Promise<string> {
    return new Promise((resolve, reject) => {
      let digest = "";

      this.docker.modem.followProgress(
        stream,
        (err: Error | null, output: Array<{ aux?: { Digest: string } }>) => {
          if (err) {
            reject(err);
            return;
          }

          // Extract digest from output
          for (const item of output) {
            if (item.aux?.Digest) {
              digest = item.aux.Digest;
            }
          }

          resolve(digest);
        },
        (event: {
          status?: string;
          progress?: string;
          progressDetail?: { current: number; total: number };
          id?: string;
        }) => {
          this.emit("push:progress", {
            status: event.status || "",
            progress: event.progress,
            progressDetail: event.progressDetail,
            id: event.id,
          });
        }
      );
    });
  }

  private parseImageTag(imageRef: string): [string, string] {
    const lastColon = imageRef.lastIndexOf(":");
    const lastSlash = imageRef.lastIndexOf("/");

    // Check if colon is part of registry (e.g., localhost:5000/image)
    if (lastColon > lastSlash) {
      return [
        imageRef.substring(0, lastColon),
        imageRef.substring(lastColon + 1),
      ];
    }

    return [imageRef, "latest"];
  }

  // ===========================================================================
  // Dockerfile Generation
  // ===========================================================================

  generateDockerfile(options: {
    baseImage: string;
    workdir?: string;
    copyFiles?: Array<{ src: string; dest: string }>;
    runCommands?: string[];
    env?: Record<string, string>;
    expose?: number[];
    user?: string;
    entrypoint?: string[];
    cmd?: string[];
    healthcheck?: {
      cmd: string;
      interval?: string;
      timeout?: string;
      retries?: number;
      startPeriod?: string;
    };
    labels?: Record<string, string>;
    multistage?: Array<{
      name: string;
      baseImage: string;
      commands: string[];
    }>;
  }): string {
    const lines: string[] = [];

    // Multi-stage builds
    if (options.multistage) {
      for (const stage of options.multistage) {
        lines.push(`FROM ${stage.baseImage} AS ${stage.name}`);
        for (const cmd of stage.commands) {
          lines.push(cmd);
        }
        lines.push("");
      }
    }

    // Final stage
    lines.push(`FROM ${options.baseImage}`);
    lines.push("");

    // Labels
    if (options.labels) {
      for (const [key, value] of Object.entries(options.labels)) {
        lines.push(`LABEL ${key}="${value}"`);
      }
      lines.push("");
    }

    // Environment
    if (options.env) {
      for (const [key, value] of Object.entries(options.env)) {
        lines.push(`ENV ${key}="${value}"`);
      }
      lines.push("");
    }

    // Workdir
    if (options.workdir) {
      lines.push(`WORKDIR ${options.workdir}`);
      lines.push("");
    }

    // Copy files
    if (options.copyFiles) {
      for (const file of options.copyFiles) {
        lines.push(`COPY ${file.src} ${file.dest}`);
      }
      lines.push("");
    }

    // Run commands
    if (options.runCommands) {
      for (const cmd of options.runCommands) {
        lines.push(`RUN ${cmd}`);
      }
      lines.push("");
    }

    // User
    if (options.user) {
      lines.push(`USER ${options.user}`);
      lines.push("");
    }

    // Expose
    if (options.expose) {
      for (const port of options.expose) {
        lines.push(`EXPOSE ${port}`);
      }
      lines.push("");
    }

    // Healthcheck
    if (options.healthcheck) {
      const hc = options.healthcheck;
      let healthLine = "HEALTHCHECK";
      if (hc.interval) healthLine += ` --interval=${hc.interval}`;
      if (hc.timeout) healthLine += ` --timeout=${hc.timeout}`;
      if (hc.retries) healthLine += ` --retries=${hc.retries}`;
      if (hc.startPeriod) healthLine += ` --start-period=${hc.startPeriod}`;
      healthLine += ` CMD ${hc.cmd}`;
      lines.push(healthLine);
      lines.push("");
    }

    // Entrypoint
    if (options.entrypoint) {
      lines.push(`ENTRYPOINT ${JSON.stringify(options.entrypoint)}`);
    }

    // CMD
    if (options.cmd) {
      lines.push(`CMD ${JSON.stringify(options.cmd)}`);
    }

    return lines.join("\n");
  }

  // ===========================================================================
  // Analysis
  // ===========================================================================

  async analyzeImage(imageRef: string): Promise<{
    size: number;
    layers: LayerInfo[];
    config: {
      env: string[];
      cmd: string[];
      entrypoint: string[];
      exposedPorts: string[];
      workdir: string;
      user: string;
    };
    history: Array<{
      created: Date;
      createdBy: string;
      size: number;
      empty: boolean;
    }>;
  }> {
    const image = this.docker.getImage(imageRef);
    const inspection = await image.inspect();
    const history = (await image.history()) as Array<{
      Id: string;
      Size: number;
      Created: number;
      CreatedBy: string;
    }>;

    const layers: LayerInfo[] = history
      .filter((h) => h.Size > 0)
      .map((h) => ({
        id: h.Id || "",
        size: h.Size,
        created: new Date(h.Created * 1000),
        createdBy: h.CreatedBy,
        cached: false,
      }));

    return {
      size: inspection.Size,
      layers,
      config: {
        env: inspection.Config.Env || [],
        cmd: inspection.Config.Cmd || [],
        entrypoint: Array.isArray(inspection.Config.Entrypoint)
          ? inspection.Config.Entrypoint
          : inspection.Config.Entrypoint
            ? [inspection.Config.Entrypoint]
            : [],
        exposedPorts: Object.keys(inspection.Config.ExposedPorts || {}),
        workdir: inspection.Config.WorkingDir,
        user: inspection.Config.User,
      },
      history: history.map((h) => ({
        created: new Date(h.Created * 1000),
        createdBy: h.CreatedBy,
        size: h.Size,
        empty: h.Size === 0,
      })),
    };
  }

  async calculateContentHash(imageRef: string): Promise<string> {
    const analysis = await this.analyzeImage(imageRef);
    const content = JSON.stringify({
      layers: analysis.layers.map((l) => l.id),
      config: analysis.config,
    });

    return crypto.createHash("sha256").update(content).digest("hex");
  }
}

// =============================================================================
// Convenience Export
// =============================================================================

export const imagePipeline = new ImagePipeline();
