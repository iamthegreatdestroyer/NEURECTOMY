/**
 * NEURECTOMY Docker Client
 *
 * @FLUX @SYNAPSE - DevOps + API Design
 *
 * Production-grade Docker Engine API client with container lifecycle management,
 * streaming logs, image management, and network operations.
 */

import Docker from "dockerode";
import { EventEmitter } from "eventemitter3";
import { PassThrough, Readable } from "stream";
import pino from "pino";
import { v4 as uuidv4 } from "uuid";

import type {
  DockerConfig,
  DockerContainerConfig,
  DockerImageConfig,
  DockerNetworkConfig,
  DockerVolumeConfig,
  ContainerInfo,
  ContainerStatus,
  HealthStatus,
  ResourceUsage,
  ContainerEvent,
  ContainerEventType,
  PortMapping,
  Mount,
  NetworkInfo,
} from "../types";

// =============================================================================
// Types
// =============================================================================

export interface DockerClientEvents {
  container: (event: ContainerEvent) => void;
  image: (event: ImageEvent) => void;
  network: (event: NetworkEvent) => void;
  volume: (event: VolumeEvent) => void;
  error: (error: Error) => void;
  connected: () => void;
  disconnected: () => void;
}

interface ImageEvent {
  type: "pull" | "push" | "build" | "delete" | "tag" | "untag";
  imageId: string;
  imageName: string;
  timestamp: Date;
}

interface NetworkEvent {
  type: "create" | "connect" | "disconnect" | "destroy";
  networkId: string;
  networkName: string;
  containerId?: string;
  timestamp: Date;
}

interface VolumeEvent {
  type: "create" | "mount" | "unmount" | "destroy";
  volumeName: string;
  timestamp: Date;
}

interface BuildProgress {
  stream?: string;
  status?: string;
  progress?: string;
  error?: string;
  aux?: { ID?: string };
}

interface ContainerStats {
  cpu_stats: {
    cpu_usage: { total_usage: number };
    system_cpu_usage: number;
    online_cpus?: number;
  };
  precpu_stats: {
    cpu_usage: { total_usage: number };
    system_cpu_usage: number;
  };
  memory_stats: {
    usage: number;
    limit: number;
    stats?: { cache?: number };
  };
  networks?: Record<string, { rx_bytes: number; tx_bytes: number }>;
  blkio_stats?: {
    io_service_bytes_recursive?: { op: string; value: number }[];
  };
  pids_stats?: { current?: number };
}

// =============================================================================
// Docker Client Implementation
// =============================================================================

export class DockerClient extends EventEmitter<DockerClientEvents> {
  private docker: Docker;
  private logger: pino.Logger;
  private eventStream?: Readable;
  private isConnected: boolean = false;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private reconnectDelay: number = 1000;

  constructor(config?: DockerConfig) {
    super();

    this.logger = pino({
      name: "docker-client",
      level: process.env.LOG_LEVEL || "info",
    });

    const dockerOptions: Docker.DockerOptions = {};

    if (config?.socketPath) {
      dockerOptions.socketPath = config.socketPath;
    } else if (config?.host) {
      dockerOptions.host = config.host;
      dockerOptions.port = config.port;
      if (config.ca && config.cert && config.key) {
        dockerOptions.ca = config.ca;
        dockerOptions.cert = config.cert;
        dockerOptions.key = config.key;
      }
    }

    if (config?.version) {
      dockerOptions.version = config.version;
    }

    if (config?.timeout) {
      dockerOptions.timeout = config.timeout;
    }

    this.docker = new Docker(dockerOptions);
  }

  // ===========================================================================
  // Connection Management
  // ===========================================================================

  async connect(): Promise<void> {
    try {
      await this.docker.ping();
      this.isConnected = true;
      this.reconnectAttempts = 0;
      this.logger.info("Connected to Docker daemon");
      this.emit("connected");
      await this.startEventStream();
    } catch (error) {
      this.logger.error({ error }, "Failed to connect to Docker daemon");
      throw new DockerConnectionError(
        "Failed to connect to Docker daemon",
        error as Error
      );
    }
  }

  async disconnect(): Promise<void> {
    if (this.eventStream) {
      this.eventStream.destroy();
      this.eventStream = undefined;
    }
    this.isConnected = false;
    this.emit("disconnected");
    this.logger.info("Disconnected from Docker daemon");
  }

  async ping(): Promise<boolean> {
    try {
      await this.docker.ping();
      return true;
    } catch {
      return false;
    }
  }

  async getInfo(): Promise<Awaited<ReturnType<Docker["info"]>>> {
    return this.docker.info();
  }

  async getVersion(): Promise<Awaited<ReturnType<Docker["version"]>>> {
    return this.docker.version();
  }

  // ===========================================================================
  // Container Lifecycle Management
  // ===========================================================================

  async createContainer(config: DockerContainerConfig): Promise<ContainerInfo> {
    this.logger.info(
      { name: config.name, image: config.image },
      "Creating container"
    );

    const containerOptions: Docker.ContainerCreateOptions = {
      name: config.name,
      Image: config.tag ? `${config.image}:${config.tag}` : config.image,
      Cmd: config.cmd,
      Entrypoint: config.entrypoint,
      Env: config.env
        ? Object.entries(config.env).map(([k, v]) => `${k}=${v}`)
        : undefined,
      Labels: {
        ...config.labels,
        "neurectomy.managed": "true",
        "neurectomy.created": new Date().toISOString(),
      },
      WorkingDir: config.workingDir,
      User: config.user,
      ExposedPorts: config.exposedPorts
        ? Object.fromEntries(config.exposedPorts.map((p) => [`${p}/tcp`, {}]))
        : undefined,
      Hostname: config.hostname,
      Domainname: config.domainname,
      Tty: config.tty,
      OpenStdin: config.stdin,
      AttachStdout: config.attachStdout,
      AttachStderr: config.attachStderr,
      HostConfig: {
        PortBindings: this.buildPortBindings(config.portBindings),
        Binds: this.buildBinds(config.mounts),
        NetworkMode: config.networkMode,
        Privileged: config.privileged,
        ReadonlyRootfs: config.readonlyRootfs,
        AutoRemove: config.autoRemove,
        RestartPolicy: config.restart
          ? {
              Name: config.restart,
              MaximumRetryCount: config.restart === "on-failure" ? 3 : 0,
            }
          : undefined,
        // Resource constraints (flattened into HostConfig per dockerode types)
        CpuShares: config.resources?.cpuShares,
        CpuPeriod: config.resources?.cpuPeriod,
        CpuQuota: config.resources?.cpuQuota,
        CpusetCpus: config.resources?.cpusetCpus,
        Memory: config.resources?.memory,
        MemorySwap: config.resources?.memorySwap,
        MemoryReservation: config.resources?.memoryReservation,
        PidsLimit: config.resources?.pidsLimit,
      },
      Healthcheck: config.healthcheck
        ? {
            Test: config.healthcheck.test,
            Interval: config.healthcheck.interval
              ? config.healthcheck.interval * 1e9
              : undefined,
            Timeout: config.healthcheck.timeout
              ? config.healthcheck.timeout * 1e9
              : undefined,
            Retries: config.healthcheck.retries,
            StartPeriod: config.healthcheck.startPeriod
              ? config.healthcheck.startPeriod * 1e9
              : undefined,
          }
        : undefined,
    };

    const container = await this.docker.createContainer(containerOptions);
    return this.inspectContainer(container.id);
  }

  async startContainer(containerId: string): Promise<void> {
    this.logger.info({ containerId }, "Starting container");
    const container = this.docker.getContainer(containerId);
    await container.start();
  }

  async stopContainer(
    containerId: string,
    timeout: number = 10
  ): Promise<void> {
    this.logger.info({ containerId, timeout }, "Stopping container");
    const container = this.docker.getContainer(containerId);
    await container.stop({ t: timeout });
  }

  async restartContainer(
    containerId: string,
    timeout: number = 10
  ): Promise<void> {
    this.logger.info({ containerId, timeout }, "Restarting container");
    const container = this.docker.getContainer(containerId);
    await container.restart({ t: timeout });
  }

  async pauseContainer(containerId: string): Promise<void> {
    this.logger.info({ containerId }, "Pausing container");
    const container = this.docker.getContainer(containerId);
    await container.pause();
  }

  async unpauseContainer(containerId: string): Promise<void> {
    this.logger.info({ containerId }, "Unpausing container");
    const container = this.docker.getContainer(containerId);
    await container.unpause();
  }

  async killContainer(
    containerId: string,
    signal: string = "SIGKILL"
  ): Promise<void> {
    this.logger.info({ containerId, signal }, "Killing container");
    const container = this.docker.getContainer(containerId);
    await container.kill({ signal });
  }

  async removeContainer(
    containerId: string,
    options: { force?: boolean; removeVolumes?: boolean } = {}
  ): Promise<void> {
    this.logger.info({ containerId, options }, "Removing container");
    const container = this.docker.getContainer(containerId);
    await container.remove({ force: options.force, v: options.removeVolumes });
  }

  async inspectContainer(containerId: string): Promise<ContainerInfo> {
    const container = this.docker.getContainer(containerId);
    const info = await container.inspect();

    return {
      id: info.Id,
      name: info.Name.replace(/^\//, ""),
      image: info.Config.Image,
      imageId: info.Image,
      runtime: "docker",
      status: this.mapContainerStatus(info.State.Status),
      health: this.mapHealthStatus(info.State.Health?.Status),
      created: new Date(info.Created),
      started: info.State.StartedAt
        ? new Date(info.State.StartedAt)
        : undefined,
      finished: info.State.FinishedAt
        ? new Date(info.State.FinishedAt)
        : undefined,
      exitCode: info.State.ExitCode,
      labels: info.Config.Labels || {},
      ports: this.extractPorts(info),
      mounts: this.extractMounts(info),
      networks: this.extractNetworks(info),
      resources: {
        cpuPercent: 0,
        memoryUsage: 0,
        memoryLimit: 0,
        memoryPercent: 0,
        networkRx: 0,
        networkTx: 0,
        blockRead: 0,
        blockWrite: 0,
        pids: 0,
      },
    };
  }

  async listContainers(all: boolean = false): Promise<ContainerInfo[]> {
    const containers = await this.docker.listContainers({ all });
    return Promise.all(containers.map((c) => this.inspectContainer(c.Id)));
  }

  // ===========================================================================
  // Container Logs (Streaming)
  // ===========================================================================

  async getLogs(
    containerId: string,
    options: {
      follow?: boolean;
      stdout?: boolean;
      stderr?: boolean;
      since?: number;
      until?: number;
      timestamps?: boolean;
      tail?: number;
    } = {}
  ): Promise<Readable> {
    const container = this.docker.getContainer(containerId);

    // Use type assertion to handle the complex overloaded types
    const logOptions = {
      follow: options.follow ?? false,
      stdout: options.stdout ?? true,
      stderr: options.stderr ?? true,
      since: options.since,
      until: options.until,
      timestamps: options.timestamps ?? true,
      tail: options.tail,
    } as const;

    const stream = (await container.logs(logOptions as any)) as unknown;

    if (typeof stream === "string") {
      const passThrough = new PassThrough();
      passThrough.end(stream);
      return passThrough;
    }

    // Docker multiplexes stdout/stderr in a special format
    const output = new PassThrough();
    this.demuxDockerStream(stream as NodeJS.ReadableStream, output);
    return output;
  }

  private demuxDockerStream(
    input: NodeJS.ReadableStream,
    output: PassThrough
  ): void {
    let header = Buffer.alloc(0);

    input.on("data", (chunk: Buffer) => {
      let buffer = Buffer.concat([header, chunk]);
      header = Buffer.alloc(0);

      while (buffer.length >= 8) {
        const size = buffer.readUInt32BE(4);

        if (buffer.length < 8 + size) {
          header = buffer;
          return;
        }

        const payload = buffer.subarray(8, 8 + size);
        output.write(payload);
        buffer = buffer.subarray(8 + size);
      }

      if (buffer.length > 0) {
        header = buffer;
      }
    });

    input.on("end", () => output.end());
    input.on("error", (err) => output.destroy(err));
  }

  // ===========================================================================
  // Container Stats (Streaming)
  // ===========================================================================

  async getStats(
    containerId: string,
    stream: boolean = false
  ): Promise<ResourceUsage> {
    const container = this.docker.getContainer(containerId);
    // Use type assertion to handle overloaded stats method
    const stats = (await container.stats({
      stream: false,
    } as { stream?: false })) as unknown as ContainerStats;
    return this.calculateResourceUsage(stats);
  }

  streamStats(
    containerId: string
  ): AsyncGenerator<ResourceUsage, void, unknown> {
    const container = this.docker.getContainer(containerId);
    const self = this;

    return (async function* () {
      const stream = await container.stats({ stream: true } as {
        stream: true;
      });

      for await (const chunk of stream as AsyncIterable<Buffer>) {
        try {
          const stats: ContainerStats = JSON.parse(chunk.toString());
          yield self.calculateResourceUsage(stats);
        } catch {
          // Skip malformed JSON
        }
      }
    })();
  }

  private calculateResourceUsage(stats: ContainerStats): ResourceUsage {
    // CPU percentage calculation
    const cpuDelta =
      stats.cpu_stats.cpu_usage.total_usage -
      stats.precpu_stats.cpu_usage.total_usage;
    const systemDelta =
      stats.cpu_stats.system_cpu_usage - stats.precpu_stats.system_cpu_usage;
    const cpuCount = stats.cpu_stats.online_cpus || 1;
    const cpuPercent =
      systemDelta > 0 ? (cpuDelta / systemDelta) * cpuCount * 100 : 0;

    // Memory calculation
    const memoryUsage =
      stats.memory_stats.usage - (stats.memory_stats.stats?.cache || 0);
    const memoryLimit = stats.memory_stats.limit;
    const memoryPercent =
      memoryLimit > 0 ? (memoryUsage / memoryLimit) * 100 : 0;

    // Network I/O
    let networkRx = 0;
    let networkTx = 0;
    if (stats.networks) {
      for (const network of Object.values(stats.networks)) {
        networkRx += network.rx_bytes;
        networkTx += network.tx_bytes;
      }
    }

    // Block I/O
    let blockRead = 0;
    let blockWrite = 0;
    if (stats.blkio_stats?.io_service_bytes_recursive) {
      for (const io of stats.blkio_stats.io_service_bytes_recursive) {
        if (io.op === "Read") blockRead += io.value;
        if (io.op === "Write") blockWrite += io.value;
      }
    }

    return {
      cpuPercent,
      memoryUsage,
      memoryLimit,
      memoryPercent,
      networkRx,
      networkTx,
      blockRead,
      blockWrite,
      pids: stats.pids_stats?.current || 0,
    };
  }

  // ===========================================================================
  // Container Exec
  // ===========================================================================

  async exec(
    containerId: string,
    command: string[],
    options: {
      attachStdin?: boolean;
      attachStdout?: boolean;
      attachStderr?: boolean;
      tty?: boolean;
      env?: string[];
      workingDir?: string;
      user?: string;
      privileged?: boolean;
    } = {}
  ): Promise<{ exitCode: number; output: string }> {
    const container = this.docker.getContainer(containerId);

    const exec = await container.exec({
      Cmd: command,
      AttachStdin: options.attachStdin ?? false,
      AttachStdout: options.attachStdout ?? true,
      AttachStderr: options.attachStderr ?? true,
      Tty: options.tty ?? false,
      Env: options.env,
      WorkingDir: options.workingDir,
      User: options.user,
      Privileged: options.privileged,
    });

    const stream = await exec.start({
      Detach: false,
      Tty: options.tty ?? false,
    });

    const output = await this.collectExecOutput(stream, options.tty ?? false);
    const inspectResult = await exec.inspect();

    return {
      exitCode: inspectResult.ExitCode ?? -1,
      output,
    };
  }

  private async collectExecOutput(
    stream: NodeJS.ReadWriteStream,
    isTty: boolean
  ): Promise<string> {
    return new Promise((resolve, reject) => {
      const chunks: Buffer[] = [];

      if (isTty) {
        stream.on("data", (chunk: Buffer) => chunks.push(chunk));
      } else {
        const stdout = new PassThrough();
        this.demuxDockerStream(stream, stdout);
        stdout.on("data", (chunk: Buffer) => chunks.push(chunk));
      }

      stream.on("end", () => resolve(Buffer.concat(chunks).toString()));
      stream.on("error", reject);
    });
  }

  // ===========================================================================
  // Image Management
  // ===========================================================================

  async pullImage(
    repository: string,
    tag: string = "latest",
    onProgress?: (progress: { status: string; progress?: string }) => void
  ): Promise<void> {
    this.logger.info({ repository, tag }, "Pulling image");

    const stream = await this.docker.pull(`${repository}:${tag}`);

    return new Promise((resolve, reject) => {
      this.docker.modem.followProgress(
        stream,
        (err: Error | null) => {
          if (err) {
            this.logger.error({ err, repository, tag }, "Failed to pull image");
            reject(err);
          } else {
            this.logger.info({ repository, tag }, "Image pulled successfully");
            this.emit("image", {
              type: "pull",
              imageId: "",
              imageName: `${repository}:${tag}`,
              timestamp: new Date(),
            });
            resolve();
          }
        },
        (event: { status: string; progress?: string }) => {
          if (onProgress) {
            onProgress(event);
          }
        }
      );
    });
  }

  async buildImage(
    config: DockerImageConfig,
    onProgress?: (progress: BuildProgress) => void
  ): Promise<string> {
    this.logger.info(
      { repository: config.repository, tag: config.tag },
      "Building image"
    );

    const buildOptions: Record<string, unknown> = {
      t: config.tag ? `${config.repository}:${config.tag}` : config.repository,
      dockerfile: config.dockerfile || "Dockerfile",
      buildargs: config.buildArgs,
      labels: config.labels,
      target: config.target,
      platform: config.platform,
      nocache: !config.cache,
      pull: config.pull,
      squash: config.squash,
    };

    const stream = await this.docker.buildImage(
      { context: config.context || "." } as unknown as NodeJS.ReadableStream,
      buildOptions
    );

    return new Promise((resolve, reject) => {
      let imageId = "";

      this.docker.modem.followProgress(
        stream,
        (err: Error | null) => {
          if (err) {
            this.logger.error({ err, config }, "Failed to build image");
            reject(err);
          } else {
            this.logger.info({ imageId, config }, "Image built successfully");
            this.emit("image", {
              type: "build",
              imageId,
              imageName: config.repository,
              timestamp: new Date(),
            });
            resolve(imageId);
          }
        },
        (event: BuildProgress) => {
          if (event.aux?.ID) {
            imageId = event.aux.ID;
          }
          if (onProgress) {
            onProgress(event);
          }
        }
      );
    });
  }

  async removeImage(imageId: string, force: boolean = false): Promise<void> {
    this.logger.info({ imageId, force }, "Removing image");
    const image = this.docker.getImage(imageId);
    await image.remove({ force });
    this.emit("image", {
      type: "delete",
      imageId,
      imageName: imageId,
      timestamp: new Date(),
    });
  }

  async listImages(): Promise<Docker.ImageInfo[]> {
    return this.docker.listImages();
  }

  async inspectImage(imageId: string): Promise<Docker.ImageInspectInfo> {
    const image = this.docker.getImage(imageId);
    return image.inspect();
  }

  async tagImage(imageId: string, repo: string, tag: string): Promise<void> {
    const image = this.docker.getImage(imageId);
    await image.tag({ repo, tag });
    this.emit("image", {
      type: "tag",
      imageId,
      imageName: `${repo}:${tag}`,
      timestamp: new Date(),
    });
  }

  // ===========================================================================
  // Network Management
  // ===========================================================================

  async createNetwork(config: DockerNetworkConfig): Promise<string> {
    this.logger.info({ name: config.name }, "Creating network");

    const network = await this.docker.createNetwork({
      Name: config.name,
      Driver: config.driver,
      Internal: config.internal,
      Attachable: config.attachable,
      Ingress: config.ingress,
      IPAM: config.ipam
        ? {
            Driver: config.ipam.driver || "default",
            Config: config.ipam.config?.map((c) => ({
              Subnet: c.subnet,
              IPRange: c.ipRange,
              Gateway: c.gateway,
              AuxiliaryAddresses: c.auxAddress,
            })),
          }
        : undefined,
      Options: config.options,
      Labels: config.labels,
    });

    // Handle the network response properly
    const networkId =
      typeof network === "object" && "id" in network
        ? (network as { id: string }).id
        : String(network);

    this.emit("network", {
      type: "create",
      networkId,
      networkName: config.name,
      timestamp: new Date(),
    });

    return networkId;
  }

  async removeNetwork(networkId: string): Promise<void> {
    this.logger.info({ networkId }, "Removing network");
    const network = this.docker.getNetwork(networkId);
    await network.remove();
    this.emit("network", {
      type: "destroy",
      networkId,
      networkName: "",
      timestamp: new Date(),
    });
  }

  async connectNetwork(networkId: string, containerId: string): Promise<void> {
    this.logger.info(
      { networkId, containerId },
      "Connecting container to network"
    );
    const network = this.docker.getNetwork(networkId);
    await network.connect({ Container: containerId });
    this.emit("network", {
      type: "connect",
      networkId,
      networkName: "",
      containerId,
      timestamp: new Date(),
    });
  }

  async disconnectNetwork(
    networkId: string,
    containerId: string,
    force: boolean = false
  ): Promise<void> {
    this.logger.info(
      { networkId, containerId },
      "Disconnecting container from network"
    );
    const network = this.docker.getNetwork(networkId);
    await network.disconnect({ Container: containerId, Force: force });
    this.emit("network", {
      type: "disconnect",
      networkId,
      networkName: "",
      containerId,
      timestamp: new Date(),
    });
  }

  async listNetworks(): Promise<Docker.NetworkInspectInfo[]> {
    return this.docker.listNetworks();
  }

  async inspectNetwork(networkId: string): Promise<Docker.NetworkInspectInfo> {
    const network = this.docker.getNetwork(networkId);
    return network.inspect();
  }

  // ===========================================================================
  // Volume Management
  // ===========================================================================

  async createVolume(config: DockerVolumeConfig): Promise<string> {
    this.logger.info({ name: config.name }, "Creating volume");

    const volume = await this.docker.createVolume({
      Name: config.name,
      Driver: config.driver || "local",
      DriverOpts: config.driverOpts,
      Labels: config.labels,
    });

    this.emit("volume", {
      type: "create",
      volumeName: config.name,
      timestamp: new Date(),
    });

    return volume.Name;
  }

  async removeVolume(
    volumeName: string,
    force: boolean = false
  ): Promise<void> {
    this.logger.info({ volumeName, force }, "Removing volume");
    const volume = this.docker.getVolume(volumeName);
    await volume.remove({ force });
    this.emit("volume", {
      type: "destroy",
      volumeName,
      timestamp: new Date(),
    });
  }

  async listVolumes(): Promise<Docker.VolumeInspectInfo[]> {
    const result = await this.docker.listVolumes();
    return result.Volumes || [];
  }

  async inspectVolume(volumeName: string): Promise<Docker.VolumeInspectInfo> {
    const volume = this.docker.getVolume(volumeName);
    return volume.inspect();
  }

  // ===========================================================================
  // Event Streaming
  // ===========================================================================

  private async startEventStream(): Promise<void> {
    try {
      const stream = await this.docker.getEvents();
      this.eventStream = stream as unknown as Readable;

      this.eventStream?.on("data", (chunk: Buffer) => {
        try {
          const event = JSON.parse(chunk.toString());
          this.handleDockerEvent(event);
        } catch {
          // Skip malformed events
        }
      });

      this.eventStream?.on("error", async (err: Error) => {
        this.logger.error({ err }, "Event stream error");
        this.emit("error", err);
        await this.handleReconnect();
      });

      this.eventStream?.on("end", async () => {
        this.logger.warn("Event stream ended");
        await this.handleReconnect();
      });
    } catch (error) {
      this.logger.error({ error }, "Failed to start event stream");
      throw error;
    }
  }

  private handleDockerEvent(event: {
    Type: string;
    Action: string;
    Actor: { ID: string; Attributes: Record<string, string> };
    time: number;
  }): void {
    const timestamp = new Date(event.time * 1000);

    switch (event.Type) {
      case "container":
        this.emit("container", {
          type: event.Action as ContainerEventType,
          containerId: event.Actor.ID,
          containerName: event.Actor.Attributes.name || "",
          runtime: "docker",
          timestamp,
          attributes: event.Actor.Attributes,
        });
        break;

      case "image":
        this.emit("image", {
          type: event.Action as ImageEvent["type"],
          imageId: event.Actor.ID,
          imageName: event.Actor.Attributes.name || "",
          timestamp,
        });
        break;

      case "network":
        this.emit("network", {
          type: event.Action as NetworkEvent["type"],
          networkId: event.Actor.ID,
          networkName: event.Actor.Attributes.name || "",
          containerId: event.Actor.Attributes.container,
          timestamp,
        });
        break;

      case "volume":
        this.emit("volume", {
          type: event.Action as VolumeEvent["type"],
          volumeName: event.Actor.Attributes.name || event.Actor.ID,
          timestamp,
        });
        break;
    }
  }

  private async handleReconnect(): Promise<void> {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.logger.error("Max reconnection attempts reached");
      this.isConnected = false;
      this.emit("disconnected");
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    this.logger.info(
      { attempt: this.reconnectAttempts, delay },
      "Attempting to reconnect"
    );

    await new Promise((resolve) => setTimeout(resolve, delay));

    try {
      await this.connect();
    } catch (error) {
      this.logger.error({ error }, "Reconnection failed");
      await this.handleReconnect();
    }
  }

  // ===========================================================================
  // Helper Methods
  // ===========================================================================

  private mapContainerStatus(status: string): ContainerStatus {
    const statusMap: Record<string, ContainerStatus> = {
      created: "created",
      running: "running",
      paused: "paused",
      restarting: "restarting",
      removing: "removing",
      exited: "exited",
      dead: "dead",
    };
    return statusMap[status] || "exited";
  }

  private mapHealthStatus(status?: string): HealthStatus {
    const healthMap: Record<string, HealthStatus> = {
      starting: "starting",
      healthy: "healthy",
      unhealthy: "unhealthy",
    };
    return status ? healthMap[status] || "none" : "none";
  }

  private buildPortBindings(
    portMappings?: PortMapping[]
  ): Record<string, { HostPort: string; HostIp?: string }[]> | undefined {
    if (!portMappings || portMappings.length === 0) {
      return undefined;
    }

    const bindings: Record<string, { HostPort: string; HostIp?: string }[]> =
      {};

    for (const mapping of portMappings) {
      const key = `${mapping.containerPort}/${mapping.protocol}`;
      bindings[key] = [
        {
          HostPort: String(mapping.hostPort),
          HostIp: mapping.hostIp,
        },
      ];
    }

    return bindings;
  }

  private buildBinds(mounts?: Mount[]): string[] | undefined {
    if (!mounts || mounts.length === 0) {
      return undefined;
    }

    return mounts.map((mount) => {
      const mode = mount.readOnly ? "ro" : "rw";
      return `${mount.source}:${mount.target}:${mode}`;
    });
  }

  private extractPorts(info: Docker.ContainerInspectInfo): PortMapping[] {
    const ports: PortMapping[] = [];
    const portBindings = info.HostConfig.PortBindings || {};

    for (const [containerPort, bindings] of Object.entries(portBindings)) {
      if (!bindings || !Array.isArray(bindings)) continue;

      const [port, protocol] = containerPort.split("/");

      for (const binding of bindings as Array<{
        HostIp: string;
        HostPort: string;
      }>) {
        ports.push({
          containerPort: parseInt(port, 10),
          hostPort: parseInt(binding.HostPort, 10),
          protocol: (protocol as "tcp" | "udp") || "tcp",
          hostIp: binding.HostIp || undefined,
        });
      }
    }

    return ports;
  }

  private extractMounts(info: Docker.ContainerInspectInfo): Mount[] {
    const mounts = info.Mounts || [];

    return mounts.map((mount) => ({
      type: mount.Type as "bind" | "volume" | "tmpfs",
      source: mount.Source,
      target: mount.Destination,
      readOnly: !mount.RW,
    }));
  }

  private extractNetworks(info: Docker.ContainerInspectInfo): NetworkInfo[] {
    const networks: NetworkInfo[] = [];
    const networkSettings = info.NetworkSettings?.Networks || {};

    for (const [name, network] of Object.entries(networkSettings)) {
      if (!network) continue;

      networks.push({
        name,
        networkId: network.NetworkID,
        ipAddress: network.IPAddress,
        gateway: network.Gateway,
        macAddress: network.MacAddress,
      });
    }

    return networks;
  }
}

// =============================================================================
// Custom Errors
// =============================================================================

export class DockerConnectionError extends Error {
  constructor(
    message: string,
    public readonly cause?: Error
  ) {
    super(message);
    this.name = "DockerConnectionError";
  }
}

export class DockerContainerError extends Error {
  constructor(
    message: string,
    public readonly containerId: string,
    public readonly cause?: Error
  ) {
    super(message);
    this.name = "DockerContainerError";
  }
}

export class DockerImageError extends Error {
  constructor(
    message: string,
    public readonly imageName: string,
    public readonly cause?: Error
  ) {
    super(message);
    this.name = "DockerImageError";
  }
}
