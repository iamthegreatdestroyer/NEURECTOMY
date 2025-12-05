/**
 * NEURECTOMY Docker Compose Orchestrator
 *
 * @FLUX @ARCHITECT - DevOps + Architecture
 *
 * Manages multi-container applications using Docker Compose semantics.
 * Supports service lifecycle, scaling, and dependency management.
 */

import { EventEmitter } from "eventemitter3";
import YAML from "yaml";
import pino from "pino";
import { v4 as uuidv4 } from "uuid";

import { DockerClient } from "./client";
import type {
  DockerContainerConfig,
  DockerNetworkConfig,
  DockerVolumeConfig,
  ContainerInfo,
  PortMapping,
  Mount,
} from "../types";

// =============================================================================
// Types
// =============================================================================

export interface ComposeFile {
  version?: string;
  name?: string;
  services: Record<string, ComposeService>;
  networks?: Record<string, ComposeNetwork>;
  volumes?: Record<string, ComposeVolume>;
  configs?: Record<string, ComposeConfig>;
  secrets?: Record<string, ComposeSecret>;
}

export interface ComposeService {
  image?: string;
  build?: ComposeBuild | string;
  container_name?: string;
  command?: string | string[];
  entrypoint?: string | string[];
  environment?: Record<string, string> | string[];
  env_file?: string | string[];
  expose?: (number | string)[];
  ports?: (string | PortConfig)[];
  volumes?: (string | VolumeConfig)[];
  networks?: (string | NetworkConfig)[] | Record<string, NetworkServiceConfig>;
  depends_on?: string[] | Record<string, DependsOnCondition>;
  healthcheck?: ComposeHealthcheck;
  restart?: "no" | "always" | "unless-stopped" | "on-failure";
  deploy?: ComposeDeploy;
  labels?: Record<string, string> | string[];
  logging?: ComposeLogging;
  working_dir?: string;
  user?: string;
  hostname?: string;
  domainname?: string;
  privileged?: boolean;
  read_only?: boolean;
  tty?: boolean;
  stdin_open?: boolean;
  cap_add?: string[];
  cap_drop?: string[];
  security_opt?: string[];
  sysctls?: Record<string, string>;
  ulimits?: Record<string, { soft: number; hard: number } | number>;
  extra_hosts?: string[];
  dns?: string | string[];
  stop_grace_period?: string;
  stop_signal?: string;
}

export interface ComposeBuild {
  context: string;
  dockerfile?: string;
  args?: Record<string, string>;
  target?: string;
  cache_from?: string[];
  labels?: Record<string, string>;
  shm_size?: string | number;
}

export interface PortConfig {
  target: number;
  published?: number | string;
  protocol?: "tcp" | "udp";
  mode?: "host" | "ingress";
}

export interface VolumeConfig {
  type: "bind" | "volume" | "tmpfs";
  source?: string;
  target: string;
  read_only?: boolean;
  bind?: { propagation?: string };
  volume?: { nocopy?: boolean };
  tmpfs?: { size?: number };
}

export interface NetworkConfig {
  aliases?: string[];
  ipv4_address?: string;
  ipv6_address?: string;
}

export interface NetworkServiceConfig {
  aliases?: string[];
  ipv4_address?: string;
  ipv6_address?: string;
}

export interface DependsOnCondition {
  condition:
    | "service_started"
    | "service_healthy"
    | "service_completed_successfully";
}

export interface ComposeHealthcheck {
  test: string | string[];
  interval?: string;
  timeout?: string;
  retries?: number;
  start_period?: string;
  disable?: boolean;
}

export interface ComposeDeploy {
  replicas?: number;
  resources?: {
    limits?: { cpus?: string; memory?: string };
    reservations?: { cpus?: string; memory?: string };
  };
  restart_policy?: {
    condition?: "none" | "on-failure" | "any";
    delay?: string;
    max_attempts?: number;
    window?: string;
  };
  update_config?: {
    parallelism?: number;
    delay?: string;
    failure_action?: "continue" | "pause" | "rollback";
    order?: "start-first" | "stop-first";
  };
  rollback_config?: {
    parallelism?: number;
    delay?: string;
    failure_action?: "continue" | "pause";
    order?: "start-first" | "stop-first";
  };
  placement?: {
    constraints?: string[];
    preferences?: { spread: string }[];
  };
}

export interface ComposeLogging {
  driver?: string;
  options?: Record<string, string>;
}

export interface ComposeNetwork {
  driver?: string;
  driver_opts?: Record<string, string>;
  external?: boolean | { name: string };
  internal?: boolean;
  attachable?: boolean;
  labels?: Record<string, string>;
  ipam?: {
    driver?: string;
    config?: {
      subnet?: string;
      ip_range?: string;
      gateway?: string;
    }[];
  };
}

export interface ComposeVolume {
  driver?: string;
  driver_opts?: Record<string, string>;
  external?: boolean | { name: string };
  labels?: Record<string, string>;
  name?: string;
}

export interface ComposeConfig {
  file?: string;
  external?: boolean | { name: string };
  name?: string;
}

export interface ComposeSecret {
  file?: string;
  external?: boolean | { name: string };
  name?: string;
}

export interface ComposeProject {
  id: string;
  name: string;
  file: ComposeFile;
  services: Map<string, ServiceInstance>;
  networks: Map<string, string>;
  volumes: Map<string, string>;
  status: "created" | "starting" | "running" | "stopping" | "stopped" | "error";
  createdAt: Date;
  startedAt?: Date;
  stoppedAt?: Date;
}

export interface ServiceInstance {
  name: string;
  replicas: ContainerInfo[];
  desiredReplicas: number;
  status: "pending" | "starting" | "running" | "stopping" | "stopped" | "error";
  lastError?: string;
}

export interface ComposeOrchestatorEvents {
  "project:created": (project: ComposeProject) => void;
  "project:started": (project: ComposeProject) => void;
  "project:stopped": (project: ComposeProject) => void;
  "project:removed": (projectId: string) => void;
  "service:starting": (projectId: string, serviceName: string) => void;
  "service:started": (projectId: string, serviceName: string) => void;
  "service:stopping": (projectId: string, serviceName: string) => void;
  "service:stopped": (projectId: string, serviceName: string) => void;
  "service:scaled": (
    projectId: string,
    serviceName: string,
    replicas: number
  ) => void;
  error: (error: Error, context?: string) => void;
}

// =============================================================================
// Compose Orchestrator
// =============================================================================

export class ComposeOrchestrator extends EventEmitter<ComposeOrchestatorEvents> {
  private docker: DockerClient;
  private logger: pino.Logger;
  private projects: Map<string, ComposeProject> = new Map();

  constructor(docker: DockerClient) {
    super();
    this.docker = docker;
    this.logger = pino({
      name: "compose-orchestrator",
      level: process.env.LOG_LEVEL || "info",
    });
  }

  // ===========================================================================
  // Project Management
  // ===========================================================================

  async createProject(
    composeContent: string | ComposeFile,
    projectName?: string
  ): Promise<ComposeProject> {
    const file =
      typeof composeContent === "string"
        ? (YAML.parse(composeContent) as ComposeFile)
        : composeContent;

    const name =
      projectName || file.name || `neurectomy-${uuidv4().slice(0, 8)}`;

    this.logger.info({ projectName: name }, "Creating compose project");

    // Validate compose file
    this.validateComposeFile(file);

    // Create project
    const project: ComposeProject = {
      id: uuidv4(),
      name,
      file,
      services: new Map(),
      networks: new Map(),
      volumes: new Map(),
      status: "created",
      createdAt: new Date(),
    };

    // Initialize services
    for (const [serviceName, serviceConfig] of Object.entries(file.services)) {
      const replicas = serviceConfig.deploy?.replicas || 1;
      project.services.set(serviceName, {
        name: serviceName,
        replicas: [],
        desiredReplicas: replicas,
        status: "pending",
      });
    }

    // Create networks
    await this.createProjectNetworks(project);

    // Create volumes
    await this.createProjectVolumes(project);

    this.projects.set(project.id, project);
    this.emit("project:created", project);

    return project;
  }

  async startProject(projectId: string): Promise<void> {
    const project = this.getProject(projectId);

    this.logger.info(
      { projectId, projectName: project.name },
      "Starting compose project"
    );

    project.status = "starting";
    project.startedAt = new Date();

    try {
      // Build dependency graph and start in order
      const startOrder = this.getStartOrder(project.file.services);

      for (const serviceName of startOrder) {
        await this.startService(project, serviceName);
      }

      project.status = "running";
      this.emit("project:started", project);
    } catch (error) {
      project.status = "error";
      this.emit(
        "error",
        error as Error,
        `Failed to start project ${project.name}`
      );
      throw error;
    }
  }

  async stopProject(projectId: string, timeout: number = 10): Promise<void> {
    const project = this.getProject(projectId);

    this.logger.info(
      { projectId, projectName: project.name },
      "Stopping compose project"
    );

    project.status = "stopping";

    try {
      // Stop services in reverse dependency order
      const startOrder = this.getStartOrder(project.file.services);
      const stopOrder = [...startOrder].reverse();

      for (const serviceName of stopOrder) {
        await this.stopService(project, serviceName, timeout);
      }

      project.status = "stopped";
      project.stoppedAt = new Date();
      this.emit("project:stopped", project);
    } catch (error) {
      project.status = "error";
      this.emit(
        "error",
        error as Error,
        `Failed to stop project ${project.name}`
      );
      throw error;
    }
  }

  async removeProject(
    projectId: string,
    removeVolumes: boolean = false
  ): Promise<void> {
    const project = this.getProject(projectId);

    this.logger.info(
      { projectId, projectName: project.name, removeVolumes },
      "Removing compose project"
    );

    // Stop if running
    if (project.status === "running" || project.status === "starting") {
      await this.stopProject(projectId);
    }

    // Remove containers
    for (const service of project.services.values()) {
      for (const container of service.replicas) {
        await this.docker.removeContainer(container.id, { force: true });
      }
    }

    // Remove networks
    for (const networkId of project.networks.values()) {
      try {
        await this.docker.removeNetwork(networkId);
      } catch (error) {
        this.logger.warn({ error, networkId }, "Failed to remove network");
      }
    }

    // Remove volumes if requested
    if (removeVolumes) {
      for (const volumeName of project.volumes.values()) {
        try {
          await this.docker.removeVolume(volumeName);
        } catch (error) {
          this.logger.warn({ error, volumeName }, "Failed to remove volume");
        }
      }
    }

    this.projects.delete(projectId);
    this.emit("project:removed", projectId);
  }

  // ===========================================================================
  // Service Management
  // ===========================================================================

  async startService(
    project: ComposeProject,
    serviceName: string
  ): Promise<void> {
    const serviceConfig = project.file.services[serviceName];
    const serviceInstance = project.services.get(serviceName);

    if (!serviceConfig || !serviceInstance) {
      throw new Error(
        `Service ${serviceName} not found in project ${project.name}`
      );
    }

    this.logger.info(
      { projectName: project.name, serviceName },
      "Starting service"
    );
    this.emit("service:starting", project.id, serviceName);

    serviceInstance.status = "starting";

    try {
      // Wait for dependencies
      await this.waitForDependencies(project, serviceName);

      // Create and start replicas
      const desiredReplicas = serviceInstance.desiredReplicas;

      for (let i = 0; i < desiredReplicas; i++) {
        const containerName = this.getContainerName(
          project.name,
          serviceName,
          i + 1
        );
        const containerConfig = await this.buildContainerConfig(
          project,
          serviceName,
          serviceConfig,
          containerName
        );

        // Check if container already exists
        const existingContainers = await this.docker.listContainers(true);
        const existing = existingContainers.find(
          (c) => c.name === containerName
        );

        let containerInfo: ContainerInfo;

        if (existing) {
          // Start existing container
          if (existing.status !== "running") {
            await this.docker.startContainer(existing.id);
          }
          containerInfo = await this.docker.inspectContainer(existing.id);
        } else {
          // Create and start new container
          containerInfo = await this.docker.createContainer(containerConfig);
          await this.docker.startContainer(containerInfo.id);
          containerInfo = await this.docker.inspectContainer(containerInfo.id);
        }

        serviceInstance.replicas.push(containerInfo);
      }

      // Wait for health checks if configured
      if (serviceConfig.healthcheck && !serviceConfig.healthcheck.disable) {
        await this.waitForHealthy(project, serviceName);
      }

      serviceInstance.status = "running";
      this.emit("service:started", project.id, serviceName);
    } catch (error) {
      serviceInstance.status = "error";
      serviceInstance.lastError = (error as Error).message;
      throw error;
    }
  }

  async stopService(
    project: ComposeProject,
    serviceName: string,
    timeout: number = 10
  ): Promise<void> {
    const serviceInstance = project.services.get(serviceName);

    if (!serviceInstance) {
      throw new Error(
        `Service ${serviceName} not found in project ${project.name}`
      );
    }

    this.logger.info(
      { projectName: project.name, serviceName },
      "Stopping service"
    );
    this.emit("service:stopping", project.id, serviceName);

    serviceInstance.status = "stopping";

    for (const container of serviceInstance.replicas) {
      try {
        await this.docker.stopContainer(container.id, timeout);
      } catch (error) {
        this.logger.warn(
          { error, containerId: container.id },
          "Failed to stop container"
        );
      }
    }

    serviceInstance.status = "stopped";
    this.emit("service:stopped", project.id, serviceName);
  }

  async scaleService(
    projectId: string,
    serviceName: string,
    replicas: number
  ): Promise<void> {
    const project = this.getProject(projectId);
    const serviceConfig = project.file.services[serviceName];
    const serviceInstance = project.services.get(serviceName);

    if (!serviceConfig || !serviceInstance) {
      throw new Error(
        `Service ${serviceName} not found in project ${project.name}`
      );
    }

    this.logger.info(
      { projectName: project.name, serviceName, replicas },
      "Scaling service"
    );

    const currentReplicas = serviceInstance.replicas.length;

    if (replicas > currentReplicas) {
      // Scale up
      for (let i = currentReplicas; i < replicas; i++) {
        const containerName = this.getContainerName(
          project.name,
          serviceName,
          i + 1
        );
        const containerConfig = await this.buildContainerConfig(
          project,
          serviceName,
          serviceConfig,
          containerName
        );

        const containerInfo =
          await this.docker.createContainer(containerConfig);
        await this.docker.startContainer(containerInfo.id);
        const updatedInfo = await this.docker.inspectContainer(
          containerInfo.id
        );
        serviceInstance.replicas.push(updatedInfo);
      }
    } else if (replicas < currentReplicas) {
      // Scale down
      const toRemove = serviceInstance.replicas.splice(replicas);
      for (const container of toRemove) {
        await this.docker.stopContainer(container.id);
        await this.docker.removeContainer(container.id);
      }
    }

    serviceInstance.desiredReplicas = replicas;
    this.emit("service:scaled", project.id, serviceName, replicas);
  }

  async restartService(
    projectId: string,
    serviceName: string,
    timeout: number = 10
  ): Promise<void> {
    const project = this.getProject(projectId);
    const serviceInstance = project.services.get(serviceName);

    if (!serviceInstance) {
      throw new Error(
        `Service ${serviceName} not found in project ${project.name}`
      );
    }

    this.logger.info(
      { projectName: project.name, serviceName },
      "Restarting service"
    );

    for (const container of serviceInstance.replicas) {
      await this.docker.restartContainer(container.id, timeout);
    }
  }

  // ===========================================================================
  // Helper Methods
  // ===========================================================================

  private getProject(projectId: string): ComposeProject {
    const project = this.projects.get(projectId);
    if (!project) {
      throw new Error(`Project ${projectId} not found`);
    }
    return project;
  }

  getProjectById(projectId: string): ComposeProject | undefined {
    return this.projects.get(projectId);
  }

  listProjects(): ComposeProject[] {
    return Array.from(this.projects.values());
  }

  private validateComposeFile(file: ComposeFile): void {
    if (!file.services || Object.keys(file.services).length === 0) {
      throw new Error("Compose file must define at least one service");
    }

    for (const [name, service] of Object.entries(file.services)) {
      if (!service.image && !service.build) {
        throw new Error(
          `Service ${name} must define either 'image' or 'build'`
        );
      }
    }
  }

  private async createProjectNetworks(project: ComposeProject): Promise<void> {
    const networks = project.file.networks || {};

    // Add default network
    if (!networks.default) {
      networks.default = {};
    }

    for (const [name, config] of Object.entries(networks)) {
      if (config.external) {
        // External network - just record the name
        const networkName =
          typeof config.external === "object" ? config.external.name : name;
        project.networks.set(name, networkName);
      } else {
        // Create network
        const networkName = `${project.name}_${name}`;
        const networkConfig: DockerNetworkConfig = {
          name: networkName,
          driver: config.driver as DockerNetworkConfig["driver"],
          internal: config.internal,
          attachable: config.attachable,
          labels: {
            ...config.labels,
            "com.docker.compose.project": project.name,
            "com.docker.compose.network": name,
            "neurectomy.managed": "true",
          },
          ipam: config.ipam,
        };

        const networkId = await this.docker.createNetwork(networkConfig);
        project.networks.set(name, networkId);
      }
    }
  }

  private async createProjectVolumes(project: ComposeProject): Promise<void> {
    const volumes = project.file.volumes || {};

    for (const [name, config] of Object.entries(volumes)) {
      if (config?.external) {
        // External volume - just record the name
        const volumeName =
          typeof config.external === "object"
            ? config.external.name
            : config.name || name;
        project.volumes.set(name, volumeName);
      } else {
        // Create volume
        const volumeName = config?.name || `${project.name}_${name}`;
        const volumeConfig: DockerVolumeConfig = {
          name: volumeName,
          driver: config?.driver,
          driverOpts: config?.driver_opts,
          labels: {
            ...config?.labels,
            "com.docker.compose.project": project.name,
            "com.docker.compose.volume": name,
            "neurectomy.managed": "true",
          },
        };

        await this.docker.createVolume(volumeConfig);
        project.volumes.set(name, volumeName);
      }
    }
  }

  private getStartOrder(services: Record<string, ComposeService>): string[] {
    const order: string[] = [];
    const visited = new Set<string>();
    const visiting = new Set<string>();

    const visit = (serviceName: string): void => {
      if (visited.has(serviceName)) return;
      if (visiting.has(serviceName)) {
        throw new Error(
          `Circular dependency detected for service: ${serviceName}`
        );
      }

      visiting.add(serviceName);

      const service = services[serviceName];
      const deps = this.getDependencies(service);

      for (const dep of deps) {
        visit(dep);
      }

      visiting.delete(serviceName);
      visited.add(serviceName);
      order.push(serviceName);
    };

    for (const serviceName of Object.keys(services)) {
      visit(serviceName);
    }

    return order;
  }

  private getDependencies(service: ComposeService): string[] {
    if (!service.depends_on) return [];

    if (Array.isArray(service.depends_on)) {
      return service.depends_on;
    }

    return Object.keys(service.depends_on);
  }

  private async waitForDependencies(
    project: ComposeProject,
    serviceName: string
  ): Promise<void> {
    const service = project.file.services[serviceName];
    const deps = service.depends_on;

    if (!deps) return;

    const dependencyEntries = Array.isArray(deps)
      ? deps.map((d) => [d, { condition: "service_started" as const }])
      : Object.entries(deps);

    for (const [depName, condition] of dependencyEntries as [
      string,
      DependsOnCondition | string,
    ][]) {
      const depInstance = project.services.get(depName);
      if (!depInstance) continue;

      const cond =
        typeof condition === "string" ? "service_started" : condition.condition;

      switch (cond) {
        case "service_started":
          await this.waitForStarted(depInstance);
          break;
        case "service_healthy":
          await this.waitForHealthy(project, depName);
          break;
        case "service_completed_successfully":
          await this.waitForCompleted(depInstance);
          break;
      }
    }
  }

  private async waitForStarted(
    service: ServiceInstance,
    timeout: number = 60000
  ): Promise<void> {
    const startTime = Date.now();

    while (service.status !== "running" && service.status !== "error") {
      if (Date.now() - startTime > timeout) {
        throw new Error(`Timeout waiting for service ${service.name} to start`);
      }
      await new Promise((resolve) => setTimeout(resolve, 500));
    }

    if (service.status === "error") {
      throw new Error(
        `Service ${service.name} failed to start: ${service.lastError}`
      );
    }
  }

  private async waitForHealthy(
    project: ComposeProject,
    serviceName: string,
    timeout: number = 120000
  ): Promise<void> {
    const serviceInstance = project.services.get(serviceName);
    if (!serviceInstance) return;

    const startTime = Date.now();

    while (true) {
      if (Date.now() - startTime > timeout) {
        throw new Error(
          `Timeout waiting for service ${serviceName} to become healthy`
        );
      }

      let allHealthy = true;

      for (const container of serviceInstance.replicas) {
        const info = await this.docker.inspectContainer(container.id);
        if (info.health !== "healthy") {
          allHealthy = false;
          break;
        }
      }

      if (allHealthy) return;

      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }

  private async waitForCompleted(
    service: ServiceInstance,
    timeout: number = 300000
  ): Promise<void> {
    const startTime = Date.now();

    while (service.status !== "stopped") {
      if (Date.now() - startTime > timeout) {
        throw new Error(
          `Timeout waiting for service ${service.name} to complete`
        );
      }
      await new Promise((resolve) => setTimeout(resolve, 500));
    }

    // Check exit codes
    for (const container of service.replicas) {
      const info = await this.docker.inspectContainer(container.id);
      if (info.exitCode !== 0) {
        throw new Error(
          `Service ${service.name} completed with exit code ${info.exitCode}`
        );
      }
    }
  }

  private getContainerName(
    projectName: string,
    serviceName: string,
    replica: number
  ): string {
    return `${projectName}-${serviceName}-${replica}`;
  }

  private async buildContainerConfig(
    project: ComposeProject,
    serviceName: string,
    service: ComposeService,
    containerName: string
  ): Promise<DockerContainerConfig> {
    // Parse environment variables
    const env: Record<string, string> = {};
    if (service.environment) {
      if (Array.isArray(service.environment)) {
        for (const e of service.environment) {
          const [key, ...valueParts] = e.split("=");
          env[key] = valueParts.join("=");
        }
      } else {
        Object.assign(env, service.environment);
      }
    }

    // Parse ports
    const portBindings = this.parsePorts(service.ports);

    // Parse volumes
    const mounts = this.parseVolumes(project, service.volumes);

    // Parse labels
    const labels: Record<string, string> = {
      "com.docker.compose.project": project.name,
      "com.docker.compose.service": serviceName,
      "neurectomy.managed": "true",
    };

    if (service.labels) {
      if (Array.isArray(service.labels)) {
        for (const l of service.labels) {
          const [key, ...valueParts] = l.split("=");
          labels[key] = valueParts.join("=");
        }
      } else {
        Object.assign(labels, service.labels);
      }
    }

    // Parse command
    const cmd =
      typeof service.command === "string"
        ? service.command.split(" ")
        : service.command;

    // Parse entrypoint
    const entrypoint =
      typeof service.entrypoint === "string"
        ? service.entrypoint.split(" ")
        : service.entrypoint;

    // Build config
    const config: DockerContainerConfig = {
      name: containerName,
      image: service.image || "",
      cmd,
      entrypoint,
      env,
      labels,
      workingDir: service.working_dir,
      user: service.user,
      hostname: service.hostname,
      domainname: service.domainname,
      privileged: service.privileged,
      readonlyRootfs: service.read_only,
      tty: service.tty,
      stdin: service.stdin_open,
      portBindings,
      mounts,
      networkMode: project.networks.has("default")
        ? project.networks.get("default")
        : undefined,
      restart: service.restart,
    };

    // Health check
    if (service.healthcheck && !service.healthcheck.disable) {
      config.healthcheck = {
        test:
          typeof service.healthcheck.test === "string"
            ? ["CMD-SHELL", service.healthcheck.test]
            : service.healthcheck.test,
        interval: this.parseDuration(service.healthcheck.interval),
        timeout: this.parseDuration(service.healthcheck.timeout),
        retries: service.healthcheck.retries,
        startPeriod: this.parseDuration(service.healthcheck.start_period),
      };
    }

    // Resource limits
    if (service.deploy?.resources) {
      const limits = service.deploy.resources.limits;
      const reservations = service.deploy.resources.reservations;

      config.resources = {
        memory: limits?.memory ? this.parseMemory(limits.memory) : undefined,
        memoryReservation: reservations?.memory
          ? this.parseMemory(reservations.memory)
          : undefined,
      };
    }

    return config;
  }

  private parsePorts(
    ports?: (string | PortConfig)[]
  ): PortMapping[] | undefined {
    if (!ports || ports.length === 0) return undefined;

    return ports.map((p) => {
      if (typeof p === "string") {
        // Parse string format: "8080:80", "8080:80/tcp", "127.0.0.1:8080:80"
        const parts = p.split(":");
        let hostIp: string | undefined;
        let hostPort: number;
        let containerPortProto: string;

        if (parts.length === 3) {
          hostIp = parts[0];
          hostPort = parseInt(parts[1], 10);
          containerPortProto = parts[2];
        } else {
          hostPort = parseInt(parts[0], 10);
          containerPortProto = parts[1];
        }

        const [containerPortStr, protocol = "tcp"] =
          containerPortProto.split("/");
        const containerPort = parseInt(containerPortStr, 10);

        return {
          hostPort,
          containerPort,
          protocol: protocol as "tcp" | "udp",
          hostIp,
        };
      } else {
        return {
          hostPort:
            typeof p.published === "string"
              ? parseInt(p.published, 10)
              : p.published || p.target,
          containerPort: p.target,
          protocol: p.protocol || "tcp",
        };
      }
    });
  }

  private parseVolumes(
    project: ComposeProject,
    volumes?: (string | VolumeConfig)[]
  ): Mount[] | undefined {
    if (!volumes || volumes.length === 0) return undefined;

    return volumes.map((v) => {
      if (typeof v === "string") {
        // Parse string format: "/host:/container", "/host:/container:ro", "volume:/container"
        const parts = v.split(":");
        const source = parts[0];
        const target = parts[1];
        const mode = parts[2];

        // Check if source is a named volume
        if (project.volumes.has(source)) {
          return {
            type: "volume" as const,
            source: project.volumes.get(source)!,
            target,
            readOnly: mode === "ro",
          };
        }

        return {
          type: "bind" as const,
          source,
          target,
          readOnly: mode === "ro",
        };
      } else {
        return {
          type: v.type,
          source:
            v.type === "volume" && v.source && project.volumes.has(v.source)
              ? project.volumes.get(v.source)!
              : v.source || "",
          target: v.target,
          readOnly: v.read_only || false,
        };
      }
    });
  }

  private parseDuration(duration?: string): number | undefined {
    if (!duration) return undefined;

    const match = duration.match(/^(\d+)(ms|s|m|h)?$/);
    if (!match) return undefined;

    const value = parseInt(match[1], 10);
    const unit = match[2] || "s";

    switch (unit) {
      case "ms":
        return value / 1000;
      case "s":
        return value;
      case "m":
        return value * 60;
      case "h":
        return value * 3600;
      default:
        return value;
    }
  }

  private parseMemory(memory: string): number {
    const match = memory.match(/^(\d+)(b|k|m|g)?$/i);
    if (!match) return 0;

    const value = parseInt(match[1], 10);
    const unit = (match[2] || "b").toLowerCase();

    switch (unit) {
      case "b":
        return value;
      case "k":
        return value * 1024;
      case "m":
        return value * 1024 * 1024;
      case "g":
        return value * 1024 * 1024 * 1024;
      default:
        return value;
    }
  }
}
