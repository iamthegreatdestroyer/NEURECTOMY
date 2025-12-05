/**
 * NEURECTOMY Model Versioning System
 * @module @neurectomy/experimentation-engine/hypothesis
 * @agent @PRISM @TENSOR
 *
 * Version control for ML models with lineage tracking,
 * staging, and deployment management.
 */

import { EventEmitter } from "eventemitter3";
import { v4 as uuidv4 } from "uuid";

// ============================================================================
// Types
// ============================================================================

export interface ModelVersion {
  id: string;
  modelId: string;
  version: string;
  runId?: string;
  stage: ModelStage;
  description?: string;
  metrics: Record<string, number>;
  parameters: Record<string, unknown>;
  artifacts: ModelArtifact[];
  tags: string[];
  signature?: ModelSignature;
  lineage: ModelLineage;
  createdAt: Date;
  updatedAt: Date;
  promotedAt?: Date;
  archivedAt?: Date;
}

export type ModelStage =
  | "none"
  | "development"
  | "staging"
  | "production"
  | "archived";

export interface ModelArtifact {
  name: string;
  path: string;
  type: "weights" | "config" | "tokenizer" | "metadata" | "other";
  format?: string;
  size?: number;
  checksum?: string;
}

export interface ModelSignature {
  inputs: SignatureField[];
  outputs: SignatureField[];
}

export interface SignatureField {
  name: string;
  dtype: string;
  shape?: (number | null)[];
  description?: string;
}

export interface ModelLineage {
  parentVersionId?: string;
  datasetId?: string;
  datasetVersion?: string;
  sourceCode?: {
    repository?: string;
    commit?: string;
    branch?: string;
  };
  dependencies?: Record<string, string>;
  environment?: Record<string, string>;
}

export interface RegisteredModel {
  id: string;
  name: string;
  description?: string;
  versions: string[];
  latestVersion?: string;
  productionVersion?: string;
  stagingVersion?: string;
  tags: string[];
  createdAt: Date;
  updatedAt: Date;
}

export interface VersioningConfig {
  autoIncrementVersion?: boolean;
  versionFormat?: "semver" | "numeric" | "timestamp";
  maxVersionsPerModel?: number;
  archiveOldVersions?: boolean;
}

export interface VersioningEvents {
  modelRegistered: (model: RegisteredModel) => void;
  versionCreated: (version: ModelVersion) => void;
  versionPromoted: (
    version: ModelVersion,
    fromStage: ModelStage,
    toStage: ModelStage
  ) => void;
  versionArchived: (version: ModelVersion) => void;
  versionDeleted: (versionId: string) => void;
}

export interface VersionComparison {
  versions: ModelVersion[];
  metricDiffs: MetricDiff[];
  parameterDiffs: ParameterDiff[];
  structuralChanges: StructuralChange[];
}

export interface MetricDiff {
  key: string;
  values: { versionId: string; value: number }[];
  delta: number;
  percentChange: number;
}

export interface ParameterDiff {
  key: string;
  changes: { versionId: string; value: unknown }[];
  isChanged: boolean;
}

export interface StructuralChange {
  type: "signature" | "artifact" | "dependency";
  description: string;
  versions: string[];
}

// ============================================================================
// ModelRegistry Class
// ============================================================================

export class ModelRegistry extends EventEmitter<VersioningEvents> {
  private models = new Map<string, RegisteredModel>();
  private versions = new Map<string, ModelVersion>();
  private config: Required<VersioningConfig>;

  constructor(config: VersioningConfig = {}) {
    super();
    this.config = {
      autoIncrementVersion: config.autoIncrementVersion ?? true,
      versionFormat: config.versionFormat ?? "semver",
      maxVersionsPerModel: config.maxVersionsPerModel ?? 100,
      archiveOldVersions: config.archiveOldVersions ?? true,
    };
  }

  // --------------------------------------------------------------------------
  // Model Registration
  // --------------------------------------------------------------------------

  /**
   * Register a new model or get existing
   */
  registerModel(
    name: string,
    options?: { description?: string; tags?: string[] }
  ): RegisteredModel {
    const existing = this.getModelByName(name);
    if (existing) {
      return existing;
    }

    const model: RegisteredModel = {
      id: uuidv4(),
      name,
      description: options?.description,
      versions: [],
      tags: options?.tags ?? [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    this.models.set(model.id, model);
    this.emit("modelRegistered", model);

    return model;
  }

  /**
   * Get model by ID
   */
  getModel(id: string): RegisteredModel | undefined {
    return this.models.get(id);
  }

  /**
   * Get model by name
   */
  getModelByName(name: string): RegisteredModel | undefined {
    return Array.from(this.models.values()).find((m) => m.name === name);
  }

  /**
   * List all registered models
   */
  listModels(filter?: { tags?: string[] }): RegisteredModel[] {
    let results = Array.from(this.models.values());

    if (filter?.tags?.length) {
      results = results.filter((m) =>
        filter.tags!.some((tag) => m.tags.includes(tag))
      );
    }

    return results;
  }

  /**
   * Delete a model and all its versions
   */
  deleteModel(id: string): boolean {
    const model = this.models.get(id);
    if (!model) {
      return false;
    }

    // Delete all versions
    for (const versionId of model.versions) {
      this.versions.delete(versionId);
    }

    return this.models.delete(id);
  }

  // --------------------------------------------------------------------------
  // Version Management
  // --------------------------------------------------------------------------

  /**
   * Create a new model version
   */
  createVersion(
    modelId: string,
    data: {
      version?: string;
      runId?: string;
      description?: string;
      metrics: Record<string, number>;
      parameters: Record<string, unknown>;
      artifacts?: ModelArtifact[];
      tags?: string[];
      signature?: ModelSignature;
      lineage?: Partial<ModelLineage>;
    }
  ): ModelVersion {
    const model = this.models.get(modelId);
    if (!model) {
      throw new Error(`Model not found: ${modelId}`);
    }

    if (model.versions.length >= this.config.maxVersionsPerModel) {
      if (this.config.archiveOldVersions) {
        this.archiveOldestVersion(model);
      } else {
        throw new Error(
          `Max versions exceeded for model: ${this.config.maxVersionsPerModel}`
        );
      }
    }

    const versionString = data.version ?? this.generateNextVersion(model);

    // Check for duplicate version
    if (this.getVersionByString(modelId, versionString)) {
      throw new Error(`Version already exists: ${versionString}`);
    }

    const version: ModelVersion = {
      id: uuidv4(),
      modelId,
      version: versionString,
      runId: data.runId,
      stage: "none",
      description: data.description,
      metrics: data.metrics,
      parameters: data.parameters,
      artifacts: data.artifacts ?? [],
      tags: data.tags ?? [],
      signature: data.signature,
      lineage: {
        ...data.lineage,
      },
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    this.versions.set(version.id, version);
    model.versions.push(version.id);
    model.latestVersion = version.id;
    model.updatedAt = new Date();

    this.emit("versionCreated", version);

    return version;
  }

  /**
   * Get version by ID
   */
  getVersion(id: string): ModelVersion | undefined {
    return this.versions.get(id);
  }

  /**
   * Get version by version string
   */
  getVersionByString(
    modelId: string,
    version: string
  ): ModelVersion | undefined {
    const model = this.models.get(modelId);
    if (!model) {
      return undefined;
    }

    for (const versionId of model.versions) {
      const v = this.versions.get(versionId);
      if (v && v.version === version) {
        return v;
      }
    }

    return undefined;
  }

  /**
   * List versions for a model
   */
  listVersions(
    modelId: string,
    filter?: { stage?: ModelStage; tags?: string[] }
  ): ModelVersion[] {
    const model = this.models.get(modelId);
    if (!model) {
      return [];
    }

    let results = model.versions
      .map((id) => this.versions.get(id))
      .filter((v): v is ModelVersion => v !== undefined);

    if (filter?.stage) {
      results = results.filter((v) => v.stage === filter.stage);
    }

    if (filter?.tags?.length) {
      results = results.filter((v) =>
        filter.tags!.some((tag) => v.tags.includes(tag))
      );
    }

    return results;
  }

  /**
   * Get the latest version of a model
   */
  getLatestVersion(modelId: string): ModelVersion | undefined {
    const model = this.models.get(modelId);
    if (!model?.latestVersion) {
      return undefined;
    }
    return this.versions.get(model.latestVersion);
  }

  /**
   * Get production version of a model
   */
  getProductionVersion(modelId: string): ModelVersion | undefined {
    const model = this.models.get(modelId);
    if (!model?.productionVersion) {
      return undefined;
    }
    return this.versions.get(model.productionVersion);
  }

  // --------------------------------------------------------------------------
  // Stage Management
  // --------------------------------------------------------------------------

  /**
   * Promote a version to a new stage
   */
  promoteVersion(versionId: string, toStage: ModelStage): ModelVersion {
    const version = this.versions.get(versionId);
    if (!version) {
      throw new Error(`Version not found: ${versionId}`);
    }

    const model = this.models.get(version.modelId);
    if (!model) {
      throw new Error(`Model not found: ${version.modelId}`);
    }

    const fromStage = version.stage;

    // Demote current version in target stage
    if (toStage === "production" && model.productionVersion) {
      const current = this.versions.get(model.productionVersion);
      if (current && current.id !== versionId) {
        current.stage = "staging";
        current.updatedAt = new Date();
      }
    }

    if (toStage === "staging" && model.stagingVersion) {
      const current = this.versions.get(model.stagingVersion);
      if (current && current.id !== versionId) {
        current.stage = "development";
        current.updatedAt = new Date();
      }
    }

    // Update version stage
    version.stage = toStage;
    version.updatedAt = new Date();
    version.promotedAt = new Date();

    // Update model references
    if (toStage === "production") {
      model.productionVersion = versionId;
    } else if (toStage === "staging") {
      model.stagingVersion = versionId;
    }

    model.updatedAt = new Date();

    this.emit("versionPromoted", version, fromStage, toStage);

    return version;
  }

  /**
   * Archive a version
   */
  archiveVersion(versionId: string): ModelVersion {
    const version = this.versions.get(versionId);
    if (!version) {
      throw new Error(`Version not found: ${versionId}`);
    }

    const model = this.models.get(version.modelId);
    if (model) {
      if (model.productionVersion === versionId) {
        model.productionVersion = undefined;
      }
      if (model.stagingVersion === versionId) {
        model.stagingVersion = undefined;
      }
    }

    version.stage = "archived";
    version.archivedAt = new Date();
    version.updatedAt = new Date();

    this.emit("versionArchived", version);

    return version;
  }

  /**
   * Delete a version
   */
  deleteVersion(versionId: string): boolean {
    const version = this.versions.get(versionId);
    if (!version) {
      return false;
    }

    const model = this.models.get(version.modelId);
    if (model) {
      model.versions = model.versions.filter((v) => v !== versionId);
      if (model.latestVersion === versionId) {
        model.latestVersion = model.versions[model.versions.length - 1];
      }
      if (model.productionVersion === versionId) {
        model.productionVersion = undefined;
      }
      if (model.stagingVersion === versionId) {
        model.stagingVersion = undefined;
      }
    }

    this.emit("versionDeleted", versionId);

    return this.versions.delete(versionId);
  }

  private archiveOldestVersion(model: RegisteredModel): void {
    const nonArchivedVersions = model.versions
      .map((id) => this.versions.get(id))
      .filter(
        (v): v is ModelVersion => v !== undefined && v.stage !== "archived"
      )
      .sort((a, b) => a.createdAt.getTime() - b.createdAt.getTime());

    if (nonArchivedVersions.length > 0) {
      this.archiveVersion(nonArchivedVersions[0].id);
    }
  }

  // --------------------------------------------------------------------------
  // Version Comparison
  // --------------------------------------------------------------------------

  /**
   * Compare multiple versions
   */
  compareVersions(versionIds: string[]): VersionComparison {
    const versions = versionIds
      .map((id) => this.versions.get(id))
      .filter((v): v is ModelVersion => v !== undefined);

    if (versions.length < 2) {
      return {
        versions,
        metricDiffs: [],
        parameterDiffs: [],
        structuralChanges: [],
      };
    }

    // Metric diffs
    const metricKeys = new Set<string>();
    for (const v of versions) {
      for (const key of Object.keys(v.metrics)) {
        metricKeys.add(key);
      }
    }

    const metricDiffs: MetricDiff[] = [];
    for (const key of metricKeys) {
      const values = versions.map((v) => ({
        versionId: v.id,
        value: v.metrics[key] ?? 0,
      }));

      const nums = values.map((v) => v.value);
      const delta = Math.max(...nums) - Math.min(...nums);
      const baseline = nums[0] || 1;
      const percentChange =
        ((nums[nums.length - 1] - baseline) / baseline) * 100;

      metricDiffs.push({
        key,
        values,
        delta,
        percentChange,
      });
    }

    // Parameter diffs
    const paramKeys = new Set<string>();
    for (const v of versions) {
      for (const key of Object.keys(v.parameters)) {
        paramKeys.add(key);
      }
    }

    const parameterDiffs: ParameterDiff[] = [];
    for (const key of paramKeys) {
      const changes = versions.map((v) => ({
        versionId: v.id,
        value: v.parameters[key],
      }));

      const uniqueValues = new Set(changes.map((c) => JSON.stringify(c.value)));
      parameterDiffs.push({
        key,
        changes,
        isChanged: uniqueValues.size > 1,
      });
    }

    // Structural changes
    const structuralChanges: StructuralChange[] = [];

    // Check signature changes
    const signatures = versions.map((v) => JSON.stringify(v.signature));
    if (new Set(signatures).size > 1) {
      structuralChanges.push({
        type: "signature",
        description: "Model signature changed between versions",
        versions: versionIds,
      });
    }

    // Check artifact changes
    const artifactCounts = versions.map((v) => v.artifacts.length);
    if (new Set(artifactCounts).size > 1) {
      structuralChanges.push({
        type: "artifact",
        description: "Number of artifacts changed between versions",
        versions: versionIds,
      });
    }

    return {
      versions,
      metricDiffs,
      parameterDiffs,
      structuralChanges,
    };
  }

  // --------------------------------------------------------------------------
  // Version Generation
  // --------------------------------------------------------------------------

  private generateNextVersion(model: RegisteredModel): string {
    const existing = model.versions
      .map((id) => this.versions.get(id)?.version)
      .filter((v): v is string => v !== undefined);

    switch (this.config.versionFormat) {
      case "semver":
        return this.nextSemver(existing);
      case "numeric":
        return this.nextNumeric(existing);
      case "timestamp":
        return this.nextTimestamp();
      default:
        return this.nextNumeric(existing);
    }
  }

  private nextSemver(existing: string[]): string {
    if (existing.length === 0) {
      return "1.0.0";
    }

    const semverRegex = /^(\d+)\.(\d+)\.(\d+)$/;
    const versions = existing
      .map((v) => {
        const match = v.match(semverRegex);
        if (match) {
          return {
            major: parseInt(match[1]),
            minor: parseInt(match[2]),
            patch: parseInt(match[3]),
          };
        }
        return null;
      })
      .filter(
        (v): v is { major: number; minor: number; patch: number } => v !== null
      );

    if (versions.length === 0) {
      return "1.0.0";
    }

    const latest = versions.reduce((a, b) => {
      if (a.major > b.major) return a;
      if (a.major < b.major) return b;
      if (a.minor > b.minor) return a;
      if (a.minor < b.minor) return b;
      return a.patch >= b.patch ? a : b;
    });

    return `${latest.major}.${latest.minor}.${latest.patch + 1}`;
  }

  private nextNumeric(existing: string[]): string {
    if (existing.length === 0) {
      return "1";
    }

    const numbers = existing.map((v) => parseInt(v)).filter((n) => !isNaN(n));

    if (numbers.length === 0) {
      return "1";
    }

    return String(Math.max(...numbers) + 1);
  }

  private nextTimestamp(): string {
    return new Date().toISOString().replace(/[:.]/g, "-");
  }

  // --------------------------------------------------------------------------
  // Utility
  // --------------------------------------------------------------------------

  /**
   * Export model registry to JSON
   */
  export(): { models: RegisteredModel[]; versions: ModelVersion[] } {
    return {
      models: Array.from(this.models.values()),
      versions: Array.from(this.versions.values()),
    };
  }

  /**
   * Import model registry from JSON
   */
  import(data: { models: RegisteredModel[]; versions: ModelVersion[] }): void {
    for (const model of data.models) {
      model.createdAt = new Date(model.createdAt);
      model.updatedAt = new Date(model.updatedAt);
      this.models.set(model.id, model);
    }

    for (const version of data.versions) {
      version.createdAt = new Date(version.createdAt);
      version.updatedAt = new Date(version.updatedAt);
      if (version.promotedAt) {
        version.promotedAt = new Date(version.promotedAt);
      }
      if (version.archivedAt) {
        version.archivedAt = new Date(version.archivedAt);
      }
      this.versions.set(version.id, version);
    }
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a new ModelRegistry instance
 */
export function createRegistry(config?: VersioningConfig): ModelRegistry {
  return new ModelRegistry(config);
}

/**
 * Create a model signature definition
 */
export function defineSignature(
  inputs: SignatureField[],
  outputs: SignatureField[]
): ModelSignature {
  return { inputs, outputs };
}

/**
 * Create a model artifact definition
 */
export function defineArtifact(
  name: string,
  path: string,
  type: ModelArtifact["type"],
  options?: { format?: string; checksum?: string }
): ModelArtifact {
  return {
    name,
    path,
    type,
    format: options?.format,
    checksum: options?.checksum,
  };
}
