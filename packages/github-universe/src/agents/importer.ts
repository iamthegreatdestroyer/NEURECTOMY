/**
 * @fileoverview GitHub Agent Importer - Import and sync agents from GitHub
 * @module @neurectomy/github-universe/agents
 *
 * @description Provides functionality for importing AI agent definitions from
 * GitHub repositories, supporting various agent specification formats and
 * enabling automatic synchronization of agent configurations.
 *
 * @agent @SYNAPSE - Integration Engineering & API Design
 * @agent @APEX - Elite Computer Science Engineering
 *
 * Features:
 * - Import agents from YAML/JSON definitions
 * - Support for multiple agent frameworks (LangChain, CrewAI, AutoGPT, etc.)
 * - Automatic dependency resolution
 * - Agent versioning and history tracking
 * - Batch import from organizations
 * - Smart caching and incremental updates
 *
 * @example
 * ```typescript
 * import { AgentImporter } from '@neurectomy/github-universe/agents';
 *
 * const importer = new AgentImporter(client);
 *
 * // Import agents from a repository
 * const agents = await importer.importFromRepo(
 *   { owner: 'org', repo: 'agents' },
 *   { recursive: true, includeExamples: true }
 * );
 *
 * // Watch for agent updates
 * await importer.watchRepo(
 *   { owner: 'org', repo: 'agents' },
 *   (agent) => console.log(`Updated: ${agent.name}`)
 * );
 * ```
 */

import { EventEmitter } from "eventemitter3";
import * as yaml from "yaml";
import type { GitHubClient } from "../client";
import {
  type RepoIdentifier,
  type AgentDefinition,
  AgentDefinitionSchema,
} from "../types";

// ============================================================================
// TYPES
// ============================================================================

/**
 * Events emitted by AgentImporter
 */
export interface AgentImporterEvents {
  "agent:discovered": (agent: DiscoveredAgent) => void;
  "agent:imported": (agent: ImportedAgent) => void;
  "agent:updated": (agent: ImportedAgent, changes: AgentChanges) => void;
  "agent:error": (path: string, error: Error) => void;
  "import:started": (repo: RepoIdentifier) => void;
  "import:completed": (repo: RepoIdentifier, stats: ImportStats) => void;
  "scan:progress": (current: number, total: number) => void;
}

/**
 * Agent discovered in repository before full import
 */
export interface DiscoveredAgent {
  path: string;
  name: string;
  format: AgentFormat;
  framework?: AgentFramework;
  sha: string;
  size: number;
}

/**
 * Fully imported agent with resolved definition
 */
export interface ImportedAgent {
  definition: AgentDefinition;
  source: AgentSource;
  metadata: AgentMetadata;
}

/**
 * Agent definition format
 */
export type AgentFormat = "yaml" | "json" | "toml" | "python" | "typescript";

/**
 * Supported agent frameworks
 */
export type AgentFramework =
  | "langchain"
  | "crewai"
  | "autogpt"
  | "autogen"
  | "semantic-kernel"
  | "llamaindex"
  | "custom"
  | "unknown";

/**
 * Agent source information
 */
export interface AgentSource {
  repo: RepoIdentifier;
  path: string;
  branch: string;
  sha: string;
  url: string;
  rawUrl: string;
}

/**
 * Agent metadata
 */
export interface AgentMetadata {
  importedAt: Date;
  lastChecked: Date;
  format: AgentFormat;
  framework: AgentFramework;
  dependencies: AgentDependency[];
  relatedFiles: string[];
  size: number;
  hash: string;
}

/**
 * Agent dependency
 */
export interface AgentDependency {
  type: "npm" | "pip" | "agent" | "model" | "tool";
  name: string;
  version?: string;
  optional?: boolean;
}

/**
 * Changes detected in agent update
 */
export interface AgentChanges {
  fields: string[];
  previousSha: string;
  newSha: string;
  diff?: string;
}

/**
 * Import statistics
 */
export interface ImportStats {
  discovered: number;
  imported: number;
  updated: number;
  failed: number;
  skipped: number;
  duration: number;
}

/**
 * Import options
 */
export interface ImportOptions {
  /** Recursively scan directories */
  recursive?: boolean;
  /** Include example/demo agents */
  includeExamples?: boolean;
  /** Include deprecated agents */
  includeDeprecated?: boolean;
  /** Filter by framework */
  frameworks?: AgentFramework[];
  /** File path patterns to include */
  include?: string[];
  /** File path patterns to exclude */
  exclude?: string[];
  /** Branch to import from */
  branch?: string;
  /** Maximum depth for recursive scan */
  maxDepth?: number;
  /** Skip agents already in cache */
  skipCached?: boolean;
  /** Validate agents against schema */
  validate?: boolean;
  /** Resolve and include dependencies */
  resolveDependencies?: boolean;
}

/**
 * Watch options
 */
export interface WatchOptions {
  /** Branch to watch */
  branch?: string;
  /** Polling interval in milliseconds */
  interval?: number;
  /** Paths to watch (defaults to all) */
  paths?: string[];
  /** Only notify on specific changes */
  changeTypes?: ("added" | "modified" | "removed")[];
}

/**
 * Agent search options
 */
export interface AgentSearchOptions {
  /** Framework filter */
  framework?: AgentFramework;
  /** Search query */
  query?: string;
  /** Capability requirements */
  capabilities?: string[];
  /** Tag filter */
  tags?: string[];
  /** Maximum results */
  limit?: number;
}

/**
 * Cached agent entry
 */
interface CachedAgent {
  agent: ImportedAgent;
  expiresAt: number;
}

// ============================================================================
// CONSTANTS
// ============================================================================

const AGENT_FILE_PATTERNS = [
  "agent.yaml",
  "agent.yml",
  "agent.json",
  "agents.yaml",
  "agents.yml",
  "agents.json",
  "*.agent.yaml",
  "*.agent.yml",
  "*.agent.json",
  "crewai.yaml",
  "crew.yaml",
  "langchain.yaml",
  ".neurectomy/agent.yaml",
  ".neurectomy/agents.yaml",
];

const DEFAULT_EXCLUDE_PATTERNS = [
  "**/node_modules/**",
  "**/venv/**",
  "**/.venv/**",
  "**/dist/**",
  "**/build/**",
  "**/__pycache__/**",
  "**/examples/**",
  "**/test/**",
  "**/tests/**",
];

const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

// ============================================================================
// AGENT IMPORTER
// ============================================================================

/**
 * Imports and manages AI agent definitions from GitHub repositories
 *
 * @extends EventEmitter
 *
 * @example
 * ```typescript
 * const importer = new AgentImporter(client);
 *
 * // Listen for events
 * importer.on('agent:imported', (agent) => {
 *   console.log(`Imported ${agent.definition.name} from ${agent.source.path}`);
 * });
 *
 * importer.on('agent:error', (path, error) => {
 *   console.error(`Failed to import ${path}: ${error.message}`);
 * });
 *
 * // Import all agents from a repo
 * const agents = await importer.importFromRepo(
 *   { owner: 'org', repo: 'ai-agents' }
 * );
 * ```
 */
export class AgentImporter extends EventEmitter<AgentImporterEvents> {
  private client: GitHubClient;
  private cache: Map<string, CachedAgent> = new Map();
  private watchIntervals: Map<string, NodeJS.Timeout> = new Map();

  constructor(client: GitHubClient) {
    super();
    this.client = client;
  }

  // ==========================================================================
  // DISCOVERY
  // ==========================================================================

  /**
   * Discover agents in a repository without importing
   *
   * @param repo - Repository identifier
   * @param options - Discovery options
   * @returns Array of discovered agents
   *
   * @example
   * ```typescript
   * const discovered = await importer.discoverAgents(
   *   { owner: 'org', repo: 'agents' },
   *   { recursive: true }
   * );
   *
   * console.log(`Found ${discovered.length} agents`);
   * discovered.forEach(a => console.log(`  - ${a.name} (${a.framework})`));
   * ```
   */
  async discoverAgents(
    repo: RepoIdentifier,
    options: ImportOptions = {}
  ): Promise<DiscoveredAgent[]> {
    const branch = options.branch || (await this.getDefaultBranch(repo));
    const discovered: DiscoveredAgent[] = [];

    // Get file tree
    const tree = await this.getFileTree(repo, branch, options);

    for (const file of tree) {
      if (this.isAgentFile(file.path, options)) {
        const agent: DiscoveredAgent = {
          path: file.path,
          name: this.extractAgentName(file.path),
          format: this.detectFormat(file.path),
          framework: this.detectFramework(file.path),
          sha: file.sha,
          size: file.size || 0,
        };

        discovered.push(agent);
        this.emit("agent:discovered", agent);
      }
    }

    return discovered;
  }

  /**
   * Search for agents across multiple repositories
   *
   * @param owner - Organization or user
   * @param options - Search options
   * @returns Array of discovered agents with repo info
   *
   * @example
   * ```typescript
   * const agents = await importer.searchAgents('my-org', {
   *   framework: 'crewai',
   *   capabilities: ['code-generation', 'analysis'],
   * });
   * ```
   */
  async searchAgents(
    owner: string,
    options: AgentSearchOptions = {}
  ): Promise<Array<DiscoveredAgent & { repo: RepoIdentifier }>> {
    // Search for agent files across org
    const query = this.buildSearchQuery(owner, options);

    const response = await this.client.rest<{
      total_count: number;
      incomplete_results: boolean;
      items: Array<{
        name: string;
        path: string;
        sha: string;
        repository?: {
          name: string;
          owner?: { login: string };
        };
      }>;
    }>("GET /search/code", {
      q: query,
      per_page: options.limit || 100,
    });

    const results: Array<DiscoveredAgent & { repo: RepoIdentifier }> = [];

    for (const item of response.items) {
      if (!item.repository) continue;

      const agent: DiscoveredAgent & { repo: RepoIdentifier } = {
        path: item.path,
        name: this.extractAgentName(item.path),
        format: this.detectFormat(item.path),
        framework: options.framework || this.detectFramework(item.path),
        sha: item.sha,
        size: 0,
        repo: {
          owner: item.repository.owner?.login || owner,
          repo: item.repository.name,
        },
      };

      results.push(agent);
    }

    return results;
  }

  // ==========================================================================
  // IMPORT
  // ==========================================================================

  /**
   * Import all agents from a repository
   *
   * @param repo - Repository identifier
   * @param options - Import options
   * @returns Array of imported agents
   *
   * @example
   * ```typescript
   * const agents = await importer.importFromRepo(
   *   { owner: 'org', repo: 'agents' },
   *   {
   *     recursive: true,
   *     frameworks: ['crewai', 'langchain'],
   *     resolveDependencies: true,
   *   }
   * );
   * ```
   */
  async importFromRepo(
    repo: RepoIdentifier,
    options: ImportOptions = {}
  ): Promise<ImportedAgent[]> {
    const startTime = Date.now();
    this.emit("import:started", repo);

    const stats: ImportStats = {
      discovered: 0,
      imported: 0,
      updated: 0,
      failed: 0,
      skipped: 0,
      duration: 0,
    };

    const branch = options.branch || (await this.getDefaultBranch(repo));
    const imported: ImportedAgent[] = [];

    // Discover agents
    const discovered = await this.discoverAgents(repo, options);
    stats.discovered = discovered.length;

    // Import each agent
    for (let i = 0; i < discovered.length; i++) {
      const agent = discovered[i];
      this.emit("scan:progress", i + 1, discovered.length);

      try {
        // Check cache
        const cacheKey = this.getCacheKey(repo, agent.path);
        const cached = this.getCached(cacheKey);

        if (cached && options.skipCached) {
          if (cached.source.sha === agent.sha) {
            imported.push(cached);
            stats.skipped++;
            continue;
          }
        }

        // Import agent
        const importedAgent = await this.importAgent(
          repo,
          agent.path,
          branch,
          options
        );

        if (cached && cached.source.sha !== importedAgent.source.sha) {
          const changes = this.detectChanges(cached, importedAgent);
          this.emit("agent:updated", importedAgent, changes);
          stats.updated++;
        }

        imported.push(importedAgent);
        this.cache.set(cacheKey, {
          agent: importedAgent,
          expiresAt: Date.now() + CACHE_TTL,
        });

        stats.imported++;
        this.emit("agent:imported", importedAgent);
      } catch (error) {
        stats.failed++;
        this.emit("agent:error", agent.path, error as Error);
      }
    }

    stats.duration = Date.now() - startTime;
    this.emit("import:completed", repo, stats);

    return imported;
  }

  /**
   * Import a single agent from a specific path
   *
   * @param repo - Repository identifier
   * @param path - Path to agent file
   * @param branch - Branch name (optional)
   * @param options - Import options
   * @returns Imported agent
   *
   * @example
   * ```typescript
   * const agent = await importer.importAgent(
   *   { owner: 'org', repo: 'agents' },
   *   'agents/researcher/agent.yaml'
   * );
   * ```
   */
  async importAgent(
    repo: RepoIdentifier,
    path: string,
    branch?: string,
    options: ImportOptions = {}
  ): Promise<ImportedAgent> {
    const ref = branch || (await this.getDefaultBranch(repo));

    // Get file content
    const response = await this.client.rest<{
      type: string;
      encoding: string;
      content: string;
      sha: string;
      size: number;
      name: string;
      path: string;
      html_url?: string;
      download_url?: string;
    }>("GET /repos/{owner}/{repo}/contents/{path}", {
      owner: repo.owner,
      repo: repo.repo,
      path,
      ref,
    });

    if (Array.isArray(response) || response.type !== "file") {
      throw new Error(`${path} is not a file`);
    }

    // Decode content
    const content = Buffer.from(response.content, "base64").toString("utf8");
    const format = this.detectFormat(path);

    // Parse definition
    let definition: AgentDefinition;
    try {
      const parsed = this.parseAgentFile(content, format);
      definition =
        options.validate !== false
          ? AgentDefinitionSchema.parse(parsed)
          : (parsed as AgentDefinition);
    } catch (error) {
      throw new Error(
        `Failed to parse agent at ${path}: ${(error as Error).message}`
      );
    }

    // Build source info
    const source: AgentSource = {
      repo,
      path,
      branch: ref,
      sha: response.sha,
      url: response.html_url || "",
      rawUrl: response.download_url || "",
    };

    // Build metadata
    const framework = this.detectFrameworkFromContent(content, format);
    const dependencies = options.resolveDependencies
      ? await this.resolveDependencies(repo, path, content, ref)
      : this.extractDependencies(content, format);

    const metadata: AgentMetadata = {
      importedAt: new Date(),
      lastChecked: new Date(),
      format,
      framework,
      dependencies,
      relatedFiles: await this.findRelatedFiles(repo, path, ref),
      size: response.size,
      hash: this.hashContent(content),
    };

    return { definition, source, metadata };
  }

  /**
   * Import agents from multiple repositories
   *
   * @param repos - Array of repository identifiers
   * @param options - Import options
   * @returns Map of repo to imported agents
   */
  async importFromRepos(
    repos: RepoIdentifier[],
    options: ImportOptions = {}
  ): Promise<Map<string, ImportedAgent[]>> {
    const results = new Map<string, ImportedAgent[]>();

    // Import in parallel with concurrency limit
    const concurrency = 5;
    const chunks = this.chunkArray(repos, concurrency);

    for (const chunk of chunks) {
      const promises = chunk.map(async (repo) => {
        const key = `${repo.owner}/${repo.repo}`;
        try {
          const agents = await this.importFromRepo(repo, options);
          results.set(key, agents);
        } catch (error) {
          results.set(key, []);
        }
      });

      await Promise.all(promises);
    }

    return results;
  }

  // ==========================================================================
  // WATCHING
  // ==========================================================================

  /**
   * Watch a repository for agent changes
   *
   * @param repo - Repository identifier
   * @param callback - Callback when agent changes
   * @param options - Watch options
   * @returns Stop function
   *
   * @example
   * ```typescript
   * const stop = await importer.watchRepo(
   *   { owner: 'org', repo: 'agents' },
   *   (agent, changeType) => {
   *     console.log(`Agent ${agent.definition.name} was ${changeType}`);
   *   },
   *   { interval: 60000 }
   * );
   *
   * // Later...
   * stop();
   * ```
   */
  async watchRepo(
    repo: RepoIdentifier,
    callback: (
      agent: ImportedAgent,
      changeType: "added" | "modified" | "removed"
    ) => void,
    options: WatchOptions = {}
  ): Promise<() => void> {
    const key = `${repo.owner}/${repo.repo}`;
    const interval = options.interval || 60000;
    const branch = options.branch || (await this.getDefaultBranch(repo));

    // Initial scan
    let lastAgents = await this.importFromRepo(repo, { branch });
    const agentMap = new Map(lastAgents.map((a) => [a.source.path, a]));

    // Polling function
    const poll = async () => {
      try {
        const currentAgents = await this.importFromRepo(repo, {
          branch,
          skipCached: false,
        });

        const currentMap = new Map(
          currentAgents.map((a) => [a.source.path, a])
        );

        // Check for added/modified
        for (const [path, agent] of currentMap) {
          const existing = agentMap.get(path);
          if (!existing) {
            if (!options.changeTypes || options.changeTypes.includes("added")) {
              callback(agent, "added");
            }
          } else if (existing.source.sha !== agent.source.sha) {
            if (
              !options.changeTypes ||
              options.changeTypes.includes("modified")
            ) {
              callback(agent, "modified");
            }
          }
        }

        // Check for removed
        for (const [path, agent] of agentMap) {
          if (!currentMap.has(path)) {
            if (
              !options.changeTypes ||
              options.changeTypes.includes("removed")
            ) {
              callback(agent, "removed");
            }
          }
        }

        // Update state
        lastAgents = currentAgents;
        agentMap.clear();
        currentAgents.forEach((a) => agentMap.set(a.source.path, a));
      } catch (error) {
        // Ignore polling errors
      }
    };

    // Start polling
    const intervalId = setInterval(poll, interval);
    this.watchIntervals.set(key, intervalId);

    // Return stop function
    return () => {
      clearInterval(intervalId);
      this.watchIntervals.delete(key);
    };
  }

  /**
   * Stop watching all repositories
   */
  stopAllWatches(): void {
    for (const [key, intervalId] of this.watchIntervals) {
      clearInterval(intervalId);
      this.watchIntervals.delete(key);
    }
  }

  // ==========================================================================
  // UTILITIES
  // ==========================================================================

  /**
   * Validate an agent definition
   *
   * @param definition - Agent definition to validate
   * @returns Validation result
   */
  validateAgent(definition: unknown): {
    valid: boolean;
    errors: string[];
  } {
    try {
      AgentDefinitionSchema.parse(definition);
      return { valid: true, errors: [] };
    } catch (error) {
      const zodError = error as {
        errors?: Array<{ path: string[]; message: string }>;
      };
      const errors = zodError.errors?.map(
        (e) => `${e.path.join(".")}: ${e.message}`
      ) || [(error as Error).message];
      return { valid: false, errors };
    }
  }

  /**
   * Export an agent to a file format
   *
   * @param agent - Imported agent
   * @param format - Export format
   * @returns Formatted string
   */
  exportAgent(agent: ImportedAgent, format: "yaml" | "json" = "yaml"): string {
    if (format === "json") {
      return JSON.stringify(agent.definition, null, 2);
    }
    return yaml.stringify(agent.definition);
  }

  /**
   * Clear the import cache
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { size: number; expired: number } {
    let expired = 0;
    const now = Date.now();

    for (const [, entry] of this.cache) {
      if (entry.expiresAt < now) {
        expired++;
      }
    }

    return { size: this.cache.size, expired };
  }

  // ==========================================================================
  // PRIVATE HELPERS
  // ==========================================================================

  private async getDefaultBranch(repo: RepoIdentifier): Promise<string> {
    const response = await this.client.rest<{ default_branch: string }>(
      "GET /repos/{owner}/{repo}",
      {
        owner: repo.owner,
        repo: repo.repo,
      }
    );
    return response.default_branch;
  }

  private async getFileTree(
    repo: RepoIdentifier,
    branch: string,
    options: ImportOptions
  ): Promise<Array<{ path: string; sha: string; size?: number }>> {
    const response = await this.client.rest<{
      sha: string;
      url: string;
      tree: Array<{
        path?: string;
        mode?: string;
        type?: string;
        sha?: string;
        size?: number;
        url?: string;
      }>;
      truncated: boolean;
    }>("GET /repos/{owner}/{repo}/git/trees/{tree_sha}", {
      owner: repo.owner,
      repo: repo.repo,
      tree_sha: branch,
      recursive: options.recursive ? "true" : undefined,
    });

    return response.tree
      .filter((item) => item.type === "blob" && item.path)
      .map((item) => ({
        path: item.path!,
        sha: item.sha!,
        size: item.size,
      }));
  }

  private isAgentFile(path: string, options: ImportOptions): boolean {
    const filename = path.split("/").pop() || "";

    // Check include patterns
    if (options.include?.length) {
      const matches = options.include.some((p) => this.matchPattern(path, p));
      if (!matches) return false;
    }

    // Check exclude patterns
    const excludes = options.exclude || DEFAULT_EXCLUDE_PATTERNS;
    if (excludes.some((p) => this.matchPattern(path, p))) {
      return false;
    }

    // Check if filename matches agent patterns
    return AGENT_FILE_PATTERNS.some((pattern) => {
      if (pattern.includes("*")) {
        const regex = new RegExp(
          "^" + pattern.replace(/\*/g, ".*").replace(/\./g, "\\.") + "$"
        );
        return regex.test(filename);
      }
      return filename === pattern || path.endsWith(`/${pattern}`);
    });
  }

  private matchPattern(path: string, pattern: string): boolean {
    const regex = new RegExp(
      "^" +
        pattern
          .replace(/\*\*/g, "<<<GLOBSTAR>>>")
          .replace(/\*/g, "[^/]*")
          .replace(/<<<GLOBSTAR>>>/g, ".*")
          .replace(/\./g, "\\.") +
        "$"
    );
    return regex.test(path);
  }

  private extractAgentName(path: string): string {
    const filename = path.split("/").pop() || "";
    const dir = path.split("/").slice(-2, -1)[0] || "";

    // Try to get name from filename
    const nameMatch = filename.match(/^(.+?)\.agent\.(yaml|yml|json)$/);
    if (nameMatch) return nameMatch[1];

    // Use directory name
    if (dir && dir !== "agents") return dir;

    // Use filename without extension
    return filename.replace(/\.(yaml|yml|json)$/, "");
  }

  private detectFormat(path: string): AgentFormat {
    if (path.endsWith(".yaml") || path.endsWith(".yml")) return "yaml";
    if (path.endsWith(".json")) return "json";
    if (path.endsWith(".toml")) return "toml";
    if (path.endsWith(".py")) return "python";
    if (path.endsWith(".ts")) return "typescript";
    return "yaml";
  }

  private detectFramework(path: string): AgentFramework {
    const lower = path.toLowerCase();
    if (lower.includes("crewai") || lower.includes("crew")) return "crewai";
    if (lower.includes("langchain")) return "langchain";
    if (lower.includes("autogpt")) return "autogpt";
    if (lower.includes("autogen")) return "autogen";
    if (lower.includes("semantic-kernel")) return "semantic-kernel";
    if (lower.includes("llamaindex")) return "llamaindex";
    return "unknown";
  }

  private detectFrameworkFromContent(
    content: string,
    format: AgentFormat
  ): AgentFramework {
    const lower = content.toLowerCase();

    // Check for framework-specific patterns
    if (lower.includes("from crewai") || lower.includes("crewai:"))
      return "crewai";
    if (lower.includes("from langchain") || lower.includes("langchain:"))
      return "langchain";
    if (lower.includes("autogpt")) return "autogpt";
    if (lower.includes("from autogen") || lower.includes("autogen:"))
      return "autogen";
    if (lower.includes("semantic_kernel") || lower.includes("semantic-kernel"))
      return "semantic-kernel";
    if (lower.includes("llama_index") || lower.includes("llamaindex"))
      return "llamaindex";

    // Check for custom Neurectomy format
    if (lower.includes("neurectomy:") || lower.includes("@neurectomy"))
      return "custom";

    return "unknown";
  }

  private parseAgentFile(content: string, format: AgentFormat): unknown {
    switch (format) {
      case "yaml":
        return yaml.parse(content);
      case "json":
        return JSON.parse(content);
      case "toml":
        // Would need toml parser
        throw new Error("TOML parsing not yet implemented");
      case "python":
      case "typescript":
        // Extract config from code
        return this.extractConfigFromCode(content, format);
      default:
        throw new Error(`Unsupported format: ${format}`);
    }
  }

  private extractConfigFromCode(content: string, format: AgentFormat): unknown {
    // Try to extract YAML/JSON config from code comments or strings
    const yamlMatch = content.match(/"""[\s\S]*?agent:\s*([\s\S]*?)"""/);
    if (yamlMatch) {
      return yaml.parse(`agent:\n${yamlMatch[1]}`);
    }

    const jsonMatch = content.match(/AGENT_CONFIG\s*=\s*({[\s\S]*?})/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[1]);
    }

    throw new Error(`Could not extract agent config from ${format} file`);
  }

  private extractDependencies(
    content: string,
    format: AgentFormat
  ): AgentDependency[] {
    const deps: AgentDependency[] = [];

    // Extract model dependencies
    const modelMatches = content.matchAll(/model[:\s]+["']?([^"'\s,}]+)/gi);
    for (const match of modelMatches) {
      deps.push({ type: "model", name: match[1] });
    }

    // Extract tool dependencies
    const toolMatches = content.matchAll(/tools?[:\s]+\[(.*?)\]/gi);
    for (const match of toolMatches) {
      const tools = match[1]
        .split(",")
        .map((t) => t.trim().replace(/["']/g, ""));
      tools.forEach((tool) => {
        if (tool) deps.push({ type: "tool", name: tool });
      });
    }

    return deps;
  }

  private async resolveDependencies(
    repo: RepoIdentifier,
    path: string,
    content: string,
    branch: string
  ): Promise<AgentDependency[]> {
    const deps = this.extractDependencies(content, this.detectFormat(path));

    // Try to find requirements.txt or package.json nearby
    const dir = path.split("/").slice(0, -1).join("/");

    try {
      // Check for Python requirements
      const reqPath = dir ? `${dir}/requirements.txt` : "requirements.txt";
      const reqResponse = await this.client
        .rest<{
          type: string;
          encoding: string;
          content: string;
        }>("GET /repos/{owner}/{repo}/contents/{path}", {
          owner: repo.owner,
          repo: repo.repo,
          path: reqPath,
          ref: branch,
        })
        .catch(() => null);

      if (
        reqResponse &&
        !Array.isArray(reqResponse) &&
        reqResponse.type === "file"
      ) {
        const reqContent = Buffer.from(reqResponse.content, "base64").toString(
          "utf8"
        );
        const lines = reqContent.split("\n");

        for (const line of lines) {
          const match = line.match(/^([a-zA-Z0-9_-]+)(?:([>=<]+)(.+))?/);
          if (match) {
            deps.push({
              type: "pip",
              name: match[1],
              version: match[3],
            });
          }
        }
      }
    } catch {
      // Ignore dependency resolution errors
    }

    return deps;
  }

  private async findRelatedFiles(
    repo: RepoIdentifier,
    agentPath: string,
    branch: string
  ): Promise<string[]> {
    const dir = agentPath.split("/").slice(0, -1).join("/") || ".";
    const related: string[] = [];

    try {
      const response = await this.client.rest<
        | Array<{
            type: string;
            name: string;
            path: string;
            sha: string;
            size: number;
          }>
        | { type: string; name: string; path: string }
      >("GET /repos/{owner}/{repo}/contents/{path}", {
        owner: repo.owner,
        repo: repo.repo,
        path: dir,
        ref: branch,
      });

      if (Array.isArray(response)) {
        for (const item of response) {
          if (item.type === "file" && item.path !== agentPath) {
            // Include common related file types
            if (/\.(py|ts|js|yaml|yml|json|md|txt)$/.test(item.name)) {
              related.push(item.path);
            }
          }
        }
      }
    } catch {
      // Ignore errors
    }

    return related;
  }

  private getCacheKey(repo: RepoIdentifier, path: string): string {
    return `${repo.owner}/${repo.repo}/${path}`;
  }

  private getCached(key: string): ImportedAgent | undefined {
    const entry = this.cache.get(key);
    if (!entry) return undefined;
    if (entry.expiresAt < Date.now()) {
      this.cache.delete(key);
      return undefined;
    }
    return entry.agent;
  }

  private detectChanges(
    previous: ImportedAgent,
    current: ImportedAgent
  ): AgentChanges {
    const fields: string[] = [];

    // Compare definition fields
    const prevDef = previous.definition;
    const currDef = current.definition;

    if (prevDef.name !== currDef.name) fields.push("name");
    if (prevDef.description !== currDef.description) fields.push("description");
    if (prevDef.version !== currDef.version) fields.push("version");
    if (
      JSON.stringify(prevDef.capabilities) !==
      JSON.stringify(currDef.capabilities)
    ) {
      fields.push("capabilities");
    }
    if (JSON.stringify(prevDef.tools) !== JSON.stringify(currDef.tools)) {
      fields.push("tools");
    }
    if (JSON.stringify(prevDef.config) !== JSON.stringify(currDef.config)) {
      fields.push("config");
    }

    return {
      fields,
      previousSha: previous.source.sha,
      newSha: current.source.sha,
    };
  }

  private hashContent(content: string): string {
    let hash = 0;
    for (let i = 0; i < content.length; i++) {
      const char = content.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16);
  }

  private buildSearchQuery(owner: string, options: AgentSearchOptions): string {
    const parts: string[] = [
      `org:${owner}`,
      "filename:agent",
      "extension:yaml OR extension:yml OR extension:json",
    ];

    if (options.query) {
      parts.push(options.query);
    }

    if (options.framework) {
      parts.push(`"${options.framework}"`);
    }

    if (options.capabilities?.length) {
      options.capabilities.forEach((cap) => {
        parts.push(`"${cap}"`);
      });
    }

    return parts.join(" ");
  }

  private chunkArray<T>(array: T[], size: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    return chunks;
  }
}
