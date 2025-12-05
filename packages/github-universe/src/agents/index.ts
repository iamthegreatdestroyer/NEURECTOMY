/**
 * @fileoverview GitHub Agent Importer module exports
 * @module @neurectomy/github-universe/agents
 *
 * @description Provides functionality for importing AI agent definitions from
 * GitHub repositories with support for multiple frameworks and automatic
 * dependency resolution.
 *
 * @example
 * ```typescript
 * import { AgentImporter, type ImportedAgent } from '@neurectomy/github-universe/agents';
 *
 * const importer = new AgentImporter(client);
 *
 * // Import agents from a repository
 * const agents = await importer.importFromRepo(
 *   { owner: 'org', repo: 'ai-agents' },
 *   { recursive: true, frameworks: ['crewai', 'langchain'] }
 * );
 *
 * // Watch for changes
 * const stop = await importer.watchRepo(
 *   { owner: 'org', repo: 'ai-agents' },
 *   (agent) => console.log(`Updated: ${agent.definition.name}`)
 * );
 * ```
 */

export { AgentImporter } from "./importer";
export type {
  AgentImporterEvents,
  DiscoveredAgent,
  ImportedAgent,
  AgentFormat,
  AgentFramework,
  AgentSource,
  AgentMetadata,
  AgentDependency,
  AgentChanges,
  ImportStats,
  ImportOptions,
  WatchOptions,
  AgentSearchOptions,
} from "./importer";
