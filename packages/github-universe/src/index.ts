/**
 * @fileoverview GitHub Universe - Comprehensive GitHub Integration Package
 * @module @neurectomy/github-universe
 *
 * @description The GitHub Universe package provides a complete suite of tools
 * for interacting with GitHub's APIs, managing repositories, branches, pull
 * requests, issues, workflows, webhooks, and importing AI agent definitions.
 *
 * @agent @SYNAPSE - Integration Engineering & API Design
 * @agent @APEX - Elite Computer Science Engineering
 *
 * Features:
 * - **GitHubClient**: Core API client with REST and GraphQL support
 * - **RepositoryManager**: Repository operations (clone, fork, create, configure)
 * - **BranchManager**: Branch operations (create, merge, delete, protect)
 * - **PRManager**: Pull request management (create, review, merge)
 * - **IssueManager**: Issue tracking (create, update, label, assign)
 * - **ActionsManager**: GitHub Actions (workflows, runs, artifacts)
 * - **WebhooksManager**: Webhook management and event verification
 * - **AgentImporter**: AI agent import from repositories
 *
 * @example Basic Usage
 * ```typescript
 * import { GitHubClient, RepositoryManager, BranchManager } from '@neurectomy/github-universe';
 *
 * // Initialize client
 * const client = new GitHubClient({
 *   auth: { type: 'token', token: process.env.GITHUB_TOKEN },
 * });
 *
 * // Use managers
 * const repos = new RepositoryManager(client);
 * const branches = new BranchManager(client);
 *
 * // Get repository info
 * const repo = await repos.getRepository({ owner: 'org', repo: 'app' });
 * console.log(`Repository: ${repo.fullName}, Stars: ${repo.stargazersCount}`);
 *
 * // Create a feature branch
 * const branch = await branches.createBranch(
 *   { owner: 'org', repo: 'app' },
 *   'feature/new-feature',
 *   'main'
 * );
 * ```
 *
 * @example Complete Workflow
 * ```typescript
 * import {
 *   GitHubClient,
 *   RepositoryManager,
 *   BranchManager,
 *   PRManager,
 *   IssueManager,
 * } from '@neurectomy/github-universe';
 *
 * const client = new GitHubClient({
 *   auth: { type: 'token', token: process.env.GITHUB_TOKEN },
 * });
 *
 * const repos = new RepositoryManager(client);
 * const branches = new BranchManager(client);
 * const prs = new PRManager(client);
 * const issues = new IssueManager(client);
 *
 * // 1. Clone repository
 * await repos.cloneRepository(
 *   { owner: 'org', repo: 'app' },
 *   './workspace/app'
 * );
 *
 * // 2. Create feature branch
 * await branches.createBranch(
 *   { owner: 'org', repo: 'app' },
 *   'feature/awesome',
 *   'main'
 * );
 *
 * // 3. After making changes, create PR
 * const pr = await prs.createPullRequest(
 *   { owner: 'org', repo: 'app' },
 *   {
 *     title: 'feat: Add awesome feature',
 *     head: 'feature/awesome',
 *     base: 'main',
 *     body: 'This PR adds an awesome new feature.',
 *   }
 * );
 *
 * // 4. Create tracking issue
 * await issues.createIssue(
 *   { owner: 'org', repo: 'app' },
 *   {
 *     title: 'Track: Awesome feature implementation',
 *     body: `Tracking PR #${pr.number}`,
 *     labels: ['enhancement', 'in-progress'],
 *   }
 * );
 * ```
 *
 * @example Webhook Handling
 * ```typescript
 * import { GitHubClient, WebhooksManager } from '@neurectomy/github-universe';
 * import express from 'express';
 *
 * const app = express();
 * const client = new GitHubClient({ auth: { type: 'token', token: '...' } });
 * const webhooks = new WebhooksManager(client);
 *
 * // Create webhook
 * await webhooks.createWebhook(
 *   { owner: 'org', repo: 'app' },
 *   {
 *     events: ['push', 'pull_request'],
 *     config: {
 *       url: 'https://api.example.com/webhooks/github',
 *       contentType: 'json',
 *       secret: process.env.WEBHOOK_SECRET,
 *     },
 *   }
 * );
 *
 * // Handle webhooks
 * app.post('/webhooks/github', express.raw({ type: '*\/*' }), (req, res) => {
 *   const payload = webhooks.parsePayload(
 *     req.body,
 *     req.headers,
 *     process.env.WEBHOOK_SECRET
 *   );
 *
 *   console.log(`Received ${payload.event} event`);
 *   res.status(200).send('OK');
 * });
 * ```
 *
 * @example Agent Import
 * ```typescript
 * import { GitHubClient, AgentImporter } from '@neurectomy/github-universe';
 *
 * const client = new GitHubClient({ auth: { type: 'token', token: '...' } });
 * const importer = new AgentImporter(client);
 *
 * // Discover agents
 * const discovered = await importer.discoverAgents(
 *   { owner: 'org', repo: 'ai-agents' },
 *   { recursive: true }
 * );
 *
 * // Import all agents
 * const agents = await importer.importFromRepo(
 *   { owner: 'org', repo: 'ai-agents' },
 *   { frameworks: ['crewai', 'langchain'] }
 * );
 *
 * // Watch for updates
 * await importer.watchRepo(
 *   { owner: 'org', repo: 'ai-agents' },
 *   (agent, changeType) => {
 *     console.log(`Agent ${agent.definition.name} was ${changeType}`);
 *   }
 * );
 * ```
 */

// ============================================================================
// CORE
// ============================================================================

export {
  GitHubClient,
  createGitHubClient,
  createEnterpriseClient,
} from "./client";
export type { GitHubClientEvents } from "./client";

// ============================================================================
// TYPES
// ============================================================================

export {
  // Auth
  GitHubAuthConfigSchema,
  type GitHubAuthConfig,

  // Repository
  VisibilitySchema,
  type Visibility,
  RepoIdentifierSchema,
  type RepoIdentifier,
  CreateRepoConfigSchema,
  type CreateRepoConfig,
  RepositorySchema,
  type Repository,

  // Branch
  BranchSchema,
  type Branch,
  BranchProtectionSchema,
  type BranchProtection,

  // Pull Request
  PRStateSchema,
  type PRState,
  CreatePRConfigSchema,
  type CreatePRConfig,
  PullRequestSchema,
  type PullRequest,

  // Review
  ReviewStateSchema,
  type ReviewState,

  // Issue
  IssueStateSchema,
  type IssueState,
  CreateIssueConfigSchema,
  type CreateIssueConfig,
  IssueSchema,
  type Issue,

  // Workflow
  WorkflowStatusSchema,
  type WorkflowStatus,
  WorkflowConclusionSchema,
  type WorkflowConclusion,
  WorkflowSchema,
  type Workflow as WorkflowType,
  WorkflowRunSchema,
  type WorkflowRun as WorkflowRunType,

  // Webhook
  WebhookEventSchema,
  type WebhookEvent,
  WebhookConfigSchema,
  type WebhookConfig,
  CreateWebhookConfigSchema,
  type CreateWebhookConfig,
  WebhookSchema,
  type Webhook,

  // Agent
  AgentDefinitionSchema,
  type AgentDefinition,
} from "./types";

// ============================================================================
// MANAGERS
// ============================================================================

// Repository Manager
export { RepositoryManager } from "./repository";
export type { RepositoryManagerEvents } from "./repository";

// Branch Manager
export { BranchManager } from "./branch";
export type {
  BranchManagerEvents,
  BranchComparison,
  MergeResult,
} from "./branch";

// PR Manager
export { PRManager } from "./pr";
export type { PRManagerEvents } from "./pr";

// Issue Manager
export { IssueManager } from "./issues";
export type { IssueManagerEvents, IssueListOptions } from "./issues";

// Actions Manager
export { ActionsManager } from "./actions";
export type {
  ActionsManagerEvents,
  Workflow,
  WorkflowRun,
  WorkflowJob,
  WorkflowStep,
  Artifact,
  WorkflowDispatchInputs,
  WorkflowListOptions,
  WorkflowRunListOptions,
  WorkflowUsage,
} from "./actions";

// Webhooks Manager
export { WebhooksManager } from "./webhooks";
export type {
  WebhooksManagerEvents,
  WebhookDelivery,
  WebhookPayload,
  UpdateWebhookConfig,
  WebhookListOptions,
  OrgIdentifier,
  WebhookRegistrationResult,
} from "./webhooks";

// Agent Importer
export { AgentImporter } from "./agents";
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
} from "./agents";

// ============================================================================
// CONVENIENCE FACTORY
// ============================================================================

import { GitHubClient, type GitHubUniverseConfig } from "./client";
import { RepositoryManager } from "./repository";
import { BranchManager } from "./branch";
import { PRManager } from "./pr";
import { IssueManager } from "./issues";
import { ActionsManager } from "./actions";
import { WebhooksManager } from "./webhooks";
import { AgentImporter } from "./agents";

/**
 * All GitHub managers in a single object
 */
export interface GitHubUniverse {
  client: GitHubClient;
  repositories: RepositoryManager;
  branches: BranchManager;
  pullRequests: PRManager;
  issues: IssueManager;
  actions: ActionsManager;
  webhooks: WebhooksManager;
  agents: AgentImporter;
}

/**
 * Create all GitHub managers with a single function
 *
 * @param config - GitHub client configuration
 * @returns All managers initialized with the same client
 *
 * @example
 * ```typescript
 * import { createGitHubUniverse } from '@neurectomy/github-universe';
 *
 * const github = createGitHubUniverse({
 *   auth: { type: 'token', token: process.env.GITHUB_TOKEN },
 * });
 *
 * // Use any manager
 * const repo = await github.repositories.getRepository({ owner: 'org', repo: 'app' });
 * const branches = await github.branches.listBranches({ owner: 'org', repo: 'app' });
 * const prs = await github.pullRequests.listPullRequests({ owner: 'org', repo: 'app' });
 * ```
 */
export function createGitHubUniverse(
  config: Partial<GitHubUniverseConfig> & { auth: GitHubUniverseConfig["auth"] }
): GitHubUniverse {
  const client = new GitHubClient(config);

  return {
    client,
    repositories: new RepositoryManager(client),
    branches: new BranchManager(client),
    pullRequests: new PRManager(client),
    issues: new IssueManager(client),
    actions: new ActionsManager(client),
    webhooks: new WebhooksManager(client),
    agents: new AgentImporter(client),
  };
}

// Export the config type
export type { GitHubUniverseConfig };

/**
 * Default export for easy import
 */
export default createGitHubUniverse;
