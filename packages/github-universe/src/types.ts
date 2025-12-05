/**
 * @fileoverview GitHub Universe Core Types
 * @module @neurectomy/github-universe/types
 *
 * Comprehensive type definitions for GitHub API integration.
 */

import { z } from "zod";

// =============================================================================
// AUTHENTICATION
// =============================================================================

/**
 * GitHub authentication configuration
 */
export const GitHubAuthConfigSchema = z.object({
  /** Personal Access Token (classic or fine-grained) */
  token: z.string().min(1),
  /** OAuth App Client ID (for OAuth flow) */
  clientId: z.string().optional(),
  /** OAuth App Client Secret */
  clientSecret: z.string().optional(),
  /** GitHub App ID (for GitHub App auth) */
  appId: z.number().optional(),
  /** GitHub App Private Key (PEM format) */
  privateKey: z.string().optional(),
  /** GitHub App Installation ID */
  installationId: z.number().optional(),
  /** Enterprise server URL (for GitHub Enterprise) */
  baseUrl: z.string().url().optional(),
  /** API version */
  apiVersion: z.string().default("2022-11-28"),
});

export type GitHubAuthConfig = z.infer<typeof GitHubAuthConfigSchema>;

/**
 * Authentication type enum
 */
export const AuthTypeSchema = z.enum(["token", "oauth", "app", "installation"]);
export type AuthType = z.infer<typeof AuthTypeSchema>;

// =============================================================================
// REPOSITORY TYPES
// =============================================================================

/**
 * Repository visibility
 */
export const VisibilitySchema = z.enum(["public", "private", "internal"]);
export type Visibility = z.infer<typeof VisibilitySchema>;

/**
 * Repository identifier
 */
export const RepoIdentifierSchema = z.object({
  owner: z.string().min(1),
  repo: z.string().min(1),
});

export type RepoIdentifier = z.infer<typeof RepoIdentifierSchema>;

/**
 * Repository configuration for creation
 */
export const CreateRepoConfigSchema = z.object({
  name: z.string().min(1).max(100),
  description: z.string().max(1000).optional(),
  visibility: VisibilitySchema.default("private"),
  autoInit: z.boolean().default(true),
  gitignoreTemplate: z.string().optional(),
  licenseTemplate: z.string().optional(),
  allowSquashMerge: z.boolean().default(true),
  allowMergeCommit: z.boolean().default(true),
  allowRebaseMerge: z.boolean().default(true),
  allowAutoMerge: z.boolean().default(false),
  deleteBranchOnMerge: z.boolean().default(true),
  hasIssues: z.boolean().default(true),
  hasProjects: z.boolean().default(true),
  hasWiki: z.boolean().default(true),
  hasDiscussions: z.boolean().default(false),
  isTemplate: z.boolean().default(false),
  homepage: z.string().url().optional(),
  topics: z.array(z.string()).max(20).optional(),
});

export type CreateRepoConfig = z.infer<typeof CreateRepoConfigSchema>;

/**
 * Repository data from GitHub API
 */
export const RepositorySchema = z.object({
  id: z.number(),
  nodeId: z.string(),
  name: z.string(),
  fullName: z.string(),
  owner: z.object({
    login: z.string(),
    id: z.number(),
    avatarUrl: z.string().optional(),
    type: z.enum(["User", "Organization"]),
  }),
  description: z.string().nullable(),
  visibility: VisibilitySchema,
  private: z.boolean(),
  fork: z.boolean(),
  archived: z.boolean(),
  disabled: z.boolean(),
  defaultBranch: z.string(),
  language: z.string().nullable(),
  languages: z.record(z.number()).optional(),
  stargazersCount: z.number(),
  watchersCount: z.number(),
  forksCount: z.number(),
  openIssuesCount: z.number(),
  size: z.number(),
  topics: z.array(z.string()),
  htmlUrl: z.string().url(),
  cloneUrl: z.string().url(),
  sshUrl: z.string(),
  gitUrl: z.string(),
  createdAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
  pushedAt: z.string().datetime().nullable(),
  license: z
    .object({
      key: z.string(),
      name: z.string(),
      spdxId: z.string().nullable(),
    })
    .nullable(),
  permissions: z
    .object({
      admin: z.boolean(),
      maintain: z.boolean().optional(),
      push: z.boolean(),
      triage: z.boolean().optional(),
      pull: z.boolean(),
    })
    .optional(),
});

export type Repository = z.infer<typeof RepositorySchema>;

/**
 * Clone options
 */
export const CloneOptionsSchema = z.object({
  /** Target directory path */
  path: z.string(),
  /** Branch to clone */
  branch: z.string().optional(),
  /** Shallow clone depth */
  depth: z.number().int().positive().optional(),
  /** Clone with submodules */
  recursive: z.boolean().default(false),
  /** Clone as mirror */
  mirror: z.boolean().default(false),
  /** Clone as bare repository */
  bare: z.boolean().default(false),
  /** Single branch clone */
  singleBranch: z.boolean().default(false),
  /** Progress callback */
  onProgress: z.function().args(z.number()).returns(z.void()).optional(),
});

export type CloneOptions = z.infer<typeof CloneOptionsSchema>;

/**
 * Fork configuration
 */
export const ForkConfigSchema = z.object({
  /** Organization to fork to (if not personal) */
  organization: z.string().optional(),
  /** New name for the fork */
  name: z.string().optional(),
  /** Copy default branch only */
  defaultBranchOnly: z.boolean().default(false),
});

export type ForkConfig = z.infer<typeof ForkConfigSchema>;

// =============================================================================
// BRANCH TYPES
// =============================================================================

/**
 * Branch protection rules
 * Supports both simplified flat structure and nested GitHub API structure
 */
export const BranchProtectionSchema = z.object({
  // === Simplified flat properties (used by manager code) ===
  /** Enable status checks (flat) */
  requireStatusChecks: z.boolean().optional(),
  /** Strict status checks (flat) */
  strictStatusChecks: z.boolean().optional(),
  /** Required status check contexts (flat, array of strings) */
  requiredStatusChecks: z
    .union([
      z
        .object({
          strict: z.boolean(),
          contexts: z.array(z.string()),
        })
        .nullable(),
      z.array(z.string()),
    ])
    .optional(),
  /** Enable reviews (flat) */
  requireReviews: z.boolean().optional(),
  /** Dismiss stale reviews (flat) */
  dismissStaleReviews: z.boolean().optional(),
  /** Require code owners (flat) */
  requireCodeOwners: z.boolean().optional(),
  /** Required reviewer count (flat) */
  requiredReviewers: z.number().optional(),
  /** Enable push restrictions (flat) */
  restrictPushes: z.boolean().optional(),
  /** Allowed pushers (flat) */
  allowedPushers: z.array(z.string()).optional(),
  /** Require linear history (flat) */
  requireLinearHistory: z.boolean().optional(),
  /** Require conversation resolution (flat) */
  requireConversationResolution: z.boolean().optional(),
  /** Require signed commits (flat) */
  requireSignedCommits: z.boolean().optional(),

  // === Nested GitHub API structure ===
  /** Enforce all configured restrictions for administrators */
  enforceAdmins: z.boolean().default(false),
  /** Require pull request reviews (nested) */
  requiredPullRequestReviews: z
    .object({
      dismissStaleReviews: z.boolean().default(true),
      requireCodeOwnerReviews: z.boolean().default(false),
      requiredApprovingReviewCount: z.number().int().min(0).max(6).default(1),
      requireLastPushApproval: z.boolean().default(false),
      dismissalRestrictions: z
        .object({
          users: z.array(z.string()).optional(),
          teams: z.array(z.string()).optional(),
        })
        .optional(),
    })
    .nullable()
    .optional(),
  /** Restrict who can push (nested) */
  restrictions: z
    .object({
      users: z.array(z.string()),
      teams: z.array(z.string()),
      apps: z.array(z.string()).optional(),
    })
    .nullable()
    .optional(),
  /** Require linear history (nested) */
  requiredLinearHistory: z.boolean().optional(),
  /** Allow force pushes */
  allowForcePushes: z.boolean().optional(),
  /** Allow deletions */
  allowDeletions: z.boolean().optional(),
  /** Block creations */
  blockCreations: z.boolean().optional(),
  /** Require conversation resolution (nested) */
  requiredConversationResolution: z.boolean().optional(),
  /** Lock branch */
  lockBranch: z.boolean().optional(),
  /** Allow fork syncing */
  allowForkSyncing: z.boolean().optional(),
  /** Require signed commits (nested) */
  requiredSignatures: z.boolean().optional(),
});

export type BranchProtection = z.infer<typeof BranchProtectionSchema>;

/**
 * Branch data
 */
export const BranchSchema = z.object({
  name: z.string(),
  sha: z.string().optional(),
  commit: z.object({
    sha: z.string(),
    url: z.string().url().optional(),
    message: z.string().optional(),
    author: z
      .object({
        name: z.string(),
        email: z.string(),
        date: z.string(),
      })
      .optional(),
  }),
  protected: z.boolean(),
  protection: BranchProtectionSchema.optional(),
});

export type Branch = z.infer<typeof BranchSchema>;

/**
 * Merge method
 */
export const MergeMethodSchema = z.enum(["merge", "squash", "rebase"]);
export type MergeMethod = z.infer<typeof MergeMethodSchema>;

// =============================================================================
// PULL REQUEST TYPES
// =============================================================================

/**
 * Pull request state
 */
export const PRStateSchema = z.enum(["open", "closed", "all"]);
export type PRState = z.infer<typeof PRStateSchema>;

/**
 * Pull request review state
 */
export const ReviewStateSchema = z.enum([
  "APPROVED",
  "CHANGES_REQUESTED",
  "COMMENTED",
  "DISMISSED",
  "PENDING",
]);
export type ReviewState = z.infer<typeof ReviewStateSchema>;

/**
 * Pull request creation config
 */
export const CreatePRConfigSchema = z.object({
  title: z.string().min(1).max(256),
  body: z.string().max(65536).optional(),
  head: z.string().min(1),
  base: z.string().min(1),
  draft: z.boolean().default(false),
  maintainerCanModify: z.boolean().default(true),
  labels: z.array(z.string()).optional(),
  assignees: z.array(z.string()).optional(),
  reviewers: z.array(z.string()).optional(),
  teamReviewers: z.array(z.string()).optional(),
  milestone: z.number().optional(),
});

export type CreatePRConfig = z.infer<typeof CreatePRConfigSchema>;

/**
 * Pull request update config (extends create with state)
 */
export const UpdatePRConfigSchema = CreatePRConfigSchema.extend({
  state: z.enum(["open", "closed"]).optional(),
});

export type UpdatePRConfig = z.infer<typeof UpdatePRConfigSchema>;

/**
 * Pull request data (supports both REST and GraphQL responses)
 */
export const PullRequestSchema = z.object({
  id: z.union([z.number(), z.string()]), // REST returns number, GraphQL returns string
  nodeId: z.string(),
  number: z.number(),
  state: z.enum(["open", "closed"]),
  locked: z.boolean(),
  title: z.string(),
  body: z.string().nullable(),
  htmlUrl: z.string().url(),
  diffUrl: z.string().url().optional(),
  patchUrl: z.string().url().optional(),
  user: z
    .object({
      login: z.string(),
      id: z.union([z.number(), z.string()]),
      avatarUrl: z.string().optional(),
    })
    .optional(),
  head: z.object({
    ref: z.string(),
    sha: z.string(),
    repo: RepoIdentifierSchema.nullable().optional(),
  }),
  base: z.object({
    ref: z.string(),
    sha: z.string(),
    repo: RepoIdentifierSchema.optional(),
  }),
  draft: z.boolean(),
  merged: z.boolean(),
  mergeable: z.boolean().nullable(),
  mergeableState: z.string().nullable(),
  mergedBy: z
    .object({
      login: z.string(),
      id: z.union([z.number(), z.string()]).optional(),
    })
    .nullable(),
  mergedAt: z.string().nullable(),
  closedAt: z.string().nullable(),
  createdAt: z.string(),
  updatedAt: z.string(),
  /** Auto-merge settings */
  autoMerge: z
    .object({
      enabledAt: z.string().optional(),
      mergeMethod: z.enum(["merge", "squash", "rebase"]).optional(),
    })
    .optional(),
  comments: z.number().optional(),
  reviewComments: z.number().optional(),
  commits: z.number().optional(),
  /** Alias for commits for GraphQL compatibility */
  commitsCount: z.number().optional(),
  /** Alias for comments for GraphQL compatibility */
  commentsCount: z.number().optional(),
  additions: z.number().optional(),
  deletions: z.number().optional(),
  changedFiles: z.number().optional(),
  labels: z.array(
    z.object({
      id: z.union([z.number(), z.string()]),
      name: z.string(),
      color: z.string(),
      description: z.string().optional(),
    })
  ),
  assignees: z.array(
    z.object({
      login: z.string(),
      id: z.union([z.number(), z.string()]).optional(),
      avatarUrl: z.string().optional(),
    })
  ),
  requestedReviewers: z.array(
    z.union([
      z.object({
        login: z.string(),
        id: z.union([z.number(), z.string()]).optional(),
      }),
      z.string(), // Can be just a string (login name)
    ])
  ),
  milestone: z
    .object({
      number: z.number(),
      title: z.string(),
      state: z.enum(["open", "closed"]).optional(),
      id: z.union([z.number(), z.string()]).optional(),
    })
    .nullable(),
});

export type PullRequest = z.infer<typeof PullRequestSchema>;

/**
 * Review request
 */
export const CreateReviewSchema = z.object({
  body: z.string().optional(),
  event: z.enum(["APPROVE", "REQUEST_CHANGES", "COMMENT"]),
  commitId: z.string().optional(), // SHA of commit to review
  comments: z
    .array(
      z.object({
        path: z.string(),
        position: z.number().optional(),
        line: z.number().optional(),
        side: z.enum(["LEFT", "RIGHT"]).optional(),
        startLine: z.number().optional(),
        startSide: z.enum(["LEFT", "RIGHT"]).optional(),
        body: z.string(),
      })
    )
    .optional(),
});

export type CreateReview = z.infer<typeof CreateReviewSchema>;

// =============================================================================
// ISSUE TYPES
// =============================================================================

/**
 * Issue state
 */
export const IssueStateSchema = z.enum(["open", "closed", "all"]);
export type IssueState = z.infer<typeof IssueStateSchema>;

/**
 * Issue state reason
 */
export const IssueStateReasonSchema = z.enum([
  "completed",
  "not_planned",
  "reopened",
]);
export type IssueStateReason = z.infer<typeof IssueStateReasonSchema>;

/**
 * Issue creation config
 */
export const CreateIssueConfigSchema = z.object({
  title: z.string().min(1).max(256),
  body: z.string().max(65536).optional(),
  labels: z.array(z.string()).optional(),
  assignees: z.array(z.string()).optional(),
  milestone: z.number().optional(),
});

export type CreateIssueConfig = z.infer<typeof CreateIssueConfigSchema>;

/**
 * Issue data
 * Supports both REST API (numeric IDs) and GraphQL (string IDs)
 */
export const IssueSchema = z.object({
  id: z.union([z.number(), z.string()]),
  nodeId: z.string(),
  number: z.number(),
  state: z.enum(["open", "closed"]),
  stateReason: z.union([IssueStateReasonSchema, z.string()]).nullable(),
  title: z.string(),
  body: z.string().nullable(),
  htmlUrl: z.string(),
  // REST API uses 'user', GraphQL uses 'author'
  user: z
    .object({
      login: z.string(),
      id: z.union([z.number(), z.string()]).optional(),
      avatarUrl: z.string().optional(),
    })
    .optional(),
  author: z
    .object({
      login: z.string(),
      avatarUrl: z.string().optional(),
    })
    .nullable()
    .optional(),
  labels: z.array(
    z.object({
      id: z.union([z.number(), z.string()]),
      name: z.string(),
      color: z.string(),
      description: z.string().nullable().optional(),
    })
  ),
  assignees: z.array(
    z.object({
      login: z.string(),
      id: z.union([z.number(), z.string()]),
      avatarUrl: z.string().optional(),
    })
  ),
  milestone: z
    .object({
      id: z.union([z.number(), z.string()]).optional(),
      number: z.number(),
      title: z.string(),
      state: z.enum(["open", "closed"]),
      dueOn: z.string().optional(),
    })
    .nullable(),
  locked: z.boolean(),
  lockReason: z.string().nullable().optional(),
  comments: z.number().optional(),
  commentsCount: z.number().optional(),
  reactionsCount: z.number().optional(),
  closedAt: z.string().nullable(),
  createdAt: z.string(),
  updatedAt: z.string(),
  closedBy: z
    .object({
      login: z.string(),
      id: z.union([z.number(), z.string()]),
    })
    .nullable()
    .optional(),
  pullRequest: z
    .object({
      url: z.string(),
    })
    .optional(),
  isPinned: z.boolean().optional(),
});

export type Issue = z.infer<typeof IssueSchema>;

/**
 * Label configuration
 */
export const LabelConfigSchema = z.object({
  name: z.string().min(1).max(50),
  color: z.string().regex(/^[0-9a-fA-F]{6}$/),
  description: z.string().max(100).optional(),
});

export type LabelConfig = z.infer<typeof LabelConfigSchema>;

// =============================================================================
// GITHUB ACTIONS TYPES
// =============================================================================

/**
 * Workflow run status
 */
export const WorkflowStatusSchema = z.enum([
  "completed",
  "action_required",
  "cancelled",
  "failure",
  "neutral",
  "skipped",
  "stale",
  "success",
  "timed_out",
  "in_progress",
  "queued",
  "requested",
  "waiting",
  "pending",
]);
export type WorkflowStatus = z.infer<typeof WorkflowStatusSchema>;

/**
 * Workflow run conclusion
 */
export const WorkflowConclusionSchema = z.enum([
  "success",
  "failure",
  "neutral",
  "cancelled",
  "skipped",
  "timed_out",
  "action_required",
  "stale",
  "startup_failure",
]);
export type WorkflowConclusion = z.infer<typeof WorkflowConclusionSchema>;

/**
 * Workflow data
 */
export const WorkflowSchema = z.object({
  id: z.number(),
  nodeId: z.string(),
  name: z.string(),
  path: z.string(),
  state: z.enum([
    "active",
    "deleted",
    "disabled_fork",
    "disabled_inactivity",
    "disabled_manually",
  ]),
  createdAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
  url: z.string().url(),
  htmlUrl: z.string().url(),
  badgeUrl: z.string().url(),
});

export type Workflow = z.infer<typeof WorkflowSchema>;

/**
 * Workflow run data
 */
export const WorkflowRunSchema = z.object({
  id: z.number(),
  nodeId: z.string(),
  name: z.string().nullable(),
  headBranch: z.string().nullable(),
  headSha: z.string(),
  path: z.string(),
  runNumber: z.number(),
  runAttempt: z.number(),
  event: z.string(),
  status: WorkflowStatusSchema.nullable(),
  conclusion: WorkflowConclusionSchema.nullable(),
  workflowId: z.number(),
  htmlUrl: z.string().url(),
  createdAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
  runStartedAt: z.string().datetime().nullable(),
  actor: z.object({
    login: z.string(),
    id: z.number(),
  }),
  triggeringActor: z
    .object({
      login: z.string(),
      id: z.number(),
    })
    .nullable(),
});

export type WorkflowRun = z.infer<typeof WorkflowRunSchema>;

/**
 * Workflow job data
 */
export const WorkflowJobSchema = z.object({
  id: z.number(),
  runId: z.number(),
  runUrl: z.string().url(),
  nodeId: z.string(),
  headSha: z.string(),
  name: z.string(),
  status: WorkflowStatusSchema,
  conclusion: WorkflowConclusionSchema.nullable(),
  createdAt: z.string().datetime(),
  startedAt: z.string().datetime().nullable(),
  completedAt: z.string().datetime().nullable(),
  steps: z.array(
    z.object({
      name: z.string(),
      status: WorkflowStatusSchema,
      conclusion: WorkflowConclusionSchema.nullable(),
      number: z.number(),
      startedAt: z.string().datetime().nullable(),
      completedAt: z.string().datetime().nullable(),
    })
  ),
  labels: z.array(z.string()),
  runnerName: z.string().nullable(),
  runnerGroupName: z.string().nullable(),
});

export type WorkflowJob = z.infer<typeof WorkflowJobSchema>;

/**
 * Workflow dispatch inputs
 */
export const WorkflowDispatchSchema = z.object({
  ref: z.string(),
  inputs: z.record(z.string()).optional(),
});

export type WorkflowDispatch = z.infer<typeof WorkflowDispatchSchema>;

// =============================================================================
// WEBHOOK TYPES
// =============================================================================

/**
 * Webhook event types
 */
export const WebhookEventSchema = z.enum([
  "branch_protection_rule",
  "check_run",
  "check_suite",
  "code_scanning_alert",
  "commit_comment",
  "create",
  "delete",
  "dependabot_alert",
  "deployment",
  "deployment_status",
  "discussion",
  "discussion_comment",
  "fork",
  "gollum",
  "issue_comment",
  "issues",
  "label",
  "member",
  "merge_group",
  "milestone",
  "page_build",
  "ping",
  "project",
  "project_card",
  "project_column",
  "public",
  "pull_request",
  "pull_request_review",
  "pull_request_review_comment",
  "pull_request_review_thread",
  "push",
  "registry_package",
  "release",
  "repository",
  "repository_dispatch",
  "repository_vulnerability_alert",
  "secret_scanning_alert",
  "star",
  "status",
  "watch",
  "workflow_dispatch",
  "workflow_job",
  "workflow_run",
]);

export type WebhookEvent = z.infer<typeof WebhookEventSchema>;

/**
 * Webhook configuration
 */
export const WebhookConfigSchema = z.object({
  url: z.string().url(),
  contentType: z.enum(["json", "form"]).default("json"),
  secret: z.string().optional(),
  insecureSsl: z.enum(["0", "1"]).default("0"),
});

export type WebhookConfig = z.infer<typeof WebhookConfigSchema>;

/**
 * Webhook creation config
 */
export const CreateWebhookConfigSchema = z.object({
  name: z.string().default("web"),
  active: z.boolean().default(true),
  events: z.array(WebhookEventSchema).min(1),
  config: WebhookConfigSchema,
});

export type CreateWebhookConfig = z.infer<typeof CreateWebhookConfigSchema>;

/**
 * Webhook data
 */
export const WebhookSchema = z.object({
  id: z.number(),
  type: z.string(),
  name: z.string(),
  active: z.boolean(),
  events: z.array(z.string()),
  config: z.object({
    url: z.string().url(),
    contentType: z.string(),
    insecureSsl: z.string(),
  }),
  createdAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
  lastResponse: z.object({
    code: z.number().nullable(),
    status: z.string().nullable(),
    message: z.string().nullable(),
  }),
});

export type Webhook = z.infer<typeof WebhookSchema>;

// =============================================================================
// AGENT IMPORT TYPES
// =============================================================================

/**
 * Agent definition schema (for importing agents from repos)
 */
export const AgentDefinitionSchema = z.object({
  id: z.string().uuid().optional(),
  name: z.string().min(1).max(100),
  version: z.string(),
  description: z.string().optional(),
  author: z.string().optional(),
  repository: z.string().url().optional(),
  license: z.string().optional(),
  tags: z.array(z.string()).optional(),
  capabilities: z.array(z.string()),
  dependencies: z.record(z.string()).optional(),
  config: z.record(z.unknown()).optional(),
  entrypoint: z.string(),
  runtime: z.enum(["node", "python", "rust", "wasm", "docker"]),
  tools: z
    .array(
      z.object({
        name: z.string(),
        description: z.string().optional(),
        parameters: z.record(z.unknown()).optional(),
      })
    )
    .optional(),
});

export type AgentDefinition = z.infer<typeof AgentDefinitionSchema>;

/**
 * Agent import result
 */
export const AgentImportResultSchema = z.object({
  success: z.boolean(),
  agentId: z.string().uuid().optional(),
  agent: AgentDefinitionSchema.optional(),
  errors: z.array(z.string()).optional(),
  warnings: z.array(z.string()).optional(),
  validationResults: z
    .array(
      z.object({
        check: z.string(),
        passed: z.boolean(),
        message: z.string().optional(),
      })
    )
    .optional(),
});

export type AgentImportResult = z.infer<typeof AgentImportResultSchema>;

/**
 * Agent search criteria
 */
export const AgentSearchCriteriaSchema = z.object({
  query: z.string().optional(),
  tags: z.array(z.string()).optional(),
  capabilities: z.array(z.string()).optional(),
  runtime: z.enum(["node", "python", "rust", "wasm", "docker"]).optional(),
  minStars: z.number().int().min(0).optional(),
  language: z.string().optional(),
  topics: z.array(z.string()).optional(),
  sort: z.enum(["stars", "forks", "updated", "relevance"]).default("relevance"),
  order: z.enum(["asc", "desc"]).default("desc"),
  perPage: z.number().int().min(1).max(100).default(30),
  page: z.number().int().min(1).default(1),
});

export type AgentSearchCriteria = z.infer<typeof AgentSearchCriteriaSchema>;

// =============================================================================
// API RESPONSE TYPES
// =============================================================================

/**
 * Pagination info
 */
export const PaginationSchema = z.object({
  page: z.number(),
  perPage: z.number(),
  totalCount: z.number(),
  totalPages: z.number(),
  hasNextPage: z.boolean(),
  hasPreviousPage: z.boolean(),
});

export type Pagination = z.infer<typeof PaginationSchema>;

/**
 * Rate limit info
 */
export const RateLimitSchema = z.object({
  limit: z.number(),
  remaining: z.number(),
  reset: z.number(),
  used: z.number(),
  resource: z.string(),
});

export type RateLimit = z.infer<typeof RateLimitSchema>;

/**
 * API error response
 */
export const APIErrorSchema = z.object({
  status: z.number(),
  message: z.string(),
  documentation_url: z.string().optional(),
  errors: z
    .array(
      z.object({
        resource: z.string().optional(),
        field: z.string().optional(),
        code: z.string(),
        message: z.string().optional(),
      })
    )
    .optional(),
});

export type APIError = z.infer<typeof APIErrorSchema>;

// =============================================================================
// CLIENT CONFIGURATION
// =============================================================================

/**
 * GitHub Universe client configuration
 */
export const GitHubUniverseConfigSchema = z.object({
  auth: GitHubAuthConfigSchema,
  /** Request timeout in milliseconds */
  timeout: z.number().int().positive().default(30000),
  /** Max retries for failed requests */
  maxRetries: z.number().int().min(0).max(10).default(3),
  /** Enable request caching */
  enableCache: z.boolean().default(true),
  /** Cache TTL in seconds */
  cacheTtl: z.number().int().positive().default(300),
  /** Enable rate limit handling */
  enableThrottling: z.boolean().default(true),
  /** Log level */
  logLevel: z
    .enum(["debug", "info", "warn", "error", "silent"])
    .default("info"),
  /** Custom user agent */
  userAgent: z.string().default("neurectomy-github-universe/1.0.0"),
});

export type GitHubUniverseConfig = z.infer<typeof GitHubUniverseConfigSchema>;
