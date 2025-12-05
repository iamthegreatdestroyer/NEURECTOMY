/**
 * @fileoverview Issue Manager
 * @module @neurectomy/github-universe/issues
 *
 * Comprehensive issue management: create, update, label, assign.
 *
 * @agents @SYNAPSE @APEX
 */

import { EventEmitter } from "eventemitter3";
import { GitHubClient } from "../client";
import {
  type Issue,
  type CreateIssueConfig,
  CreateIssueConfigSchema,
  type LabelConfig,
  LabelConfigSchema,
  type RepoIdentifier,
} from "../types";

// =============================================================================
// EVENTS
// =============================================================================

export interface IssueManagerEvents {
  "issue:created": (identifier: RepoIdentifier, issue: Issue) => void;
  "issue:updated": (identifier: RepoIdentifier, issue: Issue) => void;
  "issue:closed": (identifier: RepoIdentifier, issueNumber: number) => void;
  "issue:reopened": (identifier: RepoIdentifier, issueNumber: number) => void;
  "issue:labeled": (
    identifier: RepoIdentifier,
    issueNumber: number,
    label: string
  ) => void;
  "issue:unlabeled": (
    identifier: RepoIdentifier,
    issueNumber: number,
    label: string
  ) => void;
  "issue:assigned": (
    identifier: RepoIdentifier,
    issueNumber: number,
    assignee: string
  ) => void;
  error: (error: Error, context: string) => void;
}

// =============================================================================
// GRAPHQL QUERIES
// =============================================================================

const ISSUE_FRAGMENT = `
  fragment IssueFields on Issue {
    id
    number
    title
    body
    state
    stateReason
    author {
      login
      avatarUrl
    }
    labels(first: 20) {
      nodes {
        id
        name
        color
        description
      }
    }
    assignees(first: 10) {
      nodes {
        login
        id
        avatarUrl
      }
    }
    milestone {
      id
      title
      number
      state
      dueOn
    }
    projectItems(first: 5) {
      nodes {
        project {
          title
          number
        }
      }
    }
    comments {
      totalCount
    }
    reactions {
      totalCount
    }
    locked
    activeLockReason
    url
    createdAt
    updatedAt
    closedAt
    isPinned
  }
`;

const GET_ISSUE_QUERY = `
  ${ISSUE_FRAGMENT}
  query GetIssue($owner: String!, $name: String!, $number: Int!) {
    repository(owner: $owner, name: $name) {
      issue(number: $number) {
        ...IssueFields
      }
    }
  }
`;

const LIST_ISSUES_QUERY = `
  ${ISSUE_FRAGMENT}
  query ListIssues($owner: String!, $name: String!, $first: Int!, $after: String, $states: [IssueState!], $labels: [String!], $orderBy: IssueOrder) {
    repository(owner: $owner, name: $name) {
      issues(first: $first, after: $after, states: $states, labels: $labels, orderBy: $orderBy) {
        pageInfo {
          hasNextPage
          endCursor
        }
        totalCount
        nodes {
          ...IssueFields
        }
      }
    }
  }
`;

const SEARCH_ISSUES_QUERY = `
  ${ISSUE_FRAGMENT}
  query SearchIssues($query: String!, $first: Int!, $after: String) {
    search(query: $query, type: ISSUE, first: $first, after: $after) {
      issueCount
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        ... on Issue {
          ...IssueFields
        }
      }
    }
  }
`;

// =============================================================================
// TYPES
// =============================================================================

export interface IssueListOptions {
  state?: "open" | "closed" | "all";
  labels?: string[];
  assignee?: string;
  creator?: string;
  mentioned?: string;
  milestone?: string | number;
  sort?: "created" | "updated" | "comments";
  direction?: "asc" | "desc";
  since?: string;
  perPage?: number;
  page?: number;
}

export interface IssueComment {
  id: number;
  nodeId: string;
  body: string;
  author: { login: string; avatarUrl: string };
  createdAt: string;
  updatedAt: string;
  reactions?: Record<string, number>;
}

export interface IssueTimeline {
  event: string;
  actor?: { login: string };
  createdAt: string;
  details?: Record<string, unknown>;
}

export interface Label {
  id: string;
  name: string;
  color: string;
  description?: string;
  default?: boolean;
}

export interface Milestone {
  id: string;
  number: number;
  title: string;
  description?: string;
  state: "open" | "closed";
  dueOn?: string;
  openIssues: number;
  closedIssues: number;
  createdAt: string;
  updatedAt: string;
}

// =============================================================================
// ISSUE MANAGER
// =============================================================================

/**
 * Issue Manager
 *
 * Manages GitHub issues: creation, updates, labels, and assignments.
 *
 * @example
 * ```typescript
 * const issueManager = new IssueManager(client);
 *
 * // Create an issue
 * const issue = await issueManager.create(
 *   { owner: 'neurectomy', repo: 'core' },
 *   {
 *     title: 'Implement new agent feature',
 *     body: 'Description of the feature...',
 *     labels: ['enhancement', 'agent'],
 *     assignees: ['developer'],
 *   }
 * );
 *
 * // Update issue
 * await issueManager.update(
 *   { owner: 'neurectomy', repo: 'core' },
 *   issue.number,
 *   { state: 'closed' }
 * );
 *
 * // Search issues
 * const bugs = await issueManager.search(
 *   { owner: 'neurectomy', repo: 'core' },
 *   'is:open label:bug'
 * );
 * ```
 */
export class IssueManager extends EventEmitter<IssueManagerEvents> {
  private client: GitHubClient;

  constructor(client: GitHubClient) {
    super();
    this.client = client;
  }

  // ===========================================================================
  // ISSUE OPERATIONS
  // ===========================================================================

  /**
   * Get issue information
   */
  async get(identifier: RepoIdentifier, issueNumber: number): Promise<Issue> {
    const response = await this.client.graphql<{
      repository: { issue: Record<string, unknown> };
    }>(GET_ISSUE_QUERY, {
      owner: identifier.owner,
      name: identifier.repo,
      number: issueNumber,
    });

    return this.transformIssue(response.repository.issue);
  }

  /**
   * Create an issue
   */
  async create(
    identifier: RepoIdentifier,
    config: CreateIssueConfig
  ): Promise<Issue> {
    const validated = CreateIssueConfigSchema.parse(config);

    const response = await this.client.rest<Record<string, unknown>>(
      "POST /repos/{owner}/{repo}/issues",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        title: validated.title,
        body: validated.body,
        labels: validated.labels,
        assignees: validated.assignees,
        milestone: validated.milestone,
      }
    );

    const issue = this.transformRestIssue(response);
    this.emit("issue:created", identifier, issue);
    return issue;
  }

  /**
   * Update an issue
   */
  async update(
    identifier: RepoIdentifier,
    issueNumber: number,
    updates: Partial<CreateIssueConfig> & {
      state?: "open" | "closed";
      stateReason?: "completed" | "not_planned" | "reopened";
    }
  ): Promise<Issue> {
    const response = await this.client.rest<Record<string, unknown>>(
      "PATCH /repos/{owner}/{repo}/issues/{issue_number}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        issue_number: issueNumber,
        title: updates.title,
        body: updates.body,
        labels: updates.labels,
        assignees: updates.assignees,
        milestone: updates.milestone,
        state: updates.state,
        state_reason: updates.stateReason,
      }
    );

    const issue = this.transformRestIssue(response);
    this.emit("issue:updated", identifier, issue);
    return issue;
  }

  /**
   * Close an issue
   */
  async close(
    identifier: RepoIdentifier,
    issueNumber: number,
    reason: "completed" | "not_planned" = "completed"
  ): Promise<Issue> {
    const issue = await this.update(identifier, issueNumber, {
      state: "closed",
      stateReason: reason,
    });
    this.emit("issue:closed", identifier, issueNumber);
    return issue;
  }

  /**
   * Reopen an issue
   */
  async reopen(
    identifier: RepoIdentifier,
    issueNumber: number
  ): Promise<Issue> {
    const issue = await this.update(identifier, issueNumber, {
      state: "open",
      stateReason: "reopened",
    });
    this.emit("issue:reopened", identifier, issueNumber);
    return issue;
  }

  /**
   * Lock an issue
   */
  async lock(
    identifier: RepoIdentifier,
    issueNumber: number,
    reason?: "off-topic" | "too heated" | "resolved" | "spam"
  ): Promise<void> {
    await this.client.rest(
      "PUT /repos/{owner}/{repo}/issues/{issue_number}/lock",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        issue_number: issueNumber,
        lock_reason: reason,
      }
    );
  }

  /**
   * Unlock an issue
   */
  async unlock(identifier: RepoIdentifier, issueNumber: number): Promise<void> {
    await this.client.rest(
      "DELETE /repos/{owner}/{repo}/issues/{issue_number}/lock",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        issue_number: issueNumber,
      }
    );
  }

  /**
   * List issues
   */
  async list(
    identifier: RepoIdentifier,
    options?: IssueListOptions
  ): Promise<Issue[]> {
    const issues = await this.client.restPaginate<Record<string, unknown>>(
      "GET /repos/{owner}/{repo}/issues",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        state: options?.state ?? "open",
        labels: options?.labels?.join(","),
        assignee: options?.assignee,
        creator: options?.creator,
        mentioned: options?.mentioned,
        milestone: options?.milestone?.toString(),
        sort: options?.sort ?? "created",
        direction: options?.direction ?? "desc",
        since: options?.since,
        per_page: options?.perPage ?? 100,
        page: options?.page ?? 1,
      }
    );

    // Filter out pull requests (they also appear in issues endpoint)
    return issues
      .filter((i) => !("pull_request" in i))
      .map((i) => this.transformRestIssue(i));
  }

  /**
   * List issues using GraphQL for richer data
   */
  async listGraphQL(
    identifier: RepoIdentifier,
    options?: {
      states?: Array<"OPEN" | "CLOSED">;
      labels?: string[];
      first?: number;
      after?: string;
      orderBy?: {
        field: "CREATED_AT" | "UPDATED_AT" | "COMMENTS";
        direction: "ASC" | "DESC";
      };
    }
  ): Promise<{
    issues: Issue[];
    totalCount: number;
    pageInfo: { hasNextPage: boolean; endCursor: string };
  }> {
    const response = await this.client.graphql<{
      repository: {
        issues: {
          totalCount: number;
          pageInfo: { hasNextPage: boolean; endCursor: string };
          nodes: Record<string, unknown>[];
        };
      };
    }>(LIST_ISSUES_QUERY, {
      owner: identifier.owner,
      name: identifier.repo,
      first: options?.first ?? 20,
      after: options?.after,
      states: options?.states,
      labels: options?.labels,
      orderBy: options?.orderBy ?? { field: "UPDATED_AT", direction: "DESC" },
    });

    return {
      issues: response.repository.issues.nodes.map((n) =>
        this.transformIssue(n)
      ),
      totalCount: response.repository.issues.totalCount,
      pageInfo: response.repository.issues.pageInfo,
    };
  }

  /**
   * Search issues
   */
  async search(
    identifier: RepoIdentifier,
    query: string,
    options?: { first?: number; after?: string }
  ): Promise<{
    issues: Issue[];
    totalCount: number;
    pageInfo: { hasNextPage: boolean; endCursor: string };
  }> {
    const fullQuery = `repo:${identifier.owner}/${identifier.repo} is:issue ${query}`;

    const response = await this.client.graphql<{
      search: {
        issueCount: number;
        pageInfo: { hasNextPage: boolean; endCursor: string };
        nodes: Record<string, unknown>[];
      };
    }>(SEARCH_ISSUES_QUERY, {
      query: fullQuery,
      first: options?.first ?? 20,
      after: options?.after,
    });

    return {
      issues: response.search.nodes.map((n) => this.transformIssue(n)),
      totalCount: response.search.issueCount,
      pageInfo: response.search.pageInfo,
    };
  }

  // ===========================================================================
  // LABEL OPERATIONS
  // ===========================================================================

  /**
   * Add labels to an issue
   */
  async addLabels(
    identifier: RepoIdentifier,
    issueNumber: number,
    labels: string[]
  ): Promise<void> {
    await this.client.rest(
      "POST /repos/{owner}/{repo}/issues/{issue_number}/labels",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        issue_number: issueNumber,
        labels,
      }
    );

    for (const label of labels) {
      this.emit("issue:labeled", identifier, issueNumber, label);
    }
  }

  /**
   * Remove a label from an issue
   */
  async removeLabel(
    identifier: RepoIdentifier,
    issueNumber: number,
    label: string
  ): Promise<void> {
    await this.client.rest(
      "DELETE /repos/{owner}/{repo}/issues/{issue_number}/labels/{name}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        issue_number: issueNumber,
        name: label,
      }
    );

    this.emit("issue:unlabeled", identifier, issueNumber, label);
  }

  /**
   * Set labels (replace all)
   */
  async setLabels(
    identifier: RepoIdentifier,
    issueNumber: number,
    labels: string[]
  ): Promise<void> {
    await this.client.rest(
      "PUT /repos/{owner}/{repo}/issues/{issue_number}/labels",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        issue_number: issueNumber,
        labels,
      }
    );
  }

  /**
   * Get repository labels
   */
  async getRepoLabels(identifier: RepoIdentifier): Promise<Label[]> {
    const labels = await this.client.restPaginate<Record<string, unknown>>(
      "GET /repos/{owner}/{repo}/labels",
      {
        owner: identifier.owner,
        repo: identifier.repo,
      }
    );

    return labels.map((l) => ({
      id: String(l.id),
      name: l.name as string,
      color: l.color as string,
      description: l.description as string | undefined,
      default: l.default as boolean | undefined,
    }));
  }

  /**
   * Create a label
   */
  async createLabel(
    identifier: RepoIdentifier,
    config: LabelConfig
  ): Promise<Label> {
    const validated = LabelConfigSchema.parse(config);

    const response = await this.client.rest<Record<string, unknown>>(
      "POST /repos/{owner}/{repo}/labels",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        name: validated.name,
        color: validated.color.replace("#", ""),
        description: validated.description,
      }
    );

    return {
      id: String(response.id),
      name: response.name as string,
      color: response.color as string,
      description: response.description as string | undefined,
    };
  }

  /**
   * Update a label
   */
  async updateLabel(
    identifier: RepoIdentifier,
    name: string,
    updates: Partial<LabelConfig>
  ): Promise<Label> {
    const response = await this.client.rest<Record<string, unknown>>(
      "PATCH /repos/{owner}/{repo}/labels/{name}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        name,
        new_name: updates.name,
        color: updates.color?.replace("#", ""),
        description: updates.description,
      }
    );

    return {
      id: String(response.id),
      name: response.name as string,
      color: response.color as string,
      description: response.description as string | undefined,
    };
  }

  /**
   * Delete a label
   */
  async deleteLabel(identifier: RepoIdentifier, name: string): Promise<void> {
    await this.client.rest("DELETE /repos/{owner}/{repo}/labels/{name}", {
      owner: identifier.owner,
      repo: identifier.repo,
      name,
    });
  }

  // ===========================================================================
  // ASSIGNEE OPERATIONS
  // ===========================================================================

  /**
   * Add assignees to an issue
   */
  async addAssignees(
    identifier: RepoIdentifier,
    issueNumber: number,
    assignees: string[]
  ): Promise<void> {
    await this.client.rest(
      "POST /repos/{owner}/{repo}/issues/{issue_number}/assignees",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        issue_number: issueNumber,
        assignees,
      }
    );

    for (const assignee of assignees) {
      this.emit("issue:assigned", identifier, issueNumber, assignee);
    }
  }

  /**
   * Remove assignees from an issue
   */
  async removeAssignees(
    identifier: RepoIdentifier,
    issueNumber: number,
    assignees: string[]
  ): Promise<void> {
    await this.client.rest(
      "DELETE /repos/{owner}/{repo}/issues/{issue_number}/assignees",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        issue_number: issueNumber,
        assignees,
      }
    );
  }

  /**
   * Get available assignees for a repository
   */
  async getAvailableAssignees(
    identifier: RepoIdentifier
  ): Promise<Array<{ login: string; id: number }>> {
    return this.client.restPaginate("GET /repos/{owner}/{repo}/assignees", {
      owner: identifier.owner,
      repo: identifier.repo,
    });
  }

  // ===========================================================================
  // COMMENT OPERATIONS
  // ===========================================================================

  /**
   * Create a comment on an issue
   */
  async createComment(
    identifier: RepoIdentifier,
    issueNumber: number,
    body: string
  ): Promise<IssueComment> {
    const response = await this.client.rest<Record<string, unknown>>(
      "POST /repos/{owner}/{repo}/issues/{issue_number}/comments",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        issue_number: issueNumber,
        body,
      }
    );

    return this.transformComment(response);
  }

  /**
   * Update a comment
   */
  async updateComment(
    identifier: RepoIdentifier,
    commentId: number,
    body: string
  ): Promise<IssueComment> {
    const response = await this.client.rest<Record<string, unknown>>(
      "PATCH /repos/{owner}/{repo}/issues/comments/{comment_id}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        comment_id: commentId,
        body,
      }
    );

    return this.transformComment(response);
  }

  /**
   * Delete a comment
   */
  async deleteComment(
    identifier: RepoIdentifier,
    commentId: number
  ): Promise<void> {
    await this.client.rest(
      "DELETE /repos/{owner}/{repo}/issues/comments/{comment_id}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        comment_id: commentId,
      }
    );
  }

  /**
   * Get comments on an issue
   */
  async getComments(
    identifier: RepoIdentifier,
    issueNumber: number,
    options?: { since?: string; perPage?: number }
  ): Promise<IssueComment[]> {
    const comments = await this.client.restPaginate<Record<string, unknown>>(
      "GET /repos/{owner}/{repo}/issues/{issue_number}/comments",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        issue_number: issueNumber,
        since: options?.since,
        per_page: options?.perPage ?? 100,
      }
    );

    return comments.map((c) => this.transformComment(c));
  }

  // ===========================================================================
  // MILESTONE OPERATIONS
  // ===========================================================================

  /**
   * Get milestones
   */
  async getMilestones(
    identifier: RepoIdentifier,
    options?: {
      state?: "open" | "closed" | "all";
      sort?: "due_on" | "completeness";
    }
  ): Promise<Milestone[]> {
    const milestones = await this.client.restPaginate<Record<string, unknown>>(
      "GET /repos/{owner}/{repo}/milestones",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        state: options?.state ?? "open",
        sort: options?.sort ?? "due_on",
      }
    );

    return milestones.map((m) => ({
      id: String(m.id),
      number: m.number as number,
      title: m.title as string,
      description: m.description as string | undefined,
      state: m.state as "open" | "closed",
      dueOn: m.due_on as string | undefined,
      openIssues: m.open_issues as number,
      closedIssues: m.closed_issues as number,
      createdAt: m.created_at as string,
      updatedAt: m.updated_at as string,
    }));
  }

  /**
   * Create a milestone
   */
  async createMilestone(
    identifier: RepoIdentifier,
    config: {
      title: string;
      description?: string;
      dueOn?: string;
      state?: "open" | "closed";
    }
  ): Promise<Milestone> {
    const response = await this.client.rest<Record<string, unknown>>(
      "POST /repos/{owner}/{repo}/milestones",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        title: config.title,
        description: config.description,
        due_on: config.dueOn,
        state: config.state,
      }
    );

    return {
      id: String(response.id),
      number: response.number as number,
      title: response.title as string,
      description: response.description as string | undefined,
      state: response.state as "open" | "closed",
      dueOn: response.due_on as string | undefined,
      openIssues: response.open_issues as number,
      closedIssues: response.closed_issues as number,
      createdAt: response.created_at as string,
      updatedAt: response.updated_at as string,
    };
  }

  /**
   * Set milestone on an issue
   */
  async setMilestone(
    identifier: RepoIdentifier,
    issueNumber: number,
    milestoneNumber: number | null
  ): Promise<Issue> {
    return this.update(identifier, issueNumber, {
      milestone: milestoneNumber ?? undefined,
    });
  }

  // ===========================================================================
  // TIMELINE
  // ===========================================================================

  /**
   * Get issue timeline events
   */
  async getTimeline(
    identifier: RepoIdentifier,
    issueNumber: number
  ): Promise<IssueTimeline[]> {
    const events = await this.client.restPaginate<Record<string, unknown>>(
      "GET /repos/{owner}/{repo}/issues/{issue_number}/timeline",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        issue_number: issueNumber,
      }
    );

    return events.map((e) => ({
      event: e.event as string,
      actor: e.actor
        ? { login: (e.actor as { login: string }).login }
        : undefined,
      createdAt: e.created_at as string,
      details: e,
    }));
  }

  // ===========================================================================
  // PRIVATE METHODS
  // ===========================================================================

  private transformIssue(data: Record<string, unknown>): Issue {
    const author = data.author as { login: string; avatarUrl: string } | null;
    const labels = data.labels as
      | {
          nodes: Array<{
            id: string;
            name: string;
            color: string;
            description?: string;
          }>;
        }
      | undefined;
    const assignees = data.assignees as
      | { nodes: Array<{ login: string; id: string; avatarUrl: string }> }
      | undefined;
    const milestone = data.milestone as {
      id: string;
      title: string;
      number: number;
      state: string;
      dueOn?: string;
    } | null;
    const comments = data.comments as { totalCount: number } | undefined;
    const reactions = data.reactions as { totalCount: number } | undefined;

    return {
      id: data.id as string,
      nodeId: data.id as string,
      number: data.number as number,
      title: data.title as string,
      body: data.body as string | null,
      state: (data.state as string).toLowerCase() as "open" | "closed",
      stateReason: data.stateReason as string | null,
      author: author
        ? {
            login: author.login,
            avatarUrl: author.avatarUrl,
          }
        : null,
      labels:
        labels?.nodes.map((l) => ({
          id: l.id,
          name: l.name,
          color: l.color,
          description: l.description,
        })) ?? [],
      assignees:
        assignees?.nodes.map((a) => ({
          login: a.login,
          id: a.id,
          avatarUrl: a.avatarUrl,
        })) ?? [],
      milestone: milestone
        ? {
            id: milestone.id,
            title: milestone.title,
            number: milestone.number,
            state: milestone.state.toLowerCase() as "open" | "closed",
            dueOn: milestone.dueOn,
          }
        : null,
      commentsCount: comments?.totalCount ?? 0,
      reactionsCount: reactions?.totalCount ?? 0,
      locked: data.locked as boolean,
      lockReason: data.activeLockReason as string | null,
      htmlUrl: data.url as string,
      createdAt: data.createdAt as string,
      updatedAt: data.updatedAt as string,
      closedAt: data.closedAt as string | null,
      isPinned: (data.isPinned as boolean) ?? false,
    };
  }

  private transformRestIssue(data: Record<string, unknown>): Issue {
    const user = data.user as { login: string; avatar_url: string } | null;
    const labels = data.labels as
      | Array<{ id: number; name: string; color: string; description?: string }>
      | undefined;
    const assignees = data.assignees as
      | Array<{ login: string; id: number; avatar_url: string }>
      | undefined;
    const milestone = data.milestone as {
      id: number;
      title: string;
      number: number;
      state: string;
      due_on?: string;
    } | null;

    return {
      id: String(data.id),
      nodeId: data.node_id as string,
      number: data.number as number,
      title: data.title as string,
      body: data.body as string | null,
      state: data.state as "open" | "closed",
      stateReason: data.state_reason as string | null,
      author: user
        ? {
            login: user.login,
            avatarUrl: user.avatar_url,
          }
        : null,
      labels:
        labels?.map((l) => ({
          id: String(l.id),
          name: l.name,
          color: l.color,
          description: l.description,
        })) ?? [],
      assignees:
        assignees?.map((a) => ({
          login: a.login,
          id: String(a.id),
          avatarUrl: a.avatar_url,
        })) ?? [],
      milestone: milestone
        ? {
            id: String(milestone.id),
            title: milestone.title,
            number: milestone.number,
            state: milestone.state as "open" | "closed",
            dueOn: milestone.due_on,
          }
        : null,
      commentsCount: (data.comments as number) ?? 0,
      reactionsCount: 0,
      locked: (data.locked as boolean) ?? false,
      lockReason: data.active_lock_reason as string | null,
      htmlUrl: data.html_url as string,
      createdAt: data.created_at as string,
      updatedAt: data.updated_at as string,
      closedAt: data.closed_at as string | null,
      isPinned: false,
    };
  }

  private transformComment(data: Record<string, unknown>): IssueComment {
    const user = data.user as { login: string; avatar_url: string };

    return {
      id: data.id as number,
      nodeId: data.node_id as string,
      body: data.body as string,
      author: {
        login: user.login,
        avatarUrl: user.avatar_url,
      },
      createdAt: data.created_at as string,
      updatedAt: data.updated_at as string,
    };
  }
}

// Types are already exported where they are defined
