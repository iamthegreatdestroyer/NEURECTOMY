/**
 * @fileoverview Pull Request Manager
 * @module @neurectomy/github-universe/pr
 *
 * Comprehensive PR management: create, review, merge, manage.
 *
 * @agents @SYNAPSE @APEX
 */

import { EventEmitter } from "eventemitter3";
import { GitHubClient } from "../client";
import {
  type PullRequest,
  type CreatePRConfig,
  CreatePRConfigSchema,
  type UpdatePRConfig,
  type CreateReview,
  CreateReviewSchema,
  type ReviewState,
  type RepoIdentifier,
  type MergeMethod,
} from "../types";

// =============================================================================
// EVENTS
// =============================================================================

export interface PRManagerEvents {
  "pr:created": (identifier: RepoIdentifier, pr: PullRequest) => void;
  "pr:updated": (identifier: RepoIdentifier, pr: PullRequest) => void;
  "pr:merged": (identifier: RepoIdentifier, prNumber: number) => void;
  "pr:closed": (identifier: RepoIdentifier, prNumber: number) => void;
  "pr:reviewed": (
    identifier: RepoIdentifier,
    prNumber: number,
    state: ReviewState
  ) => void;
  error: (error: Error, context: string) => void;
}

// =============================================================================
// GRAPHQL QUERIES
// =============================================================================

const PR_FRAGMENT = `
  fragment PRFields on PullRequest {
    id
    number
    title
    body
    state
    isDraft
    merged
    mergeable
    mergeStateStatus
    headRefName
    baseRefName
    headRefOid
    baseRefOid
    additions
    deletions
    changedFiles
    author {
      login
      avatarUrl
      ... on User {
        id
      }
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
    reviewRequests(first: 10) {
      nodes {
        requestedReviewer {
          ... on User {
            login
            id
          }
          ... on Team {
            name
            slug
          }
        }
      }
    }
    reviews(last: 10) {
      nodes {
        id
        state
        body
        author {
          login
        }
        submittedAt
      }
    }
    comments(first: 10) {
      totalCount
    }
    commits(last: 1) {
      totalCount
      nodes {
        commit {
          statusCheckRollup {
            state
            contexts(first: 10) {
              nodes {
                ... on StatusContext {
                  context
                  state
                }
                ... on CheckRun {
                  name
                  conclusion
                }
              }
            }
          }
        }
      }
    }
    milestone {
      id
      title
      number
    }
    autoMergeRequest {
      enabledAt
      mergeMethod
    }
    url
    createdAt
    updatedAt
    closedAt
    mergedAt
    mergedBy {
      login
    }
  }
`;

const GET_PR_QUERY = `
  ${PR_FRAGMENT}
  query GetPullRequest($owner: String!, $name: String!, $number: Int!) {
    repository(owner: $owner, name: $name) {
      pullRequest(number: $number) {
        ...PRFields
      }
    }
  }
`;

const LIST_PRS_QUERY = `
  ${PR_FRAGMENT}
  query ListPullRequests($owner: String!, $name: String!, $first: Int!, $after: String, $states: [PullRequestState!], $baseRefName: String, $headRefName: String) {
    repository(owner: $owner, name: $name) {
      pullRequests(first: $first, after: $after, states: $states, baseRefName: $baseRefName, headRefName: $headRefName, orderBy: {field: UPDATED_AT, direction: DESC}) {
        pageInfo {
          hasNextPage
          endCursor
        }
        totalCount
        nodes {
          ...PRFields
        }
      }
    }
  }
`;

// =============================================================================
// TYPES
// =============================================================================

export interface PRListOptions {
  state?: "open" | "closed" | "all";
  base?: string;
  head?: string;
  sort?: "created" | "updated" | "popularity" | "long-running";
  direction?: "asc" | "desc";
  perPage?: number;
  page?: number;
}

export interface PRReview {
  id: number;
  nodeId: string;
  state: ReviewState;
  body: string;
  author: { login: string };
  submittedAt: string;
  commitId: string;
}

export interface PRComment {
  id: number;
  nodeId: string;
  body: string;
  author: { login: string };
  path?: string;
  line?: number;
  side?: "LEFT" | "RIGHT";
  createdAt: string;
  updatedAt: string;
}

export interface PRFile {
  filename: string;
  status:
    | "added"
    | "removed"
    | "modified"
    | "renamed"
    | "copied"
    | "changed"
    | "unchanged";
  additions: number;
  deletions: number;
  changes: number;
  patch?: string;
  previousFilename?: string;
}

export interface MergeOptions {
  commitTitle?: string;
  commitMessage?: string;
  mergeMethod?: MergeMethod;
  sha?: string;
}

// =============================================================================
// PR MANAGER
// =============================================================================

/**
 * Pull Request Manager
 *
 * Manages GitHub pull requests: creation, review, merging, and management.
 *
 * @example
 * ```typescript
 * const prManager = new PRManager(client);
 *
 * // Create a pull request
 * const pr = await prManager.create(
 *   { owner: 'neurectomy', repo: 'core' },
 *   {
 *     title: 'feat: Add new agent capabilities',
 *     head: 'feature/new-agent',
 *     base: 'main',
 *     body: 'Implements new agent features...',
 *   }
 * );
 *
 * // Request reviewers
 * await prManager.requestReviewers(
 *   { owner: 'neurectomy', repo: 'core' },
 *   pr.number,
 *   { reviewers: ['alice', 'bob'] }
 * );
 *
 * // Approve and merge
 * await prManager.createReview(
 *   { owner: 'neurectomy', repo: 'core' },
 *   pr.number,
 *   { event: 'APPROVE', body: 'LGTM!' }
 * );
 * await prManager.merge(
 *   { owner: 'neurectomy', repo: 'core' },
 *   pr.number,
 *   { mergeMethod: 'squash' }
 * );
 * ```
 */
export class PRManager extends EventEmitter<PRManagerEvents> {
  private client: GitHubClient;

  constructor(client: GitHubClient) {
    super();
    this.client = client;
  }

  // ===========================================================================
  // PR OPERATIONS
  // ===========================================================================

  /**
   * Get pull request information
   */
  async get(
    identifier: RepoIdentifier,
    prNumber: number
  ): Promise<PullRequest> {
    const response = await this.client.graphql<{
      repository: { pullRequest: Record<string, unknown> };
    }>(GET_PR_QUERY, {
      owner: identifier.owner,
      name: identifier.repo,
      number: prNumber,
    });

    return this.transformPR(response.repository.pullRequest);
  }

  /**
   * Create a pull request
   */
  async create(
    identifier: RepoIdentifier,
    config: CreatePRConfig
  ): Promise<PullRequest> {
    const validated = CreatePRConfigSchema.parse(config);

    const response = await this.client.rest<Record<string, unknown>>(
      "POST /repos/{owner}/{repo}/pulls",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        title: validated.title,
        head: validated.head,
        base: validated.base,
        body: validated.body,
        draft: validated.draft,
        maintainer_can_modify: validated.maintainerCanModify,
      }
    );

    const pr = this.transformRestPR(response);

    // Add labels if provided
    if (validated.labels && validated.labels.length > 0) {
      await this.addLabels(identifier, pr.number, validated.labels);
    }

    // Add assignees if provided
    if (validated.assignees && validated.assignees.length > 0) {
      await this.addAssignees(identifier, pr.number, validated.assignees);
    }

    // Request reviewers if provided
    if (validated.reviewers && validated.reviewers.length > 0) {
      await this.requestReviewers(identifier, pr.number, {
        reviewers: validated.reviewers,
      });
    }

    this.emit("pr:created", identifier, pr);
    return pr;
  }

  /**
   * Update a pull request
   */
  async update(
    identifier: RepoIdentifier,
    prNumber: number,
    updates: Partial<UpdatePRConfig>
  ): Promise<PullRequest> {
    const response = await this.client.rest<Record<string, unknown>>(
      "PATCH /repos/{owner}/{repo}/pulls/{pull_number}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        pull_number: prNumber,
        title: updates.title,
        body: updates.body,
        base: updates.base,
        state: updates.state,
        maintainer_can_modify: updates.maintainerCanModify,
      }
    );

    const pr = this.transformRestPR(response);
    this.emit("pr:updated", identifier, pr);
    return pr;
  }

  /**
   * Close a pull request
   */
  async close(
    identifier: RepoIdentifier,
    prNumber: number
  ): Promise<PullRequest> {
    const pr = await this.update(identifier, prNumber, { state: "closed" });
    this.emit("pr:closed", identifier, prNumber);
    return pr;
  }

  /**
   * Reopen a pull request
   */
  async reopen(
    identifier: RepoIdentifier,
    prNumber: number
  ): Promise<PullRequest> {
    return this.update(identifier, prNumber, { state: "open" });
  }

  /**
   * Convert draft PR to ready for review
   */
  async markReadyForReview(
    identifier: RepoIdentifier,
    prNumber: number
  ): Promise<PullRequest> {
    // Get PR node ID
    const pr = await this.get(identifier, prNumber);

    await this.client.graphql(
      `mutation MarkReadyForReview($pullRequestId: ID!) {
        markPullRequestReadyForReview(input: { pullRequestId: $pullRequestId }) {
          pullRequest { id }
        }
      }`,
      { pullRequestId: pr.nodeId }
    );

    return this.get(identifier, prNumber);
  }

  /**
   * Convert PR to draft
   */
  async convertToDraft(
    identifier: RepoIdentifier,
    prNumber: number
  ): Promise<PullRequest> {
    const pr = await this.get(identifier, prNumber);

    await this.client.graphql(
      `mutation ConvertToDraft($pullRequestId: ID!) {
        convertPullRequestToDraft(input: { pullRequestId: $pullRequestId }) {
          pullRequest { id }
        }
      }`,
      { pullRequestId: pr.nodeId }
    );

    return this.get(identifier, prNumber);
  }

  /**
   * List pull requests
   */
  async list(
    identifier: RepoIdentifier,
    options?: PRListOptions
  ): Promise<PullRequest[]> {
    const prs = await this.client.restPaginate<Record<string, unknown>>(
      "GET /repos/{owner}/{repo}/pulls",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        state: options?.state ?? "open",
        base: options?.base,
        head: options?.head,
        sort: options?.sort,
        direction: options?.direction ?? "desc",
        per_page: options?.perPage ?? 100,
        page: options?.page ?? 1,
      }
    );

    return prs.map((pr) => this.transformRestPR(pr));
  }

  /**
   * List PRs using GraphQL for richer data
   */
  async listGraphQL(
    identifier: RepoIdentifier,
    options?: {
      states?: Array<"OPEN" | "CLOSED" | "MERGED">;
      base?: string;
      head?: string;
      first?: number;
      after?: string;
    }
  ): Promise<{
    pullRequests: PullRequest[];
    totalCount: number;
    pageInfo: { hasNextPage: boolean; endCursor: string };
  }> {
    const response = await this.client.graphql<{
      repository: {
        pullRequests: {
          totalCount: number;
          pageInfo: { hasNextPage: boolean; endCursor: string };
          nodes: Record<string, unknown>[];
        };
      };
    }>(LIST_PRS_QUERY, {
      owner: identifier.owner,
      name: identifier.repo,
      first: options?.first ?? 20,
      after: options?.after,
      states: options?.states,
      baseRefName: options?.base,
      headRefName: options?.head,
    });

    return {
      pullRequests: response.repository.pullRequests.nodes.map((n) =>
        this.transformPR(n)
      ),
      totalCount: response.repository.pullRequests.totalCount,
      pageInfo: response.repository.pullRequests.pageInfo,
    };
  }

  // ===========================================================================
  // MERGE OPERATIONS
  // ===========================================================================

  /**
   * Merge a pull request
   */
  async merge(
    identifier: RepoIdentifier,
    prNumber: number,
    options?: MergeOptions
  ): Promise<{ merged: boolean; sha?: string; message?: string }> {
    try {
      const response = await this.client.rest<{
        sha: string;
        merged: boolean;
        message: string;
      }>("PUT /repos/{owner}/{repo}/pulls/{pull_number}/merge", {
        owner: identifier.owner,
        repo: identifier.repo,
        pull_number: prNumber,
        commit_title: options?.commitTitle,
        commit_message: options?.commitMessage,
        merge_method: options?.mergeMethod ?? "merge",
        sha: options?.sha,
      });

      this.emit("pr:merged", identifier, prNumber);
      return {
        merged: response.merged,
        sha: response.sha,
        message: response.message,
      };
    } catch (error) {
      const err = error as Error & { status?: number };
      if (err.status === 405) {
        return { merged: false, message: "Pull request is not mergeable" };
      }
      if (err.status === 409) {
        return { merged: false, message: "Head branch was modified" };
      }
      throw error;
    }
  }

  /**
   * Check if PR is mergeable
   */
  async isMergeable(
    identifier: RepoIdentifier,
    prNumber: number
  ): Promise<{ mergeable: boolean; reason?: string }> {
    const pr = await this.get(identifier, prNumber);

    if (pr.merged) {
      return { mergeable: false, reason: "Already merged" };
    }

    if (pr.state !== "open") {
      return { mergeable: false, reason: "PR is closed" };
    }

    if (!pr.mergeable) {
      return { mergeable: false, reason: pr.mergeableState ?? undefined };
    }

    return { mergeable: true };
  }

  /**
   * Enable auto-merge
   */
  async enableAutoMerge(
    identifier: RepoIdentifier,
    prNumber: number,
    mergeMethod: MergeMethod = "squash"
  ): Promise<void> {
    const pr = await this.get(identifier, prNumber);

    const mergeMethodMap = {
      merge: "MERGE",
      squash: "SQUASH",
      rebase: "REBASE",
    };

    await this.client.graphql(
      `mutation EnableAutoMerge($pullRequestId: ID!, $mergeMethod: PullRequestMergeMethod!) {
        enablePullRequestAutoMerge(input: { pullRequestId: $pullRequestId, mergeMethod: $mergeMethod }) {
          pullRequest { id }
        }
      }`,
      {
        pullRequestId: pr.nodeId,
        mergeMethod: mergeMethodMap[mergeMethod],
      }
    );
  }

  /**
   * Disable auto-merge
   */
  async disableAutoMerge(
    identifier: RepoIdentifier,
    prNumber: number
  ): Promise<void> {
    const pr = await this.get(identifier, prNumber);

    await this.client.graphql(
      `mutation DisableAutoMerge($pullRequestId: ID!) {
        disablePullRequestAutoMerge(input: { pullRequestId: $pullRequestId }) {
          pullRequest { id }
        }
      }`,
      { pullRequestId: pr.nodeId }
    );
  }

  /**
   * Update branch (merge base into head)
   */
  async updateBranch(
    identifier: RepoIdentifier,
    prNumber: number,
    expectedHeadSha?: string
  ): Promise<{ updated: boolean; message?: string }> {
    try {
      await this.client.rest(
        "PUT /repos/{owner}/{repo}/pulls/{pull_number}/update-branch",
        {
          owner: identifier.owner,
          repo: identifier.repo,
          pull_number: prNumber,
          expected_head_sha: expectedHeadSha,
        }
      );

      return { updated: true };
    } catch (error) {
      const err = error as Error & { status?: number };
      return { updated: false, message: err.message };
    }
  }

  // ===========================================================================
  // REVIEW OPERATIONS
  // ===========================================================================

  /**
   * Create a review
   */
  async createReview(
    identifier: RepoIdentifier,
    prNumber: number,
    review: CreateReview
  ): Promise<PRReview> {
    const validated = CreateReviewSchema.parse(review);

    const response = await this.client.rest<Record<string, unknown>>(
      "POST /repos/{owner}/{repo}/pulls/{pull_number}/reviews",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        pull_number: prNumber,
        body: validated.body,
        event: validated.event,
        comments: validated.comments?.map((c) => ({
          path: c.path,
          line: c.line,
          body: c.body,
          side: c.side,
          start_line: c.startLine,
          start_side: c.startSide,
        })),
        commit_id: validated.commitId,
      }
    );

    // Map review event to ReviewState
    const reviewStateMap: Record<string, ReviewState> = {
      APPROVE: "APPROVED",
      REQUEST_CHANGES: "CHANGES_REQUESTED",
      COMMENT: "COMMENTED",
    };
    const reviewState = reviewStateMap[validated.event] ?? "PENDING";
    this.emit("pr:reviewed", identifier, prNumber, reviewState);
    return this.transformReview(response);
  }

  /**
   * Get reviews
   */
  async getReviews(
    identifier: RepoIdentifier,
    prNumber: number
  ): Promise<PRReview[]> {
    const reviews = await this.client.restPaginate<Record<string, unknown>>(
      "GET /repos/{owner}/{repo}/pulls/{pull_number}/reviews",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        pull_number: prNumber,
      }
    );

    return reviews.map((r) => this.transformReview(r));
  }

  /**
   * Request reviewers
   */
  async requestReviewers(
    identifier: RepoIdentifier,
    prNumber: number,
    request: { reviewers?: string[]; teamReviewers?: string[] }
  ): Promise<void> {
    await this.client.rest(
      "POST /repos/{owner}/{repo}/pulls/{pull_number}/requested_reviewers",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        pull_number: prNumber,
        reviewers: request.reviewers,
        team_reviewers: request.teamReviewers,
      }
    );
  }

  /**
   * Remove reviewer request
   */
  async removeReviewers(
    identifier: RepoIdentifier,
    prNumber: number,
    request: { reviewers?: string[]; teamReviewers?: string[] }
  ): Promise<void> {
    await this.client.rest(
      "DELETE /repos/{owner}/{repo}/pulls/{pull_number}/requested_reviewers",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        pull_number: prNumber,
        reviewers: request.reviewers,
        team_reviewers: request.teamReviewers,
      }
    );
  }

  /**
   * Dismiss a review
   */
  async dismissReview(
    identifier: RepoIdentifier,
    prNumber: number,
    reviewId: number,
    message: string
  ): Promise<void> {
    await this.client.rest(
      "PUT /repos/{owner}/{repo}/pulls/{pull_number}/reviews/{review_id}/dismissals",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        pull_number: prNumber,
        review_id: reviewId,
        message,
      }
    );
  }

  // ===========================================================================
  // COMMENT OPERATIONS
  // ===========================================================================

  /**
   * Create a comment on the PR
   */
  async createComment(
    identifier: RepoIdentifier,
    prNumber: number,
    body: string
  ): Promise<{ id: number; body: string }> {
    const response = await this.client.rest<{ id: number; body: string }>(
      "POST /repos/{owner}/{repo}/issues/{issue_number}/comments",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        issue_number: prNumber,
        body,
      }
    );

    return { id: response.id, body: response.body };
  }

  /**
   * Get comments on the PR
   */
  async getComments(
    identifier: RepoIdentifier,
    prNumber: number
  ): Promise<
    Array<{
      id: number;
      body: string;
      author: { login: string };
      createdAt: string;
    }>
  > {
    const comments = await this.client.restPaginate<{
      id: number;
      body: string;
      user: { login: string };
      created_at: string;
    }>("GET /repos/{owner}/{repo}/issues/{issue_number}/comments", {
      owner: identifier.owner,
      repo: identifier.repo,
      issue_number: prNumber,
    });

    return comments.map((c) => ({
      id: c.id,
      body: c.body,
      author: { login: c.user.login },
      createdAt: c.created_at,
    }));
  }

  /**
   * Get review comments (line-level)
   */
  async getReviewComments(
    identifier: RepoIdentifier,
    prNumber: number
  ): Promise<PRComment[]> {
    const comments = await this.client.restPaginate<Record<string, unknown>>(
      "GET /repos/{owner}/{repo}/pulls/{pull_number}/comments",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        pull_number: prNumber,
      }
    );

    return comments.map((c) => ({
      id: c.id as number,
      nodeId: c.node_id as string,
      body: c.body as string,
      author: { login: (c.user as { login: string }).login },
      path: c.path as string,
      line: c.line as number | undefined,
      side: c.side as "LEFT" | "RIGHT" | undefined,
      createdAt: c.created_at as string,
      updatedAt: c.updated_at as string,
    }));
  }

  // ===========================================================================
  // FILES & COMMITS
  // ===========================================================================

  /**
   * Get files changed in PR
   */
  async getFiles(
    identifier: RepoIdentifier,
    prNumber: number
  ): Promise<PRFile[]> {
    const files = await this.client.restPaginate<Record<string, unknown>>(
      "GET /repos/{owner}/{repo}/pulls/{pull_number}/files",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        pull_number: prNumber,
      }
    );

    return files.map((f) => ({
      filename: f.filename as string,
      status: f.status as PRFile["status"],
      additions: f.additions as number,
      deletions: f.deletions as number,
      changes: f.changes as number,
      patch: f.patch as string | undefined,
      previousFilename: f.previous_filename as string | undefined,
    }));
  }

  /**
   * Get commits in PR
   */
  async getCommits(
    identifier: RepoIdentifier,
    prNumber: number
  ): Promise<
    Array<{
      sha: string;
      message: string;
      author: { name: string; email: string; date: string };
    }>
  > {
    const commits = await this.client.restPaginate<{
      sha: string;
      commit: {
        message: string;
        author: { name: string; email: string; date: string };
      };
    }>("GET /repos/{owner}/{repo}/pulls/{pull_number}/commits", {
      owner: identifier.owner,
      repo: identifier.repo,
      pull_number: prNumber,
    });

    return commits.map((c) => ({
      sha: c.sha,
      message: c.commit.message,
      author: c.commit.author,
    }));
  }

  // ===========================================================================
  // LABELS & ASSIGNEES
  // ===========================================================================

  /**
   * Add labels to PR
   */
  async addLabels(
    identifier: RepoIdentifier,
    prNumber: number,
    labels: string[]
  ): Promise<void> {
    await this.client.rest(
      "POST /repos/{owner}/{repo}/issues/{issue_number}/labels",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        issue_number: prNumber,
        labels,
      }
    );
  }

  /**
   * Remove label from PR
   */
  async removeLabel(
    identifier: RepoIdentifier,
    prNumber: number,
    label: string
  ): Promise<void> {
    await this.client.rest(
      "DELETE /repos/{owner}/{repo}/issues/{issue_number}/labels/{name}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        issue_number: prNumber,
        name: label,
      }
    );
  }

  /**
   * Add assignees to PR
   */
  async addAssignees(
    identifier: RepoIdentifier,
    prNumber: number,
    assignees: string[]
  ): Promise<void> {
    await this.client.rest(
      "POST /repos/{owner}/{repo}/issues/{issue_number}/assignees",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        issue_number: prNumber,
        assignees,
      }
    );
  }

  /**
   * Remove assignees from PR
   */
  async removeAssignees(
    identifier: RepoIdentifier,
    prNumber: number,
    assignees: string[]
  ): Promise<void> {
    await this.client.rest(
      "DELETE /repos/{owner}/{repo}/issues/{issue_number}/assignees",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        issue_number: prNumber,
        assignees,
      }
    );
  }

  // ===========================================================================
  // PRIVATE METHODS
  // ===========================================================================

  private transformPR(data: Record<string, unknown>): PullRequest {
    const author = data.author as {
      login: string;
      avatarUrl: string;
      id: string;
    } | null;
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
    const reviewRequests = data.reviewRequests as
      | {
          nodes: Array<{
            requestedReviewer: { login?: string; name?: string; slug?: string };
          }>;
        }
      | undefined;
    const reviews = data.reviews as
      | {
          nodes: Array<{
            id: string;
            state: string;
            body: string;
            author: { login: string };
            submittedAt: string;
          }>;
        }
      | undefined;
    const comments = data.comments as { totalCount: number } | undefined;
    const commits = data.commits as
      | {
          totalCount: number;
          nodes: Array<{
            commit: {
              statusCheckRollup?: {
                state: string;
                contexts?: { nodes: unknown[] };
              };
            };
          }>;
        }
      | undefined;
    const milestone = data.milestone as {
      id: string;
      title: string;
      number: number;
    } | null;
    const autoMerge = data.autoMergeRequest as {
      enabledAt: string;
      mergeMethod: string;
    } | null;
    const mergedBy = data.mergedBy as { login: string } | null;

    return {
      id: data.id as string,
      nodeId: data.id as string,
      number: data.number as number,
      title: data.title as string,
      body: data.body as string | null,
      state: (data.state as string).toLowerCase() as "open" | "closed",
      locked: (data.locked as boolean) ?? false,
      draft: data.isDraft as boolean,
      merged: data.merged as boolean,
      mergeable: data.mergeable === "MERGEABLE",
      mergeableState: data.mergeStateStatus as string,
      head: {
        ref: data.headRefName as string,
        sha: data.headRefOid as string,
      },
      base: {
        ref: data.baseRefName as string,
        sha: data.baseRefOid as string,
      },
      user: author
        ? {
            login: author.login,
            id: author.id,
            avatarUrl: author.avatarUrl,
          }
        : undefined,
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
      requestedReviewers:
        reviewRequests?.nodes
          .map(
            (r) => r.requestedReviewer?.login ?? r.requestedReviewer?.slug ?? ""
          )
          .filter(Boolean) ?? [],
      additions: data.additions as number,
      deletions: data.deletions as number,
      changedFiles: data.changedFiles as number,
      commitsCount: commits?.totalCount ?? 0,
      commentsCount: comments?.totalCount ?? 0,
      htmlUrl: data.url as string,
      createdAt: data.createdAt as string,
      updatedAt: data.updatedAt as string,
      closedAt: data.closedAt as string | null,
      mergedAt: data.mergedAt as string | null,
      mergedBy: mergedBy ? { login: mergedBy.login } : null,
      milestone: milestone
        ? {
            id: milestone.id,
            title: milestone.title,
            number: milestone.number,
          }
        : null,
      autoMerge: autoMerge
        ? {
            enabledAt: autoMerge.enabledAt,
            mergeMethod: autoMerge.mergeMethod.toLowerCase() as MergeMethod,
          }
        : undefined,
    };
  }

  private transformRestPR(data: Record<string, unknown>): PullRequest {
    const user = data.user as {
      login: string;
      id: number;
      avatar_url: string;
    } | null;
    const head = data.head as { ref: string; sha: string };
    const base = data.base as { ref: string; sha: string };
    const labels = data.labels as
      | Array<{ id: number; name: string; color: string; description?: string }>
      | undefined;
    const assignees = data.assignees as
      | Array<{ login: string; id: number; avatar_url: string }>
      | undefined;
    const requestedReviewers = data.requested_reviewers as
      | Array<{ login: string }>
      | undefined;
    const milestone = data.milestone as {
      id: number;
      title: string;
      number: number;
    } | null;
    const mergedBy = data.merged_by as { login: string } | null;
    const autoMerge = data.auto_merge as {
      enabled_at: string;
      merge_method: string;
    } | null;

    return {
      id: String(data.id),
      nodeId: data.node_id as string,
      number: data.number as number,
      title: data.title as string,
      body: data.body as string | null,
      state: data.state as "open" | "closed",
      locked: (data.locked as boolean) ?? false,
      draft: (data.draft as boolean) ?? false,
      merged: (data.merged as boolean) ?? false,
      mergeable: (data.mergeable as boolean | null) ?? null,
      mergeableState: (data.mergeable_state as string) ?? "unknown",
      head: { ref: head.ref, sha: head.sha },
      base: { ref: base.ref, sha: base.sha },
      user: user
        ? { login: user.login, id: user.id, avatarUrl: user.avatar_url }
        : undefined,
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
      requestedReviewers:
        requestedReviewers?.map((r) => ({ login: r.login })) ?? [],
      additions: (data.additions as number) ?? 0,
      deletions: (data.deletions as number) ?? 0,
      changedFiles: (data.changed_files as number) ?? 0,
      commitsCount: (data.commits as number) ?? 0,
      commentsCount: (data.comments as number) ?? 0,
      htmlUrl: data.html_url as string,
      createdAt: data.created_at as string,
      updatedAt: data.updated_at as string,
      closedAt: data.closed_at as string | null,
      mergedAt: data.merged_at as string | null,
      mergedBy: mergedBy ? { login: mergedBy.login } : null,
      milestone: milestone
        ? {
            id: String(milestone.id),
            title: milestone.title,
            number: milestone.number,
          }
        : null,
      autoMerge: autoMerge
        ? {
            enabledAt: autoMerge.enabled_at,
            mergeMethod: autoMerge.merge_method as MergeMethod,
          }
        : undefined,
    };
  }

  private transformReview(data: Record<string, unknown>): PRReview {
    const user = data.user as { login: string };

    return {
      id: data.id as number,
      nodeId: data.node_id as string,
      state: (data.state as string).toUpperCase() as ReviewState,
      body: (data.body as string) ?? "",
      author: { login: user.login },
      submittedAt: data.submitted_at as string,
      commitId: data.commit_id as string,
    };
  }
}

// Types are already exported where they are defined
