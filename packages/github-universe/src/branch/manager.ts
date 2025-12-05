/**
 * @fileoverview Branch Manager
 * @module @neurectomy/github-universe/branch
 *
 * Comprehensive branch management: create, merge, delete, protect.
 *
 * @agents @SYNAPSE @APEX
 */

import { EventEmitter } from "eventemitter3";
import { GitHubClient } from "../client";
import {
  type Branch,
  type BranchProtection,
  type RepoIdentifier,
  type MergeMethod,
  BranchProtectionSchema,
} from "../types";

// =============================================================================
// EVENTS
// =============================================================================

export interface BranchManagerEvents {
  "branch:created": (identifier: RepoIdentifier, branch: Branch) => void;
  "branch:deleted": (identifier: RepoIdentifier, branchName: string) => void;
  "branch:merged": (
    identifier: RepoIdentifier,
    source: string,
    target: string
  ) => void;
  "branch:protected": (
    identifier: RepoIdentifier,
    branch: string,
    rules: BranchProtection
  ) => void;
  "branch:unprotected": (identifier: RepoIdentifier, branch: string) => void;
  error: (error: Error, context: string) => void;
}

// =============================================================================
// GRAPHQL QUERIES
// =============================================================================

const BRANCH_FRAGMENT = `
  fragment BranchFields on Ref {
    name
    prefix
    target {
      ... on Commit {
        oid
        message
        author {
          name
          email
          date
        }
        committedDate
        additions
        deletions
        changedFiles
        history(first: 1) {
          totalCount
        }
      }
    }
    branchProtectionRule {
      id
      pattern
      requiresApprovingReviews
      requiredApprovingReviewCount
      requiresStatusChecks
      requiredStatusCheckContexts
      requiresStrictStatusChecks
      requiresCodeOwnerReviews
      requiresConversationResolution
      isAdminEnforced
      allowsForcePushes
      allowsDeletions
      restrictsPushes
      restrictsReviewDismissals
      dismissesStaleReviews
      requiresLinearHistory
      requiresCommitSignatures
    }
  }
`;

const GET_BRANCH_QUERY = `
  ${BRANCH_FRAGMENT}
  query GetBranch($owner: String!, $name: String!, $branch: String!) {
    repository(owner: $owner, name: $name) {
      ref(qualifiedName: $branch) {
        ...BranchFields
      }
    }
  }
`;

const LIST_BRANCHES_QUERY = `
  ${BRANCH_FRAGMENT}
  query ListBranches($owner: String!, $name: String!, $first: Int!, $after: String) {
    repository(owner: $owner, name: $name) {
      refs(refPrefix: "refs/heads/", first: $first, after: $after, orderBy: {field: TAG_COMMIT_DATE, direction: DESC}) {
        pageInfo {
          hasNextPage
          endCursor
        }
        totalCount
        nodes {
          ...BranchFields
        }
      }
    }
  }
`;

const COMPARE_BRANCHES_QUERY = `
  query CompareBranches($owner: String!, $name: String!, $base: String!, $head: String!) {
    repository(owner: $owner, name: $name) {
      ref(qualifiedName: $base) {
        compare(headRef: $head) {
          aheadBy
          behindBy
          status
        }
      }
    }
  }
`;

// =============================================================================
// TYPES
// =============================================================================

export interface BranchComparison {
  aheadBy: number;
  behindBy: number;
  status: "IDENTICAL" | "AHEAD" | "BEHIND" | "DIVERGED";
  mergeBaseCommit?: string;
  commits?: Array<{
    sha: string;
    message: string;
    author: string;
    date: string;
  }>;
  files?: Array<{
    filename: string;
    status: "added" | "removed" | "modified" | "renamed";
    additions: number;
    deletions: number;
  }>;
}

export interface MergeResult {
  merged: boolean;
  sha?: string;
  message?: string;
  error?: string;
}

export interface CreateBranchOptions {
  /** Base branch or commit SHA */
  from: string;
  /** Create from tag instead of branch */
  fromTag?: boolean;
}

// =============================================================================
// BRANCH MANAGER
// =============================================================================

/**
 * Branch Manager
 *
 * Manages GitHub branches: creation, deletion, protection, and merging.
 *
 * @example
 * ```typescript
 * const branchManager = new BranchManager(client);
 *
 * // Create a feature branch
 * const branch = await branchManager.create(
 *   { owner: 'neurectomy', repo: 'core' },
 *   'feature/new-agent',
 *   { from: 'main' }
 * );
 *
 * // Set branch protection
 * await branchManager.protect(
 *   { owner: 'neurectomy', repo: 'core' },
 *   'main',
 *   {
 *     requireReviews: true,
 *     requiredReviewers: 2,
 *     requireStatusChecks: ['build', 'test'],
 *   }
 * );
 *
 * // Merge branches
 * await branchManager.merge(
 *   { owner: 'neurectomy', repo: 'core' },
 *   'feature/new-agent',
 *   'main',
 *   'squash'
 * );
 * ```
 */
export class BranchManager extends EventEmitter<BranchManagerEvents> {
  private client: GitHubClient;

  constructor(client: GitHubClient) {
    super();
    this.client = client;
  }

  // ===========================================================================
  // BRANCH OPERATIONS
  // ===========================================================================

  /**
   * Get branch information
   */
  async get(identifier: RepoIdentifier, branchName: string): Promise<Branch> {
    const response = await this.client.graphql<{
      repository: { ref: Record<string, unknown> | null };
    }>(GET_BRANCH_QUERY, {
      owner: identifier.owner,
      name: identifier.repo,
      branch: `refs/heads/${branchName}`,
    });

    if (!response.repository.ref) {
      throw new Error(`Branch '${branchName}' not found`);
    }

    return this.transformBranch(response.repository.ref);
  }

  /**
   * Check if branch exists
   */
  async exists(
    identifier: RepoIdentifier,
    branchName: string
  ): Promise<boolean> {
    try {
      await this.get(identifier, branchName);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Create a new branch
   */
  async create(
    identifier: RepoIdentifier,
    branchName: string,
    options: CreateBranchOptions
  ): Promise<Branch> {
    // Get the SHA to branch from
    let sha: string;

    if (options.fromTag) {
      // Get tag SHA
      const response = await this.client.rest<{ object: { sha: string } }>(
        "GET /repos/{owner}/{repo}/git/refs/tags/{ref}",
        {
          owner: identifier.owner,
          repo: identifier.repo,
          ref: options.from,
        }
      );
      sha = response.object.sha;
    } else if (
      options.from.length === 40 &&
      /^[a-f0-9]+$/i.test(options.from)
    ) {
      // Already a SHA
      sha = options.from;
    } else {
      // Get branch SHA
      const response = await this.client.rest<{ object: { sha: string } }>(
        "GET /repos/{owner}/{repo}/git/refs/heads/{ref}",
        {
          owner: identifier.owner,
          repo: identifier.repo,
          ref: options.from,
        }
      );
      sha = response.object.sha;
    }

    // Create the branch
    await this.client.rest("POST /repos/{owner}/{repo}/git/refs", {
      owner: identifier.owner,
      repo: identifier.repo,
      ref: `refs/heads/${branchName}`,
      sha,
    });

    // Get the created branch
    const branch = await this.get(identifier, branchName);
    this.emit("branch:created", identifier, branch);
    return branch;
  }

  /**
   * Delete a branch
   */
  async delete(identifier: RepoIdentifier, branchName: string): Promise<void> {
    await this.client.rest(
      "DELETE /repos/{owner}/{repo}/git/refs/heads/{ref}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        ref: branchName,
      }
    );

    this.emit("branch:deleted", identifier, branchName);
  }

  /**
   * Rename a branch
   */
  async rename(
    identifier: RepoIdentifier,
    oldName: string,
    newName: string
  ): Promise<Branch> {
    await this.client.rest(
      "POST /repos/{owner}/{repo}/branches/{branch}/rename",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        branch: oldName,
        new_name: newName,
      }
    );

    return this.get(identifier, newName);
  }

  /**
   * List branches
   */
  async list(
    identifier: RepoIdentifier,
    options?: { protected?: boolean; perPage?: number; page?: number }
  ): Promise<Branch[]> {
    const branches = await this.client.restPaginate<Record<string, unknown>>(
      "GET /repos/{owner}/{repo}/branches",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        protected: options?.protected,
        per_page: options?.perPage ?? 100,
        page: options?.page ?? 1,
      }
    );

    return branches.map((b) => this.transformRestBranch(b));
  }

  /**
   * List branches using GraphQL for richer data
   */
  async listGraphQL(
    identifier: RepoIdentifier,
    options?: { first?: number; after?: string }
  ): Promise<{
    branches: Branch[];
    totalCount: number;
    pageInfo: { hasNextPage: boolean; endCursor: string };
  }> {
    const response = await this.client.graphql<{
      repository: {
        refs: {
          totalCount: number;
          pageInfo: { hasNextPage: boolean; endCursor: string };
          nodes: Record<string, unknown>[];
        };
      };
    }>(LIST_BRANCHES_QUERY, {
      owner: identifier.owner,
      name: identifier.repo,
      first: options?.first ?? 50,
      after: options?.after,
    });

    return {
      branches: response.repository.refs.nodes.map((n) =>
        this.transformBranch(n)
      ),
      totalCount: response.repository.refs.totalCount,
      pageInfo: response.repository.refs.pageInfo,
    };
  }

  // ===========================================================================
  // MERGE OPERATIONS
  // ===========================================================================

  /**
   * Merge branches
   */
  async merge(
    identifier: RepoIdentifier,
    head: string,
    base: string,
    method: MergeMethod = "merge",
    commitMessage?: string
  ): Promise<MergeResult> {
    try {
      const response = await this.client.rest<{
        sha: string;
        merged: boolean;
        message: string;
      }>("POST /repos/{owner}/{repo}/merges", {
        owner: identifier.owner,
        repo: identifier.repo,
        base,
        head,
        commit_message: commitMessage,
      });

      this.emit("branch:merged", identifier, head, base);

      return {
        merged: response.merged,
        sha: response.sha,
        message: response.message,
      };
    } catch (error) {
      const err = error as Error & { status?: number };
      if (err.status === 409) {
        return {
          merged: false,
          error: "Merge conflict",
        };
      }
      throw error;
    }
  }

  /**
   * Compare two branches
   */
  async compare(
    identifier: RepoIdentifier,
    base: string,
    head: string,
    options?: { includeCommits?: boolean; includeFiles?: boolean }
  ): Promise<BranchComparison> {
    // Get basic comparison via GraphQL
    const graphqlResponse = await this.client.graphql<{
      repository: {
        ref: {
          compare: {
            aheadBy: number;
            behindBy: number;
            status: string;
          };
        };
      };
    }>(COMPARE_BRANCHES_QUERY, {
      owner: identifier.owner,
      name: identifier.repo,
      base: `refs/heads/${base}`,
      head: `refs/heads/${head}`,
    });

    const compare = graphqlResponse.repository.ref.compare;
    const result: BranchComparison = {
      aheadBy: compare.aheadBy,
      behindBy: compare.behindBy,
      status: compare.status as BranchComparison["status"],
    };

    // Get detailed comparison via REST if requested
    if (options?.includeCommits || options?.includeFiles) {
      const restResponse = await this.client.rest<{
        merge_base_commit: { sha: string };
        commits: Array<{
          sha: string;
          commit: { message: string; author: { name: string; date: string } };
        }>;
        files: Array<{
          filename: string;
          status: string;
          additions: number;
          deletions: number;
        }>;
      }>("GET /repos/{owner}/{repo}/compare/{basehead}", {
        owner: identifier.owner,
        repo: identifier.repo,
        basehead: `${base}...${head}`,
      });

      result.mergeBaseCommit = restResponse.merge_base_commit.sha;

      if (options.includeCommits) {
        result.commits = restResponse.commits.map((c) => ({
          sha: c.sha,
          message: c.commit.message,
          author: c.commit.author.name,
          date: c.commit.author.date,
        }));
      }

      if (options.includeFiles) {
        result.files = restResponse.files.map((f) => ({
          filename: f.filename,
          status: f.status as "added" | "removed" | "modified" | "renamed",
          additions: f.additions,
          deletions: f.deletions,
        }));
      }
    }

    return result;
  }

  /**
   * Sync branch with upstream (for forks)
   */
  async syncWithUpstream(
    identifier: RepoIdentifier,
    branchName: string
  ): Promise<MergeResult> {
    try {
      const response = await this.client.rest<{
        merge_type: string;
        base_branch: string;
      }>("POST /repos/{owner}/{repo}/merge-upstream", {
        owner: identifier.owner,
        repo: identifier.repo,
        branch: branchName,
      });

      return {
        merged: true,
        message: `Synced ${response.base_branch} with upstream`,
      };
    } catch (error) {
      const err = error as Error & { status?: number };
      return {
        merged: false,
        error: err.message,
      };
    }
  }

  // ===========================================================================
  // BRANCH PROTECTION
  // ===========================================================================

  /**
   * Get branch protection rules
   */
  async getProtection(
    identifier: RepoIdentifier,
    branchName: string
  ): Promise<BranchProtection | null> {
    try {
      const response = await this.client.rest<Record<string, unknown>>(
        "GET /repos/{owner}/{repo}/branches/{branch}/protection",
        {
          owner: identifier.owner,
          repo: identifier.repo,
          branch: branchName,
        }
      );

      return this.transformProtection(response);
    } catch (error) {
      const err = error as Error & { status?: number };
      if (err.status === 404) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Set branch protection rules
   */
  async protect(
    identifier: RepoIdentifier,
    branchName: string,
    rules: BranchProtection
  ): Promise<BranchProtection> {
    const validated = BranchProtectionSchema.parse(rules);

    await this.client.rest(
      "PUT /repos/{owner}/{repo}/branches/{branch}/protection",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        branch: branchName,
        required_status_checks: validated.requireStatusChecks
          ? {
              strict: validated.strictStatusChecks ?? false,
              contexts: validated.requiredStatusChecks ?? [],
            }
          : null,
        enforce_admins: validated.enforceAdmins ?? false,
        required_pull_request_reviews: validated.requireReviews
          ? {
              dismiss_stale_reviews: validated.dismissStaleReviews ?? false,
              require_code_owner_reviews: validated.requireCodeOwners ?? false,
              required_approving_review_count: validated.requiredReviewers ?? 1,
            }
          : null,
        restrictions: validated.restrictPushes
          ? {
              users: validated.allowedPushers ?? [],
              teams: [],
            }
          : null,
        required_linear_history: validated.requireLinearHistory ?? false,
        allow_force_pushes: validated.allowForcePushes ?? false,
        allow_deletions: validated.allowDeletions ?? false,
        required_conversation_resolution:
          validated.requireConversationResolution ?? false,
        required_signatures: validated.requireSignedCommits ?? false,
      }
    );

    this.emit("branch:protected", identifier, branchName, validated);
    return validated;
  }

  /**
   * Remove branch protection
   */
  async unprotect(
    identifier: RepoIdentifier,
    branchName: string
  ): Promise<void> {
    await this.client.rest(
      "DELETE /repos/{owner}/{repo}/branches/{branch}/protection",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        branch: branchName,
      }
    );

    this.emit("branch:unprotected", identifier, branchName);
  }

  /**
   * List protected branches
   */
  async listProtected(identifier: RepoIdentifier): Promise<Branch[]> {
    return this.list(identifier, { protected: true });
  }

  // ===========================================================================
  // STATUS CHECKS
  // ===========================================================================

  /**
   * Get required status checks for a branch
   */
  async getRequiredStatusChecks(
    identifier: RepoIdentifier,
    branchName: string
  ): Promise<{ strict: boolean; contexts: string[] } | null> {
    try {
      const response = await this.client.rest<{
        strict: boolean;
        contexts: string[];
      }>(
        "GET /repos/{owner}/{repo}/branches/{branch}/protection/required_status_checks",
        {
          owner: identifier.owner,
          repo: identifier.repo,
          branch: branchName,
        }
      );

      return response;
    } catch (error) {
      const err = error as Error & { status?: number };
      if (err.status === 404) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Update required status checks
   */
  async updateRequiredStatusChecks(
    identifier: RepoIdentifier,
    branchName: string,
    checks: { strict?: boolean; contexts?: string[] }
  ): Promise<void> {
    await this.client.rest(
      "PATCH /repos/{owner}/{repo}/branches/{branch}/protection/required_status_checks",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        branch: branchName,
        strict: checks.strict,
        contexts: checks.contexts,
      }
    );
  }

  // ===========================================================================
  // COMMIT OPERATIONS
  // ===========================================================================

  /**
   * Get the latest commit on a branch
   */
  async getLatestCommit(
    identifier: RepoIdentifier,
    branchName: string
  ): Promise<{
    sha: string;
    message: string;
    author: { name: string; email: string; date: string };
    committer: { name: string; email: string; date: string };
  }> {
    const response = await this.client.rest<{
      sha: string;
      commit: {
        message: string;
        author: { name: string; email: string; date: string };
        committer: { name: string; email: string; date: string };
      };
    }>("GET /repos/{owner}/{repo}/commits/{ref}", {
      owner: identifier.owner,
      repo: identifier.repo,
      ref: branchName,
    });

    return {
      sha: response.sha,
      message: response.commit.message,
      author: response.commit.author,
      committer: response.commit.committer,
    };
  }

  /**
   * Get commits on a branch
   */
  async getCommits(
    identifier: RepoIdentifier,
    branchName: string,
    options?: {
      since?: string;
      until?: string;
      author?: string;
      perPage?: number;
      page?: number;
    }
  ): Promise<
    Array<{
      sha: string;
      message: string;
      author: { name: string; email: string; date: string };
      url: string;
    }>
  > {
    const commits = await this.client.restPaginate<{
      sha: string;
      commit: {
        message: string;
        author: { name: string; email: string; date: string };
      };
      html_url: string;
    }>("GET /repos/{owner}/{repo}/commits", {
      owner: identifier.owner,
      repo: identifier.repo,
      sha: branchName,
      since: options?.since,
      until: options?.until,
      author: options?.author,
      per_page: options?.perPage ?? 100,
      page: options?.page ?? 1,
    });

    return commits.map((c) => ({
      sha: c.sha,
      message: c.commit.message,
      author: c.commit.author,
      url: c.html_url,
    }));
  }

  // ===========================================================================
  // PRIVATE METHODS
  // ===========================================================================

  private transformBranch(data: Record<string, unknown>): Branch {
    const target = data.target as Record<string, unknown>;
    const author = target.author as {
      name: string;
      email: string;
      date: string;
    };
    const history = target.history as { totalCount: number };
    const protection = data.branchProtectionRule as Record<
      string,
      unknown
    > | null;

    return {
      name: (data.name as string).replace("refs/heads/", ""),
      sha: target.oid as string,
      protected: !!protection,
      commit: {
        sha: target.oid as string,
        message: target.message as string,
        author: {
          name: author.name,
          email: author.email,
          date: author.date,
        },
      },
      protection: protection
        ? this.transformProtectionFromGraphQL(protection)
        : undefined,
    };
  }

  private transformRestBranch(data: Record<string, unknown>): Branch {
    const commit = data.commit as { sha: string; url: string };

    return {
      name: data.name as string,
      sha: commit.sha,
      protected: data.protected as boolean,
      commit: {
        sha: commit.sha,
        message: "", // REST doesn't include message
        author: {
          name: "",
          email: "",
          date: "",
        },
      },
    };
  }

  private transformProtection(data: Record<string, unknown>): BranchProtection {
    const requiredStatusChecks = data.required_status_checks as {
      strict: boolean;
      contexts: string[];
    } | null;
    const requiredPullRequestReviews = data.required_pull_request_reviews as {
      dismiss_stale_reviews: boolean;
      require_code_owner_reviews: boolean;
      required_approving_review_count: number;
    } | null;
    const restrictions = data.restrictions as {
      users: Array<{ login: string }>;
    } | null;
    const enforceAdmins = data.enforce_admins as { enabled: boolean } | null;

    return {
      requireReviews: !!requiredPullRequestReviews,
      requiredReviewers:
        requiredPullRequestReviews?.required_approving_review_count,
      dismissStaleReviews: requiredPullRequestReviews?.dismiss_stale_reviews,
      requireCodeOwners: requiredPullRequestReviews?.require_code_owner_reviews,
      requireStatusChecks: !!requiredStatusChecks,
      requiredStatusChecks: requiredStatusChecks?.contexts,
      strictStatusChecks: requiredStatusChecks?.strict,
      enforceAdmins: enforceAdmins?.enabled ?? false,
      restrictPushes: !!restrictions,
      allowedPushers: restrictions?.users.map((u) => u.login),
      requireLinearHistory: (data.required_linear_history as boolean) ?? false,
      allowForcePushes: (data.allow_force_pushes as boolean) ?? false,
      allowDeletions: (data.allow_deletions as boolean) ?? false,
      requireConversationResolution:
        (data.required_conversation_resolution as boolean) ?? false,
      requireSignedCommits: (data.required_signatures as boolean) ?? false,
    };
  }

  private transformProtectionFromGraphQL(
    data: Record<string, unknown>
  ): BranchProtection {
    return {
      requireReviews: (data.requiresApprovingReviews as boolean) ?? false,
      requiredReviewers: data.requiredApprovingReviewCount as number,
      dismissStaleReviews: data.dismissesStaleReviews as boolean,
      requireCodeOwners: data.requiresCodeOwnerReviews as boolean,
      requireStatusChecks: (data.requiresStatusChecks as boolean) ?? false,
      requiredStatusChecks: data.requiredStatusCheckContexts as string[],
      strictStatusChecks: data.requiresStrictStatusChecks as boolean,
      enforceAdmins: (data.isAdminEnforced as boolean) ?? false,
      restrictPushes: (data.restrictsPushes as boolean) ?? false,
      requireLinearHistory: (data.requiresLinearHistory as boolean) ?? false,
      allowForcePushes: (data.allowsForcePushes as boolean) ?? false,
      allowDeletions: (data.allowsDeletions as boolean) ?? false,
      requireConversationResolution:
        (data.requiresConversationResolution as boolean) ?? false,
      requireSignedCommits: (data.requiresCommitSignatures as boolean) ?? false,
    };
  }
}
