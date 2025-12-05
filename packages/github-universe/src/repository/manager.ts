/**
 * @fileoverview Repository Manager
 * @module @neurectomy/github-universe/repository
 *
 * Comprehensive repository management: clone, create, fork, configure.
 *
 * @agents @SYNAPSE @APEX
 */

import { EventEmitter } from "eventemitter3";
import { v4 as uuidv4 } from "uuid";
import {
  simpleGit,
  SimpleGit,
  CloneOptions as GitCloneOptions,
} from "simple-git";
import * as path from "path";
import * as fs from "fs/promises";
import { GitHubClient } from "../client";
import {
  type Repository,
  type RepoIdentifier,
  type CreateRepoConfig,
  CreateRepoConfigSchema,
  type CloneOptions,
  CloneOptionsSchema,
  type ForkConfig,
  ForkConfigSchema,
  type Visibility,
} from "../types";

// =============================================================================
// EVENTS
// =============================================================================

export interface RepositoryManagerEvents {
  "repo:created": (repo: Repository) => void;
  "repo:deleted": (identifier: RepoIdentifier) => void;
  "repo:forked": (original: RepoIdentifier, fork: Repository) => void;
  "repo:cloned": (identifier: RepoIdentifier, path: string) => void;
  "repo:updated": (repo: Repository) => void;
  "clone:progress": (identifier: RepoIdentifier, progress: number) => void;
  error: (error: Error, context: string) => void;
}

// =============================================================================
// GRAPHQL QUERIES
// =============================================================================

const REPOSITORY_FRAGMENT = `
  fragment RepositoryFields on Repository {
    id
    databaseId
    name
    nameWithOwner
    description
    url
    homepageUrl
    isPrivate
    isFork
    isArchived
    isDisabled
    isTemplate
    visibility
    defaultBranchRef {
      name
    }
    primaryLanguage {
      name
    }
    languages(first: 10) {
      edges {
        size
        node {
          name
        }
      }
    }
    stargazerCount
    forkCount
    watchers {
      totalCount
    }
    issues(states: OPEN) {
      totalCount
    }
    diskUsage
    repositoryTopics(first: 20) {
      nodes {
        topic {
          name
        }
      }
    }
    licenseInfo {
      key
      name
      spdxId
    }
    createdAt
    updatedAt
    pushedAt
    owner {
      login
      ... on User {
        id
        avatarUrl
      }
      ... on Organization {
        id
        avatarUrl
      }
    }
    viewerPermission
    viewerCanAdminister
  }
`;

const GET_REPOSITORY_QUERY = `
  ${REPOSITORY_FRAGMENT}
  query GetRepository($owner: String!, $name: String!) {
    repository(owner: $owner, name: $name) {
      ...RepositoryFields
    }
  }
`;

const LIST_USER_REPOS_QUERY = `
  ${REPOSITORY_FRAGMENT}
  query ListUserRepositories($login: String!, $first: Int!, $after: String) {
    user(login: $login) {
      repositories(first: $first, after: $after, orderBy: {field: UPDATED_AT, direction: DESC}) {
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          ...RepositoryFields
        }
      }
    }
  }
`;

const SEARCH_REPOS_QUERY = `
  ${REPOSITORY_FRAGMENT}
  query SearchRepositories($query: String!, $first: Int!, $after: String) {
    search(query: $query, type: REPOSITORY, first: $first, after: $after) {
      repositoryCount
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        ... on Repository {
          ...RepositoryFields
        }
      }
    }
  }
`;

// =============================================================================
// REPOSITORY MANAGER
// =============================================================================

/**
 * Repository Manager
 *
 * Manages GitHub repositories: creation, cloning, forking, and configuration.
 *
 * @example
 * ```typescript
 * const repoManager = new RepositoryManager(client);
 *
 * // Create a new repository
 * const repo = await repoManager.create({
 *   name: 'my-agent',
 *   description: 'An AI agent for task automation',
 *   visibility: 'private',
 *   autoInit: true,
 * });
 *
 * // Clone a repository
 * await repoManager.clone(
 *   { owner: 'neurectomy', repo: 'core' },
 *   { path: './workspace/core', depth: 1 }
 * );
 *
 * // Fork a repository
 * const fork = await repoManager.fork(
 *   { owner: 'someorg', repo: 'cool-agent' },
 *   { name: 'my-cool-agent' }
 * );
 * ```
 */
export class RepositoryManager extends EventEmitter<RepositoryManagerEvents> {
  private client: GitHubClient;
  private git: SimpleGit;
  private cloneOperations = new Map<
    string,
    { progress: number; cancel: () => void }
  >();

  constructor(client: GitHubClient) {
    super();
    this.client = client;
    this.git = simpleGit();
  }

  // ===========================================================================
  // REPOSITORY OPERATIONS
  // ===========================================================================

  /**
   * Get repository information
   */
  async get(identifier: RepoIdentifier): Promise<Repository> {
    const response = await this.client.graphql<{
      repository: Record<string, unknown>;
    }>(GET_REPOSITORY_QUERY, {
      owner: identifier.owner,
      name: identifier.repo,
    });

    return this.transformRepository(response.repository);
  }

  /**
   * Check if repository exists
   */
  async exists(identifier: RepoIdentifier): Promise<boolean> {
    try {
      await this.get(identifier);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Create a new repository
   */
  async create(config: CreateRepoConfig, org?: string): Promise<Repository> {
    const validated = CreateRepoConfigSchema.parse(config);

    const endpoint = org ? "POST /orgs/{org}/repos" : "POST /user/repos";

    const response = await this.client.rest<Record<string, unknown>>(endpoint, {
      ...(org && { org }),
      name: validated.name,
      description: validated.description,
      private: validated.visibility === "private",
      visibility: validated.visibility,
      auto_init: validated.autoInit,
      gitignore_template: validated.gitignoreTemplate,
      license_template: validated.licenseTemplate,
      allow_squash_merge: validated.allowSquashMerge,
      allow_merge_commit: validated.allowMergeCommit,
      allow_rebase_merge: validated.allowRebaseMerge,
      allow_auto_merge: validated.allowAutoMerge,
      delete_branch_on_merge: validated.deleteBranchOnMerge,
      has_issues: validated.hasIssues,
      has_projects: validated.hasProjects,
      has_wiki: validated.hasWiki,
      has_discussions: validated.hasDiscussions,
      is_template: validated.isTemplate,
      homepage: validated.homepage,
    });

    const repo = this.transformRestRepository(response);

    // Set topics if provided
    if (validated.topics && validated.topics.length > 0) {
      await this.setTopics(
        { owner: repo.owner.login, repo: repo.name },
        validated.topics
      );
      repo.topics = validated.topics;
    }

    this.emit("repo:created", repo);
    return repo;
  }

  /**
   * Create repository from template
   */
  async createFromTemplate(
    template: RepoIdentifier,
    config: CreateRepoConfig,
    org?: string
  ): Promise<Repository> {
    const validated = CreateRepoConfigSchema.parse(config);

    const response = await this.client.rest<Record<string, unknown>>(
      "POST /repos/{template_owner}/{template_repo}/generate",
      {
        template_owner: template.owner,
        template_repo: template.repo,
        owner: org,
        name: validated.name,
        description: validated.description,
        private: validated.visibility === "private",
        include_all_branches: false,
      }
    );

    const repo = this.transformRestRepository(response);
    this.emit("repo:created", repo);
    return repo;
  }

  /**
   * Update repository settings
   */
  async update(
    identifier: RepoIdentifier,
    updates: Partial<CreateRepoConfig>
  ): Promise<Repository> {
    const response = await this.client.rest<Record<string, unknown>>(
      "PATCH /repos/{owner}/{repo}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        name: updates.name,
        description: updates.description,
        private: updates.visibility === "private",
        visibility: updates.visibility,
        allow_squash_merge: updates.allowSquashMerge,
        allow_merge_commit: updates.allowMergeCommit,
        allow_rebase_merge: updates.allowRebaseMerge,
        allow_auto_merge: updates.allowAutoMerge,
        delete_branch_on_merge: updates.deleteBranchOnMerge,
        has_issues: updates.hasIssues,
        has_projects: updates.hasProjects,
        has_wiki: updates.hasWiki,
        has_discussions: updates.hasDiscussions,
        is_template: updates.isTemplate,
        homepage: updates.homepage,
      }
    );

    const repo = this.transformRestRepository(response);

    // Update topics if provided
    if (updates.topics) {
      await this.setTopics(identifier, updates.topics);
      repo.topics = updates.topics;
    }

    this.emit("repo:updated", repo);
    return repo;
  }

  /**
   * Delete repository
   */
  async delete(identifier: RepoIdentifier): Promise<void> {
    await this.client.rest("DELETE /repos/{owner}/{repo}", {
      owner: identifier.owner,
      repo: identifier.repo,
    });

    this.emit("repo:deleted", identifier);
  }

  /**
   * Archive repository
   */
  async archive(identifier: RepoIdentifier): Promise<Repository> {
    const response = await this.client.rest<Record<string, unknown>>(
      "PATCH /repos/{owner}/{repo}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        archived: true,
      }
    );

    const repo = this.transformRestRepository(response);
    this.emit("repo:updated", repo);
    return repo;
  }

  /**
   * Unarchive repository
   */
  async unarchive(identifier: RepoIdentifier): Promise<Repository> {
    const response = await this.client.rest<Record<string, unknown>>(
      "PATCH /repos/{owner}/{repo}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        archived: false,
      }
    );

    const repo = this.transformRestRepository(response);
    this.emit("repo:updated", repo);
    return repo;
  }

  /**
   * Transfer repository to another owner
   */
  async transfer(
    identifier: RepoIdentifier,
    newOwner: string,
    teamIds?: number[]
  ): Promise<Repository> {
    const response = await this.client.rest<Record<string, unknown>>(
      "POST /repos/{owner}/{repo}/transfer",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        new_owner: newOwner,
        ...(teamIds && { team_ids: teamIds }),
      }
    );

    return this.transformRestRepository(response);
  }

  // ===========================================================================
  // FORK OPERATIONS
  // ===========================================================================

  /**
   * Fork a repository
   */
  async fork(
    identifier: RepoIdentifier,
    config?: ForkConfig
  ): Promise<Repository> {
    const validated: ForkConfig = config
      ? ForkConfigSchema.parse(config)
      : { defaultBranchOnly: false };

    const response = await this.client.rest<Record<string, unknown>>(
      "POST /repos/{owner}/{repo}/forks",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        organization: validated.organization,
        name: validated.name,
        default_branch_only: validated.defaultBranchOnly,
      }
    );

    const fork = this.transformRestRepository(response);
    this.emit("repo:forked", identifier, fork);
    return fork;
  }

  /**
   * List forks of a repository
   */
  async listForks(
    identifier: RepoIdentifier,
    options?: {
      sort?: "newest" | "oldest" | "stargazers" | "watchers";
      perPage?: number;
    }
  ): Promise<Repository[]> {
    const forks = await this.client.restPaginate<Record<string, unknown>>(
      "GET /repos/{owner}/{repo}/forks",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        sort: options?.sort ?? "newest",
        per_page: options?.perPage ?? 100,
      }
    );

    return forks.map((f) => this.transformRestRepository(f));
  }

  // ===========================================================================
  // CLONE OPERATIONS
  // ===========================================================================

  /**
   * Clone a repository
   */
  async clone(
    identifier: RepoIdentifier,
    options: CloneOptions
  ): Promise<string> {
    const validated = CloneOptionsSchema.parse(options);
    const operationId = uuidv4();

    // Get clone URL
    const repo = await this.get(identifier);
    const cloneUrl = this.getAuthenticatedCloneUrl(repo.cloneUrl);

    // Prepare clone options
    const gitOptions: GitCloneOptions = {
      "--progress": null,
    };

    if (validated.branch) {
      gitOptions["--branch"] = validated.branch;
    }
    if (validated.depth) {
      gitOptions["--depth"] = validated.depth;
    }
    if (validated.recursive) {
      gitOptions["--recursive"] = null;
    }
    if (validated.mirror) {
      gitOptions["--mirror"] = null;
    }
    if (validated.bare) {
      gitOptions["--bare"] = null;
    }
    if (validated.singleBranch) {
      gitOptions["--single-branch"] = null;
    }

    // Ensure directory exists
    const targetPath = path.resolve(validated.path);
    await fs.mkdir(path.dirname(targetPath), { recursive: true });

    // Set up progress tracking
    let cancelled = false;
    this.cloneOperations.set(operationId, {
      progress: 0,
      cancel: () => {
        cancelled = true;
      },
    });

    try {
      // Clone repository
      await this.git.clone(cloneUrl, targetPath, gitOptions);

      if (cancelled) {
        // Clean up if cancelled
        await fs.rm(targetPath, { recursive: true, force: true });
        throw new Error("Clone operation cancelled");
      }

      this.emit("repo:cloned", identifier, targetPath);
      return targetPath;
    } finally {
      this.cloneOperations.delete(operationId);
    }
  }

  /**
   * Cancel a clone operation
   */
  cancelClone(operationId: string): boolean {
    const operation = this.cloneOperations.get(operationId);
    if (operation) {
      operation.cancel();
      return true;
    }
    return false;
  }

  /**
   * Get download URL for repository archive
   */
  async getArchiveUrl(
    identifier: RepoIdentifier,
    format: "tarball" | "zipball" = "tarball",
    ref?: string
  ): Promise<string> {
    const response = await this.client.rest<{ url: string }>(
      `GET /repos/{owner}/{repo}/${format}/{ref}`,
      {
        owner: identifier.owner,
        repo: identifier.repo,
        ref: ref ?? "",
      }
    );

    return response.url;
  }

  // ===========================================================================
  // LISTING & SEARCH
  // ===========================================================================

  /**
   * List repositories for authenticated user
   */
  async listUserRepos(options?: {
    type?: "all" | "owner" | "public" | "private" | "member";
    sort?: "created" | "updated" | "pushed" | "full_name";
    direction?: "asc" | "desc";
    perPage?: number;
    page?: number;
  }): Promise<Repository[]> {
    const repos = await this.client.restPaginate<Record<string, unknown>>(
      "GET /user/repos",
      {
        type: options?.type ?? "all",
        sort: options?.sort ?? "updated",
        direction: options?.direction ?? "desc",
        per_page: options?.perPage ?? 100,
        page: options?.page ?? 1,
      }
    );

    return repos.map((r) => this.transformRestRepository(r));
  }

  /**
   * List repositories for a user
   */
  async listReposForUser(
    username: string,
    options?: {
      type?: "all" | "owner" | "member";
      sort?: "created" | "updated" | "pushed" | "full_name";
      perPage?: number;
    }
  ): Promise<Repository[]> {
    const repos = await this.client.restPaginate<Record<string, unknown>>(
      "GET /users/{username}/repos",
      {
        username,
        type: options?.type ?? "all",
        sort: options?.sort ?? "updated",
        per_page: options?.perPage ?? 100,
      }
    );

    return repos.map((r) => this.transformRestRepository(r));
  }

  /**
   * List repositories for an organization
   */
  async listOrgRepos(
    org: string,
    options?: {
      type?: "all" | "public" | "private" | "forks" | "sources" | "member";
      sort?: "created" | "updated" | "pushed" | "full_name";
      perPage?: number;
    }
  ): Promise<Repository[]> {
    const repos = await this.client.restPaginate<Record<string, unknown>>(
      "GET /orgs/{org}/repos",
      {
        org,
        type: options?.type ?? "all",
        sort: options?.sort ?? "updated",
        per_page: options?.perPage ?? 100,
      }
    );

    return repos.map((r) => this.transformRestRepository(r));
  }

  /**
   * Search repositories
   */
  async search(
    query: string,
    options?: {
      sort?: "stars" | "forks" | "help-wanted-issues" | "updated";
      order?: "asc" | "desc";
      perPage?: number;
      page?: number;
    }
  ): Promise<{ totalCount: number; items: Repository[] }> {
    const response = await this.client.rest<{
      total_count: number;
      incomplete_results: boolean;
      items: Record<string, unknown>[];
    }>("GET /search/repositories", {
      q: query,
      sort: options?.sort,
      order: options?.order ?? "desc",
      per_page: options?.perPage ?? 30,
      page: options?.page ?? 1,
    });

    return {
      totalCount: response.total_count,
      items: response.items.map((r) => this.transformRestRepository(r)),
    };
  }

  /**
   * Search repositories using GraphQL for richer data
   */
  async searchGraphQL(
    query: string,
    options?: { first?: number; after?: string }
  ): Promise<{
    totalCount: number;
    items: Repository[];
    pageInfo: { hasNextPage: boolean; endCursor: string };
  }> {
    const response = await this.client.graphql<{
      search: {
        repositoryCount: number;
        pageInfo: { hasNextPage: boolean; endCursor: string };
        nodes: Record<string, unknown>[];
      };
    }>(SEARCH_REPOS_QUERY, {
      query,
      first: options?.first ?? 20,
      after: options?.after,
    });

    return {
      totalCount: response.search.repositoryCount,
      items: response.search.nodes.map((n) => this.transformRepository(n)),
      pageInfo: response.search.pageInfo,
    };
  }

  // ===========================================================================
  // TOPICS & METADATA
  // ===========================================================================

  /**
   * Get repository topics
   */
  async getTopics(identifier: RepoIdentifier): Promise<string[]> {
    const response = await this.client.rest<{ names: string[] }>(
      "GET /repos/{owner}/{repo}/topics",
      {
        owner: identifier.owner,
        repo: identifier.repo,
      }
    );

    return response.names;
  }

  /**
   * Set repository topics
   */
  async setTopics(
    identifier: RepoIdentifier,
    topics: string[]
  ): Promise<string[]> {
    const response = await this.client.rest<{ names: string[] }>(
      "PUT /repos/{owner}/{repo}/topics",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        names: topics,
      }
    );

    return response.names;
  }

  /**
   * Get repository languages
   */
  async getLanguages(
    identifier: RepoIdentifier
  ): Promise<Record<string, number>> {
    return this.client.rest<Record<string, number>>(
      "GET /repos/{owner}/{repo}/languages",
      {
        owner: identifier.owner,
        repo: identifier.repo,
      }
    );
  }

  /**
   * Get repository contributors
   */
  async getContributors(
    identifier: RepoIdentifier,
    options?: { anon?: boolean; perPage?: number }
  ): Promise<Array<{ login: string; id: number; contributions: number }>> {
    return this.client.restPaginate("GET /repos/{owner}/{repo}/contributors", {
      owner: identifier.owner,
      repo: identifier.repo,
      anon: options?.anon ? "true" : undefined,
      per_page: options?.perPage ?? 100,
    });
  }

  /**
   * Get repository README
   */
  async getReadme(
    identifier: RepoIdentifier,
    ref?: string
  ): Promise<{ content: string; encoding: string; path: string }> {
    const response = await this.client.rest<{
      content: string;
      encoding: string;
      path: string;
    }>("GET /repos/{owner}/{repo}/readme", {
      owner: identifier.owner,
      repo: identifier.repo,
      ...(ref && { ref }),
    });

    // Decode content if base64
    if (response.encoding === "base64") {
      response.content = Buffer.from(response.content, "base64").toString(
        "utf-8"
      );
    }

    return response;
  }

  // ===========================================================================
  // COLLABORATORS
  // ===========================================================================

  /**
   * List repository collaborators
   */
  async listCollaborators(
    identifier: RepoIdentifier,
    options?: {
      affiliation?: "outside" | "direct" | "all";
      permission?: string;
    }
  ): Promise<
    Array<{ login: string; id: number; permissions: Record<string, boolean> }>
  > {
    return this.client.restPaginate("GET /repos/{owner}/{repo}/collaborators", {
      owner: identifier.owner,
      repo: identifier.repo,
      affiliation: options?.affiliation ?? "all",
      permission: options?.permission,
    });
  }

  /**
   * Add collaborator to repository
   */
  async addCollaborator(
    identifier: RepoIdentifier,
    username: string,
    permission: "pull" | "triage" | "push" | "maintain" | "admin" = "push"
  ): Promise<void> {
    await this.client.rest(
      "PUT /repos/{owner}/{repo}/collaborators/{username}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        username,
        permission,
      }
    );
  }

  /**
   * Remove collaborator from repository
   */
  async removeCollaborator(
    identifier: RepoIdentifier,
    username: string
  ): Promise<void> {
    await this.client.rest(
      "DELETE /repos/{owner}/{repo}/collaborators/{username}",
      {
        owner: identifier.owner,
        repo: identifier.repo,
        username,
      }
    );
  }

  /**
   * Check if user is a collaborator
   */
  async isCollaborator(
    identifier: RepoIdentifier,
    username: string
  ): Promise<boolean> {
    try {
      await this.client.rest(
        "GET /repos/{owner}/{repo}/collaborators/{username}",
        {
          owner: identifier.owner,
          repo: identifier.repo,
          username,
        }
      );
      return true;
    } catch {
      return false;
    }
  }

  // ===========================================================================
  // PRIVATE METHODS
  // ===========================================================================

  private getAuthenticatedCloneUrl(cloneUrl: string): string {
    const token = this.client.getConfig().auth.token;
    const url = new URL(cloneUrl);
    url.username = "x-access-token";
    url.password = token;
    return url.toString();
  }

  private transformRepository(data: Record<string, unknown>): Repository {
    const owner = data.owner as Record<string, unknown>;
    const defaultBranchRef = data.defaultBranchRef as { name: string } | null;
    const primaryLanguage = data.primaryLanguage as { name: string } | null;
    const licenseInfo = data.licenseInfo as {
      key: string;
      name: string;
      spdxId: string | null;
    } | null;
    const languages = data.languages as
      | { edges: Array<{ size: number; node: { name: string } }> }
      | undefined;
    const topics = data.repositoryTopics as
      | { nodes: Array<{ topic: { name: string } }> }
      | undefined;
    const watchers = data.watchers as { totalCount: number } | undefined;
    const issues = data.issues as { totalCount: number } | undefined;

    return {
      id: data.databaseId as number,
      nodeId: data.id as string,
      name: data.name as string,
      fullName: data.nameWithOwner as string,
      owner: {
        login: owner.login as string,
        id: owner.id as number,
        avatarUrl: owner.avatarUrl as string,
        type: "User", // GraphQL doesn't easily expose this
      },
      description: data.description as string | null,
      visibility: (data.visibility as string).toLowerCase() as Visibility,
      private: data.isPrivate as boolean,
      fork: data.isFork as boolean,
      archived: data.isArchived as boolean,
      disabled: data.isDisabled as boolean,
      defaultBranch: defaultBranchRef?.name ?? "main",
      language: primaryLanguage?.name ?? null,
      languages: languages?.edges.reduce(
        (acc, e) => {
          acc[e.node.name] = e.size;
          return acc;
        },
        {} as Record<string, number>
      ),
      stargazersCount: data.stargazerCount as number,
      watchersCount: watchers?.totalCount ?? 0,
      forksCount: data.forkCount as number,
      openIssuesCount: issues?.totalCount ?? 0,
      size: data.diskUsage as number,
      topics: topics?.nodes.map((n) => n.topic.name) ?? [],
      htmlUrl: data.url as string,
      cloneUrl: `${data.url}.git`,
      sshUrl: `git@github.com:${data.nameWithOwner}.git`,
      gitUrl: `git://github.com/${data.nameWithOwner}.git`,
      createdAt: data.createdAt as string,
      updatedAt: data.updatedAt as string,
      pushedAt: data.pushedAt as string | null,
      license: licenseInfo
        ? {
            key: licenseInfo.key,
            name: licenseInfo.name,
            spdxId: licenseInfo.spdxId,
          }
        : null,
      permissions: data.viewerCanAdminister
        ? {
            admin: data.viewerCanAdminister as boolean,
            push: true,
            pull: true,
          }
        : undefined,
    };
  }

  private transformRestRepository(data: Record<string, unknown>): Repository {
    const owner = data.owner as Record<string, unknown>;
    const license = data.license as {
      key: string;
      name: string;
      spdx_id: string;
    } | null;

    return {
      id: data.id as number,
      nodeId: data.node_id as string,
      name: data.name as string,
      fullName: data.full_name as string,
      owner: {
        login: owner.login as string,
        id: owner.id as number,
        avatarUrl: owner.avatar_url as string,
        type: owner.type as "User" | "Organization",
      },
      description: data.description as string | null,
      visibility:
        (data.visibility as Visibility) ??
        (data.private ? "private" : "public"),
      private: data.private as boolean,
      fork: data.fork as boolean,
      archived: data.archived as boolean,
      disabled: data.disabled as boolean,
      defaultBranch: data.default_branch as string,
      language: data.language as string | null,
      stargazersCount: data.stargazers_count as number,
      watchersCount: data.watchers_count as number,
      forksCount: data.forks_count as number,
      openIssuesCount: data.open_issues_count as number,
      size: data.size as number,
      topics: (data.topics as string[]) ?? [],
      htmlUrl: data.html_url as string,
      cloneUrl: data.clone_url as string,
      sshUrl: data.ssh_url as string,
      gitUrl: data.git_url as string,
      createdAt: data.created_at as string,
      updatedAt: data.updated_at as string,
      pushedAt: data.pushed_at as string | null,
      license: license
        ? {
            key: license.key,
            name: license.name,
            spdxId: license.spdx_id,
          }
        : null,
      permissions: data.permissions as Repository["permissions"],
    };
  }
}

// Types are already exported where they are defined
