/**
 * NEURECTOMY Discovery Engine - Repository Scanner
 *
 * @VANGUARD @PRISM - Comprehensive repository scanning with GitHub API integration
 *
 * Features:
 * - Repository metadata extraction
 * - Health metrics calculation
 * - Dependency file detection
 * - Activity analysis
 * - Community health scoring
 *
 * @packageDocumentation
 */

import { Octokit } from "@octokit/rest";
import { EventEmitter } from "eventemitter3";
import pLimit from "p-limit";

import type {
  RepositoryInfo,
  RepositoryHealth,
  HealthFactor,
  HealthCategory,
  LicenseInfo,
  ScannerConfig,
  RepositorySearchQuery,
  RepositorySearchResult,
  DiscoveryEvents,
} from "../types";

// ==============================================================================
// Scanner Configuration
// ==============================================================================

const DEFAULT_CONFIG: Required<ScannerConfig> = {
  githubToken: "",
  rateLimit: 30,
  concurrency: 5,
  cacheTtl: 3600,
  includeArchived: false,
  includeForks: false,
  minStars: 0,
  languages: [],
  topics: [],
};

// ==============================================================================
// Repository Scanner
// ==============================================================================

/**
 * Repository scanner with GitHub API integration
 *
 * @example
 * ```typescript
 * const scanner = new RepositoryScanner({ githubToken: process.env.GITHUB_TOKEN });
 *
 * // Scan a single repository
 * const repo = await scanner.scanRepository("facebook", "react");
 *
 * // Search repositories
 * const results = await scanner.searchRepositories({
 *   language: "typescript",
 *   minStars: 1000,
 *   topic: "react",
 * });
 *
 * // Calculate health score
 * const health = await scanner.calculateHealth(repo);
 * ```
 */
export class RepositoryScanner extends EventEmitter<DiscoveryEvents> {
  private octokit: Octokit;
  private config: Required<ScannerConfig>;
  private limiter: ReturnType<typeof pLimit>;
  private cache: Map<string, { data: unknown; expiresAt: number }> = new Map();

  constructor(config: ScannerConfig = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.octokit = new Octokit({
      auth: this.config.githubToken || undefined,
      throttle: {
        onRateLimit: (retryAfter: number, options: unknown) => {
          this.emit("error", {
            error: new Error(`Rate limit hit, retrying after ${retryAfter}s`),
            context: "rate-limit",
          });
          return true;
        },
        onSecondaryRateLimit: (retryAfter: number, options: unknown) => {
          this.emit("error", {
            error: new Error(`Secondary rate limit hit`),
            context: "secondary-rate-limit",
          });
          return false;
        },
      },
    });
    this.limiter = pLimit(this.config.concurrency);
  }

  // ============================================================================
  // Repository Scanning
  // ============================================================================

  /**
   * Scan a single repository
   */
  async scanRepository(owner: string, repo: string): Promise<RepositoryInfo> {
    const cacheKey = `repo:${owner}/${repo}`;
    const cached = this.getFromCache<RepositoryInfo>(cacheKey);
    if (cached) return cached;

    try {
      const [repoData, languagesData] = await Promise.all([
        this.octokit.repos.get({ owner, repo }),
        this.octokit.repos.listLanguages({ owner, repo }),
      ]);

      const repository = this.mapRepositoryData(
        repoData.data,
        languagesData.data
      );
      this.setCache(cacheKey, repository);
      this.emit("repository:scanned", { repository });
      return repository;
    } catch (error) {
      this.emit("error", {
        error: error as Error,
        context: `scanRepository:${owner}/${repo}`,
      });
      throw error;
    }
  }

  /**
   * Scan multiple repositories
   */
  async scanRepositories(
    repos: Array<{ owner: string; repo: string }>
  ): Promise<RepositoryInfo[]> {
    const total = repos.length;
    let current = 0;

    const results = await Promise.all(
      repos.map((r) =>
        this.limiter(async () => {
          const result = await this.scanRepository(r.owner, r.repo);
          current++;
          this.emit("progress", {
            current,
            total,
            message: `Scanned ${r.owner}/${r.repo}`,
          });
          return result;
        })
      )
    );

    return results;
  }

  /**
   * Search repositories with filters
   */
  async searchRepositories(
    query: RepositorySearchQuery
  ): Promise<RepositorySearchResult> {
    const q = this.buildSearchQuery(query);
    const page = query.page ?? 1;
    const perPage = Math.min(query.perPage ?? 30, 100);

    try {
      const response = await this.octokit.search.repos({
        q,
        sort: query.sort ?? "stars",
        order: query.order ?? "desc",
        page,
        per_page: perPage,
      });

      const repositories = await Promise.all(
        response.data.items.map(async (item) => {
          const languages = await this.octokit.repos.listLanguages({
            owner: item.owner?.login ?? "",
            repo: item.name,
          });
          return this.mapRepositoryData(item as any, languages.data);
        })
      );

      return {
        totalCount: response.data.total_count,
        incompleteResults: response.data.incomplete_results,
        repositories,
        page,
        perPage,
        hasMore: page * perPage < response.data.total_count,
      };
    } catch (error) {
      this.emit("error", {
        error: error as Error,
        context: "searchRepositories",
      });
      throw error;
    }
  }

  /**
   * Get trending repositories
   */
  async getTrending(
    options: {
      language?: string;
      since?: "daily" | "weekly" | "monthly";
      limit?: number;
    } = {}
  ): Promise<RepositoryInfo[]> {
    const { language, since = "weekly", limit = 25 } = options;

    // Calculate date based on 'since'
    const date = new Date();
    switch (since) {
      case "daily":
        date.setDate(date.getDate() - 1);
        break;
      case "weekly":
        date.setDate(date.getDate() - 7);
        break;
      case "monthly":
        date.setMonth(date.getMonth() - 1);
        break;
    }

    const result = await this.searchRepositories({
      language,
      createdAfter: date,
      sort: "stars",
      order: "desc",
      perPage: limit,
    });

    return result.repositories;
  }

  // ============================================================================
  // Health Analysis
  // ============================================================================

  /**
   * Calculate repository health score
   */
  async calculateHealth(repository: RepositoryInfo): Promise<RepositoryHealth> {
    const factors: HealthFactor[] = [];

    // Activity Factor
    factors.push(this.calculateActivityFactor(repository));

    // Community Factor
    factors.push(await this.calculateCommunityFactor(repository));

    // Maintenance Factor
    factors.push(await this.calculateMaintenanceFactor(repository));

    // Documentation Factor
    factors.push(await this.calculateDocumentationFactor(repository));

    // Security Factor
    factors.push(await this.calculateSecurityFactor(repository));

    // Quality Factor
    factors.push(this.calculateQualityFactor(repository));

    // Calculate overall score
    const totalWeight = factors.reduce((sum, f) => sum + f.weight, 0);
    const weightedScore = factors.reduce(
      (sum, f) => sum + f.score * f.weight,
      0
    );
    const score = Math.round(weightedScore / totalWeight);

    const health: RepositoryHealth = {
      score,
      factors,
      calculatedAt: new Date(),
    };

    this.emit("repository:health", { repository, health });
    return health;
  }

  /**
   * Calculate activity factor
   */
  private calculateActivityFactor(repository: RepositoryInfo): HealthFactor {
    const daysSinceUpdate = Math.floor(
      (Date.now() - repository.updatedAt.getTime()) / (1000 * 60 * 60 * 24)
    );
    const daysSincePush = Math.floor(
      (Date.now() - repository.pushedAt.getTime()) / (1000 * 60 * 60 * 24)
    );

    let score = 100;

    // Penalize for inactivity
    if (daysSincePush > 365) score -= 50;
    else if (daysSincePush > 180) score -= 30;
    else if (daysSincePush > 90) score -= 15;
    else if (daysSincePush > 30) score -= 5;

    // Archived repos get low score
    if (repository.isArchived) score = Math.min(score, 20);

    return {
      name: "Activity",
      category: "activity",
      score: Math.max(0, score),
      weight: 25,
      description: `Last activity ${daysSincePush} days ago`,
      recommendation:
        daysSincePush > 90
          ? "Repository has been inactive for a while"
          : undefined,
    };
  }

  /**
   * Calculate community factor
   */
  private async calculateCommunityFactor(
    repository: RepositoryInfo
  ): Promise<HealthFactor> {
    let score = 0;

    // Stars contribute to score
    if (repository.stars > 10000) score += 30;
    else if (repository.stars > 1000) score += 25;
    else if (repository.stars > 100) score += 20;
    else if (repository.stars > 10) score += 10;

    // Forks indicate adoption
    if (repository.forks > 1000) score += 20;
    else if (repository.forks > 100) score += 15;
    else if (repository.forks > 10) score += 10;

    // Watchers show interest
    if (repository.watchers > 100) score += 15;
    else if (repository.watchers > 10) score += 10;
    else score += 5;

    // Issues enabled shows openness
    if (repository.hasIssues) score += 10;

    // Topics/tags help discoverability
    if (repository.topics.length >= 5) score += 15;
    else if (repository.topics.length >= 3) score += 10;
    else if (repository.topics.length >= 1) score += 5;

    return {
      name: "Community",
      category: "community",
      score: Math.min(100, score),
      weight: 20,
      description: `${repository.stars} stars, ${repository.forks} forks`,
      recommendation:
        repository.topics.length < 3
          ? "Add more topics to improve discoverability"
          : undefined,
    };
  }

  /**
   * Calculate maintenance factor
   */
  private async calculateMaintenanceFactor(
    repository: RepositoryInfo
  ): Promise<HealthFactor> {
    try {
      // Get recent commits
      const commits = await this.octokit.repos.listCommits({
        owner: repository.owner,
        repo: repository.name,
        per_page: 100,
      });

      // Get contributors
      const contributors = await this.octokit.repos.listContributors({
        owner: repository.owner,
        repo: repository.name,
        per_page: 100,
      });

      let score = 0;

      // Recent commits
      const recentCommits = commits.data.filter((c) => {
        const commitDate = new Date(c.commit.author?.date ?? 0);
        const daysSince =
          (Date.now() - commitDate.getTime()) / (1000 * 60 * 60 * 24);
        return daysSince <= 90;
      }).length;

      if (recentCommits > 50) score += 35;
      else if (recentCommits > 20) score += 30;
      else if (recentCommits > 10) score += 20;
      else if (recentCommits > 0) score += 10;

      // Number of contributors
      const contributorCount = contributors.data.length;
      if (contributorCount > 50) score += 35;
      else if (contributorCount > 20) score += 30;
      else if (contributorCount > 5) score += 20;
      else if (contributorCount > 1) score += 10;

      // Regular release cycle (check tags)
      const tags = await this.octokit.repos.listTags({
        owner: repository.owner,
        repo: repository.name,
        per_page: 10,
      });

      if (tags.data.length > 0) score += 30;

      return {
        name: "Maintenance",
        category: "maintenance",
        score: Math.min(100, score),
        weight: 25,
        description: `${recentCommits} commits in 90 days, ${contributorCount} contributors`,
        recommendation:
          contributorCount <= 1
            ? "Single maintainer is a risk factor"
            : undefined,
      };
    } catch (error) {
      return {
        name: "Maintenance",
        category: "maintenance",
        score: 50,
        weight: 25,
        description: "Unable to analyze maintenance",
      };
    }
  }

  /**
   * Calculate documentation factor
   */
  private async calculateDocumentationFactor(
    repository: RepositoryInfo
  ): Promise<HealthFactor> {
    let score = 0;

    try {
      // Check for README
      try {
        const readme = await this.octokit.repos.getReadme({
          owner: repository.owner,
          repo: repository.name,
        });
        const readmeSize = readme.data.size ?? 0;
        if (readmeSize > 5000) score += 30;
        else if (readmeSize > 1000) score += 20;
        else if (readmeSize > 0) score += 10;
      } catch {
        // No README
      }

      // Check for common documentation files
      const docFiles = [
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "CHANGELOG.md",
        "docs/",
        ".github/ISSUE_TEMPLATE",
        ".github/PULL_REQUEST_TEMPLATE.md",
      ];

      for (const file of docFiles) {
        try {
          await this.octokit.repos.getContent({
            owner: repository.owner,
            repo: repository.name,
            path: file,
          });
          score += 10;
        } catch {
          // File doesn't exist
        }
      }

      // Has description
      if (repository.description && repository.description.length > 20) {
        score += 10;
      }

      // Has wiki
      if (repository.hasWiki) score += 10;
    } catch {
      // Fallback
    }

    return {
      name: "Documentation",
      category: "documentation",
      score: Math.min(100, score),
      weight: 15,
      description:
        score > 50 ? "Well documented" : "Documentation could be improved",
      recommendation:
        score < 50
          ? "Add CONTRIBUTING.md, CHANGELOG.md, and improve README"
          : undefined,
    };
  }

  /**
   * Calculate security factor
   */
  private async calculateSecurityFactor(
    repository: RepositoryInfo
  ): Promise<HealthFactor> {
    let score = 50; // Base score

    try {
      // Check for security policy
      try {
        await this.octokit.repos.getContent({
          owner: repository.owner,
          repo: repository.name,
          path: "SECURITY.md",
        });
        score += 20;
      } catch {
        // No security policy
      }

      // Check for Dependabot config
      try {
        await this.octokit.repos.getContent({
          owner: repository.owner,
          repo: repository.name,
          path: ".github/dependabot.yml",
        });
        score += 15;
      } catch {
        // No Dependabot
      }

      // Check for GitHub Actions (CI)
      try {
        await this.octokit.repos.getContent({
          owner: repository.owner,
          repo: repository.name,
          path: ".github/workflows",
        });
        score += 15;
      } catch {
        // No GitHub Actions
      }

      // License provides legal clarity
      if (repository.license) score += 10;
    } catch {
      // Fallback
    }

    return {
      name: "Security",
      category: "security",
      score: Math.min(100, score),
      weight: 10,
      description:
        score > 70
          ? "Good security practices"
          : "Security practices could be improved",
      recommendation:
        score < 70 ? "Add SECURITY.md and enable Dependabot" : undefined,
    };
  }

  /**
   * Calculate quality factor
   */
  private calculateQualityFactor(repository: RepositoryInfo): HealthFactor {
    let score = 50; // Base score

    // Has primary language (not just docs)
    if (repository.language) score += 15;

    // Multiple languages might indicate comprehensive project
    const langCount = Object.keys(repository.languages).length;
    if (langCount > 3) score += 15;
    else if (langCount > 1) score += 10;

    // Not too many open issues relative to stars
    const issueRatio =
      repository.stars > 0
        ? repository.openIssues / repository.stars
        : repository.openIssues;
    if (issueRatio < 0.01) score += 20;
    else if (issueRatio < 0.05) score += 10;
    else if (issueRatio > 0.2) score -= 10;

    // Not a fork (original work)
    if (!repository.isFork) score += 10;

    return {
      name: "Quality",
      category: "quality",
      score: Math.min(100, Math.max(0, score)),
      weight: 5,
      description: `Issue ratio: ${(issueRatio * 100).toFixed(1)}%`,
    };
  }

  // ============================================================================
  // Dependency Detection
  // ============================================================================

  /**
   * Detect dependency files in repository
   */
  async detectDependencyFiles(
    owner: string,
    repo: string
  ): Promise<DependencyFileInfo[]> {
    const dependencyFiles: DependencyFileInfo[] = [];

    const filesToCheck = [
      // JavaScript/TypeScript
      { path: "package.json", ecosystem: "npm" as const },
      { path: "package-lock.json", ecosystem: "npm" as const },
      { path: "yarn.lock", ecosystem: "npm" as const },
      { path: "pnpm-lock.yaml", ecosystem: "npm" as const },
      // Python
      { path: "requirements.txt", ecosystem: "pypi" as const },
      { path: "Pipfile", ecosystem: "pypi" as const },
      { path: "pyproject.toml", ecosystem: "pypi" as const },
      { path: "setup.py", ecosystem: "pypi" as const },
      // Rust
      { path: "Cargo.toml", ecosystem: "cargo" as const },
      { path: "Cargo.lock", ecosystem: "cargo" as const },
      // Go
      { path: "go.mod", ecosystem: "go" as const },
      { path: "go.sum", ecosystem: "go" as const },
      // Ruby
      { path: "Gemfile", ecosystem: "rubygems" as const },
      { path: "Gemfile.lock", ecosystem: "rubygems" as const },
      // Java
      { path: "pom.xml", ecosystem: "maven" as const },
      { path: "build.gradle", ecosystem: "maven" as const },
      // .NET
      { path: "*.csproj", ecosystem: "nuget" as const },
      { path: "packages.config", ecosystem: "nuget" as const },
    ];

    await Promise.all(
      filesToCheck.map(async ({ path, ecosystem }) => {
        try {
          const response = await this.octokit.repos.getContent({
            owner,
            repo,
            path,
          });

          if (!Array.isArray(response.data) && response.data.type === "file") {
            dependencyFiles.push({
              path,
              ecosystem,
              size: response.data.size,
              sha: response.data.sha,
            });
          }
        } catch {
          // File doesn't exist
        }
      })
    );

    return dependencyFiles;
  }

  /**
   * Get dependency file content
   */
  async getDependencyFileContent(
    owner: string,
    repo: string,
    path: string
  ): Promise<string> {
    const response = await this.octokit.repos.getContent({
      owner,
      repo,
      path,
    });

    if (Array.isArray(response.data) || response.data.type !== "file") {
      throw new Error(`${path} is not a file`);
    }

    const content = Buffer.from(response.data.content, "base64").toString(
      "utf-8"
    );
    return content;
  }

  // ============================================================================
  // Helper Methods
  // ============================================================================

  /**
   * Build GitHub search query
   */
  private buildSearchQuery(query: RepositorySearchQuery): string {
    const parts: string[] = [];

    if (query.query) parts.push(query.query);
    if (query.language) parts.push(`language:${query.language}`);
    if (query.topic) parts.push(`topic:${query.topic}`);
    if (query.minStars) parts.push(`stars:>=${query.minStars}`);
    if (query.maxStars) parts.push(`stars:<=${query.maxStars}`);
    if (query.createdAfter) {
      parts.push(`created:>=${query.createdAfter.toISOString().split("T")[0]}`);
    }
    if (query.updatedAfter) {
      parts.push(`pushed:>=${query.updatedAfter.toISOString().split("T")[0]}`);
    }

    // Default filters from config
    if (!this.config.includeArchived) parts.push("archived:false");
    if (!this.config.includeForks) parts.push("fork:false");
    if (this.config.minStars > 0) parts.push(`stars:>=${this.config.minStars}`);

    return parts.join(" ");
  }

  /**
   * Map GitHub API response to RepositoryInfo
   */
  private mapRepositoryData(
    data: any,
    languages: Record<string, number>
  ): RepositoryInfo {
    return {
      id: String(data.id),
      name: data.name,
      fullName: data.full_name,
      owner: data.owner?.login ?? "",
      description: data.description,
      language: data.language,
      languages,
      stars: data.stargazers_count ?? 0,
      forks: data.forks_count ?? 0,
      watchers: data.watchers_count ?? 0,
      openIssues: data.open_issues_count ?? 0,
      topics: data.topics ?? [],
      license: data.license
        ? {
            key: data.license.key,
            name: data.license.name,
            spdxId: data.license.spdx_id,
            url: data.license.url,
          }
        : null,
      defaultBranch: data.default_branch ?? "main",
      createdAt: new Date(data.created_at),
      updatedAt: new Date(data.updated_at),
      pushedAt: new Date(data.pushed_at),
      url: data.html_url,
      cloneUrl: data.clone_url,
      isPrivate: data.private ?? false,
      isArchived: data.archived ?? false,
      isFork: data.fork ?? false,
      hasWiki: data.has_wiki ?? false,
      hasIssues: data.has_issues ?? true,
      size: data.size ?? 0,
    };
  }

  /**
   * Get from cache
   */
  private getFromCache<T>(key: string): T | null {
    const cached = this.cache.get(key);
    if (cached && cached.expiresAt > Date.now()) {
      return cached.data as T;
    }
    this.cache.delete(key);
    return null;
  }

  /**
   * Set cache
   */
  private setCache(key: string, data: unknown): void {
    this.cache.set(key, {
      data,
      expiresAt: Date.now() + this.config.cacheTtl * 1000,
    });
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Get rate limit status
   */
  async getRateLimit(): Promise<{
    limit: number;
    remaining: number;
    reset: Date;
  }> {
    const response = await this.octokit.rateLimit.get();
    return {
      limit: response.data.rate.limit,
      remaining: response.data.rate.remaining,
      reset: new Date(response.data.rate.reset * 1000),
    };
  }
}

// ==============================================================================
// Supporting Types
// ==============================================================================

interface DependencyFileInfo {
  path: string;
  ecosystem: "npm" | "pypi" | "cargo" | "go" | "rubygems" | "maven" | "nuget";
  size: number;
  sha: string;
}

// ==============================================================================
// Factory Function
// ==============================================================================

/**
 * Create a repository scanner instance
 */
export function createRepositoryScanner(
  config?: ScannerConfig
): RepositoryScanner {
  return new RepositoryScanner(config);
}
