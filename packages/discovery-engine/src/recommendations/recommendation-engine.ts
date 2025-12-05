/**
 * NEURECTOMY Discovery Engine - Recommendation Engine
 *
 * @ORACLE @PRISM - ML-powered library recommendations with trending analysis
 *
 * Features:
 * - Trending library detection
 * - Community health scoring
 * - Alternative library suggestions
 * - Technology radar generation
 * - Migration effort estimation
 *
 * @packageDocumentation
 */

import { EventEmitter } from "eventemitter3";
import pLimit from "p-limit";

import type {
  LibraryRecommendation,
  PackageInfo,
  PackageComparison,
  ComparisonMetric,
  RecommendationReason,
  MigrationDifficulty,
  TypeScriptSupport,
  RecommendationConfig,
  PackageSearchQuery,
  PackageSearchResult,
  TechnologyRadar,
  RadarEntry,
  RadarQuadrant,
  RadarRing,
  RadarMovement,
  DiscoveryEvents,
} from "../types";

// ==============================================================================
// Default Configuration
// ==============================================================================

const DEFAULT_CONFIG: Required<RecommendationConfig> = {
  minPopularity: 100,
  minMaintenance: 50,
  preferTypeScript: true,
  maxBundleSize: 500000, // 500KB
  categories: [],
  recommendationsPerCategory: 5,
};

// ==============================================================================
// Package Categories
// ==============================================================================

const PACKAGE_CATEGORIES: Record<string, string[]> = {
  "state-management": [
    "redux",
    "zustand",
    "jotai",
    "recoil",
    "mobx",
    "valtio",
    "xstate",
  ],
  "http-client": ["axios", "ky", "got", "node-fetch", "undici", "ofetch"],
  testing: ["jest", "vitest", "mocha", "ava", "tap", "uvu"],
  validation: ["zod", "yup", "joi", "ajv", "superstruct", "valibot"],
  orm: ["prisma", "typeorm", "sequelize", "drizzle-orm", "mikro-orm", "knex"],
  "ui-components": [
    "@mui/material",
    "@chakra-ui/react",
    "antd",
    "@mantine/core",
    "radix-ui",
  ],
  animation: [
    "framer-motion",
    "react-spring",
    "@react-spring/web",
    "animejs",
    "gsap",
  ],
  "date-time": ["date-fns", "dayjs", "luxon", "moment", "temporal-polyfill"],
  "css-in-js": [
    "styled-components",
    "@emotion/react",
    "linaria",
    "vanilla-extract",
    "stitches",
  ],
  bundler: ["vite", "esbuild", "webpack", "rollup", "parcel", "turbopack"],
  forms: ["react-hook-form", "formik", "@tanstack/react-form", "final-form"],
  routing: [
    "react-router-dom",
    "@tanstack/react-router",
    "wouter",
    "next/router",
  ],
};

// ==============================================================================
// Recommendation Engine
// ==============================================================================

/**
 * ML-powered library recommendation engine
 *
 * @example
 * ```typescript
 * const engine = new RecommendationEngine();
 *
 * // Get recommendations for a category
 * const stateManagement = await engine.getRecommendations("state-management");
 *
 * // Find alternatives to a package
 * const alternatives = await engine.findAlternatives("moment");
 *
 * // Generate technology radar
 * const radar = await engine.generateTechnologyRadar();
 * ```
 */
export class RecommendationEngine extends EventEmitter<DiscoveryEvents> {
  private config: Required<RecommendationConfig>;
  private limiter: ReturnType<typeof pLimit>;
  private cache: Map<string, { data: unknown; expiresAt: number }> = new Map();
  private packageDataCache: Map<string, PackageInfo> = new Map();

  constructor(config: RecommendationConfig = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.limiter = pLimit(5);
  }

  // ============================================================================
  // Recommendations
  // ============================================================================

  /**
   * Get library recommendations for a category
   */
  async getRecommendations(category: string): Promise<LibraryRecommendation[]> {
    const packages = PACKAGE_CATEGORIES[category] || [];
    if (packages.length === 0) {
      return [];
    }

    const packageInfos = await this.fetchPackageInfos(packages);
    const scored = await this.scorePackages(packageInfos);

    // Sort by score and take top N
    const sorted = scored.sort((a, b) => b.score - a.score);
    const topN = sorted.slice(0, this.config.recommendationsPerCategory);

    const recommendations: LibraryRecommendation[] = [];

    for (const { packageInfo, score } of topN) {
      const reason = this.determineRecommendationReason(packageInfo, score);

      recommendations.push({
        package: packageInfo,
        score,
        reason,
        explanation: this.generateExplanation(packageInfo, reason, score),
      });

      this.emit("recommendation:generated", {
        recommendation: recommendations[recommendations.length - 1],
      });
    }

    return recommendations;
  }

  /**
   * Find alternatives to a specific package
   */
  async findAlternatives(
    packageName: string,
    limit: number = 5
  ): Promise<LibraryRecommendation[]> {
    // Find category for this package
    let category: string | null = null;
    for (const [cat, packages] of Object.entries(PACKAGE_CATEGORIES)) {
      if (packages.includes(packageName)) {
        category = cat;
        break;
      }
    }

    if (!category) {
      // Try to find similar packages via npm search
      return this.searchSimilarPackages(packageName, limit);
    }

    // Get current package info for comparison
    const currentInfo = await this.fetchPackageInfo(packageName);
    if (!currentInfo) {
      return [];
    }

    // Get recommendations excluding the current package
    const packages = PACKAGE_CATEGORIES[category].filter(
      (p) => p !== packageName
    );
    const packageInfos = await this.fetchPackageInfos(packages);
    const scored = await this.scorePackages(packageInfos);

    const sorted = scored.sort((a, b) => b.score - a.score).slice(0, limit);

    const alternatives: LibraryRecommendation[] = [];

    for (const { packageInfo, score } of sorted) {
      const comparison = this.comparePackages(packageInfo, currentInfo);
      const migrationDifficulty = this.estimateMigrationDifficulty(
        currentInfo,
        packageInfo
      );

      alternatives.push({
        package: packageInfo,
        score,
        reason:
          comparison.winner === "recommended" ? "better-maintained" : "popular",
        explanation: this.generateAlternativeExplanation(
          packageInfo,
          currentInfo,
          comparison
        ),
        comparison,
        migrationDifficulty,
        estimatedEffort: this.estimateMigrationEffort(migrationDifficulty),
      });
    }

    return alternatives;
  }

  /**
   * Search for similar packages via npm
   */
  private async searchSimilarPackages(
    packageName: string,
    limit: number
  ): Promise<LibraryRecommendation[]> {
    const currentInfo = await this.fetchPackageInfo(packageName);
    if (!currentInfo) return [];

    // Use keywords to find similar packages
    const keywords = currentInfo.keywords.slice(0, 3);
    const searchResults = await this.searchNpm({
      query: keywords.join(" "),
      perPage: limit * 2,
    });

    const alternatives: LibraryRecommendation[] = [];

    for (const pkg of searchResults.packages) {
      if (pkg.name === packageName) continue;
      if (alternatives.length >= limit) break;

      const score = this.calculatePackageScore(pkg);
      if (score < this.config.minPopularity) continue;

      alternatives.push({
        package: pkg,
        score,
        reason: "complementary",
        explanation: `Similar to ${packageName} based on shared keywords`,
      });
    }

    return alternatives;
  }

  /**
   * Get trending packages
   */
  async getTrending(
    ecosystem: "npm" | "pypi" | "cargo" = "npm",
    limit: number = 20
  ): Promise<LibraryRecommendation[]> {
    // For npm, use npms.io API for quality scores
    if (ecosystem === "npm") {
      const trending = await this.fetchNpmTrending(limit);
      return trending.map((pkg) => ({
        package: pkg,
        score: this.calculatePackageScore(pkg),
        reason: "trending" as RecommendationReason,
        explanation: `Trending package with ${pkg.weeklyDownloads.toLocaleString()} weekly downloads`,
      }));
    }

    return [];
  }

  // ============================================================================
  // Technology Radar
  // ============================================================================

  /**
   * Generate a technology radar
   */
  async generateTechnologyRadar(): Promise<TechnologyRadar> {
    const entries: RadarEntry[] = [];

    // Analyze packages in each category
    for (const [category, packages] of Object.entries(PACKAGE_CATEGORIES)) {
      const quadrant = this.categoryToQuadrant(category);
      const packageInfos = await this.fetchPackageInfos(packages);

      for (const info of packageInfos) {
        const score = this.calculatePackageScore(info);
        const ring = this.scoreToRing(score);
        const movement = await this.calculateMovement(info);

        entries.push({
          name: info.name,
          quadrant,
          ring,
          movement,
          description: info.description,
          useCases: this.generateUseCases(info, category),
          related: packages.filter((p) => p !== info.name).slice(0, 3),
          score,
        });
      }
    }

    return {
      entries,
      generatedAt: new Date(),
      sources: ["npm", "github"],
      version: "1.0",
    };
  }

  /**
   * Map category to radar quadrant
   */
  private categoryToQuadrant(category: string): RadarQuadrant {
    const mapping: Record<string, RadarQuadrant> = {
      "state-management": "languages-frameworks",
      "http-client": "languages-frameworks",
      testing: "tools",
      validation: "languages-frameworks",
      orm: "platforms",
      "ui-components": "languages-frameworks",
      animation: "techniques",
      "date-time": "languages-frameworks",
      "css-in-js": "techniques",
      bundler: "tools",
      forms: "languages-frameworks",
      routing: "languages-frameworks",
    };

    return mapping[category] || "tools";
  }

  /**
   * Convert score to radar ring
   */
  private scoreToRing(score: number): RadarRing {
    if (score >= 80) return "adopt";
    if (score >= 60) return "trial";
    if (score >= 40) return "assess";
    return "hold";
  }

  /**
   * Calculate movement trend
   */
  private async calculateMovement(info: PackageInfo): Promise<RadarMovement> {
    // In a real implementation, we'd compare with historical data
    // For now, use heuristics based on last publish date
    const daysSincePublish = Math.floor(
      (Date.now() - info.lastPublished.getTime()) / (1000 * 60 * 60 * 24)
    );

    if (daysSincePublish < 30) return "up";
    if (daysSincePublish > 365) return "down";
    return "stable";
  }

  /**
   * Generate use cases for a package
   */
  private generateUseCases(info: PackageInfo, category: string): string[] {
    const useCases: string[] = [];

    // Add general use case based on category
    const categoryUseCases: Record<string, string[]> = {
      "state-management": ["Application state management", "Complex UI state"],
      "http-client": ["REST API calls", "Data fetching"],
      testing: ["Unit testing", "Integration testing", "E2E testing"],
      validation: ["Form validation", "API input validation", "Type safety"],
      orm: ["Database queries", "Data modeling", "Migrations"],
    };

    if (categoryUseCases[category]) {
      useCases.push(...categoryUseCases[category]);
    }

    // Add based on keywords
    if (info.keywords.includes("typescript")) {
      useCases.push("TypeScript projects");
    }
    if (info.keywords.includes("react")) {
      useCases.push("React applications");
    }

    return useCases.slice(0, 4);
  }

  // ============================================================================
  // Package Scoring
  // ============================================================================

  /**
   * Score packages for recommendations
   */
  private async scorePackages(
    packages: PackageInfo[]
  ): Promise<Array<{ packageInfo: PackageInfo; score: number }>> {
    return packages.map((packageInfo) => ({
      packageInfo,
      score: this.calculatePackageScore(packageInfo),
    }));
  }

  /**
   * Calculate comprehensive package score
   */
  private calculatePackageScore(info: PackageInfo): number {
    let score = 0;

    // Popularity (30 points max)
    if (info.weeklyDownloads > 10000000) score += 30;
    else if (info.weeklyDownloads > 1000000) score += 25;
    else if (info.weeklyDownloads > 100000) score += 20;
    else if (info.weeklyDownloads > 10000) score += 15;
    else if (info.weeklyDownloads > 1000) score += 10;
    else score += 5;

    // GitHub Stars (20 points max)
    if (info.stars > 50000) score += 20;
    else if (info.stars > 10000) score += 17;
    else if (info.stars > 5000) score += 14;
    else if (info.stars > 1000) score += 10;
    else if (info.stars > 100) score += 5;

    // Maintenance (20 points max)
    const daysSincePublish = Math.floor(
      (Date.now() - info.lastPublished.getTime()) / (1000 * 60 * 60 * 24)
    );
    if (daysSincePublish < 30) score += 20;
    else if (daysSincePublish < 90) score += 15;
    else if (daysSincePublish < 180) score += 10;
    else if (daysSincePublish < 365) score += 5;

    // TypeScript Support (15 points max)
    if (this.config.preferTypeScript) {
      switch (info.typescript) {
        case "native":
          score += 15;
          break;
        case "bundled":
          score += 12;
          break;
        case "definitelyTyped":
          score += 8;
          break;
        case "none":
          score += 0;
          break;
      }
    } else {
      score += 15; // Give full points if TS not preferred
    }

    // Bundle Size (10 points max)
    if (info.bundleSize) {
      if (info.bundleSize < 10000) score += 10;
      else if (info.bundleSize < 50000) score += 8;
      else if (info.bundleSize < 100000) score += 5;
      else if (info.bundleSize < this.config.maxBundleSize) score += 2;
    } else {
      score += 5; // Unknown size gets middle score
    }

    // Tree-shakeable (5 points)
    if (info.treeShakeable) score += 5;

    return Math.min(100, score);
  }

  /**
   * Determine recommendation reason
   */
  private determineRecommendationReason(
    info: PackageInfo,
    score: number
  ): RecommendationReason {
    const daysSincePublish = Math.floor(
      (Date.now() - info.lastPublished.getTime()) / (1000 * 60 * 60 * 24)
    );

    if (daysSincePublish < 14 && score > 70) return "trending";
    if (info.weeklyDownloads > 1000000) return "popular";
    if (daysSincePublish < 30) return "better-maintained";
    if (info.typescript === "native") return "better-typescript";
    if (info.bundleSize && info.bundleSize < 20000) return "smaller-bundle";
    if (info.stars > 5000) return "active-community";

    return "popular";
  }

  /**
   * Generate recommendation explanation
   */
  private generateExplanation(
    info: PackageInfo,
    reason: RecommendationReason,
    score: number
  ): string {
    const parts: string[] = [];

    parts.push(`Score: ${score}/100`);

    switch (reason) {
      case "trending":
        parts.push("Recently gaining popularity");
        break;
      case "popular":
        parts.push(`${info.weeklyDownloads.toLocaleString()} weekly downloads`);
        break;
      case "better-maintained":
        parts.push("Actively maintained with recent updates");
        break;
      case "better-typescript":
        parts.push("Excellent TypeScript support");
        break;
      case "smaller-bundle":
        parts.push(
          `Small bundle size: ${(info.bundleSize! / 1000).toFixed(1)}KB`
        );
        break;
      case "active-community":
        parts.push(`${info.stars.toLocaleString()} GitHub stars`);
        break;
    }

    return parts.join(". ");
  }

  /**
   * Generate alternative explanation
   */
  private generateAlternativeExplanation(
    alternative: PackageInfo,
    current: PackageInfo,
    comparison: PackageComparison
  ): string {
    const wins = comparison.metrics.filter((m) => m.winner === "recommended");
    const advantages = wins.map((m) => m.name.toLowerCase()).join(", ");

    if (advantages) {
      return `Better than ${current.name} in: ${advantages}`;
    }

    return `Alternative to ${current.name} with different trade-offs`;
  }

  // ============================================================================
  // Package Comparison
  // ============================================================================

  /**
   * Compare two packages
   */
  private comparePackages(
    recommended: PackageInfo,
    current: PackageInfo
  ): PackageComparison {
    const metrics: ComparisonMetric[] = [];

    // Downloads
    metrics.push({
      name: "Weekly Downloads",
      recommended: recommended.weeklyDownloads,
      current: current.weeklyDownloads,
      winner: this.compareValues(
        recommended.weeklyDownloads,
        current.weeklyDownloads,
        "higher"
      ),
      difference: this.calculateDifference(
        recommended.weeklyDownloads,
        current.weeklyDownloads
      ),
    });

    // Stars
    metrics.push({
      name: "GitHub Stars",
      recommended: recommended.stars,
      current: current.stars,
      winner: this.compareValues(recommended.stars, current.stars, "higher"),
      difference: this.calculateDifference(recommended.stars, current.stars),
    });

    // Bundle Size
    if (recommended.bundleSize && current.bundleSize) {
      metrics.push({
        name: "Bundle Size",
        recommended: `${(recommended.bundleSize / 1000).toFixed(1)}KB`,
        current: `${(current.bundleSize / 1000).toFixed(1)}KB`,
        winner: this.compareValues(
          recommended.bundleSize,
          current.bundleSize,
          "lower"
        ),
        difference: this.calculateDifference(
          current.bundleSize,
          recommended.bundleSize
        ),
      });
    }

    // TypeScript Support
    const tsRanking = { native: 4, bundled: 3, definitelyTyped: 2, none: 1 };
    metrics.push({
      name: "TypeScript Support",
      recommended: recommended.typescript,
      current: current.typescript,
      winner: this.compareValues(
        tsRanking[recommended.typescript],
        tsRanking[current.typescript],
        "higher"
      ),
    });

    // Last Published
    metrics.push({
      name: "Last Update",
      recommended: recommended.lastPublished.toLocaleDateString(),
      current: current.lastPublished.toLocaleDateString(),
      winner: this.compareValues(
        recommended.lastPublished.getTime(),
        current.lastPublished.getTime(),
        "higher"
      ),
    });

    // Determine overall winner
    const recWins = metrics.filter((m) => m.winner === "recommended").length;
    const curWins = metrics.filter((m) => m.winner === "current").length;

    return {
      compareTo: current.name,
      metrics,
      winner:
        recWins > curWins
          ? "recommended"
          : recWins < curWins
            ? "current"
            : "tie",
    };
  }

  /**
   * Compare values
   */
  private compareValues(
    rec: number,
    cur: number,
    preference: "higher" | "lower"
  ): "recommended" | "current" | "tie" {
    if (rec === cur) return "tie";
    if (preference === "higher") {
      return rec > cur ? "recommended" : "current";
    }
    return rec < cur ? "recommended" : "current";
  }

  /**
   * Calculate percentage difference
   */
  private calculateDifference(a: number, b: number): number {
    if (b === 0) return 100;
    return Math.round(((a - b) / b) * 100);
  }

  // ============================================================================
  // Migration Estimation
  // ============================================================================

  /**
   * Estimate migration difficulty
   */
  private estimateMigrationDifficulty(
    from: PackageInfo,
    to: PackageInfo
  ): MigrationDifficulty {
    // Simple heuristic based on package characteristics
    const fromKeywords = new Set(from.keywords);
    const toKeywords = new Set(to.keywords);
    const commonKeywords = [...fromKeywords].filter((k) => toKeywords.has(k));

    const similarity = commonKeywords.length / Math.max(fromKeywords.size, 1);

    if (similarity > 0.5) return "easy";
    if (similarity > 0.3) return "moderate";
    if (similarity > 0.1) return "hard";
    return "major";
  }

  /**
   * Estimate migration effort in hours
   */
  private estimateMigrationEffort(difficulty: MigrationDifficulty): number {
    const effortMap: Record<MigrationDifficulty, number> = {
      trivial: 1,
      easy: 4,
      moderate: 16,
      hard: 40,
      major: 80,
    };

    return effortMap[difficulty];
  }

  // ============================================================================
  // NPM API Integration
  // ============================================================================

  /**
   * Fetch multiple package infos
   */
  private async fetchPackageInfos(names: string[]): Promise<PackageInfo[]> {
    const results = await Promise.all(
      names.map((name) =>
        this.limiter(async () => {
          const info = await this.fetchPackageInfo(name);
          return info;
        })
      )
    );

    return results.filter((r): r is PackageInfo => r !== null);
  }

  /**
   * Fetch single package info from npm
   */
  private async fetchPackageInfo(name: string): Promise<PackageInfo | null> {
    if (this.packageDataCache.has(name)) {
      return this.packageDataCache.get(name)!;
    }

    try {
      const [registryData, downloadsData] = await Promise.all([
        fetch(`https://registry.npmjs.org/${name}`).then((r) =>
          r.ok ? r.json() : null
        ),
        fetch(`https://api.npmjs.org/downloads/point/last-week/${name}`).then(
          (r) => (r.ok ? r.json() : null)
        ),
      ]);

      if (!registryData) return null;

      const latestVersion = registryData["dist-tags"]?.latest;
      const versionData = registryData.versions?.[latestVersion] || {};

      const info: PackageInfo = {
        name,
        version: latestVersion,
        description: registryData.description || "",
        weeklyDownloads: downloadsData?.downloads || 0,
        stars: 0, // Would need GitHub API for this
        license: registryData.license || "UNKNOWN",
        lastPublished: new Date(
          registryData.time?.[latestVersion] || Date.now()
        ),
        repository: registryData.repository?.url,
        homepage: registryData.homepage,
        keywords: registryData.keywords || [],
        maintainers: (registryData.maintainers || []).map((m: any) => m.name),
        typescript: this.detectTypeScriptSupport(versionData),
        bundleSize: undefined, // Would need bundlephobia API
        treeShakeable: versionData.module !== undefined,
      };

      this.packageDataCache.set(name, info);
      return info;
    } catch (error) {
      return null;
    }
  }

  /**
   * Detect TypeScript support level
   */
  private detectTypeScriptSupport(versionData: any): TypeScriptSupport {
    if (versionData.types || versionData.typings) {
      return "bundled";
    }

    // Check if it's a TS-first package
    if (
      versionData.devDependencies?.typescript ||
      versionData.peerDependencies?.typescript
    ) {
      return "native";
    }

    // Would need to check DefinitelyTyped
    return "none";
  }

  /**
   * Search npm packages
   */
  private async searchNpm(
    query: PackageSearchQuery
  ): Promise<PackageSearchResult> {
    const url = new URL("https://registry.npmjs.org/-/v1/search");
    url.searchParams.set("text", query.query);
    url.searchParams.set("size", String(query.perPage || 20));

    try {
      const response = await fetch(url.toString());
      if (!response.ok) throw new Error("Search failed");

      const data = await response.json();

      const packages: PackageInfo[] = data.objects.map((obj: any) => ({
        name: obj.package.name,
        version: obj.package.version,
        description: obj.package.description || "",
        weeklyDownloads: obj.downloads?.weekly || 0,
        stars: 0,
        license: obj.package.license || "UNKNOWN",
        lastPublished: new Date(obj.package.date),
        repository: obj.package.links?.repository,
        homepage: obj.package.links?.homepage,
        keywords: obj.package.keywords || [],
        maintainers: obj.package.maintainers?.map((m: any) => m.username) || [],
        typescript: "none" as TypeScriptSupport,
      }));

      return {
        totalCount: data.total,
        packages,
        page: 1,
        perPage: query.perPage || 20,
        hasMore: packages.length < data.total,
      };
    } catch (error) {
      return {
        totalCount: 0,
        packages: [],
        page: 1,
        perPage: query.perPage || 20,
        hasMore: false,
      };
    }
  }

  /**
   * Fetch trending packages
   */
  private async fetchNpmTrending(limit: number): Promise<PackageInfo[]> {
    // Use npms.io for quality-based recommendations
    try {
      const response = await fetch(
        `https://api.npms.io/v2/search?q=not:deprecated&size=${limit}`
      );

      if (!response.ok) throw new Error("Failed to fetch trending");

      const data = await response.json();

      return data.results.map((result: any) => ({
        name: result.package.name,
        version: result.package.version,
        description: result.package.description || "",
        weeklyDownloads: result.package.downloads?.weekly || 0,
        stars: result.package.links?.repository ? 0 : 0,
        license: result.package.license || "UNKNOWN",
        lastPublished: new Date(result.package.date),
        repository: result.package.links?.repository,
        homepage: result.package.links?.homepage,
        keywords: result.package.keywords || [],
        maintainers: result.package.publisher?.username
          ? [result.package.publisher.username]
          : [],
        typescript: "none" as TypeScriptSupport,
      }));
    } catch (error) {
      return [];
    }
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.cache.clear();
    this.packageDataCache.clear();
  }
}

// ==============================================================================
// Factory Function
// ==============================================================================

/**
 * Create a recommendation engine instance
 */
export function createRecommendationEngine(
  config?: RecommendationConfig
): RecommendationEngine {
  return new RecommendationEngine(config);
}
