/**
 * NEURECTOMY Discovery Engine - Core Types
 *
 * @packageDocumentation
 */

// ==============================================================================
// Repository Types
// ==============================================================================

/**
 * Repository metadata from scanning
 */
export interface RepositoryInfo {
  /** Unique identifier */
  id: string;
  /** Repository name */
  name: string;
  /** Full repository name (owner/repo) */
  fullName: string;
  /** Repository owner */
  owner: string;
  /** Repository description */
  description: string | null;
  /** Primary programming language */
  language: string | null;
  /** All languages used */
  languages: Record<string, number>;
  /** Star count */
  stars: number;
  /** Fork count */
  forks: number;
  /** Watcher count */
  watchers: number;
  /** Open issues count */
  openIssues: number;
  /** Repository topics/tags */
  topics: string[];
  /** License information */
  license: LicenseInfo | null;
  /** Default branch */
  defaultBranch: string;
  /** Creation date */
  createdAt: Date;
  /** Last update date */
  updatedAt: Date;
  /** Last push date */
  pushedAt: Date;
  /** Repository URL */
  url: string;
  /** Clone URL */
  cloneUrl: string;
  /** Is private repository */
  isPrivate: boolean;
  /** Is archived */
  isArchived: boolean;
  /** Is fork */
  isFork: boolean;
  /** Has wiki */
  hasWiki: boolean;
  /** Has issues enabled */
  hasIssues: boolean;
  /** Size in KB */
  size: number;
}

/**
 * License information
 */
export interface LicenseInfo {
  /** SPDX license key */
  key: string;
  /** License name */
  name: string;
  /** SPDX identifier */
  spdxId: string | null;
  /** License URL */
  url: string | null;
}

/**
 * Repository health metrics
 */
export interface RepositoryHealth {
  /** Overall health score (0-100) */
  score: number;
  /** Individual health factors */
  factors: HealthFactor[];
  /** Calculated at timestamp */
  calculatedAt: Date;
}

/**
 * Individual health factor
 */
export interface HealthFactor {
  /** Factor name */
  name: string;
  /** Factor category */
  category: HealthCategory;
  /** Score contribution (0-100) */
  score: number;
  /** Weight in overall calculation */
  weight: number;
  /** Description */
  description: string;
  /** Recommendation for improvement */
  recommendation?: string;
}

/**
 * Health factor categories
 */
export type HealthCategory =
  | "maintenance"
  | "community"
  | "documentation"
  | "security"
  | "activity"
  | "quality";

// ==============================================================================
// Dependency Types
// ==============================================================================

/**
 * Package dependency information
 */
export interface Dependency {
  /** Package name */
  name: string;
  /** Specified version constraint */
  version: string;
  /** Resolved version (if available) */
  resolvedVersion?: string;
  /** Dependency type */
  type: DependencyType;
  /** Package ecosystem */
  ecosystem: PackageEcosystem;
  /** Is optional dependency */
  optional: boolean;
  /** Package registry URL */
  registryUrl?: string;
  /** Package homepage */
  homepage?: string;
  /** Package repository */
  repository?: string;
}

/**
 * Dependency type
 */
export type DependencyType =
  | "production"
  | "development"
  | "peer"
  | "optional"
  | "bundled";

/**
 * Package ecosystem/registry
 */
export type PackageEcosystem =
  | "npm"
  | "pypi"
  | "cargo"
  | "maven"
  | "nuget"
  | "go"
  | "rubygems"
  | "packagist"
  | "hex"
  | "pub";

/**
 * Dependency tree node
 */
export interface DependencyNode {
  /** Dependency info */
  dependency: Dependency;
  /** Child dependencies */
  children: DependencyNode[];
  /** Depth in tree */
  depth: number;
  /** Is circular reference */
  isCircular: boolean;
  /** Path to this node */
  path: string[];
}

/**
 * Dependency analysis result
 */
export interface DependencyAnalysis {
  /** Total dependencies */
  total: number;
  /** Production dependencies */
  production: number;
  /** Development dependencies */
  development: number;
  /** Direct dependencies */
  direct: number;
  /** Transitive dependencies */
  transitive: number;
  /** Outdated dependencies */
  outdated: OutdatedDependency[];
  /** Dependencies with vulnerabilities */
  vulnerable: VulnerableDependency[];
  /** Deprecated dependencies */
  deprecated: DeprecatedDependency[];
  /** Duplicate dependencies (different versions) */
  duplicates: DuplicateDependency[];
  /** Dependency tree */
  tree: DependencyNode[];
  /** Analysis timestamp */
  analyzedAt: Date;
}

/**
 * Outdated dependency info
 */
export interface OutdatedDependency {
  /** Dependency info */
  dependency: Dependency;
  /** Current version */
  current: string;
  /** Latest version */
  latest: string;
  /** Latest stable version */
  latestStable: string;
  /** Breaking changes between versions */
  hasBreakingChanges: boolean;
  /** Upgrade urgency */
  urgency: UpgradeUrgency;
}

/**
 * Upgrade urgency level
 */
export type UpgradeUrgency = "low" | "medium" | "high" | "critical";

/**
 * Vulnerable dependency info
 */
export interface VulnerableDependency {
  /** Dependency info */
  dependency: Dependency;
  /** Vulnerabilities found */
  vulnerabilities: Vulnerability[];
  /** Highest severity */
  highestSeverity: VulnerabilitySeverity;
  /** Fix available */
  fixAvailable: boolean;
  /** Fixed in version */
  fixedIn?: string;
}

/**
 * Vulnerability information
 */
export interface Vulnerability {
  /** Vulnerability ID (CVE, GHSA, etc.) */
  id: string;
  /** Vulnerability source */
  source: VulnerabilitySource;
  /** Severity level */
  severity: VulnerabilitySeverity;
  /** CVSS score */
  cvssScore?: number;
  /** Title/summary */
  title: string;
  /** Description */
  description: string;
  /** Affected version range */
  affectedVersions: string;
  /** Fixed version */
  fixedVersion?: string;
  /** References/links */
  references: string[];
  /** Published date */
  publishedAt: Date;
  /** Last updated */
  updatedAt: Date;
}

/**
 * Vulnerability source
 */
export type VulnerabilitySource =
  | "nvd"
  | "github"
  | "snyk"
  | "npm"
  | "osv"
  | "other";

/**
 * Vulnerability severity
 */
export type VulnerabilitySeverity =
  | "critical"
  | "high"
  | "medium"
  | "low"
  | "unknown";

/**
 * Deprecated dependency info
 */
export interface DeprecatedDependency {
  /** Dependency info */
  dependency: Dependency;
  /** Deprecation reason */
  reason: string;
  /** Replacement package (if any) */
  replacement?: string;
  /** Deprecated since */
  deprecatedAt?: Date;
}

/**
 * Duplicate dependency info
 */
export interface DuplicateDependency {
  /** Package name */
  name: string;
  /** All versions found */
  versions: string[];
  /** Paths where each version is required */
  paths: Record<string, string[]>;
  /** Recommended resolution */
  recommendation: string;
}

// ==============================================================================
// Recommendation Types
// ==============================================================================

/**
 * Library recommendation
 */
export interface LibraryRecommendation {
  /** Recommended package */
  package: PackageInfo;
  /** Recommendation score (0-100) */
  score: number;
  /** Recommendation reason */
  reason: RecommendationReason;
  /** Detailed explanation */
  explanation: string;
  /** Comparison with current/alternative */
  comparison?: PackageComparison;
  /** Migration difficulty (if replacing) */
  migrationDifficulty?: MigrationDifficulty;
  /** Estimated migration effort in hours */
  estimatedEffort?: number;
}

/**
 * Package information for recommendations
 */
export interface PackageInfo {
  /** Package name */
  name: string;
  /** Current version */
  version: string;
  /** Description */
  description: string;
  /** Weekly downloads */
  weeklyDownloads: number;
  /** GitHub stars */
  stars: number;
  /** License */
  license: string;
  /** Last publish date */
  lastPublished: Date;
  /** Repository URL */
  repository?: string;
  /** Homepage */
  homepage?: string;
  /** Keywords/tags */
  keywords: string[];
  /** Maintainers */
  maintainers: string[];
  /** TypeScript support */
  typescript: TypeScriptSupport;
  /** Bundle size (bytes) */
  bundleSize?: number;
  /** Tree-shakeable */
  treeShakeable?: boolean;
}

/**
 * TypeScript support level
 */
export type TypeScriptSupport =
  | "native"
  | "definitelyTyped"
  | "bundled"
  | "none";

/**
 * Recommendation reason
 */
export type RecommendationReason =
  | "trending"
  | "popular"
  | "better-maintained"
  | "more-secure"
  | "smaller-bundle"
  | "better-performance"
  | "better-typescript"
  | "active-community"
  | "replacement"
  | "complementary";

/**
 * Package comparison
 */
export interface PackageComparison {
  /** Package being compared to */
  compareTo: string;
  /** Metric comparisons */
  metrics: ComparisonMetric[];
  /** Overall winner */
  winner: "recommended" | "current" | "tie";
}

/**
 * Comparison metric
 */
export interface ComparisonMetric {
  /** Metric name */
  name: string;
  /** Recommended package value */
  recommended: number | string;
  /** Current/comparison package value */
  current: number | string;
  /** Winner for this metric */
  winner: "recommended" | "current" | "tie";
  /** Percentage difference */
  difference?: number;
}

/**
 * Migration difficulty
 */
export type MigrationDifficulty =
  | "trivial"
  | "easy"
  | "moderate"
  | "hard"
  | "major";

// ==============================================================================
// Scanner Configuration
// ==============================================================================

/**
 * Scanner configuration
 */
export interface ScannerConfig {
  /** GitHub access token */
  githubToken?: string;
  /** Rate limit per minute */
  rateLimit?: number;
  /** Concurrent requests */
  concurrency?: number;
  /** Cache TTL in seconds */
  cacheTtl?: number;
  /** Include archived repositories */
  includeArchived?: boolean;
  /** Include forks */
  includeForks?: boolean;
  /** Minimum stars filter */
  minStars?: number;
  /** Languages to include */
  languages?: string[];
  /** Topics to include */
  topics?: string[];
}

/**
 * Analyzer configuration
 */
export interface AnalyzerConfig {
  /** Vulnerability database sources */
  vulnerabilitySources?: VulnerabilitySource[];
  /** Check for outdated packages */
  checkOutdated?: boolean;
  /** Check for deprecated packages */
  checkDeprecated?: boolean;
  /** Check for vulnerabilities */
  checkVulnerabilities?: boolean;
  /** Maximum tree depth to analyze */
  maxDepth?: number;
  /** Cache TTL in seconds */
  cacheTtl?: number;
  /** Severity threshold for alerts */
  severityThreshold?: VulnerabilitySeverity;
}

/**
 * Recommendation engine configuration
 */
export interface RecommendationConfig {
  /** Minimum popularity score */
  minPopularity?: number;
  /** Minimum maintenance score */
  minMaintenance?: number;
  /** Prefer TypeScript support */
  preferTypeScript?: boolean;
  /** Maximum bundle size (bytes) */
  maxBundleSize?: number;
  /** Categories to recommend for */
  categories?: string[];
  /** Number of recommendations per category */
  recommendationsPerCategory?: number;
}

// ==============================================================================
// Event Types
// ==============================================================================

/**
 * Discovery engine events
 */
export interface DiscoveryEvents {
  /** Repository scanned */
  "repository:scanned": { repository: RepositoryInfo };
  /** Repository health calculated */
  "repository:health": { repository: RepositoryInfo; health: RepositoryHealth };
  /** Dependencies analyzed */
  "dependencies:analyzed": { analysis: DependencyAnalysis };
  /** Vulnerability found */
  "vulnerability:found": { dependency: VulnerableDependency };
  /** Recommendation generated */
  "recommendation:generated": { recommendation: LibraryRecommendation };
  /** Error occurred */
  error: { error: Error; context: string };
  /** Progress update */
  progress: { current: number; total: number; message: string };
}

// ==============================================================================
// Search Types
// ==============================================================================

/**
 * Repository search query
 */
export interface RepositorySearchQuery {
  /** Search query string */
  query?: string;
  /** Language filter */
  language?: string;
  /** Topic filter */
  topic?: string;
  /** Minimum stars */
  minStars?: number;
  /** Maximum stars */
  maxStars?: number;
  /** Created after date */
  createdAfter?: Date;
  /** Updated after date */
  updatedAfter?: Date;
  /** Sort field */
  sort?: "stars" | "forks" | "updated" | "created" | "best-match";
  /** Sort order */
  order?: "asc" | "desc";
  /** Page number */
  page?: number;
  /** Results per page */
  perPage?: number;
}

/**
 * Repository search result
 */
export interface RepositorySearchResult {
  /** Total count */
  totalCount: number;
  /** Incomplete results */
  incompleteResults: boolean;
  /** Repositories found */
  repositories: RepositoryInfo[];
  /** Current page */
  page: number;
  /** Results per page */
  perPage: number;
  /** Has more pages */
  hasMore: boolean;
}

/**
 * Package search query
 */
export interface PackageSearchQuery {
  /** Search query */
  query: string;
  /** Package ecosystem */
  ecosystem?: PackageEcosystem;
  /** Keywords filter */
  keywords?: string[];
  /** Sort field */
  sort?: "popularity" | "quality" | "maintenance" | "relevance";
  /** Page number */
  page?: number;
  /** Results per page */
  perPage?: number;
}

/**
 * Package search result
 */
export interface PackageSearchResult {
  /** Total count */
  totalCount: number;
  /** Packages found */
  packages: PackageInfo[];
  /** Current page */
  page: number;
  /** Results per page */
  perPage: number;
  /** Has more pages */
  hasMore: boolean;
}

// ==============================================================================
// Technology Radar Types
// ==============================================================================

/**
 * Technology radar entry
 */
export interface RadarEntry {
  /** Technology name */
  name: string;
  /** Quadrant */
  quadrant: RadarQuadrant;
  /** Ring (adoption level) */
  ring: RadarRing;
  /** Movement from previous */
  movement: RadarMovement;
  /** Description */
  description: string;
  /** Use cases */
  useCases: string[];
  /** Related technologies */
  related: string[];
  /** Score (0-100) */
  score: number;
}

/**
 * Radar quadrant
 */
export type RadarQuadrant =
  | "languages-frameworks"
  | "tools"
  | "platforms"
  | "techniques";

/**
 * Radar ring (adoption level)
 */
export type RadarRing = "adopt" | "trial" | "assess" | "hold";

/**
 * Movement direction
 */
export type RadarMovement = "up" | "down" | "stable" | "new";

/**
 * Technology radar snapshot
 */
export interface TechnologyRadar {
  /** Radar entries */
  entries: RadarEntry[];
  /** Generated timestamp */
  generatedAt: Date;
  /** Data sources used */
  sources: string[];
  /** Version */
  version: string;
}
