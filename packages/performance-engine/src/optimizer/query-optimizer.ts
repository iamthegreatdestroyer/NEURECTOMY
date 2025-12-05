/**
 * @neurectomy/performance-engine - Query Optimizer
 *
 * @elite-agent-collective @VELOCITY @VERTEX
 *
 * Intelligent query optimization for SQL, GraphQL, and NoSQL queries
 * with index recommendations and execution plan analysis.
 */

import * as crypto from "crypto";
import type { QueryOptimization } from "../types.js";

// ============================================================================
// TYPES
// ============================================================================

type QueryType = "sql" | "graphql" | "mongodb" | "elasticsearch" | "custom";

interface QueryAnalysis {
  originalQuery: string;
  type: QueryType;
  complexity: number;
  issues: QueryIssue[];
  tableAccess: TableAccess[];
  joins: JoinInfo[];
  aggregations: string[];
  subqueries: number;
  parameters: string[];
}

interface QueryIssue {
  type: string;
  severity: "low" | "medium" | "high" | "critical";
  description: string;
  location?: { start: number; end: number };
  suggestion: string;
}

interface TableAccess {
  table: string;
  alias?: string;
  accessType: "full_scan" | "index_scan" | "index_seek" | "unique_scan";
  columns: string[];
  predicates: string[];
}

interface JoinInfo {
  type: "inner" | "left" | "right" | "full" | "cross";
  leftTable: string;
  rightTable: string;
  condition: string;
  estimated: boolean;
}

interface IndexRecommendation {
  table: string;
  columns: string[];
  type: "btree" | "hash" | "gin" | "gist" | "brin";
  reason: string;
  estimatedImpact: number;
}

// ============================================================================
// QUERY OPTIMIZER IMPLEMENTATION
// ============================================================================

/**
 * Multi-database query optimizer with intelligent recommendations
 *
 * @elite-agent-collective @VELOCITY - Performance optimization patterns
 * @elite-agent-collective @VERTEX - Graph and relational query expertise
 */
export class QueryOptimizer {
  private queryCache: Map<string, QueryOptimization> = new Map();
  private indexStats: Map<string, { usage: number; lastUsed: number }> =
    new Map();

  /**
   * Optimize a query and return recommendations
   */
  async optimizeQuery(
    query: string,
    type: QueryType
  ): Promise<QueryOptimization> {
    // Check cache
    const cacheKey = this.generateCacheKey(query, type);
    const cached = this.queryCache.get(cacheKey);
    if (cached) return cached;

    // Analyze query
    const analysis = this.analyzeQuery(query, type);

    // Generate optimizations based on type
    const optimization = await this.generateOptimization(query, type, analysis);

    // Cache result
    this.queryCache.set(cacheKey, optimization);
    if (this.queryCache.size > 1000) {
      // Evict oldest entries
      const firstKey = this.queryCache.keys().next().value;
      if (firstKey) this.queryCache.delete(firstKey);
    }

    return optimization;
  }

  /**
   * Analyze query structure and patterns
   */
  private analyzeQuery(query: string, type: QueryType): QueryAnalysis {
    switch (type) {
      case "sql":
        return this.analyzeSQLQuery(query);
      case "graphql":
        return this.analyzeGraphQLQuery(query);
      case "mongodb":
        return this.analyzeMongoDBQuery(query);
      case "elasticsearch":
        return this.analyzeElasticsearchQuery(query);
      default:
        return this.analyzeGenericQuery(query, type);
    }
  }

  /**
   * Analyze SQL query
   */
  private analyzeSQLQuery(query: string): QueryAnalysis {
    const normalized = query.toLowerCase().trim();
    const issues: QueryIssue[] = [];
    const tableAccess: TableAccess[] = [];
    const joins: JoinInfo[] = [];
    const aggregations: string[] = [];

    // Detect SELECT *
    if (/select\s+\*\s+from/i.test(query)) {
      issues.push({
        type: "select_star",
        severity: "medium",
        description:
          "SELECT * retrieves all columns, potentially transferring unnecessary data",
        suggestion: "Specify only the columns you need",
      });
    }

    // Detect missing WHERE clause in UPDATE/DELETE
    if (/^(update|delete)/i.test(normalized) && !/where/i.test(normalized)) {
      issues.push({
        type: "missing_where",
        severity: "critical",
        description: "UPDATE/DELETE without WHERE clause affects all rows",
        suggestion: "Add a WHERE clause to limit affected rows",
      });
    }

    // Detect LIKE with leading wildcard
    if (/%[^']*'/i.test(query) && /'%/i.test(query)) {
      const likeMatch = query.match(/like\s+'%/i);
      if (likeMatch) {
        issues.push({
          type: "leading_wildcard",
          severity: "high",
          description: "LIKE with leading wildcard prevents index usage",
          suggestion: "Use full-text search or restructure the query",
        });
      }
    }

    // Detect functions on indexed columns
    const functionOnColumn =
      /where\s+(?:lower|upper|trim|substring|cast)\s*\([^)]+\)/i;
    if (functionOnColumn.test(query)) {
      issues.push({
        type: "function_on_column",
        severity: "high",
        description: "Functions on columns in WHERE clause prevent index usage",
        suggestion: "Use computed columns or expression indexes",
      });
    }

    // Detect OR conditions
    if (/where\s+[^(]*\s+or\s+/i.test(normalized)) {
      issues.push({
        type: "or_condition",
        severity: "medium",
        description: "OR conditions may prevent optimal index usage",
        suggestion: "Consider using UNION or restructuring with IN clause",
      });
    }

    // Detect NOT IN with subquery
    if (/not\s+in\s*\(\s*select/i.test(query)) {
      issues.push({
        type: "not_in_subquery",
        severity: "high",
        description:
          "NOT IN with subquery can be slow and handle NULLs unexpectedly",
        suggestion: "Use NOT EXISTS or LEFT JOIN IS NULL pattern",
      });
    }

    // Detect implicit type conversion
    if (/where\s+\w+\s*=\s*'?\d+'?/i.test(query)) {
      issues.push({
        type: "potential_type_mismatch",
        severity: "low",
        description: "Potential implicit type conversion in comparison",
        suggestion: "Ensure data types match to avoid conversion overhead",
      });
    }

    // Detect ORDER BY without LIMIT
    if (/order\s+by/i.test(normalized) && !/limit\s+\d+/i.test(normalized)) {
      issues.push({
        type: "unbounded_sort",
        severity: "medium",
        description: "ORDER BY without LIMIT sorts entire result set",
        suggestion: "Add LIMIT clause for pagination",
      });
    }

    // Detect DISTINCT
    if (/select\s+distinct/i.test(query)) {
      issues.push({
        type: "distinct_usage",
        severity: "low",
        description: "DISTINCT requires sorting/hashing for deduplication",
        suggestion: "Ensure DISTINCT is necessary or restructure query",
      });
    }

    // Extract tables
    const fromMatch = normalized.match(/from\s+(\w+)(?:\s+(?:as\s+)?(\w+))?/g);
    if (fromMatch) {
      for (const match of fromMatch) {
        const parts = match.replace(/^from\s+/i, "").split(/\s+(?:as\s+)?/i);
        tableAccess.push({
          table: parts[0],
          alias: parts[1],
          accessType: "full_scan", // Default assumption
          columns: [],
          predicates: [],
        });
      }
    }

    // Extract joins
    const joinRegex =
      /(inner|left|right|full|cross)?\s*join\s+(\w+)(?:\s+(?:as\s+)?(\w+))?\s+on\s+([^where]+?)(?=\s+(?:inner|left|right|full|cross|where|group|order|limit|$))/gi;
    let joinMatch;
    while ((joinMatch = joinRegex.exec(query)) !== null) {
      joins.push({
        type: (joinMatch[1]?.toLowerCase() || "inner") as JoinInfo["type"],
        leftTable: tableAccess[tableAccess.length - 1]?.table || "unknown",
        rightTable: joinMatch[2],
        condition: joinMatch[4].trim(),
        estimated: false,
      });
    }

    // Extract aggregations
    const aggFunctions = [
      "count",
      "sum",
      "avg",
      "min",
      "max",
      "group_concat",
      "string_agg",
    ];
    for (const agg of aggFunctions) {
      const regex = new RegExp(`${agg}\\s*\\([^)]+\\)`, "gi");
      const matches = query.match(regex);
      if (matches) {
        aggregations.push(...matches);
      }
    }

    // Count subqueries
    const subqueries = (query.match(/\(\s*select/gi) || []).length;

    // Extract parameters
    const parameters = [...query.matchAll(/[:@$]\w+|\?/g)].map((m) => m[0]);

    // Calculate complexity score
    const complexity = this.calculateSQLComplexity(
      query,
      issues,
      joins,
      subqueries
    );

    return {
      originalQuery: query,
      type: "sql",
      complexity,
      issues,
      tableAccess,
      joins,
      aggregations,
      subqueries,
      parameters,
    };
  }

  /**
   * Analyze GraphQL query
   */
  private analyzeGraphQLQuery(query: string): QueryAnalysis {
    const issues: QueryIssue[] = [];

    // Detect deep nesting
    const depth = this.calculateGraphQLDepth(query);
    if (depth > 5) {
      issues.push({
        type: "deep_nesting",
        severity: "high",
        description: `Query has nesting depth of ${depth}, which may cause performance issues`,
        suggestion: "Flatten query or use pagination at nested levels",
      });
    }

    // Detect missing arguments for list fields
    if (
      !/\(\s*first\s*:|limit\s*:|take\s*:/i.test(query) &&
      /\{[^}]*\{/i.test(query)
    ) {
      issues.push({
        type: "unbounded_list",
        severity: "medium",
        description: "List fields without pagination may return large datasets",
        suggestion: "Add first/limit argument to list fields",
      });
    }

    // Detect N+1 potential
    const fieldGroups = query.match(/\{[^{}]+\}/g) || [];
    if (fieldGroups.length > 3) {
      issues.push({
        type: "n_plus_one_risk",
        severity: "medium",
        description: "Multiple nested selections may cause N+1 query problem",
        suggestion: "Use DataLoader or batching at resolver level",
      });
    }

    return {
      originalQuery: query,
      type: "graphql",
      complexity: depth * 10 + fieldGroups.length,
      issues,
      tableAccess: [],
      joins: [],
      aggregations: [],
      subqueries: 0,
      parameters: [],
    };
  }

  /**
   * Analyze MongoDB query
   */
  private analyzeMongoDBQuery(query: string): QueryAnalysis {
    const issues: QueryIssue[] = [];

    // Try to parse as JSON
    let queryObj: Record<string, unknown> = {};
    try {
      queryObj = JSON.parse(query);
    } catch {
      // Not valid JSON, analyze as string
    }

    // Detect $where usage
    if (query.includes("$where")) {
      issues.push({
        type: "where_operator",
        severity: "critical",
        description:
          "$where operator executes JavaScript and cannot use indexes",
        suggestion: "Use standard query operators instead of $where",
      });
    }

    // Detect regex without anchor
    if (/\$regex.*[^$]"[^"]*"/.test(query) && !/\$regex.*\^/.test(query)) {
      issues.push({
        type: "unanchored_regex",
        severity: "high",
        description: "Regex without anchor (^) cannot use indexes efficiently",
        suggestion: "Anchor regex patterns at the beginning when possible",
      });
    }

    // Detect $nin
    if (query.includes("$nin")) {
      issues.push({
        type: "nin_operator",
        severity: "medium",
        description: "$nin operator may perform full collection scan",
        suggestion: "Consider restructuring query or using $in with complement",
      });
    }

    // Detect $or without indexes
    if (query.includes("$or")) {
      issues.push({
        type: "or_operator",
        severity: "medium",
        description: "$or requires indexes on all clauses for efficiency",
        suggestion: "Ensure compound indexes exist for all $or branches",
      });
    }

    return {
      originalQuery: query,
      type: "mongodb",
      complexity: issues.length * 20 + query.length / 100,
      issues,
      tableAccess: [],
      joins: [],
      aggregations: [],
      subqueries: 0,
      parameters: [],
    };
  }

  /**
   * Analyze Elasticsearch query
   */
  private analyzeElasticsearchQuery(query: string): QueryAnalysis {
    const issues: QueryIssue[] = [];

    // Detect wildcard queries
    if (query.includes('"wildcard"') || query.includes('"prefix"')) {
      issues.push({
        type: "wildcard_query",
        severity: "medium",
        description: "Wildcard/prefix queries can be slow on large indices",
        suggestion: "Use ngrams or consider edge_ngram tokenizer",
      });
    }

    // Detect script usage
    if (query.includes('"script"')) {
      issues.push({
        type: "script_usage",
        severity: "high",
        description:
          "Scripts in queries prevent caching and are computationally expensive",
        suggestion: "Use runtime fields or indexed computed values",
      });
    }

    // Detect high from offset
    const fromMatch = query.match(/"from"\s*:\s*(\d+)/);
    if (fromMatch && parseInt(fromMatch[1]) > 10000) {
      issues.push({
        type: "deep_pagination",
        severity: "high",
        description: "Deep pagination with from/size is inefficient",
        suggestion: "Use search_after or scroll API for deep pagination",
      });
    }

    // Detect large size
    const sizeMatch = query.match(/"size"\s*:\s*(\d+)/);
    if (sizeMatch && parseInt(sizeMatch[1]) > 1000) {
      issues.push({
        type: "large_result_set",
        severity: "medium",
        description: "Retrieving large result sets impacts performance",
        suggestion: "Use pagination or aggregations for large datasets",
      });
    }

    return {
      originalQuery: query,
      type: "elasticsearch",
      complexity: issues.length * 15 + query.length / 150,
      issues,
      tableAccess: [],
      joins: [],
      aggregations: [],
      subqueries: 0,
      parameters: [],
    };
  }

  /**
   * Analyze generic query
   */
  private analyzeGenericQuery(query: string, type: QueryType): QueryAnalysis {
    return {
      originalQuery: query,
      type,
      complexity: query.length / 50,
      issues: [],
      tableAccess: [],
      joins: [],
      aggregations: [],
      subqueries: 0,
      parameters: [],
    };
  }

  /**
   * Generate optimization based on analysis
   */
  private async generateOptimization(
    query: string,
    type: QueryType,
    analysis: QueryAnalysis
  ): Promise<QueryOptimization> {
    const optimizedQuery = this.rewriteQuery(query, type, analysis);
    const improvements = this.generateImprovements(analysis);
    const indexSuggestions = this.generateIndexSuggestions(analysis);
    const warnings = analysis.issues
      .filter((i) => i.severity === "critical" || i.severity === "high")
      .map((i) => i.description);

    // Estimate speedup based on issues fixed
    const estimatedSpeedup = this.estimateSpeedup(analysis, improvements);

    return {
      originalQuery: query,
      optimizedQuery,
      queryType: type,
      improvements,
      estimatedSpeedup,
      indexSuggestions,
      warnings,
    };
  }

  /**
   * Rewrite query with optimizations
   */
  private rewriteQuery(
    query: string,
    type: QueryType,
    analysis: QueryAnalysis
  ): string {
    if (type !== "sql") {
      // For non-SQL, return with comments
      return query;
    }

    let optimized = query;

    // Replace NOT IN with NOT EXISTS
    if (analysis.issues.some((i) => i.type === "not_in_subquery")) {
      optimized = this.rewriteNotInToNotExists(optimized);
    }

    // Add index hints for OR conditions
    if (analysis.issues.some((i) => i.type === "or_condition")) {
      optimized = this.rewriteOrToUnion(optimized);
    }

    return optimized;
  }

  /**
   * Rewrite NOT IN to NOT EXISTS
   */
  private rewriteNotInToNotExists(query: string): string {
    // Simplified rewrite - production would use proper SQL parser
    return query.replace(
      /(\w+)\s+not\s+in\s*\(\s*select\s+(\w+)\s+from\s+(\w+)(?:\s+where\s+([^)]+))?\)/gi,
      (_, col, subCol, subTable, subWhere) => {
        const whereClause = subWhere ? ` AND ${subWhere}` : "";
        return `NOT EXISTS (SELECT 1 FROM ${subTable} WHERE ${subTable}.${subCol} = ${col}${whereClause})`;
      }
    );
  }

  /**
   * Rewrite OR to UNION
   */
  private rewriteOrToUnion(query: string): string {
    // Return original - full implementation would parse and restructure
    return (
      query + "\n-- Consider rewriting OR as UNION for better index utilization"
    );
  }

  /**
   * Generate improvement descriptions
   */
  private generateImprovements(
    analysis: QueryAnalysis
  ): QueryOptimization["improvements"] {
    return analysis.issues.map((issue) => ({
      type: issue.type,
      description: issue.suggestion,
      impact:
        issue.severity === "critical"
          ? "high"
          : issue.severity === "high"
            ? "high"
            : issue.severity === "medium"
              ? "medium"
              : "low",
    }));
  }

  /**
   * Generate index suggestions
   */
  private generateIndexSuggestions(
    analysis: QueryAnalysis
  ): QueryOptimization["indexSuggestions"] {
    const suggestions: QueryOptimization["indexSuggestions"] = [];

    // Suggest indexes for tables accessed
    for (const table of analysis.tableAccess) {
      if (table.accessType === "full_scan" && table.predicates.length > 0) {
        suggestions.push({
          table: table.table,
          columns: table.predicates,
          type: "btree",
          reason: `Add index to avoid full table scan on ${table.table}`,
        });
      }
    }

    // Suggest composite indexes for joins
    for (const join of analysis.joins) {
      const cols =
        join.condition.match(/\w+\.(\w+)/g)?.map((c) => c.split(".")[1]) || [];
      if (cols.length > 0) {
        suggestions.push({
          table: join.rightTable,
          columns: cols,
          type: "btree",
          reason: `Add index for join condition with ${join.leftTable}`,
        });
      }
    }

    return suggestions;
  }

  /**
   * Estimate speedup from optimizations
   */
  private estimateSpeedup(
    analysis: QueryAnalysis,
    improvements: QueryOptimization["improvements"]
  ): number {
    let speedup = 1.0;

    for (const improvement of improvements) {
      switch (improvement.impact) {
        case "high":
          speedup *= 2.0;
          break;
        case "medium":
          speedup *= 1.5;
          break;
        case "low":
          speedup *= 1.1;
          break;
      }
    }

    // Factor in complexity
    if (analysis.complexity > 100) {
      speedup *= 1.2;
    }

    return Math.min(speedup, 10.0); // Cap at 10x
  }

  /**
   * Calculate GraphQL query depth
   */
  private calculateGraphQLDepth(query: string): number {
    let maxDepth = 0;
    let currentDepth = 0;

    for (const char of query) {
      if (char === "{") {
        currentDepth++;
        maxDepth = Math.max(maxDepth, currentDepth);
      } else if (char === "}") {
        currentDepth--;
      }
    }

    return maxDepth;
  }

  /**
   * Calculate SQL query complexity
   */
  private calculateSQLComplexity(
    query: string,
    issues: QueryIssue[],
    joins: JoinInfo[],
    subqueries: number
  ): number {
    let complexity = 0;

    // Base complexity from length
    complexity += query.length / 100;

    // Issues add complexity
    complexity += issues.reduce((sum, i) => {
      switch (i.severity) {
        case "critical":
          return sum + 30;
        case "high":
          return sum + 20;
        case "medium":
          return sum + 10;
        case "low":
          return sum + 5;
      }
    }, 0);

    // Joins add complexity
    complexity += joins.length * 15;

    // Subqueries add significant complexity
    complexity += subqueries * 25;

    return complexity;
  }

  /**
   * Generate cache key for query
   */
  private generateCacheKey(query: string, type: QueryType): string {
    // Normalize whitespace
    const normalized = query.replace(/\s+/g, " ").trim().toLowerCase();
    const hash = crypto
      .createHash("sha256")
      .update(normalized)
      .digest("hex")
      .slice(0, 16);
    return `${type}:${hash}`;
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export { QueryOptimizer as default };
export type {
  QueryType,
  QueryAnalysis,
  QueryIssue,
  TableAccess,
  JoinInfo,
  IndexRecommendation,
};
