/**
 * Migration Tools - Schema Version Transition Utilities
 *
 * Provides comprehensive tools for managing schema migrations:
 * - Migration guide generation and execution
 * - Codemod support for automated transformations
 * - Query/mutation transformation utilities
 * - Rollback capabilities
 *
 * @packageDocumentation
 * @module @neurectomy/api-client/graphql
 */

import { EventEmitter } from "events";
import type { GraphQLClient } from "./client";
import type {
  SchemaRegistry,
  CompatibilityReport,
  BreakingChange,
  BreakingChangeType,
  SchemaVersionInfo,
} from "./schema-registry";

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Migration step definition
 */
export interface MigrationStep {
  id: string;
  order: number;
  title: string;
  description: string;
  isRequired: boolean;
  hasCodemod: boolean;
  codemod?: Codemod;
  manualSteps?: string[];
  verificationQuery?: string;
  rollbackSteps?: string[];
  estimatedMinutes: number;
  affectedPaths: string[];
  status: MigrationStepStatus;
}

/**
 * Migration step status
 */
export enum MigrationStepStatus {
  PENDING = "PENDING",
  IN_PROGRESS = "IN_PROGRESS",
  COMPLETED = "COMPLETED",
  FAILED = "FAILED",
  SKIPPED = "SKIPPED",
  ROLLED_BACK = "ROLLED_BACK",
}

/**
 * Codemod definition
 */
export interface Codemod {
  id: string;
  name: string;
  description: string;
  language: CodemodLanguage;
  pattern: string;
  replacement: string;
  isReversible: boolean;
  testCases: CodemodTestCase[];
}

/**
 * Supported codemod languages
 */
export enum CodemodLanguage {
  TYPESCRIPT = "typescript",
  JAVASCRIPT = "javascript",
  GRAPHQL = "graphql",
  PYTHON = "python",
  RUST = "rust",
  GO = "go",
}

/**
 * Codemod test case
 */
export interface CodemodTestCase {
  input: string;
  expectedOutput: string;
  description: string;
}

/**
 * Migration guide definition
 */
export interface MigrationGuide {
  id: string;
  fromVersion: string;
  toVersion: string;
  title: string;
  description: string;
  estimatedEffort: MigrationEffort;
  steps: MigrationStep[];
  preflightChecks: PreflightCheck[];
  rollbackPlan: RollbackPlan;
  createdAt: Date;
  updatedAt: Date;
}

/**
 * Migration effort estimate
 */
export interface MigrationEffort {
  hours: number;
  complexity: "TRIVIAL" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  automatedPercentage: number;
  developerDays: number;
  riskLevel: "LOW" | "MEDIUM" | "HIGH";
}

/**
 * Preflight check definition
 */
export interface PreflightCheck {
  id: string;
  name: string;
  description: string;
  query?: string;
  expectedResult?: unknown;
  validator?: (result: unknown) => boolean;
  isCritical: boolean;
  status: PreflightStatus;
  message?: string;
}

/**
 * Preflight check status
 */
export enum PreflightStatus {
  PENDING = "PENDING",
  PASSED = "PASSED",
  FAILED = "FAILED",
  WARNING = "WARNING",
  SKIPPED = "SKIPPED",
}

/**
 * Rollback plan definition
 */
export interface RollbackPlan {
  isAutomatic: boolean;
  steps: RollbackStep[];
  estimatedMinutes: number;
  dataLossRisk: "NONE" | "MINIMAL" | "PARTIAL" | "COMPLETE";
}

/**
 * Rollback step definition
 */
export interface RollbackStep {
  order: number;
  description: string;
  command?: string;
  isManual: boolean;
}

/**
 * Migration execution context
 */
export interface MigrationContext {
  sourceVersion: string;
  targetVersion: string;
  dryRun: boolean;
  force: boolean;
  skipPreflightChecks: boolean;
  autoRollbackOnFailure: boolean;
  logLevel: "debug" | "info" | "warn" | "error";
}

/**
 * Migration execution result
 */
export interface MigrationResult {
  success: boolean;
  fromVersion: string;
  toVersion: string;
  completedSteps: MigrationStep[];
  failedSteps: MigrationStep[];
  skippedSteps: MigrationStep[];
  preflightResults: PreflightCheck[];
  duration: number;
  errors: MigrationError[];
  warnings: string[];
  rollbackPerformed: boolean;
}

/**
 * Migration error definition
 */
export interface MigrationError {
  step: MigrationStep;
  error: Error;
  timestamp: Date;
  context: Record<string, unknown>;
}

/**
 * Query transformation rule
 */
export interface QueryTransformation {
  id: string;
  name: string;
  description: string;
  matchPattern: RegExp;
  transform: (
    query: string,
    variables?: Record<string, unknown>
  ) => TransformResult;
  applicableVersions: {
    from: string;
    to: string;
  };
}

/**
 * Transform result
 */
export interface TransformResult {
  query: string;
  variables?: Record<string, unknown>;
  warnings: string[];
  wasModified: boolean;
}

/**
 * Migration tools configuration
 */
export interface MigrationToolsConfig {
  client: GraphQLClient;
  schemaRegistry: SchemaRegistry;
  enableDryRun?: boolean;
  autoBackup?: boolean;
  maxRetries?: number;
  retryDelay?: number;
}

/**
 * Migration tools events
 */
export interface MigrationToolsEvents {
  "migration:started": (
    guide: MigrationGuide,
    context: MigrationContext
  ) => void;
  "migration:completed": (result: MigrationResult) => void;
  "migration:failed": (result: MigrationResult) => void;
  "step:started": (step: MigrationStep) => void;
  "step:completed": (step: MigrationStep) => void;
  "step:failed": (step: MigrationStep, error: Error) => void;
  "preflight:started": (check: PreflightCheck) => void;
  "preflight:completed": (check: PreflightCheck) => void;
  "preflight:failed": (check: PreflightCheck, reason: string) => void;
  "rollback:started": (plan: RollbackPlan) => void;
  "rollback:completed": () => void;
  "codemod:applied": (codemod: Codemod, filesModified: number) => void;
  warning: (message: string) => void;
  error: (error: Error) => void;
}

// ============================================================================
// Migration Tools Implementation
// ============================================================================

/**
 * Migration Tools for managing schema version transitions
 *
 * @example
 * ```typescript
 * const tools = new MigrationTools({
 *   client: graphqlClient,
 *   schemaRegistry: registry,
 *   enableDryRun: true
 * });
 *
 * // Get migration guide
 * const guide = await tools.getMigrationGuide('1.0.0', '2.0.0');
 *
 * // Execute migration
 * const result = await tools.executeMigration(guide, {
 *   dryRun: true,
 *   autoRollbackOnFailure: true
 * });
 * ```
 */
export class MigrationTools extends EventEmitter {
  private readonly config: Required<MigrationToolsConfig>;
  private readonly transformations: Map<string, QueryTransformation> =
    new Map();
  private readonly activeCodemods: Map<string, Codemod> = new Map();

  constructor(config: MigrationToolsConfig) {
    super();
    this.config = {
      enableDryRun: true,
      autoBackup: true,
      maxRetries: 3,
      retryDelay: 1000,
      ...config,
    };

    // Register built-in transformations
    this.registerBuiltInTransformations();
  }

  // --------------------------------------------------------------------------
  // Migration Guide Management
  // --------------------------------------------------------------------------

  /**
   * Get migration guide between two versions
   */
  async getMigrationGuide(
    fromVersion: string,
    toVersion: string
  ): Promise<MigrationGuide> {
    const query = `
      query GetMigrationGuide($from: String!, $to: String!) {
        migrationGuide(fromVersion: $from, toVersion: $to) {
          fromVersion
          toVersion
          estimatedEffort
          steps {
            order
            title
            description
            codeExample {
              before
              after
              language
            }
            isRequired
            hasCodemod
          }
          codemods {
            id
            name
            description
            language
            command
            package
          }
          verificationSteps
        }
      }
    `;

    const result = await this.config.client.query<{
      migrationGuide: {
        fromVersion: string;
        toVersion: string;
        estimatedEffort: string;
        steps: Array<{
          order: number;
          title: string;
          description: string;
          codeExample?: {
            before: string;
            after: string;
            language: string;
          };
          isRequired: boolean;
          hasCodemod: boolean;
        }>;
        codemods: Array<{
          id: string;
          name: string;
          description: string;
          language: string;
          command: string;
          package: string;
        }>;
        verificationSteps: string[];
      };
    }>(query, { from: fromVersion, to: toVersion });

    // Transform the result into our internal format
    return this.buildMigrationGuide(
      result.migrationGuide,
      fromVersion,
      toVersion
    );
  }

  /**
   * Generate migration guide from compatibility report
   */
  async generateMigrationGuide(
    compatibilityReport: CompatibilityReport
  ): Promise<MigrationGuide> {
    const steps = this.generateStepsFromBreakingChanges(
      compatibilityReport.breakingChanges
    );
    const preflightChecks = this.generatePreflightChecks(compatibilityReport);
    const rollbackPlan = this.generateRollbackPlan(steps);

    const effort = this.calculateEffort(compatibilityReport);

    return {
      id: `migration-${compatibilityReport.sourceVersion}-to-${compatibilityReport.targetVersion}`,
      fromVersion: compatibilityReport.sourceVersion,
      toVersion: compatibilityReport.targetVersion,
      title: `Migration from ${compatibilityReport.sourceVersion} to ${compatibilityReport.targetVersion}`,
      description: `Automated migration guide with ${steps.length} steps`,
      estimatedEffort: effort,
      steps,
      preflightChecks,
      rollbackPlan,
      createdAt: new Date(),
      updatedAt: new Date(),
    };
  }

  // --------------------------------------------------------------------------
  // Migration Execution
  // --------------------------------------------------------------------------

  /**
   * Execute a migration
   */
  async executeMigration(
    guide: MigrationGuide,
    context: Partial<MigrationContext> = {}
  ): Promise<MigrationResult> {
    const fullContext: MigrationContext = {
      sourceVersion: guide.fromVersion,
      targetVersion: guide.toVersion,
      dryRun: false,
      force: false,
      skipPreflightChecks: false,
      autoRollbackOnFailure: true,
      logLevel: "info",
      ...context,
    };

    const result: MigrationResult = {
      success: false,
      fromVersion: guide.fromVersion,
      toVersion: guide.toVersion,
      completedSteps: [],
      failedSteps: [],
      skippedSteps: [],
      preflightResults: [],
      duration: 0,
      errors: [],
      warnings: [],
      rollbackPerformed: false,
    };

    const startTime = Date.now();

    this.emit("migration:started", guide, fullContext);

    try {
      // Run preflight checks
      if (!fullContext.skipPreflightChecks) {
        result.preflightResults = await this.runPreflightChecks(
          guide.preflightChecks,
          fullContext
        );

        const criticalFailures = result.preflightResults.filter(
          (c) => c.isCritical && c.status === PreflightStatus.FAILED
        );

        if (criticalFailures.length > 0 && !fullContext.force) {
          const error = new Error(
            `Critical preflight check(s) failed: ${criticalFailures.map((c) => c.name).join(", ")}`
          );
          this.emit("error", error);
          result.errors.push({
            step: guide.steps[0],
            error,
            timestamp: new Date(),
            context: { preflightResults: criticalFailures },
          });
          result.duration = Date.now() - startTime;
          this.emit("migration:failed", result);
          return result;
        }
      }

      // Execute migration steps
      for (const step of guide.steps) {
        try {
          this.emit("step:started", step);

          if (fullContext.dryRun) {
            // In dry-run mode, just log what would happen
            this.log(
              fullContext.logLevel,
              `[DRY RUN] Would execute step: ${step.title}`
            );
            step.status = MigrationStepStatus.SKIPPED;
            result.skippedSteps.push(step);
          } else {
            await this.executeStep(step, fullContext);
            step.status = MigrationStepStatus.COMPLETED;
            result.completedSteps.push(step);
          }

          this.emit("step:completed", step);
        } catch (error) {
          step.status = MigrationStepStatus.FAILED;
          result.failedSteps.push(step);
          result.errors.push({
            step,
            error: error as Error,
            timestamp: new Date(),
            context: {},
          });

          this.emit("step:failed", step, error as Error);

          // Attempt rollback if configured
          if (fullContext.autoRollbackOnFailure && !fullContext.dryRun) {
            await this.executeRollback(
              guide.rollbackPlan,
              result.completedSteps
            );
            result.rollbackPerformed = true;
          }

          result.duration = Date.now() - startTime;
          this.emit("migration:failed", result);
          return result;
        }
      }

      result.success = true;
      result.duration = Date.now() - startTime;
      this.emit("migration:completed", result);
    } catch (error) {
      result.errors.push({
        step: guide.steps[0],
        error: error as Error,
        timestamp: new Date(),
        context: {},
      });
      result.duration = Date.now() - startTime;
      this.emit("migration:failed", result);
    }

    return result;
  }

  /**
   * Execute a single migration step
   */
  private async executeStep(
    step: MigrationStep,
    context: MigrationContext
  ): Promise<void> {
    this.log(context.logLevel, `Executing step ${step.order}: ${step.title}`);

    // Apply codemod if available
    if (step.hasCodemod && step.codemod) {
      await this.applyCodemod(step.codemod);
    }

    // Execute manual steps (log for user)
    if (step.manualSteps && step.manualSteps.length > 0) {
      this.emit("warning", `Manual steps required for "${step.title}":`);
      step.manualSteps.forEach((ms, i) => {
        this.emit("warning", `  ${i + 1}. ${ms}`);
      });
    }

    // Verify step completion
    if (step.verificationQuery) {
      const verified = await this.verifyStep(step);
      if (!verified) {
        throw new Error(`Verification failed for step: ${step.title}`);
      }
    }

    step.status = MigrationStepStatus.COMPLETED;
  }

  /**
   * Verify a migration step completed successfully
   */
  private async verifyStep(step: MigrationStep): Promise<boolean> {
    if (!step.verificationQuery) {
      return true;
    }

    try {
      const result = await this.config.client.query(step.verificationQuery);
      // Basic verification - more sophisticated validation could be added
      return result !== null && result !== undefined;
    } catch {
      return false;
    }
  }

  // --------------------------------------------------------------------------
  // Preflight Checks
  // --------------------------------------------------------------------------

  /**
   * Run all preflight checks
   */
  private async runPreflightChecks(
    checks: PreflightCheck[],
    context: MigrationContext
  ): Promise<PreflightCheck[]> {
    const results: PreflightCheck[] = [];

    for (const check of checks) {
      this.emit("preflight:started", check);

      try {
        const result = await this.runPreflightCheck(check, context);
        results.push(result);

        if (result.status === PreflightStatus.FAILED) {
          this.emit(
            "preflight:failed",
            result,
            result.message || "Unknown failure"
          );
        } else {
          this.emit("preflight:completed", result);
        }
      } catch (error) {
        check.status = PreflightStatus.FAILED;
        check.message = (error as Error).message;
        results.push(check);
        this.emit("preflight:failed", check, check.message);
      }
    }

    return results;
  }

  /**
   * Run a single preflight check
   */
  private async runPreflightCheck(
    check: PreflightCheck,
    context: MigrationContext
  ): Promise<PreflightCheck> {
    const result = { ...check };

    if (context.dryRun) {
      result.status = PreflightStatus.SKIPPED;
      result.message = "Skipped in dry-run mode";
      return result;
    }

    if (check.query) {
      try {
        const queryResult = await this.config.client.query(check.query);

        if (check.validator) {
          const isValid = check.validator(queryResult);
          result.status = isValid
            ? PreflightStatus.PASSED
            : PreflightStatus.FAILED;
          result.message = isValid ? "Validation passed" : "Validation failed";
        } else if (check.expectedResult !== undefined) {
          const isMatch =
            JSON.stringify(queryResult) ===
            JSON.stringify(check.expectedResult);
          result.status = isMatch
            ? PreflightStatus.PASSED
            : PreflightStatus.FAILED;
          result.message = isMatch
            ? "Result matches expected"
            : "Result does not match expected";
        } else {
          result.status = PreflightStatus.PASSED;
        }
      } catch (error) {
        result.status = PreflightStatus.FAILED;
        result.message = (error as Error).message;
      }
    } else {
      result.status = PreflightStatus.PASSED;
    }

    return result;
  }

  // --------------------------------------------------------------------------
  // Rollback Support
  // --------------------------------------------------------------------------

  /**
   * Execute rollback plan
   */
  private async executeRollback(
    plan: RollbackPlan,
    completedSteps: MigrationStep[]
  ): Promise<void> {
    this.emit("rollback:started", plan);

    // Reverse the completed steps
    const reversedSteps = [...completedSteps].reverse();

    for (const step of reversedSteps) {
      if (step.rollbackSteps && step.rollbackSteps.length > 0) {
        for (const rollbackStep of step.rollbackSteps) {
          this.log("info", `Rolling back: ${rollbackStep}`);
          // Execute rollback step - implementation depends on step type
        }
        step.status = MigrationStepStatus.ROLLED_BACK;
      }
    }

    this.emit("rollback:completed");
  }

  /**
   * Create a rollback checkpoint
   */
  async createCheckpoint(name: string): Promise<string> {
    const checkpointId = `checkpoint-${name}-${Date.now()}`;
    // Implementation would save current state
    this.log("info", `Created checkpoint: ${checkpointId}`);
    return checkpointId;
  }

  /**
   * Restore from a checkpoint
   */
  async restoreCheckpoint(checkpointId: string): Promise<void> {
    this.log("info", `Restoring checkpoint: ${checkpointId}`);
    // Implementation would restore saved state
  }

  // --------------------------------------------------------------------------
  // Query Transformation
  // --------------------------------------------------------------------------

  /**
   * Register a query transformation rule
   */
  registerTransformation(transformation: QueryTransformation): void {
    this.transformations.set(transformation.id, transformation);
  }

  /**
   * Transform a query for version migration
   */
  transformQuery(
    query: string,
    variables: Record<string, unknown> | undefined,
    fromVersion: string,
    toVersion: string
  ): TransformResult {
    let result: TransformResult = {
      query,
      variables,
      warnings: [],
      wasModified: false,
    };

    // Apply all applicable transformations
    for (const [, transformation] of this.transformations) {
      if (
        transformation.applicableVersions.from === fromVersion &&
        transformation.applicableVersions.to === toVersion
      ) {
        if (transformation.matchPattern.test(result.query)) {
          const transformed = transformation.transform(
            result.query,
            result.variables
          );
          result = {
            query: transformed.query,
            variables: transformed.variables,
            warnings: [...result.warnings, ...transformed.warnings],
            wasModified: true,
          };
        }
      }
    }

    return result;
  }

  /**
   * Analyze a query for potential migration issues
   */
  analyzeQuery(
    query: string,
    targetVersion: string
  ): {
    hasDeprecatedFields: boolean;
    deprecatedFields: string[];
    hasBreakingChanges: boolean;
    breakingChanges: string[];
    suggestions: string[];
  } {
    const result = {
      hasDeprecatedFields: false,
      deprecatedFields: [] as string[],
      hasBreakingChanges: false,
      breakingChanges: [] as string[],
      suggestions: [] as string[],
    };

    // This would require schema introspection for full implementation
    // Simplified version checks for common patterns

    // Check for deprecated field patterns (example)
    const deprecatedPatterns = [
      {
        pattern: /oldFieldName/g,
        field: "oldFieldName",
        replacement: "newFieldName",
      },
    ];

    for (const { pattern, field, replacement } of deprecatedPatterns) {
      if (pattern.test(query)) {
        result.hasDeprecatedFields = true;
        result.deprecatedFields.push(field);
        result.suggestions.push(`Replace '${field}' with '${replacement}'`);
      }
    }

    return result;
  }

  // --------------------------------------------------------------------------
  // Codemod Support
  // --------------------------------------------------------------------------

  /**
   * Register a codemod
   */
  registerCodemod(codemod: Codemod): void {
    this.activeCodemods.set(codemod.id, codemod);
  }

  /**
   * Apply a codemod
   */
  async applyCodemod(codemod: Codemod): Promise<number> {
    this.log("info", `Applying codemod: ${codemod.name}`);

    // Validate codemod with test cases
    for (const testCase of codemod.testCases) {
      const result = this.applyCodemodTransform(testCase.input, codemod);
      if (result !== testCase.expectedOutput) {
        throw new Error(`Codemod test case failed: ${testCase.description}`);
      }
    }

    // In a real implementation, this would:
    // 1. Scan project files for matching patterns
    // 2. Apply transformations
    // 3. Write modified files
    const filesModified = 0; // Placeholder

    this.emit("codemod:applied", codemod, filesModified);
    return filesModified;
  }

  /**
   * Apply codemod transformation to content
   */
  private applyCodemodTransform(content: string, codemod: Codemod): string {
    const regex = new RegExp(codemod.pattern, "g");
    return content.replace(regex, codemod.replacement);
  }

  /**
   * Preview codemod changes without applying
   */
  previewCodemod(
    content: string,
    codemod: Codemod
  ): {
    original: string;
    modified: string;
    diff: string[];
  } {
    const modified = this.applyCodemodTransform(content, codemod);
    const diff = this.generateDiff(content, modified);

    return {
      original: content,
      modified,
      diff,
    };
  }

  // --------------------------------------------------------------------------
  // Helper Generation Methods
  // --------------------------------------------------------------------------

  /**
   * Generate migration steps from breaking changes
   */
  private generateStepsFromBreakingChanges(
    breakingChanges: BreakingChange[]
  ): MigrationStep[] {
    const steps: MigrationStep[] = [];

    for (let i = 0; i < breakingChanges.length; i++) {
      const change = breakingChanges[i];

      const step: MigrationStep = {
        id: `step-${i + 1}`,
        order: i + 1,
        title: `Fix ${change.type}: ${change.path}`,
        description: change.description,
        isRequired: true,
        hasCodemod: this.hasCodemodForChange(change),
        codemod: this.getCodemodForChange(change),
        manualSteps: change.migrationSteps,
        estimatedMinutes: this.estimateStepTime(change),
        affectedPaths: [change.path],
        status: MigrationStepStatus.PENDING,
      };

      steps.push(step);
    }

    return steps;
  }

  /**
   * Generate preflight checks from compatibility report
   */
  private generatePreflightChecks(
    report: CompatibilityReport
  ): PreflightCheck[] {
    const checks: PreflightCheck[] = [];

    // Check schema version availability
    checks.push({
      id: "check-target-version",
      name: "Target Version Available",
      description: `Verify schema version ${report.targetVersion} is available`,
      isCritical: true,
      status: PreflightStatus.PENDING,
    });

    // Check for active connections
    checks.push({
      id: "check-no-active-subscriptions",
      name: "No Active Critical Subscriptions",
      description:
        "Verify no critical subscriptions are active during migration",
      isCritical: false,
      status: PreflightStatus.PENDING,
    });

    // Add deprecation checks
    for (const deprecation of report.deprecations) {
      checks.push({
        id: `check-deprecation-${deprecation}`,
        name: `Deprecation: ${deprecation}`,
        description: `Verify usage of deprecated element: ${deprecation}`,
        isCritical: false,
        status: PreflightStatus.PENDING,
      });
    }

    return checks;
  }

  /**
   * Generate rollback plan from steps
   */
  private generateRollbackPlan(steps: MigrationStep[]): RollbackPlan {
    const rollbackSteps: RollbackStep[] = [];

    for (let i = steps.length - 1; i >= 0; i--) {
      const step = steps[i];
      if (step.codemod?.isReversible) {
        rollbackSteps.push({
          order: rollbackSteps.length + 1,
          description: `Reverse: ${step.title}`,
          isManual: false,
        });
      } else if (step.rollbackSteps) {
        for (const rb of step.rollbackSteps) {
          rollbackSteps.push({
            order: rollbackSteps.length + 1,
            description: rb,
            isManual: true,
          });
        }
      }
    }

    return {
      isAutomatic: rollbackSteps.every((s) => !s.isManual),
      steps: rollbackSteps,
      estimatedMinutes: rollbackSteps.length * 5,
      dataLossRisk: "NONE",
    };
  }

  /**
   * Calculate migration effort
   */
  private calculateEffort(report: CompatibilityReport): MigrationEffort {
    const breakingCount = report.breakingChanges.length;
    const deprecationCount = report.deprecations.length;

    let hours = breakingCount * 2 + deprecationCount * 0.5;
    let complexity: MigrationEffort["complexity"];
    let riskLevel: MigrationEffort["riskLevel"];

    if (breakingCount === 0) {
      complexity = "TRIVIAL";
      riskLevel = "LOW";
    } else if (breakingCount <= 3) {
      complexity = "LOW";
      riskLevel = "LOW";
    } else if (breakingCount <= 10) {
      complexity = "MEDIUM";
      riskLevel = "MEDIUM";
    } else if (breakingCount <= 20) {
      complexity = "HIGH";
      riskLevel = "HIGH";
    } else {
      complexity = "CRITICAL";
      riskLevel = "HIGH";
    }

    return {
      hours: Math.max(hours, 1),
      complexity,
      automatedPercentage: this.calculateAutomationPercentage(
        report.breakingChanges
      ),
      developerDays: Math.ceil(hours / 8),
      riskLevel,
    };
  }

  /**
   * Calculate automation percentage
   */
  private calculateAutomationPercentage(changes: BreakingChange[]): number {
    if (changes.length === 0) return 100;

    const automatable = changes.filter((c) =>
      this.hasCodemodForChange(c)
    ).length;
    return Math.round((automatable / changes.length) * 100);
  }

  /**
   * Check if codemod exists for a breaking change
   */
  private hasCodemodForChange(change: BreakingChange): boolean {
    // Check registered codemods
    for (const [, codemod] of this.activeCodemods) {
      if (codemod.pattern && new RegExp(codemod.pattern).test(change.path)) {
        return true;
      }
    }

    // Check if change type is typically automatable
    const automatableTypes: BreakingChangeType[] = [
      BreakingChangeType.FIELD_REMOVED as BreakingChangeType,
      BreakingChangeType.ENUM_VALUE_REMOVED as BreakingChangeType,
    ];

    return automatableTypes.includes(change.type);
  }

  /**
   * Get codemod for a breaking change
   */
  private getCodemodForChange(change: BreakingChange): Codemod | undefined {
    for (const [, codemod] of this.activeCodemods) {
      if (codemod.pattern && new RegExp(codemod.pattern).test(change.path)) {
        return codemod;
      }
    }
    return undefined;
  }

  /**
   * Estimate time for a migration step
   */
  private estimateStepTime(change: BreakingChange): number {
    const timeByType: Record<string, number> = {
      FIELD_REMOVED: 30,
      TYPE_REMOVED: 120,
      ENUM_VALUE_REMOVED: 15,
      REQUIRED_ARGUMENT_ADDED: 45,
      TYPE_CHANGED: 90,
      NULLABLE_TO_NON_NULL: 60,
      DEFAULT_VALUE_CHANGED: 15,
      INTERFACE_IMPLEMENTATION_REMOVED: 90,
      DIRECTIVE_REMOVED: 30,
      DIRECTIVE_ARGUMENT_REMOVED: 20,
      UNION_MEMBER_REMOVED: 45,
    };

    return timeByType[change.type] || 60;
  }

  /**
   * Build migration guide from API response
   */
  private buildMigrationGuide(
    apiGuide: {
      fromVersion: string;
      toVersion: string;
      estimatedEffort: string;
      steps: Array<{
        order: number;
        title: string;
        description: string;
        codeExample?: {
          before: string;
          after: string;
          language: string;
        };
        isRequired: boolean;
        hasCodemod: boolean;
      }>;
      codemods: Array<{
        id: string;
        name: string;
        description: string;
        language: string;
        command: string;
        package: string;
      }>;
      verificationSteps: string[];
    },
    fromVersion: string,
    toVersion: string
  ): MigrationGuide {
    const steps: MigrationStep[] = apiGuide.steps.map((s) => ({
      id: `step-${s.order}`,
      order: s.order,
      title: s.title,
      description: s.description,
      isRequired: s.isRequired,
      hasCodemod: s.hasCodemod,
      estimatedMinutes: 30,
      affectedPaths: [],
      status: MigrationStepStatus.PENDING,
    }));

    return {
      id: `guide-${fromVersion}-${toVersion}`,
      fromVersion,
      toVersion,
      title: `Migration Guide: ${fromVersion} → ${toVersion}`,
      description: `Step-by-step guide for migrating from ${fromVersion} to ${toVersion}`,
      estimatedEffort: {
        hours: parseFloat(apiGuide.estimatedEffort) || 4,
        complexity: "MEDIUM",
        automatedPercentage: 50,
        developerDays: 1,
        riskLevel: "MEDIUM",
      },
      steps,
      preflightChecks: [],
      rollbackPlan: {
        isAutomatic: false,
        steps: [],
        estimatedMinutes: 30,
        dataLossRisk: "NONE",
      },
      createdAt: new Date(),
      updatedAt: new Date(),
    };
  }

  /**
   * Register built-in query transformations
   */
  private registerBuiltInTransformations(): void {
    // Example transformation for field rename
    this.registerTransformation({
      id: "rename-deprecated-fields",
      name: "Rename Deprecated Fields",
      description:
        "Automatically renames deprecated fields to their replacements",
      matchPattern: /\boldFieldName\b/g,
      transform: (query, variables) => ({
        query: query.replace(/\boldFieldName\b/g, "newFieldName"),
        variables,
        warnings: ["Renamed oldFieldName to newFieldName"],
        wasModified: true,
      }),
      applicableVersions: { from: "1.0.0", to: "2.0.0" },
    });
  }

  /**
   * Generate a simple diff between two strings
   */
  private generateDiff(original: string, modified: string): string[] {
    const diff: string[] = [];
    const originalLines = original.split("\n");
    const modifiedLines = modified.split("\n");

    const maxLines = Math.max(originalLines.length, modifiedLines.length);

    for (let i = 0; i < maxLines; i++) {
      const origLine = originalLines[i] || "";
      const modLine = modifiedLines[i] || "";

      if (origLine !== modLine) {
        if (origLine) diff.push(`- ${origLine}`);
        if (modLine) diff.push(`+ ${modLine}`);
      }
    }

    return diff;
  }

  /**
   * Log a message at the specified level
   */
  private log(level: string, message: string): void {
    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] [${level.toUpperCase()}] ${message}`;

    switch (level) {
      case "debug":
        console.debug(logMessage);
        break;
      case "info":
        console.info(logMessage);
        break;
      case "warn":
        console.warn(logMessage);
        this.emit("warning", message);
        break;
      case "error":
        console.error(logMessage);
        break;
    }
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create migration tools with default configuration
 */
export function createMigrationTools(
  client: GraphQLClient,
  schemaRegistry: SchemaRegistry
): MigrationTools {
  return new MigrationTools({ client, schemaRegistry });
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Format a migration guide as a human-readable string
 */
export function formatMigrationGuide(guide: MigrationGuide): string {
  const lines: string[] = [];

  lines.push(`# ${guide.title}`);
  lines.push("");
  lines.push(`From: ${guide.fromVersion}`);
  lines.push(`To: ${guide.toVersion}`);
  lines.push("");
  lines.push(`## Effort Estimate`);
  lines.push(`- Hours: ~${guide.estimatedEffort.hours}`);
  lines.push(`- Complexity: ${guide.estimatedEffort.complexity}`);
  lines.push(`- Automation: ${guide.estimatedEffort.automatedPercentage}%`);
  lines.push(`- Risk Level: ${guide.estimatedEffort.riskLevel}`);
  lines.push("");
  lines.push(`## Migration Steps`);

  for (const step of guide.steps) {
    lines.push("");
    lines.push(`### ${step.order}. ${step.title}`);
    lines.push("");
    lines.push(step.description);

    if (step.hasCodemod) {
      lines.push("");
      lines.push("✅ Codemod available for automatic migration");
    }

    if (step.manualSteps && step.manualSteps.length > 0) {
      lines.push("");
      lines.push("Manual steps:");
      step.manualSteps.forEach((ms, i) => {
        lines.push(`  ${i + 1}. ${ms}`);
      });
    }
  }

  if (guide.preflightChecks.length > 0) {
    lines.push("");
    lines.push(`## Preflight Checks`);
    for (const check of guide.preflightChecks) {
      const icon = check.isCritical ? "🔴" : "🟡";
      lines.push(`${icon} ${check.name}: ${check.description}`);
    }
  }

  lines.push("");
  lines.push(`## Rollback Plan`);
  lines.push(`- Automatic: ${guide.rollbackPlan.isAutomatic ? "Yes" : "No"}`);
  lines.push(
    `- Estimated Time: ${guide.rollbackPlan.estimatedMinutes} minutes`
  );
  lines.push(`- Data Loss Risk: ${guide.rollbackPlan.dataLossRisk}`);

  return lines.join("\n");
}

/**
 * Format a migration result as a human-readable string
 */
export function formatMigrationResult(result: MigrationResult): string {
  const lines: string[] = [];

  const statusIcon = result.success ? "✅" : "❌";
  lines.push(
    `${statusIcon} Migration ${result.success ? "Completed" : "Failed"}`
  );
  lines.push("");
  lines.push(`From: ${result.fromVersion}`);
  lines.push(`To: ${result.toVersion}`);
  lines.push(`Duration: ${(result.duration / 1000).toFixed(2)}s`);
  lines.push("");
  lines.push(`## Step Summary`);
  lines.push(`- Completed: ${result.completedSteps.length}`);
  lines.push(`- Failed: ${result.failedSteps.length}`);
  lines.push(`- Skipped: ${result.skippedSteps.length}`);

  if (result.errors.length > 0) {
    lines.push("");
    lines.push(`## Errors`);
    for (const error of result.errors) {
      lines.push(`- [${error.step.title}] ${error.error.message}`);
    }
  }

  if (result.warnings.length > 0) {
    lines.push("");
    lines.push(`## Warnings`);
    result.warnings.forEach((w) => lines.push(`- ${w}`));
  }

  if (result.rollbackPerformed) {
    lines.push("");
    lines.push("⚠️ Rollback was performed due to migration failure");
  }

  return lines.join("\n");
}
