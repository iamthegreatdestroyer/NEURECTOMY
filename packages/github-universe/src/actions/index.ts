/**
 * @fileoverview GitHub Actions Manager module exports
 * @module @neurectomy/github-universe/actions
 *
 * @description Provides comprehensive GitHub Actions management including
 * workflow triggering, run monitoring, and artifact handling.
 *
 * @example
 * ```typescript
 * import { ActionsManager, type Workflow, type WorkflowRun } from '@neurectomy/github-universe/actions';
 *
 * const actions = new ActionsManager(client);
 *
 * // Trigger a workflow
 * await actions.triggerWorkflow(
 *   { owner: 'org', repo: 'app' },
 *   'ci.yml',
 *   'main',
 *   { environment: 'staging' }
 * );
 *
 * // List workflow runs
 * const runs = await actions.listWorkflowRuns(
 *   { owner: 'org', repo: 'app' },
 *   'ci.yml'
 * );
 *
 * // Download artifacts
 * const artifacts = await actions.downloadArtifacts(
 *   { owner: 'org', repo: 'app' },
 *   runs[0].id,
 *   './artifacts'
 * );
 * ```
 */

export { ActionsManager } from "./manager";
export type {
  ActionsManagerEvents,
  Workflow,
  WorkflowRun,
  WorkflowJob,
  WorkflowStep,
  Artifact,
  WorkflowDispatchInputs,
  WorkflowListOptions,
  WorkflowRunListOptions,
  WorkflowUsage,
} from "./manager";
