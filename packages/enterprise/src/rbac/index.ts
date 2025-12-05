/**
 * @fileoverview RBAC Module
 * @module @neurectomy/enterprise/rbac
 *
 * @description
 * Role-Based Access Control with:
 * - Hierarchical roles
 * - Permission inheritance
 * - Policy-based access control
 * - Attribute-based conditions
 *
 * @FORTRESS Enterprise access control
 */

// Role Manager
export {
  RoleManager,
  createRoleManager,
  type RoleNode,
  type AssignmentRequest,
  type RoleHierarchy,
} from "./role-manager.js";

// Permission Evaluator
export {
  PermissionEvaluator,
  createPermissionEvaluator,
  type PermissionEvaluatorConfig,
  type EvaluationContext,
  type EvaluationResult,
  type PermissionPattern,
  type CacheStats,
} from "./permission-evaluator.js";

// Re-export types
export type {
  Role,
  Permission,
  RoleAssignment,
  RBACConfig,
  PolicyRule,
  PolicyCondition,
} from "../types.js";

/**
 * Create full RBAC system
 */
export interface RBACSystemConfig {
  rbac: import("../types.js").RBACConfig;
  evaluator?: import("./permission-evaluator.js").PermissionEvaluatorConfig;
}

export interface RBACSystem {
  roles: import("./role-manager.js").RoleManager;
  evaluator: import("./permission-evaluator.js").PermissionEvaluator;
}

export async function createRBACSystem(
  config: RBACSystemConfig
): Promise<RBACSystem> {
  const { RoleManager } = await import("./role-manager.js");
  const { PermissionEvaluator } = await import("./permission-evaluator.js");

  const roles = new RoleManager(config.rbac);
  const evaluator = new PermissionEvaluator(config.evaluator || {});

  // Initialize
  await roles.initialize();
  await evaluator.initialize();

  return { roles, evaluator };
}

/**
 * Quick permission check helper
 */
export async function checkAccess(
  system: RBACSystem,
  principalId: string,
  tenantId: string,
  resource: { type: string; id: string; owner?: string },
  action: string
): Promise<boolean> {
  const userRoles = system.roles.getRoles(principalId, tenantId);

  const result = await system.evaluator.evaluate({
    principal: {
      id: principalId,
      type: "user",
      roles: userRoles.map((r) => r.id),
      attributes: {},
    },
    resource: {
      ...resource,
      attributes: {},
    },
    action,
    environment: {
      timestamp: new Date(),
      tenantId,
      attributes: {},
    },
  });

  return result.allowed;
}
