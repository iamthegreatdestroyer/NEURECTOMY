/**
 * @fileoverview Role-Based Access Control Manager
 * @module @neurectomy/enterprise/rbac/role-manager
 *
 * @description
 * Enterprise RBAC with hierarchical roles:
 * - Role definition and hierarchy
 * - Permission inheritance
 * - Role assignment management
 * - Audit integration
 *
 * @FORTRESS Security-first access control
 */

import { EventEmitter } from "events";
import type { Role, Permission, RoleAssignment, RBACConfig } from "../types.js";

/**
 * Role hierarchy node
 */
export interface RoleNode {
  /** Role */
  role: Role;
  /** Parent roles */
  parents: string[];
  /** Child roles */
  children: string[];
  /** Effective permissions (inherited + direct) */
  effectivePermissions: Set<string>;
}

/**
 * Role assignment request
 */
export interface AssignmentRequest {
  /** User or entity ID */
  principalId: string;
  /** Principal type */
  principalType: "user" | "group" | "service";
  /** Role to assign */
  roleId: string;
  /** Tenant context */
  tenantId: string;
  /** Resource scope (optional) */
  scope?: string;
  /** Expiration (optional) */
  expiresAt?: Date;
  /** Assignment reason */
  reason?: string;
  /** Assigned by */
  assignedBy: string;
}

/**
 * Role Manager
 *
 * Manages role definitions, hierarchies, and assignments
 * with full audit trail support.
 */
export class RoleManager extends EventEmitter {
  private roles: Map<string, RoleNode> = new Map();
  private assignments: Map<string, RoleAssignment[]> = new Map();
  private permissionCache: Map<string, Set<string>> = new Map();
  private initialized: boolean = false;

  constructor(private config: RBACConfig) {
    super();
  }

  /**
   * Initialize role manager
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    this.emit("initializing");

    // Load default roles
    await this.loadDefaultRoles();

    // Build permission cache
    this.rebuildPermissionCache();

    this.initialized = true;
    this.emit("initialized");
  }

  /**
   * Create a new role
   */
  async createRole(role: Role): Promise<Role> {
    if (this.roles.has(role.id)) {
      throw new Error(`Role already exists: ${role.id}`);
    }

    const node: RoleNode = {
      role,
      parents: role.inheritsFrom || [],
      children: [],
      effectivePermissions: new Set(role.permissions),
    };

    // Update parent roles to include this as child
    for (const parentId of node.parents) {
      const parent = this.roles.get(parentId);
      if (parent) {
        parent.children.push(role.id);
      }
    }

    this.roles.set(role.id, node);

    // Rebuild effective permissions
    this.rebuildPermissionCache();

    this.emit("role:created", role);
    return role;
  }

  /**
   * Update a role
   */
  async updateRole(roleId: string, updates: Partial<Role>): Promise<Role> {
    const node = this.roles.get(roleId);
    if (!node) {
      throw new Error(`Role not found: ${roleId}`);
    }

    // Update role properties
    Object.assign(node.role, updates);

    // Update direct permissions
    if (updates.permissions) {
      node.effectivePermissions = new Set(updates.permissions);
    }

    // Update inheritance if changed
    if (updates.inheritsFrom) {
      // Remove from old parents
      for (const oldParentId of node.parents) {
        const oldParent = this.roles.get(oldParentId);
        if (oldParent) {
          oldParent.children = oldParent.children.filter((id) => id !== roleId);
        }
      }

      // Add to new parents
      node.parents = updates.inheritsFrom;
      for (const newParentId of node.parents) {
        const newParent = this.roles.get(newParentId);
        if (newParent) {
          newParent.children.push(roleId);
        }
      }
    }

    // Rebuild cache
    this.rebuildPermissionCache();

    this.emit("role:updated", node.role);
    return node.role;
  }

  /**
   * Delete a role
   */
  async deleteRole(roleId: string): Promise<void> {
    const node = this.roles.get(roleId);
    if (!node) {
      throw new Error(`Role not found: ${roleId}`);
    }

    // Check for existing assignments
    const assignmentsWithRole = this.findAssignmentsByRole(roleId);
    if (assignmentsWithRole.length > 0) {
      throw new Error(
        `Cannot delete role with ${assignmentsWithRole.length} active assignments`
      );
    }

    // Check for child roles
    if (node.children.length > 0) {
      throw new Error(
        `Cannot delete role with child roles: ${node.children.join(", ")}`
      );
    }

    // Remove from parent's children
    for (const parentId of node.parents) {
      const parent = this.roles.get(parentId);
      if (parent) {
        parent.children = parent.children.filter((id) => id !== roleId);
      }
    }

    this.roles.delete(roleId);
    this.rebuildPermissionCache();

    this.emit("role:deleted", { roleId });
  }

  /**
   * Assign role to principal
   */
  async assignRole(request: AssignmentRequest): Promise<RoleAssignment> {
    const role = this.roles.get(request.roleId);
    if (!role) {
      throw new Error(`Role not found: ${request.roleId}`);
    }

    const assignment: RoleAssignment = {
      id: this.generateAssignmentId(),
      principalId: request.principalId,
      principalType: request.principalType,
      roleId: request.roleId,
      tenantId: request.tenantId,
      scope: request.scope,
      assignedAt: new Date(),
      assignedBy: request.assignedBy,
      expiresAt: request.expiresAt,
      reason: request.reason,
      active: true,
    };

    // Store assignment
    const key = this.getAssignmentKey(request.principalId, request.tenantId);
    const existing = this.assignments.get(key) || [];

    // Check for duplicate
    const duplicate = existing.find(
      (a) =>
        a.roleId === request.roleId && a.scope === request.scope && a.active
    );
    if (duplicate) {
      throw new Error("Assignment already exists");
    }

    existing.push(assignment);
    this.assignments.set(key, existing);

    // Invalidate permission cache for principal
    this.permissionCache.delete(key);

    this.emit("role:assigned", assignment);
    return assignment;
  }

  /**
   * Revoke role from principal
   */
  async revokeRole(
    principalId: string,
    roleId: string,
    tenantId: string,
    revokedBy: string,
    reason?: string
  ): Promise<void> {
    const key = this.getAssignmentKey(principalId, tenantId);
    const assignments = this.assignments.get(key) || [];

    const assignment = assignments.find((a) => a.roleId === roleId && a.active);

    if (!assignment) {
      throw new Error("Assignment not found");
    }

    assignment.active = false;
    assignment.revokedAt = new Date();
    assignment.revokedBy = revokedBy;
    assignment.revokeReason = reason;

    // Invalidate permission cache
    this.permissionCache.delete(key);

    this.emit("role:revoked", { assignment, revokedBy, reason });
  }

  /**
   * Get roles for principal
   */
  getRoles(principalId: string, tenantId: string): Role[] {
    const key = this.getAssignmentKey(principalId, tenantId);
    const assignments = this.assignments.get(key) || [];

    return assignments
      .filter((a) => a.active && !this.isExpired(a))
      .map((a) => this.roles.get(a.roleId)?.role)
      .filter((r): r is Role => r !== undefined);
  }

  /**
   * Get effective permissions for principal
   */
  getEffectivePermissions(principalId: string, tenantId: string): Set<string> {
    const key = this.getAssignmentKey(principalId, tenantId);

    // Check cache
    const cached = this.permissionCache.get(key);
    if (cached) return cached;

    // Calculate effective permissions
    const permissions = new Set<string>();
    const roles = this.getRoles(principalId, tenantId);

    for (const role of roles) {
      const node = this.roles.get(role.id);
      if (node) {
        for (const perm of node.effectivePermissions) {
          permissions.add(perm);
        }
      }
    }

    // Cache result
    this.permissionCache.set(key, permissions);

    return permissions;
  }

  /**
   * Check if principal has permission
   */
  hasPermission(
    principalId: string,
    tenantId: string,
    permission: string
  ): boolean {
    const permissions = this.getEffectivePermissions(principalId, tenantId);

    // Check exact match
    if (permissions.has(permission)) return true;

    // Check wildcard permissions
    const parts = permission.split(":");
    for (let i = parts.length - 1; i >= 0; i--) {
      const wildcard = [...parts.slice(0, i), "*"].join(":");
      if (permissions.has(wildcard)) return true;
    }

    // Check global wildcard
    return permissions.has("*");
  }

  /**
   * Get role by ID
   */
  getRole(roleId: string): Role | undefined {
    return this.roles.get(roleId)?.role;
  }

  /**
   * Get all roles
   */
  getAllRoles(): Role[] {
    return Array.from(this.roles.values()).map((n) => n.role);
  }

  /**
   * Get role hierarchy
   */
  getRoleHierarchy(roleId: string): RoleHierarchy {
    const node = this.roles.get(roleId);
    if (!node) {
      throw new Error(`Role not found: ${roleId}`);
    }

    return {
      role: node.role,
      parents: node.parents.map((id) => this.getRoleHierarchy(id)),
      children: node.children.map((id) => ({
        role: this.roles.get(id)!.role,
        parents: [],
        children: [],
      })),
    };
  }

  /**
   * Get assignments for principal
   */
  getAssignments(principalId: string, tenantId: string): RoleAssignment[] {
    const key = this.getAssignmentKey(principalId, tenantId);
    return this.assignments.get(key) || [];
  }

  /**
   * Get all assignments for a tenant
   */
  getTenantAssignments(tenantId: string): RoleAssignment[] {
    const result: RoleAssignment[] = [];

    for (const [key, assignments] of this.assignments) {
      if (key.endsWith(`:${tenantId}`)) {
        result.push(...assignments.filter((a) => a.active));
      }
    }

    return result;
  }

  /**
   * Cleanup expired assignments
   */
  async cleanupExpired(): Promise<number> {
    let cleaned = 0;

    for (const [key, assignments] of this.assignments) {
      for (const assignment of assignments) {
        if (assignment.active && this.isExpired(assignment)) {
          assignment.active = false;
          assignment.revokedAt = new Date();
          assignment.revokeReason = "Expired";
          cleaned++;
        }
      }

      // Invalidate cache
      this.permissionCache.delete(key);
    }

    if (cleaned > 0) {
      this.emit("assignments:cleaned", { count: cleaned });
    }

    return cleaned;
  }

  /**
   * Shutdown role manager
   */
  async shutdown(): Promise<void> {
    this.emit("shutting-down");
    this.permissionCache.clear();
    this.initialized = false;
    this.emit("shutdown");
  }

  // Private methods

  private async loadDefaultRoles(): Promise<void> {
    const defaultRoles: Role[] = [
      {
        id: "super_admin",
        name: "Super Administrator",
        description: "Full system access",
        permissions: ["*"],
        inheritsFrom: [],
        system: true,
      },
      {
        id: "tenant_admin",
        name: "Tenant Administrator",
        description: "Full tenant access",
        permissions: [
          "tenant:*",
          "users:*",
          "roles:read",
          "roles:assign",
          "agents:*",
          "experiments:*",
          "models:*",
        ],
        inheritsFrom: [],
        system: true,
      },
      {
        id: "developer",
        name: "Developer",
        description: "Development access",
        permissions: [
          "agents:read",
          "agents:create",
          "agents:update",
          "experiments:read",
          "experiments:create",
          "experiments:update",
          "models:read",
          "models:create",
        ],
        inheritsFrom: [],
        system: true,
      },
      {
        id: "analyst",
        name: "Analyst",
        description: "Read-only analytics access",
        permissions: [
          "agents:read",
          "experiments:read",
          "models:read",
          "analytics:*",
          "reports:read",
        ],
        inheritsFrom: [],
        system: true,
      },
      {
        id: "viewer",
        name: "Viewer",
        description: "Read-only access",
        permissions: ["agents:read", "experiments:read", "models:read"],
        inheritsFrom: [],
        system: true,
      },
    ];

    for (const role of defaultRoles) {
      const node: RoleNode = {
        role,
        parents: role.inheritsFrom || [],
        children: [],
        effectivePermissions: new Set(role.permissions),
      };
      this.roles.set(role.id, node);
    }
  }

  private rebuildPermissionCache(): void {
    // Clear cache
    this.permissionCache.clear();

    // Rebuild effective permissions for all roles
    for (const [roleId, node] of this.roles) {
      node.effectivePermissions = this.calculateEffectivePermissions(
        roleId,
        new Set()
      );
    }
  }

  private calculateEffectivePermissions(
    roleId: string,
    visited: Set<string>
  ): Set<string> {
    if (visited.has(roleId)) {
      // Circular inheritance detected
      return new Set();
    }

    visited.add(roleId);

    const node = this.roles.get(roleId);
    if (!node) return new Set();

    const permissions = new Set(node.role.permissions);

    // Add inherited permissions
    for (const parentId of node.parents) {
      const parentPerms = this.calculateEffectivePermissions(parentId, visited);
      for (const perm of parentPerms) {
        permissions.add(perm);
      }
    }

    return permissions;
  }

  private findAssignmentsByRole(roleId: string): RoleAssignment[] {
    const result: RoleAssignment[] = [];

    for (const assignments of this.assignments.values()) {
      for (const assignment of assignments) {
        if (assignment.roleId === roleId && assignment.active) {
          result.push(assignment);
        }
      }
    }

    return result;
  }

  private getAssignmentKey(principalId: string, tenantId: string): string {
    return `${principalId}:${tenantId}`;
  }

  private isExpired(assignment: RoleAssignment): boolean {
    if (!assignment.expiresAt) return false;
    return assignment.expiresAt < new Date();
  }

  private generateAssignmentId(): string {
    return `assign_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }
}

/**
 * Role hierarchy type
 */
export interface RoleHierarchy {
  role: Role;
  parents: RoleHierarchy[];
  children: RoleHierarchy[];
}

/**
 * Factory function
 */
export function createRoleManager(config: RBACConfig): RoleManager {
  return new RoleManager(config);
}

export default RoleManager;
