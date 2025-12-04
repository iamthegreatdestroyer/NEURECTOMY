//! Role-Based Access Control (RBAC)
//!
//! Implements a hierarchical permission system:
//! - Roles with inherited permissions
//! - Fine-grained permission checking
//! - Dynamic permission assignment
//!
//! @FORTRESS - Defense in Depth

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::str::FromStr;

use super::AuthError;

/// User roles in NEURECTOMY
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// System administrator - full access
    Admin,
    /// Developer - can create and manage own agents
    Developer,
    /// Viewer - read-only access to shared agents
    Viewer,
    /// Service account - for automated systems
    Service,
    /// Guest - very limited access
    Guest,
}

impl Role {
    /// Get all permissions for this role
    pub fn permissions(&self) -> Vec<Permission> {
        match self {
            Role::Admin => Permission::all(),
            Role::Developer => vec![
                // Agent permissions
                Permission::AgentCreate,
                Permission::AgentRead,
                Permission::AgentUpdate,
                Permission::AgentDelete,
                Permission::AgentDeploy,
                Permission::AgentTrain,
                // Container permissions
                Permission::ContainerCreate,
                Permission::ContainerRead,
                Permission::ContainerUpdate,
                Permission::ContainerStart,
                Permission::ContainerStop,
                Permission::ContainerLogs,
                // Conversation permissions
                Permission::ConversationCreate,
                Permission::ConversationRead,
                Permission::ConversationDelete,
                // Knowledge base
                Permission::KnowledgeCreate,
                Permission::KnowledgeRead,
                Permission::KnowledgeUpdate,
                Permission::KnowledgeDelete,
                // Training
                Permission::TrainingCreate,
                Permission::TrainingRead,
                Permission::TrainingCancel,
                // Tools
                Permission::ToolCreate,
                Permission::ToolRead,
                Permission::ToolUpdate,
                Permission::ToolDelete,
                // API keys
                Permission::ApiKeyCreate,
                Permission::ApiKeyRead,
                Permission::ApiKeyRevoke,
                // Profile
                Permission::ProfileRead,
                Permission::ProfileUpdate,
            ],
            Role::Viewer => vec![
                Permission::AgentRead,
                Permission::ContainerRead,
                Permission::ContainerLogs,
                Permission::ConversationRead,
                Permission::KnowledgeRead,
                Permission::TrainingRead,
                Permission::ToolRead,
                Permission::ProfileRead,
            ],
            Role::Service => vec![
                Permission::AgentRead,
                Permission::ContainerRead,
                Permission::ContainerStart,
                Permission::ContainerStop,
                Permission::ConversationCreate,
                Permission::ConversationRead,
                Permission::TrainingRead,
            ],
            Role::Guest => vec![
                Permission::AgentRead,
            ],
        }
    }

    /// Check if this role has admin privileges
    pub fn is_admin(&self) -> bool {
        matches!(self, Role::Admin)
    }

    /// Check if this role can manage other users
    pub fn can_manage_users(&self) -> bool {
        matches!(self, Role::Admin)
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::Admin => write!(f, "admin"),
            Role::Developer => write!(f, "developer"),
            Role::Viewer => write!(f, "viewer"),
            Role::Service => write!(f, "service"),
            Role::Guest => write!(f, "guest"),
        }
    }
}

impl FromStr for Role {
    type Err = AuthError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "admin" => Ok(Role::Admin),
            "developer" => Ok(Role::Developer),
            "viewer" => Ok(Role::Viewer),
            "service" => Ok(Role::Service),
            "guest" => Ok(Role::Guest),
            _ => Err(AuthError::InternalError(format!("Unknown role: {}", s))),
        }
    }
}

/// Fine-grained permissions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    // Agent permissions
    AgentCreate,
    AgentRead,
    AgentUpdate,
    AgentDelete,
    AgentDeploy,
    AgentTrain,
    AgentShare,

    // Container permissions
    ContainerCreate,
    ContainerRead,
    ContainerUpdate,
    ContainerDelete,
    ContainerStart,
    ContainerStop,
    ContainerLogs,
    ContainerExec,

    // Conversation permissions
    ConversationCreate,
    ConversationRead,
    ConversationUpdate,
    ConversationDelete,
    ConversationExport,

    // Knowledge base permissions
    KnowledgeCreate,
    KnowledgeRead,
    KnowledgeUpdate,
    KnowledgeDelete,
    KnowledgeImport,
    KnowledgeExport,

    // Training permissions
    TrainingCreate,
    TrainingRead,
    TrainingUpdate,
    TrainingCancel,
    TrainingDelete,

    // Tool permissions
    ToolCreate,
    ToolRead,
    ToolUpdate,
    ToolDelete,
    ToolExecute,

    // API key permissions
    ApiKeyCreate,
    ApiKeyRead,
    ApiKeyUpdate,
    ApiKeyRevoke,

    // User/Profile permissions
    ProfileRead,
    ProfileUpdate,

    // Admin permissions
    UserCreate,
    UserRead,
    UserUpdate,
    UserDelete,
    UserImpersonate,
    SystemConfig,
    AuditRead,
    MetricsRead,
}

impl Permission {
    /// Get all permissions
    pub fn all() -> Vec<Permission> {
        vec![
            // Agents
            Permission::AgentCreate,
            Permission::AgentRead,
            Permission::AgentUpdate,
            Permission::AgentDelete,
            Permission::AgentDeploy,
            Permission::AgentTrain,
            Permission::AgentShare,
            // Containers
            Permission::ContainerCreate,
            Permission::ContainerRead,
            Permission::ContainerUpdate,
            Permission::ContainerDelete,
            Permission::ContainerStart,
            Permission::ContainerStop,
            Permission::ContainerLogs,
            Permission::ContainerExec,
            // Conversations
            Permission::ConversationCreate,
            Permission::ConversationRead,
            Permission::ConversationUpdate,
            Permission::ConversationDelete,
            Permission::ConversationExport,
            // Knowledge
            Permission::KnowledgeCreate,
            Permission::KnowledgeRead,
            Permission::KnowledgeUpdate,
            Permission::KnowledgeDelete,
            Permission::KnowledgeImport,
            Permission::KnowledgeExport,
            // Training
            Permission::TrainingCreate,
            Permission::TrainingRead,
            Permission::TrainingUpdate,
            Permission::TrainingCancel,
            Permission::TrainingDelete,
            // Tools
            Permission::ToolCreate,
            Permission::ToolRead,
            Permission::ToolUpdate,
            Permission::ToolDelete,
            Permission::ToolExecute,
            // API Keys
            Permission::ApiKeyCreate,
            Permission::ApiKeyRead,
            Permission::ApiKeyUpdate,
            Permission::ApiKeyRevoke,
            // Profile
            Permission::ProfileRead,
            Permission::ProfileUpdate,
            // Admin
            Permission::UserCreate,
            Permission::UserRead,
            Permission::UserUpdate,
            Permission::UserDelete,
            Permission::UserImpersonate,
            Permission::SystemConfig,
            Permission::AuditRead,
            Permission::MetricsRead,
        ]
    }
}

impl std::fmt::Display for Permission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// RBAC service for permission checking
#[derive(Clone)]
pub struct RbacService {
    /// Additional permissions per user (beyond role)
    user_permissions: std::sync::Arc<tokio::sync::RwLock<std::collections::HashMap<uuid::Uuid, HashSet<Permission>>>>,
    /// Denied permissions per user
    user_denied: std::sync::Arc<tokio::sync::RwLock<std::collections::HashMap<uuid::Uuid, HashSet<Permission>>>>,
}

impl RbacService {
    /// Create new RBAC service
    pub fn new() -> Self {
        Self {
            user_permissions: std::sync::Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            user_denied: std::sync::Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Get all effective permissions for a user
    pub async fn get_permissions(&self, user_id: uuid::Uuid, role: &Role) -> Vec<Permission> {
        let mut permissions: HashSet<Permission> = role.permissions().into_iter().collect();

        // Add any additional permissions
        if let Some(additional) = self.user_permissions.read().await.get(&user_id) {
            permissions.extend(additional.iter().cloned());
        }

        // Remove any denied permissions
        if let Some(denied) = self.user_denied.read().await.get(&user_id) {
            for d in denied {
                permissions.remove(d);
            }
        }

        permissions.into_iter().collect()
    }

    /// Check if user has a specific permission
    pub async fn has_permission(
        &self,
        user_id: uuid::Uuid,
        role: &Role,
        permission: Permission,
    ) -> bool {
        // Admin always has all permissions
        if role.is_admin() {
            return true;
        }

        // Check denied first
        if let Some(denied) = self.user_denied.read().await.get(&user_id) {
            if denied.contains(&permission) {
                return false;
            }
        }

        // Check role permissions
        if role.permissions().contains(&permission) {
            return true;
        }

        // Check additional permissions
        if let Some(additional) = self.user_permissions.read().await.get(&user_id) {
            if additional.contains(&permission) {
                return true;
            }
        }

        false
    }

    /// Grant additional permission to user
    pub async fn grant_permission(&self, user_id: uuid::Uuid, permission: Permission) {
        self.user_permissions
            .write()
            .await
            .entry(user_id)
            .or_default()
            .insert(permission);
    }

    /// Revoke additional permission from user
    pub async fn revoke_permission(&self, user_id: uuid::Uuid, permission: Permission) {
        if let Some(perms) = self.user_permissions.write().await.get_mut(&user_id) {
            perms.remove(&permission);
        }
    }

    /// Deny a permission for user (overrides role)
    pub async fn deny_permission(&self, user_id: uuid::Uuid, permission: Permission) {
        self.user_denied
            .write()
            .await
            .entry(user_id)
            .or_default()
            .insert(permission);
    }

    /// Remove denial for user
    pub async fn remove_denial(&self, user_id: uuid::Uuid, permission: Permission) {
        if let Some(denied) = self.user_denied.write().await.get_mut(&user_id) {
            denied.remove(&permission);
        }
    }
}

impl Default for RbacService {
    fn default() -> Self {
        Self::new()
    }
}

/// Resource ownership check
pub trait Owned {
    fn owner_id(&self) -> uuid::Uuid;
}

/// Check if user can access owned resource
pub fn can_access_resource<T: Owned>(
    user_id: uuid::Uuid,
    role: &Role,
    resource: &T,
    required_permission: Permission,
) -> bool {
    // Admin can access all
    if role.is_admin() {
        return true;
    }

    // Owner can access their own
    if resource.owner_id() == user_id {
        return role.permissions().contains(&required_permission);
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_permissions() {
        let admin_perms = Role::Admin.permissions();
        let dev_perms = Role::Developer.permissions();
        let viewer_perms = Role::Viewer.permissions();

        // Admin has all permissions
        assert!(admin_perms.len() > dev_perms.len());
        // Developer has more than viewer
        assert!(dev_perms.len() > viewer_perms.len());
        // Viewer has read permissions
        assert!(viewer_perms.contains(&Permission::AgentRead));
        assert!(!viewer_perms.contains(&Permission::AgentCreate));
    }

    #[tokio::test]
    async fn test_rbac_service() {
        let service = RbacService::new();
        let user_id = uuid::Uuid::new_v4();

        // Developer should have AgentCreate
        assert!(
            service
                .has_permission(user_id, &Role::Developer, Permission::AgentCreate)
                .await
        );

        // Viewer should not have AgentCreate
        assert!(
            !service
                .has_permission(user_id, &Role::Viewer, Permission::AgentCreate)
                .await
        );

        // Grant additional permission to viewer
        service.grant_permission(user_id, Permission::AgentCreate).await;
        assert!(
            service
                .has_permission(user_id, &Role::Viewer, Permission::AgentCreate)
                .await
        );

        // Deny permission
        service.deny_permission(user_id, Permission::AgentRead).await;
        assert!(
            !service
                .has_permission(user_id, &Role::Viewer, Permission::AgentRead)
                .await
        );
    }

    #[test]
    fn test_role_parsing() {
        assert_eq!(Role::from_str("admin").unwrap(), Role::Admin);
        assert_eq!(Role::from_str("DEVELOPER").unwrap(), Role::Developer);
        assert_eq!(Role::from_str("Viewer").unwrap(), Role::Viewer);
    }
}
