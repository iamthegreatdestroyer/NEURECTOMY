//! Application State Management
//!
//! Global state shared across the Tauri application.

use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// Recent project entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentProject {
    pub path: String,
    pub name: String,
    pub last_opened: chrono::DateTime<chrono::Utc>,
}

/// Application settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub theme: String,
    pub auto_save: bool,
    pub auto_save_interval: u64,
    pub gpu_acceleration: bool,
    pub max_recent_projects: usize,
    pub telemetry_enabled: bool,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            theme: "dark".to_string(),
            auto_save: true,
            auto_save_interval: 60, // seconds
            gpu_acceleration: true,
            max_recent_projects: 10,
            telemetry_enabled: false,
        }
    }
}

/// Inner state data
#[derive(Debug, Default)]
struct StateData {
    current_project: Option<String>,
    recent_projects: Vec<RecentProject>,
    settings: AppSettings,
    is_dirty: bool,
}

/// Application state wrapper
#[derive(Debug)]
pub struct AppState {
    inner: Arc<RwLock<StateData>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(StateData::default())),
        }
    }
    
    /// Get current project path
    pub async fn current_project(&self) -> Option<String> {
        self.inner.read().await.current_project.clone()
    }
    
    /// Set current project
    pub async fn set_current_project(&self, path: Option<String>) {
        let mut state = self.inner.write().await;
        state.current_project = path.clone();
        
        // Add to recent projects
        if let Some(p) = path {
            let name = std::path::Path::new(&p)
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "Untitled".to_string());
            
            // Remove if already exists
            state.recent_projects.retain(|r| r.path != p);
            
            // Add to front
            state.recent_projects.insert(0, RecentProject {
                path: p,
                name,
                last_opened: chrono::Utc::now(),
            });
            
            // Trim to max
            let max = state.settings.max_recent_projects;
            state.recent_projects.truncate(max);
        }
    }
    
    /// Get recent projects
    pub async fn recent_projects(&self) -> Vec<RecentProject> {
        self.inner.read().await.recent_projects.clone()
    }
    
    /// Get settings
    pub async fn settings(&self) -> AppSettings {
        self.inner.read().await.settings.clone()
    }
    
    /// Update settings
    pub async fn update_settings(&self, settings: AppSettings) {
        self.inner.write().await.settings = settings;
    }
    
    /// Check if state is dirty (has unsaved changes)
    pub async fn is_dirty(&self) -> bool {
        self.inner.read().await.is_dirty
    }
    
    /// Mark state as dirty
    pub async fn set_dirty(&self, dirty: bool) {
        self.inner.write().await.is_dirty = dirty;
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
