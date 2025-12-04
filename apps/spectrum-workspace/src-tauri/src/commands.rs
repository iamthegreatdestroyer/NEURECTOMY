//! Tauri Commands for NEURECTOMY Desktop
//!
//! Native commands exposed to the frontend for system operations,
//! file management, and platform integration.

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

use crate::state::AppState;

/// System information structure
#[derive(Debug, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub os_version: String,
    pub arch: String,
    pub hostname: String,
    pub cpu_count: usize,
    pub memory_total: u64,
}

/// Get system information
#[tauri::command]
pub async fn get_system_info() -> Result<SystemInfo, String> {
    let info = SystemInfo {
        os: std::env::consts::OS.to_string(),
        os_version: os_info::get().version().to_string(),
        arch: std::env::consts::ARCH.to_string(),
        hostname: hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| "unknown".to_string()),
        cpu_count: num_cpus::get(),
        memory_total: sys_info::mem_info()
            .map(|m| m.total * 1024)
            .unwrap_or(0),
    };
    
    tracing::debug!("System info: {:?}", info);
    Ok(info)
}

/// GPU information structure
#[derive(Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vendor: String,
    pub driver_version: String,
    pub webgpu_supported: bool,
    pub vulkan_supported: bool,
}

/// Get GPU information
#[tauri::command]
pub async fn get_gpu_info() -> Result<GpuInfo, String> {
    // This is a placeholder - actual GPU detection would use wgpu
    let info = GpuInfo {
        name: "GPU Detection Pending".to_string(),
        vendor: "Unknown".to_string(),
        driver_version: "Unknown".to_string(),
        webgpu_supported: true,
        vulkan_supported: cfg!(any(target_os = "windows", target_os = "linux")),
    };
    
    Ok(info)
}

/// Open file dialog configuration
#[derive(Debug, Deserialize)]
pub struct FileDialogOptions {
    pub title: Option<String>,
    pub default_path: Option<String>,
    pub filters: Option<Vec<FileFilter>>,
}

#[derive(Debug, Deserialize)]
pub struct FileFilter {
    pub name: String,
    pub extensions: Vec<String>,
}

/// Open a file selection dialog
#[tauri::command]
pub async fn open_file_dialog(
    app: AppHandle,
    options: Option<FileDialogOptions>,
) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    
    let mut dialog = app.dialog().file();
    
    if let Some(opts) = options {
        if let Some(title) = opts.title {
            dialog = dialog.set_title(&title);
        }
        if let Some(filters) = opts.filters {
            for filter in filters {
                dialog = dialog.add_filter(&filter.name, &filter.extensions.iter().map(|s| s.as_str()).collect::<Vec<_>>());
            }
        }
    }
    
    let result = dialog.blocking_pick_file();
    
    Ok(result.map(|p| p.to_string()))
}

/// Open a save file dialog
#[tauri::command]
pub async fn save_file_dialog(
    app: AppHandle,
    options: Option<FileDialogOptions>,
) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    
    let mut dialog = app.dialog().file();
    
    if let Some(opts) = options {
        if let Some(title) = opts.title {
            dialog = dialog.set_title(&title);
        }
        if let Some(filters) = opts.filters {
            for filter in filters {
                dialog = dialog.add_filter(&filter.name, &filter.extensions.iter().map(|s| s.as_str()).collect::<Vec<_>>());
            }
        }
    }
    
    let result = dialog.blocking_save_file();
    
    Ok(result.map(|p| p.to_string()))
}

/// Read a project file from the file system
#[tauri::command]
pub async fn read_project_file(path: String) -> Result<String, String> {
    tokio::fs::read_to_string(&path)
        .await
        .map_err(|e| format!("Failed to read file: {}", e))
}

/// Write content to a project file
#[tauri::command]
pub async fn write_project_file(path: String, content: String) -> Result<(), String> {
    // Ensure parent directories exist
    if let Some(parent) = std::path::Path::new(&path).parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("Failed to create directories: {}", e))?;
    }
    
    tokio::fs::write(&path, content)
        .await
        .map_err(|e| format!("Failed to write file: {}", e))
}

/// Get the application data directory
#[tauri::command]
pub async fn get_app_data_dir(app: AppHandle) -> Result<String, String> {
    app.path()
        .app_data_dir()
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| format!("Failed to get app data dir: {}", e))
}

/// Show a native notification
#[tauri::command]
pub async fn show_notification(
    app: AppHandle,
    title: String,
    body: String,
) -> Result<(), String> {
    use tauri_plugin_notification::NotificationExt;
    
    app.notification()
        .builder()
        .title(&title)
        .body(&body)
        .show()
        .map_err(|e| format!("Failed to show notification: {}", e))
}
