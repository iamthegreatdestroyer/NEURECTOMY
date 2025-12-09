//! Tauri Commands for NEURECTOMY Desktop
//!
//! Native commands exposed to the frontend for system operations,
//! file management, and platform integration.

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};
use tokio::fs;

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

/// GitHub repository clone options
#[derive(Debug, Deserialize)]
pub struct GitCloneOptions {
    pub url: String,
    pub target_path: String,
    pub auth_token: Option<String>,
}



/// Get directory contents for file browser
#[derive(Debug, Serialize)]
pub struct DirectoryEntry {
    pub name: String,
    pub path: String,
    pub is_directory: bool,
    pub size: Option<u64>,
    pub modified: Option<String>,
}

#[tauri::command]
pub async fn read_directory(path: String) -> Result<Vec<DirectoryEntry>, String> {
    use tokio::fs;
    use std::path::Path;
    
    let mut entries = Vec::new();
    let mut dir_entries = fs::read_dir(&path)
        .await
        .map_err(|e| format!("Failed to read directory: {}", e))?;
    
    while let Some(entry) = dir_entries.next_entry()
        .await
        .map_err(|e| format!("Failed to read directory entry: {}", e))? {
        
        let metadata = entry.metadata()
            .await
            .map_err(|e| format!("Failed to get metadata: {}", e))?;
        
        let entry_path = entry.path();
        let name = entry_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        entries.push(DirectoryEntry {
            name,
            path: entry_path.to_string_lossy().to_string(),
            is_directory: metadata.is_dir(),
            size: if metadata.is_file() { Some(metadata.len()) } else { None },
            modified: metadata.modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs().to_string()),
        });
    }
    
    // Sort: directories first, then files, alphabetically
    entries.sort_by(|a, b| {
        match (a.is_directory, b.is_directory) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.name.cmp(&b.name),
        }
    });
    
    Ok(entries)
}

/// Get list of drives (Windows-specific)
#[tauri::command]
pub async fn get_drives() -> Result<Vec<String>, String> {
    #[cfg(target_os = "windows")]
    {
        use std::process::Command;
        
        let output = Command::new("wmic")
            .args(&["logicaldisk", "get", "name"])
            .output()
            .map_err(|e| format!("Failed to get drives: {}", e))?;
        
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let drives: Vec<String> = stdout
                .lines()
                .skip(1) // Skip header
                .filter_map(|line| {
                    let drive = line.trim();
                    if drive.ends_with(':') {
                        Some(format!("{}\\", drive))
                    } else {
                        None
                    }
                })
                .collect();
            Ok(drives)
        } else {
            Err("Failed to get drive list".to_string())
        }
    }
    
    #[cfg(not(target_os = "windows"))]
    {
        // For non-Windows systems, return root
        Ok(vec!["/".to_string()])
    }
}

/// Clone a GitHub repository
#[tauri::command]
pub async fn clone_github_repo(repo_url: String, target_path: String) -> Result<String, String> {
    use std::process::Command;

    tracing::info!("Cloning GitHub repo: {} -> {}", repo_url, target_path);

    // Ensure target directory exists
    std::fs::create_dir_all(&target_path)
        .map_err(|e| format!("Failed to create target directory: {}", e))?;

    // Run git clone command
    let output = Command::new("git")
        .args(&["clone", &repo_url, &target_path])
        .output()
        .map_err(|e| format!("Failed to execute git clone: {}", e))?;

    if output.status.success() {
        let message = format!("Successfully cloned repository to {}", target_path);
        tracing::info!("{}", message);
        Ok(message)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let error = format!("Git clone failed: {}", stderr);
        tracing::error!("{}", error);
        Err(error)
    }
}

/// Copy local files/directories to target path
#[tauri::command]
pub async fn copy_local_files(source_paths: Vec<String>, target_path: String) -> Result<String, String> {

    tracing::info!("Copying local files: {:?} -> {}", source_paths, target_path);

    // Ensure target directory exists
    fs::create_dir_all(&target_path)
        .await
        .map_err(|e| format!("Failed to create target directory: {}", e))?;

    let mut copied_files = 0;
    let mut copied_dirs = 0;

    for source_path in source_paths {
        let source_path = std::path::Path::new(&source_path);
        let file_name = source_path
            .file_name()
            .ok_or_else(|| format!("Invalid source path: {}", source_path.display()))?;

        let target_item_path = std::path::Path::new(&target_path).join(file_name);

        if source_path.is_dir() {
            // Copy directory recursively
            copy_dir_recursive(source_path, &target_item_path).await?;
            copied_dirs += 1;
        } else {
            // Copy file
            fs::copy(source_path, &target_item_path)
                .await
                .map_err(|e| format!("Failed to copy file {}: {}", source_path.display(), e))?;
            copied_files += 1;
        }
    }

    let message = format!(
        "Successfully copied {} files and {} directories to {}",
        copied_files, copied_dirs, target_path
    );
    tracing::info!("{}", message);
    Ok(message)
}

/// Recursively copy a directory
async fn copy_dir_recursive(
    source: &std::path::Path,
    target: &std::path::Path,
) -> Result<(), String> {
    use tokio::fs;

    // Create target directory
    fs::create_dir_all(target)
        .await
        .map_err(|e| format!("Failed to create directory {}: {}", target.display(), e))?;

    let mut entries = fs::read_dir(source)
        .await
        .map_err(|e| format!("Failed to read directory {}: {}", source.display(), e))?;

    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|e| format!("Failed to read directory entry: {}", e))?
    {
        let entry_path = entry.path();
        let target_path = target.join(entry.file_name());

        let metadata = entry
            .metadata()
            .await
            .map_err(|e| format!("Failed to get metadata for {}: {}", entry_path.display(), e))?;

        if metadata.is_dir() {
            Box::pin(copy_dir_recursive(&entry_path, &target_path)).await?;
        } else {
            fs::copy(&entry_path, &target_path)
                .await
                .map_err(|e| format!("Failed to copy file {}: {}", entry_path.display(), e))?;
        }
    }

    Ok(())
}
