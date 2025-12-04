//! GPU Detection and WebGPU Support
//!
//! Native GPU capabilities detection for optimal 3D rendering.

use serde::{Deserialize, Serialize};

/// GPU memory information
#[derive(Debug, Serialize, Deserialize)]
pub struct GpuMemoryInfo {
    pub total_mb: u64,
    pub available_mb: u64,
    pub used_mb: u64,
}

/// Check if WebGPU is supported on this system
#[tauri::command]
pub async fn check_webgpu_support() -> Result<bool, String> {
    // WebGPU support check - in production this would use wgpu
    // For now, assume support on modern systems
    
    #[cfg(target_os = "windows")]
    {
        // Windows 10 version 2004+ supports WebGPU via D3D12
        Ok(true)
    }
    
    #[cfg(target_os = "macos")]
    {
        // macOS 10.15+ supports WebGPU via Metal
        Ok(true)
    }
    
    #[cfg(target_os = "linux")]
    {
        // Linux support depends on Vulkan
        // Check if Vulkan is available
        Ok(std::path::Path::new("/usr/lib/libvulkan.so").exists() ||
           std::path::Path::new("/usr/lib/x86_64-linux-gnu/libvulkan.so").exists())
    }
    
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        Ok(false)
    }
}

/// Get GPU memory information
#[tauri::command]
pub async fn get_gpu_memory_info() -> Result<GpuMemoryInfo, String> {
    // This is a placeholder - actual implementation would query GPU
    // Using wgpu or platform-specific APIs
    
    // For now, return placeholder values
    // In production, this would use:
    // - NVML for NVIDIA
    // - ADL for AMD
    // - Metal API for Apple
    
    Ok(GpuMemoryInfo {
        total_mb: 8192,  // 8GB placeholder
        available_mb: 6144,
        used_mb: 2048,
    })
}

/// GPU adapter information for WebGPU
#[derive(Debug, Serialize, Deserialize)]
pub struct GpuAdapter {
    pub name: String,
    pub vendor: String,
    pub device_type: String,
    pub backend: String,
    pub features: Vec<String>,
    pub limits: GpuLimits,
}

/// GPU limits for WebGPU
#[derive(Debug, Serialize, Deserialize)]
pub struct GpuLimits {
    pub max_texture_dimension_1d: u32,
    pub max_texture_dimension_2d: u32,
    pub max_texture_dimension_3d: u32,
    pub max_buffer_size: u64,
    pub max_bind_groups: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
}

impl Default for GpuLimits {
    fn default() -> Self {
        Self {
            max_texture_dimension_1d: 8192,
            max_texture_dimension_2d: 8192,
            max_texture_dimension_3d: 2048,
            max_buffer_size: 1 << 30, // 1GB
            max_bind_groups: 4,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
        }
    }
}
