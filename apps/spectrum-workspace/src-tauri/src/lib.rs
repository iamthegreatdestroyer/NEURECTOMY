//! NEURECTOMY Desktop Application
//!
//! Tauri 2.0 backend for the NEURECTOMY platform with native GPU access,
//! file system operations, and system integration.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod gpu;
mod services;
mod state;

use tauri::Manager;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub fn run() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "neurectomy_desktop=debug,tauri=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting NEURECTOMY Desktop Application");

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_os::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .setup(|app| {
            // Initialize application state
            let state = state::AppState::new();
            app.manage(state);
            
            // Open devtools in debug builds
            #[cfg(debug_assertions)]
            {
                if let Some(window) = app.get_webview_window("main") {
                    window.open_devtools();
                    tracing::info!("DevTools opened for debugging");
                }
            }

            // Start embedded backend services
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let mut service_manager = services::ServiceManager::new();
                
                match service_manager.start_all(&handle).await {
                    Ok(_) => {
                        tracing::info!("✓ All embedded backend services started successfully");
                    }
                    Err(e) => {
                        tracing::error!("Failed to start backend services: {}", e);
                        // Don't fail the app, just log the error
                        // The desktop app can still function with limited capabilities
                    }
                }

                // Keep service manager alive for the lifetime of the app
                handle.manage(service_manager);
            });

            tracing::info!("NEURECTOMY Desktop initialized successfully");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_system_info,
            commands::get_gpu_info,
            commands::open_file_dialog,
            commands::save_file_dialog,
            commands::read_project_file,
            commands::write_project_file,
            commands::get_app_data_dir,
            commands::show_notification,
            gpu::check_webgpu_support,
            gpu::get_gpu_memory_info,
        ])
        .run(tauri::generate_context!())
        .expect("error while running NEURECTOMY Desktop");
}
