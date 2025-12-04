/**
 * Tauri Desktop Integration
 *
 * Native commands and utilities for the NEURECTOMY desktop application.
 * These functions are only available when running as a Tauri app.
 */

import { invoke } from "@tauri-apps/api/core";

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface SystemInfo {
  os: string;
  os_version: string;
  arch: string;
  hostname: string;
  cpu_count: number;
  memory_total: number;
}

export interface GpuInfo {
  name: string;
  vendor: string;
  driver_version: string;
  webgpu_supported: boolean;
  vulkan_supported: boolean;
}

export interface GpuMemoryInfo {
  total_mb: number;
  available_mb: number;
  used_mb: number;
}

export interface FileDialogOptions {
  title?: string;
  default_path?: string;
  filters?: FileFilter[];
}

export interface FileFilter {
  name: string;
  extensions: string[];
}

// ============================================================================
// ENVIRONMENT DETECTION
// ============================================================================

/**
 * Check if running in Tauri environment
 */
export function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI__" in window;
}

/**
 * Get Tauri version if available
 */
export async function getTauriVersion(): Promise<string | null> {
  if (!isTauri()) return null;
  try {
    const { getVersion } = await import("@tauri-apps/api/app");
    return await getVersion();
  } catch {
    return null;
  }
}

// ============================================================================
// SYSTEM COMMANDS
// ============================================================================

/**
 * Get system information
 */
export async function getSystemInfo(): Promise<SystemInfo | null> {
  if (!isTauri()) return null;
  try {
    return await invoke<SystemInfo>("get_system_info");
  } catch (error) {
    console.error("Failed to get system info:", error);
    return null;
  }
}

/**
 * Get GPU information
 */
export async function getGpuInfo(): Promise<GpuInfo | null> {
  if (!isTauri()) return null;
  try {
    return await invoke<GpuInfo>("get_gpu_info");
  } catch (error) {
    console.error("Failed to get GPU info:", error);
    return null;
  }
}

/**
 * Check WebGPU support
 */
export async function checkWebGpuSupport(): Promise<boolean> {
  if (!isTauri()) {
    // Fallback to browser detection
    return "gpu" in navigator;
  }
  try {
    return await invoke<boolean>("check_webgpu_support");
  } catch {
    return false;
  }
}

/**
 * Get GPU memory info
 */
export async function getGpuMemoryInfo(): Promise<GpuMemoryInfo | null> {
  if (!isTauri()) return null;
  try {
    return await invoke<GpuMemoryInfo>("get_gpu_memory_info");
  } catch (error) {
    console.error("Failed to get GPU memory info:", error);
    return null;
  }
}

// ============================================================================
// FILE OPERATIONS
// ============================================================================

/**
 * Open a file selection dialog
 */
export async function openFileDialog(
  options?: FileDialogOptions
): Promise<string | null> {
  if (!isTauri()) {
    // Fallback to browser file input
    return new Promise((resolve) => {
      const input = document.createElement("input");
      input.type = "file";
      if (options?.filters) {
        input.accept = options.filters
          .flatMap((f) => f.extensions.map((e) => `.${e}`))
          .join(",");
      }
      input.onchange = () => {
        const file = input.files?.[0];
        resolve(file?.name ?? null);
      };
      input.click();
    });
  }

  try {
    return await invoke<string | null>("open_file_dialog", { options });
  } catch (error) {
    console.error("Failed to open file dialog:", error);
    return null;
  }
}

/**
 * Open a save file dialog
 */
export async function saveFileDialog(
  options?: FileDialogOptions
): Promise<string | null> {
  if (!isTauri()) {
    console.warn("Save file dialog not available in browser");
    return null;
  }

  try {
    return await invoke<string | null>("save_file_dialog", { options });
  } catch (error) {
    console.error("Failed to open save dialog:", error);
    return null;
  }
}

/**
 * Read a project file
 */
export async function readProjectFile(path: string): Promise<string | null> {
  if (!isTauri()) {
    console.warn("File reading not available in browser");
    return null;
  }

  try {
    return await invoke<string>("read_project_file", { path });
  } catch (error) {
    console.error("Failed to read file:", error);
    return null;
  }
}

/**
 * Write content to a project file
 */
export async function writeProjectFile(
  path: string,
  content: string
): Promise<boolean> {
  if (!isTauri()) {
    console.warn("File writing not available in browser");
    return false;
  }

  try {
    await invoke("write_project_file", { path, content });
    return true;
  } catch (error) {
    console.error("Failed to write file:", error);
    return false;
  }
}

/**
 * Get the application data directory
 */
export async function getAppDataDir(): Promise<string | null> {
  if (!isTauri()) return null;
  try {
    return await invoke<string>("get_app_data_dir");
  } catch (error) {
    console.error("Failed to get app data dir:", error);
    return null;
  }
}

// ============================================================================
// NOTIFICATIONS
// ============================================================================

/**
 * Show a native notification
 */
export async function showNotification(
  title: string,
  body: string
): Promise<boolean> {
  if (!isTauri()) {
    // Fallback to browser notifications
    if ("Notification" in window) {
      const permission = await Notification.requestPermission();
      if (permission === "granted") {
        new Notification(title, { body });
        return true;
      }
    }
    return false;
  }

  try {
    await invoke("show_notification", { title, body });
    return true;
  } catch (error) {
    console.error("Failed to show notification:", error);
    return false;
  }
}

// ============================================================================
// WINDOW MANAGEMENT
// ============================================================================

/**
 * Minimize the window
 */
export async function minimizeWindow(): Promise<void> {
  if (!isTauri()) return;
  try {
    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    await getCurrentWindow().minimize();
  } catch (error) {
    console.error("Failed to minimize window:", error);
  }
}

/**
 * Maximize/restore the window
 */
export async function toggleMaximize(): Promise<void> {
  if (!isTauri()) return;
  try {
    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    await getCurrentWindow().toggleMaximize();
  } catch (error) {
    console.error("Failed to toggle maximize:", error);
  }
}

/**
 * Close the window
 */
export async function closeWindow(): Promise<void> {
  if (!isTauri()) return;
  try {
    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    await getCurrentWindow().close();
  } catch (error) {
    console.error("Failed to close window:", error);
  }
}

/**
 * Set window title
 */
export async function setWindowTitle(title: string): Promise<void> {
  if (!isTauri()) {
    document.title = title;
    return;
  }
  try {
    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    await getCurrentWindow().setTitle(title);
  } catch (error) {
    console.error("Failed to set window title:", error);
  }
}

/**
 * Toggle fullscreen
 */
export async function toggleFullscreen(): Promise<void> {
  if (!isTauri()) {
    // Browser fullscreen
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      await document.documentElement.requestFullscreen();
    }
    return;
  }
  try {
    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    const win = getCurrentWindow();
    const isFullscreen = await win.isFullscreen();
    await win.setFullscreen(!isFullscreen);
  } catch (error) {
    console.error("Failed to toggle fullscreen:", error);
  }
}

// ============================================================================
// CLIPBOARD
// ============================================================================

/**
 * Copy text to clipboard
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (isTauri()) {
      const { writeText } =
        await import("@tauri-apps/plugin-clipboard-manager");
      await writeText(text);
    } else {
      await navigator.clipboard.writeText(text);
    }
    return true;
  } catch (error) {
    console.error("Failed to copy to clipboard:", error);
    return false;
  }
}

/**
 * Read text from clipboard
 */
export async function readFromClipboard(): Promise<string | null> {
  try {
    if (isTauri()) {
      const { readText } = await import("@tauri-apps/plugin-clipboard-manager");
      return await readText();
    } else {
      return await navigator.clipboard.readText();
    }
  } catch (error) {
    console.error("Failed to read from clipboard:", error);
    return null;
  }
}

// ============================================================================
// AUTO-UPDATE
// ============================================================================

export interface UpdateInfo {
  version: string;
  date: string;
  body: string;
}

/**
 * Check for updates
 */
export async function checkForUpdates(): Promise<UpdateInfo | null> {
  if (!isTauri()) return null;
  try {
    const { check } = await import("@tauri-apps/plugin-updater");
    const update = await check();
    if (update?.available) {
      return {
        version: update.version,
        date: update.date ?? "",
        body: update.body ?? "",
      };
    }
    return null;
  } catch (error) {
    console.error("Failed to check for updates:", error);
    return null;
  }
}

/**
 * Download and install an update
 */
export async function installUpdate(): Promise<boolean> {
  if (!isTauri()) return false;
  try {
    const { check } = await import("@tauri-apps/plugin-updater");
    const update = await check();
    if (update?.available) {
      await update.downloadAndInstall();
      return true;
    }
    return false;
  } catch (error) {
    console.error("Failed to install update:", error);
    return false;
  }
}
