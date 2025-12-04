/**
 * Tauri React Hooks
 *
 * React hooks for integrating with Tauri native functionality.
 */

import { useState, useEffect, useCallback } from "react";
import {
  isTauri,
  getSystemInfo,
  getGpuInfo,
  getGpuMemoryInfo,
  checkForUpdates,
  type SystemInfo,
  type GpuInfo,
  type GpuMemoryInfo,
  type UpdateInfo,
} from "./index";

// ============================================================================
// PLATFORM DETECTION HOOK
// ============================================================================

export interface PlatformInfo {
  isTauri: boolean;
  isWeb: boolean;
  platform: "windows" | "macos" | "linux" | "web" | "unknown";
}

/**
 * Hook to detect the current platform
 */
export function usePlatform(): PlatformInfo {
  const [platform, setPlatform] = useState<PlatformInfo>({
    isTauri: false,
    isWeb: true,
    platform: "web",
  });

  useEffect(() => {
    const detect = async () => {
      const inTauri = isTauri();
      if (inTauri) {
        const { platform: osPlatform } = await import("@tauri-apps/plugin-os");
        const os = await osPlatform();
        setPlatform({
          isTauri: true,
          isWeb: false,
          platform: os as "windows" | "macos" | "linux",
        });
      } else {
        setPlatform({
          isTauri: false,
          isWeb: true,
          platform: "web",
        });
      }
    };
    detect();
  }, []);

  return platform;
}

// ============================================================================
// SYSTEM INFO HOOK
// ============================================================================

export interface UseSystemInfoResult {
  systemInfo: SystemInfo | null;
  gpuInfo: GpuInfo | null;
  gpuMemory: GpuMemoryInfo | null;
  loading: boolean;
  error: Error | null;
  refresh: () => void;
}

/**
 * Hook to get system and GPU information
 */
export function useSystemInfo(): UseSystemInfoResult {
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [gpuInfo, setGpuInfo] = useState<GpuInfo | null>(null);
  const [gpuMemory, setGpuMemory] = useState<GpuMemoryInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchInfo = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [sys, gpu, mem] = await Promise.all([
        getSystemInfo(),
        getGpuInfo(),
        getGpuMemoryInfo(),
      ]);
      setSystemInfo(sys);
      setGpuInfo(gpu);
      setGpuMemory(mem);
    } catch (e) {
      setError(e instanceof Error ? e : new Error("Unknown error"));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchInfo();
  }, [fetchInfo]);

  return {
    systemInfo,
    gpuInfo,
    gpuMemory,
    loading,
    error,
    refresh: fetchInfo,
  };
}

// ============================================================================
// AUTO-UPDATE HOOK
// ============================================================================

export interface UseUpdateResult {
  updateAvailable: boolean;
  updateInfo: UpdateInfo | null;
  checking: boolean;
  installing: boolean;
  checkNow: () => void;
  install: () => void;
}

/**
 * Hook to manage application updates
 */
export function useAutoUpdate(): UseUpdateResult {
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null);
  const [checking, setChecking] = useState(false);
  const [installing, setInstalling] = useState(false);

  const checkNow = useCallback(async () => {
    if (!isTauri()) return;
    setChecking(true);
    try {
      const update = await checkForUpdates();
      setUpdateInfo(update);
    } catch (e) {
      console.error("Update check failed:", e);
    } finally {
      setChecking(false);
    }
  }, []);

  const install = useCallback(async () => {
    if (!isTauri() || !updateInfo) return;
    setInstalling(true);
    try {
      const { check } = await import("@tauri-apps/plugin-updater");
      const update = await check();
      if (update?.available) {
        await update.downloadAndInstall();
      }
    } catch (e) {
      console.error("Update installation failed:", e);
    } finally {
      setInstalling(false);
    }
  }, [updateInfo]);

  // Check on mount
  useEffect(() => {
    checkNow();
    // Check every hour
    const interval = setInterval(checkNow, 60 * 60 * 1000);
    return () => clearInterval(interval);
  }, [checkNow]);

  return {
    updateAvailable: !!updateInfo,
    updateInfo,
    checking,
    installing,
    checkNow,
    install,
  };
}

// ============================================================================
// WINDOW STATE HOOK
// ============================================================================

export interface WindowState {
  isMaximized: boolean;
  isFullscreen: boolean;
  isFocused: boolean;
}

/**
 * Hook to track window state
 */
export function useWindowState(): WindowState {
  const [state, setState] = useState<WindowState>({
    isMaximized: false,
    isFullscreen: false,
    isFocused: true,
  });

  useEffect(() => {
    if (!isTauri()) return;

    let unlisten: (() => void) | undefined;

    const setup = async () => {
      const { getCurrentWindow } = await import("@tauri-apps/api/window");
      const win = getCurrentWindow();

      // Get initial state
      const [isMaximized, isFullscreen, isFocused] = await Promise.all([
        win.isMaximized(),
        win.isFullscreen(),
        win.isFocused(),
      ]);
      setState({ isMaximized, isFullscreen, isFocused });

      // Listen for changes
      const unlistenResize = await win.onResized(async () => {
        const [max, full] = await Promise.all([
          win.isMaximized(),
          win.isFullscreen(),
        ]);
        setState((s) => ({ ...s, isMaximized: max, isFullscreen: full }));
      });

      const unlistenFocus = await win.onFocusChanged(({ payload }) => {
        setState((s) => ({ ...s, isFocused: payload }));
      });

      unlisten = () => {
        unlistenResize();
        unlistenFocus();
      };
    };

    setup();

    return () => unlisten?.();
  }, []);

  return state;
}

// ============================================================================
// KEYBOARD SHORTCUTS HOOK
// ============================================================================

type ShortcutCallback = () => void;
type ShortcutMap = Record<string, ShortcutCallback>;

/**
 * Hook to register keyboard shortcuts
 */
export function useKeyboardShortcuts(shortcuts: ShortcutMap): void {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const key = getShortcutKey(event);
      const callback = shortcuts[key];
      if (callback) {
        event.preventDefault();
        callback();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [shortcuts]);
}

function getShortcutKey(event: KeyboardEvent): string {
  const parts: string[] = [];
  if (event.ctrlKey || event.metaKey) parts.push("Ctrl");
  if (event.altKey) parts.push("Alt");
  if (event.shiftKey) parts.push("Shift");
  parts.push(event.key.toUpperCase());
  return parts.join("+");
}
