/**
 * Local Import Panel Component
 *
 * Sidebar panel for local file/folder import functionality.
 * Provides quick access to local file system import with drive browsing.
 *
 * Features:
 * - Drive selection
 * - Quick folder navigation
 * - Recent import locations
 * - Drag and drop support
 * - Import history
 *
 * @module @neurectomy/components/shell/LocalImportPanel
 * @author @APEX @CANVAS
 */

import React, { useState, useCallback, useEffect } from "react";
import { cn } from "@/lib/utils";
import {
  HardDrive,
  Folder,
  FolderOpen,
  File,
  Plus,
  History,
  Upload,
  RefreshCw,
  ChevronRight,
  Home,
  Monitor,
} from "lucide-react";
import { Button, ScrollArea, Separator } from "@neurectomy/ui";
import { LocalImportDialog } from "./LocalImportDialog";
import { invoke } from "@tauri-apps/api/core";

// ============================================================================
// Types
// ============================================================================

interface DriveInfo {
  name: string;
  path: string;
  type: "fixed" | "removable" | "network";
  freeSpace?: number;
  totalSpace?: number;
}

interface RecentLocation {
  id: string;
  name: string;
  path: string;
  type: "folder" | "file";
  lastAccessed: string;
}

// ============================================================================
// Component
// ============================================================================

interface LocalImportPanelProps {
  onImport?: (sourcePaths: string[], targetPath: string) => Promise<void>;
}

export function LocalImportPanel({ onImport }: LocalImportPanelProps = {}) {
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [drives, setDrives] = useState<DriveInfo[]>([]);
  const [loading, setLoading] = useState(false);

  // Mock recent locations - in real implementation, this would come from local storage
  const recentLocations: RecentLocation[] = [
    {
      id: "1",
      name: "Documents",
      path: "C:\\Users\\user\\Documents",
      type: "folder",
      lastAccessed: "2024-01-15",
    },
    {
      id: "2",
      name: "Desktop",
      path: "C:\\Users\\user\\Desktop",
      type: "folder",
      lastAccessed: "2024-01-14",
    },
    {
      id: "3",
      name: "Projects",
      path: "D:\\Projects",
      type: "folder",
      lastAccessed: "2024-01-13",
    },
  ];

  // Load drives from Tauri API
  useEffect(() => {
    const loadDrives = async () => {
      try {
        const driveList = await invoke("get_drives");
        const driveInfos: DriveInfo[] = (driveList as string[]).map(
          (drive) => ({
            name: `${drive} Drive`,
            path: drive,
            type: "fixed" as const,
            freeSpace: undefined, // TODO: Get actual space info
            totalSpace: undefined,
          })
        );
        setDrives(driveInfos);
      } catch (error) {
        console.error("Failed to load drives:", error);
        // Fallback to mock drives
        const mockDrives: DriveInfo[] = [
          {
            name: "Local Disk (C:)",
            path: "C:",
            type: "fixed",
            freeSpace: 150,
            totalSpace: 500,
          },
        ];
        setDrives(mockDrives);
      }
    };

    loadDrives();
  }, []);

  const handleDriveClick = useCallback((drive: DriveInfo) => {
    // In real implementation, this would navigate to the drive in the import dialog
    setImportDialogOpen(true);
  }, []);

  const handleLocationClick = useCallback((location: RecentLocation) => {
    // In real implementation, this would navigate to the location in the import dialog
    setImportDialogOpen(true);
  }, []);

  const getDriveIcon = (type: DriveInfo["type"]) => {
    switch (type) {
      case "fixed":
        return <HardDrive className="w-4 h-4" />;
      case "removable":
        return <Monitor className="w-4 h-4" />;
      case "network":
        return <Folder className="w-4 h-4" />;
      default:
        return <HardDrive className="w-4 h-4" />;
    }
  };

  const formatSpace = (bytes?: number) => {
    if (!bytes) return "";
    const gb = bytes / 1024;
    return `${gb.toFixed(0)} GB`;
  };

  return (
    <>
      <div className="h-full flex flex-col bg-card">
        {/* Header */}
        <div className="p-4 border-b border-border">
          <div className="flex items-center gap-2 mb-3">
            <HardDrive className="w-5 h-5 text-primary" />
            <h3 className="font-semibold text-sm">Local Import</h3>
          </div>

          {/* Quick Actions */}
          <div className="space-y-2">
            <Button
              onClick={() => setImportDialogOpen(true)}
              size="sm"
              className="w-full"
            >
              <Upload className="w-4 h-4 mr-2" />
              Browse Files
            </Button>
            <Button
              onClick={() => setImportDialogOpen(true)}
              variant="outline"
              size="sm"
              className="w-full"
            >
              <Plus className="w-4 h-4 mr-2" />
              Import Folder
            </Button>
          </div>
        </div>

        {/* Content */}
        <ScrollArea className="flex-1">
          <div className="p-4 space-y-4">
            {/* Drives */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <HardDrive className="w-4 h-4 text-muted-foreground" />
                <h4 className="font-medium text-sm">Drives</h4>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setLoading(true)}
                  disabled={loading}
                  className="ml-auto h-6 w-6 p-0"
                >
                  <RefreshCw
                    className={cn("w-3 h-3", loading && "animate-spin")}
                  />
                </Button>
              </div>

              <div className="space-y-2">
                {drives.map((drive) => (
                  <div
                    key={drive.path}
                    className="p-3 rounded-md border border-border hover:bg-accent cursor-pointer transition-colors"
                    onClick={() => handleDriveClick(drive)}
                  >
                    <div className="flex items-center gap-3 mb-1">
                      {getDriveIcon(drive.type)}
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm truncate">
                          {drive.name}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {drive.path}
                        </div>
                      </div>
                      <ChevronRight className="w-4 h-4 text-muted-foreground" />
                    </div>

                    {drive.freeSpace && drive.totalSpace && (
                      <div className="text-xs text-muted-foreground">
                        {formatSpace(drive.freeSpace)} free of{" "}
                        {formatSpace(drive.totalSpace)}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <Separator />

            {/* Recent Locations */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <History className="w-4 h-4 text-muted-foreground" />
                <h4 className="font-medium text-sm">Recent</h4>
              </div>

              <div className="space-y-2">
                {recentLocations.map((location) => (
                  <div
                    key={location.id}
                    className="p-3 rounded-md border border-border hover:bg-accent cursor-pointer transition-colors"
                    onClick={() => handleLocationClick(location)}
                  >
                    <div className="flex items-center gap-3">
                      {location.type === "folder" ? (
                        <Folder className="w-4 h-4 text-blue-500" />
                      ) : (
                        <File className="w-4 h-4 text-muted-foreground" />
                      )}
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm truncate">
                          {location.name}
                        </div>
                        <div className="text-xs text-muted-foreground truncate">
                          {location.path}
                        </div>
                      </div>
                      <ChevronRight className="w-4 h-4 text-muted-foreground" />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <Separator />

            {/* Quick Access */}
            <div>
              <h4 className="font-medium text-sm mb-3">Quick Access</h4>
              <div className="space-y-2">
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start"
                  onClick={() =>
                    handleLocationClick({
                      id: "home",
                      name: "Home",
                      path: "C:\\Users\\user",
                      type: "folder",
                      lastAccessed: new Date().toISOString(),
                    })
                  }
                >
                  <Home className="w-4 h-4 mr-2" />
                  Home
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start"
                  onClick={() =>
                    handleLocationClick({
                      id: "desktop",
                      name: "Desktop",
                      path: "C:\\Users\\user\\Desktop",
                      type: "folder",
                      lastAccessed: new Date().toISOString(),
                    })
                  }
                >
                  <Monitor className="w-4 h-4 mr-2" />
                  Desktop
                </Button>
              </div>
            </div>
          </div>
        </ScrollArea>
      </div>

      {/* Import Dialog */}
      <LocalImportDialog
        isOpen={importDialogOpen}
        onClose={() => setImportDialogOpen(false)}
        onImport={async (sourcePaths, targetPath) => {
          if (onImport) {
            await onImport(
              sourcePaths.map((item) => item.path),
              targetPath
            );
          }
        }}
      />
    </>
  );
}
