/**
 * Local Import Dialog Component
 *
 * Modal dialog for importing local files and folders with:
 * - File/folder browser with tree view
 * - Drag & drop support
 * - Multiple selection
 * - File type filtering
 * - Recent locations
 *
 * @module @neurectomy/components
 * @author @CANVAS @APEX
 */

import { useState, useCallback, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Folder,
  File,
  FileText,
  Image,
  Code,
  Archive,
  Upload,
  X,
  ChevronRight,
  ChevronDown,
  Home,
  HardDrive,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Button,
  Input,
  Label,
  Checkbox,
  Progress,
  Alert,
  AlertDescription,
  Badge,
  ScrollArea,
} from "@neurectomy/ui";
import { MotionDiv } from "@/lib/motion";
import { invoke } from "@tauri-apps/api/core";

// =============================================================================
// Types
// =============================================================================

export interface FileSystemItem {
  name: string;
  path: string;
  type: "file" | "folder";
  size?: number;
  modified?: Date;
  extension?: string;
  children?: FileSystemItem[];
  expanded?: boolean;
  selected?: boolean;
}

export interface ImportProgress {
  phase: "scanning" | "copying" | "indexing" | "complete";
  progress: number;
  message: string;
  currentItem?: string;
  totalItems?: number;
  processedItems?: number;
}

export interface LocalImportDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onImport: (items: FileSystemItem[], destinationPath: string) => Promise<void>;
}

// =============================================================================
// File System API Integration
// =============================================================================

class FileSystemAPI {
  async getDrives(): Promise<string[]> {
    // Call Tauri command to get available drives
    const result = await invoke("get_drives");
    return result as string[];
  }

  async readDirectory(path: string): Promise<FileSystemItem[]> {
    // Call Tauri command to read directory
    const result = await invoke("read_directory", { path });
    return result as FileSystemItem[];
  }

  async getFileInfo(path: string): Promise<FileSystemItem> {
    // Call Tauri command to get file info
    const result = await invoke("read_directory", { path });
    // For single file, return the first item
    const items = result as FileSystemItem[];
    return items[0];
  }

  async copyItems(items: FileSystemItem[], destination: string): Promise<void> {
    // Call Tauri command to copy files
    const sourcePaths = items.map((item) => item.path);
    await invoke("copy_local_files", {
      source_paths: sourcePaths,
      target_path: destination,
    });
  }
}

// =============================================================================
// File Icon Component
// =============================================================================

interface FileIconProps {
  filename: string;
  className?: string;
}

function FileIcon({ filename, className }: FileIconProps) {
  const extension = filename.split(".").pop()?.toLowerCase();

  const iconMap: Record<string, any> = {
    // Code files
    js: Code,
    ts: Code,
    jsx: Code,
    tsx: Code,
    py: Code,
    java: Code,
    cpp: Code,
    c: Code,
    cs: Code,
    php: Code,
    rb: Code,
    go: Code,
    rs: Code,
    swift: Code,
    kt: Code,

    // Documents
    txt: FileText,
    md: FileText,
    pdf: FileText,
    doc: FileText,
    docx: FileText,

    // Images
    jpg: Image,
    jpeg: Image,
    png: Image,
    gif: Image,
    svg: Image,
    webp: Image,

    // Archives
    zip: Archive,
    rar: Archive,
    tar: Archive,
    gz: Archive,
    "7z": Archive,
  };

  const Icon = iconMap[extension || ""] || File;

  return <Icon className={cn("w-4 h-4", className)} />;
}

// =============================================================================
// File Tree Node Component
// =============================================================================

interface FileTreeNodeProps {
  item: FileSystemItem;
  level: number;
  onToggle: (path: string) => void;
  onSelect: (path: string, selected: boolean) => void;
  selectedPaths: Set<string>;
}

function FileTreeNode({
  item,
  level,
  onToggle,
  onSelect,
  selectedPaths,
}: FileTreeNodeProps) {
  const isExpanded = item.expanded;
  const isSelected = selectedPaths.has(item.path);
  const hasChildren = item.children && item.children.length > 0;

  const handleToggle = useCallback(() => {
    if (hasChildren) {
      onToggle(item.path);
    }
  }, [hasChildren, item.path, onToggle]);

  const handleSelect = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onSelect(item.path, !isSelected);
    },
    [item.path, isSelected, onSelect]
  );

  return (
    <div>
      <div
        className={cn(
          "flex items-center gap-2 py-1 px-2 hover:bg-muted/50 cursor-pointer rounded-sm",
          isSelected && "bg-primary/10"
        )}
        style={{ paddingLeft: `${level * 16 + 8}px` }}
        onClick={handleToggle}
      >
        {/* Expand/Collapse Icon */}
        {item.type === "folder" && hasChildren && (
          <div className="w-4 h-4 flex items-center justify-center">
            {isExpanded ? (
              <ChevronDown className="w-3 h-3" />
            ) : (
              <ChevronRight className="w-3 h-3" />
            )}
          </div>
        )}
        {item.type === "folder" && !hasChildren && <div className="w-4 h-4" />}

        {/* Selection Checkbox */}
        <Checkbox
          checked={isSelected}
          onCheckedChange={(checked) => onSelect(item.path, checked as boolean)}
          onClick={(e) => e.stopPropagation()}
        />

        {/* File/Folder Icon */}
        {item.type === "folder" ? (
          <Folder className="w-4 h-4 text-blue-500" />
        ) : (
          <FileIcon filename={item.name} />
        )}

        {/* Name */}
        <span className="text-sm truncate flex-1">{item.name}</span>

        {/* Size */}
        {item.size && (
          <span className="text-xs text-muted-foreground">
            {(item.size / 1024).toFixed(1)} KB
          </span>
        )}
      </div>

      {/* Children */}
      {isExpanded && item.children && (
        <div>
          {item.children.map((child) => (
            <FileTreeNode
              key={child.path}
              item={child}
              level={level + 1}
              onToggle={onToggle}
              onSelect={onSelect}
              selectedPaths={selectedPaths}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// File Tree Component
// =============================================================================

interface FileTreeProps {
  rootPath: string;
  items: FileSystemItem[];
  selectedPaths: Set<string>;
  onToggle: (path: string) => void;
  onSelect: (path: string, selected: boolean) => void;
  onRefresh: () => void;
  isLoading: boolean;
}

function FileTree({
  rootPath,
  items,
  selectedPaths,
  onToggle,
  onSelect,
  onRefresh,
  isLoading,
}: FileTreeProps) {
  return (
    <div className="border border-border rounded-lg">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-border">
        <div className="flex items-center gap-2">
          <HardDrive className="w-4 h-4" />
          <span className="text-sm font-medium">{rootPath}</span>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={onRefresh}
          disabled={isLoading}
        >
          <RefreshCw className={cn("w-4 h-4", isLoading && "animate-spin")} />
        </Button>
      </div>

      {/* Tree */}
      <ScrollArea className="h-64">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-6 h-6 animate-spin" />
          </div>
        ) : (
          <div className="p-2">
            {items.map((item) => (
              <FileTreeNode
                key={item.path}
                item={item}
                level={0}
                onToggle={onToggle}
                onSelect={onSelect}
                selectedPaths={selectedPaths}
              />
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  );
}

// =============================================================================
// Main Dialog Component
// =============================================================================

export function LocalImportDialog({
  isOpen,
  onClose,
  onImport,
}: LocalImportDialogProps) {
  // --------------------------------
  // State
  // --------------------------------
  const [currentPath, setCurrentPath] = useState("C:");
  const [fileTree, setFileTree] = useState<FileSystemItem[]>([]);
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set());
  const [destinationPath, setDestinationPath] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [importProgress, setImportProgress] = useState<ImportProgress | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);
  const [drives, setDrives] = useState<string[]>([]);

  const fileSystemAPI = new FileSystemAPI();
  const fileInputRef = useRef<HTMLInputElement>(null);

  // --------------------------------
  // Effects
  // --------------------------------
  useEffect(() => {
    if (isOpen) {
      // Reset state when dialog opens
      setSelectedPaths(new Set());
      setError(null);
      setImportProgress(null);
      setIsImporting(false);
      loadDrives();
      loadDirectory("C:");
    }
  }, [isOpen]);

  // --------------------------------
  // API Calls
  // --------------------------------
  const loadDrives = useCallback(async () => {
    try {
      const driveList = await fileSystemAPI.getDrives();
      setDrives(driveList);
    } catch (err) {
      setError("Failed to load drives");
    }
  }, []);

  const loadDirectory = useCallback(async (path: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const items = await fileSystemAPI.readDirectory(path);
      setFileTree(items);
      setCurrentPath(path);
    } catch (err) {
      setError(`Failed to load directory: ${path}`);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // --------------------------------
  // Handlers
  // --------------------------------
  const handleToggle = useCallback((path: string) => {
    setFileTree((prev) =>
      prev.map((item) => {
        if (item.path === path) {
          return { ...item, expanded: !item.expanded };
        }
        return item;
      })
    );
  }, []);

  const handleSelect = useCallback((path: string, selected: boolean) => {
    setSelectedPaths((prev) => {
      const newSet = new Set(prev);
      if (selected) {
        newSet.add(path);
      } else {
        newSet.delete(path);
      }
      return newSet;
    });
  }, []);

  const handleDriveChange = useCallback(
    (drive: string) => {
      loadDirectory(drive);
    },
    [loadDirectory]
  );

  const handleFileUpload = useCallback((files: FileList | null) => {
    if (!files) return;

    const fileItems: FileSystemItem[] = Array.from(files).map((file) => ({
      name: file.name,
      path: file.name, // Temporary path for uploaded files
      type: "file",
      size: file.size,
      extension: file.name.split(".").pop(),
      modified: new Date(file.lastModified),
    }));

    // Add uploaded files to selection
    setSelectedPaths((prev) => {
      const newSet = new Set(prev);
      fileItems.forEach((item) => newSet.add(item.path));
      return newSet;
    });
  }, []);

  const handleImport = useCallback(async () => {
    if (selectedPaths.size === 0 || !destinationPath.trim()) return;

    const selectedItems = Array.from(selectedPaths).map((path) => {
      // Find the item in the file tree or create from uploaded files
      const treeItem = fileTree.find((item) => item.path === path);
      if (treeItem) return treeItem;

      // Handle uploaded files
      return {
        name: path,
        path,
        type: "file" as const,
        size: 0,
        extension: path.split(".").pop(),
      };
    });

    setIsImporting(true);
    setError(null);
    setImportProgress({
      phase: "scanning",
      progress: 0,
      message: "Scanning files...",
      totalItems: selectedItems.length,
      processedItems: 0,
    });

    try {
      await onImport(selectedItems, destinationPath);
      setImportProgress({
        phase: "complete",
        progress: 100,
        message: "Import completed successfully!",
        totalItems: selectedItems.length,
        processedItems: selectedItems.length,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to import files");
      setImportProgress(null);
    } finally {
      setIsImporting(false);
    }
  }, [selectedPaths, destinationPath, fileTree, onImport]);

  const handleClose = useCallback(() => {
    if (isImporting) return; // Prevent closing during import
    onClose();
  }, [isImporting, onClose]);

  // --------------------------------
  // Render
  // --------------------------------
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <MotionDiv
            className="fixed inset-0 bg-black/50 z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={handleClose}
          />

          {/* Dialog */}
          <MotionDiv
            className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-4xl max-h-[90vh] bg-background border border-border rounded-lg shadow-xl z-50 overflow-hidden"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-border">
              <div className="flex items-center gap-3">
                <Folder className="w-6 h-6" />
                <h2 className="text-xl font-semibold">
                  Import Local Files & Folders
                </h2>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleClose}
                disabled={isImporting}
              >
                <X className="w-4 h-4" />
              </Button>
            </div>

            {/* Content */}
            <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
              {/* Drive Selector */}
              <div className="mb-6">
                <Label className="text-sm font-medium mb-2 block">
                  Select Drive
                </Label>
                <div className="flex gap-2">
                  {drives.map((drive) => (
                    <Button
                      key={drive}
                      variant={
                        currentPath.startsWith(drive) ? "default" : "outline"
                      }
                      size="sm"
                      onClick={() => handleDriveChange(drive)}
                    >
                      <HardDrive className="w-4 h-4 mr-2" />
                      {drive}
                    </Button>
                  ))}
                </div>
              </div>

              {/* File Tree */}
              <div className="mb-6">
                <Label className="text-sm font-medium mb-2 block">
                  Browse Files & Folders
                </Label>
                <FileTree
                  rootPath={currentPath}
                  items={fileTree}
                  selectedPaths={selectedPaths}
                  onToggle={handleToggle}
                  onSelect={handleSelect}
                  onRefresh={() => loadDirectory(currentPath)}
                  isLoading={isLoading}
                />
              </div>

              {/* Drag & Drop / File Upload */}
              <div className="mb-6">
                <Label className="text-sm font-medium mb-2 block">
                  Or Upload Files
                </Label>
                <div
                  className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary/50 transition-colors cursor-pointer"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="w-8 h-8 mx-auto mb-4 text-muted-foreground" />
                  <p className="text-sm text-muted-foreground mb-2">
                    Click to select files or drag and drop here
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Supports multiple files and folders
                  </p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    className="hidden"
                    onChange={(e) => handleFileUpload(e.target.files)}
                  />
                </div>
              </div>

              {/* Selected Items Summary */}
              {selectedPaths.size > 0 && (
                <MotionDiv
                  className="mb-6 p-4 border border-primary/20 rounded-lg bg-primary/5"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <h3 className="font-medium mb-2">
                    Selected Items ({selectedPaths.size})
                  </h3>
                  <ScrollArea className="max-h-32">
                    <div className="space-y-1">
                      {Array.from(selectedPaths).map((path) => (
                        <div
                          key={path}
                          className="flex items-center gap-2 text-sm"
                        >
                          <FileIcon filename={path} className="w-3 h-3" />
                          <span className="truncate">
                            {path.split("\\").pop()}
                          </span>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </MotionDiv>
              )}

              {/* Destination Path */}
              <div className="mb-6">
                <Label
                  htmlFor="destination"
                  className="text-sm font-medium mb-2 block"
                >
                  Destination Path
                </Label>
                <div className="flex gap-2">
                  <Input
                    id="destination"
                    placeholder="C:\Projects\workspace"
                    value={destinationPath}
                    onChange={(e) => setDestinationPath(e.target.value)}
                  />
                  <Button variant="outline" size="sm">
                    Browse...
                  </Button>
                </div>
              </div>

              {/* Import Progress */}
              {importProgress && (
                <MotionDiv
                  className="mb-6 space-y-4"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <Label className="text-sm font-medium">
                        Import Progress
                      </Label>
                      <Badge variant="outline">{importProgress.phase}</Badge>
                    </div>
                    <Progress
                      value={importProgress.progress}
                      className="mb-2"
                    />
                    <p className="text-sm text-muted-foreground">
                      {importProgress.message}
                    </p>
                    {importProgress.currentItem && (
                      <p className="text-xs text-muted-foreground mt-1">
                        {importProgress.currentItem}
                      </p>
                    )}
                    {importProgress.totalItems &&
                      importProgress.processedItems && (
                        <p className="text-xs text-muted-foreground mt-1">
                          {importProgress.processedItems} /{" "}
                          {importProgress.totalItems} items
                        </p>
                      )}
                  </div>
                </MotionDiv>
              )}

              {/* Error Display */}
              {error && (
                <Alert className="mb-6">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-end gap-3 p-6 border-t border-border">
              <Button
                variant="outline"
                onClick={handleClose}
                disabled={isImporting}
              >
                Cancel
              </Button>
              <Button
                onClick={handleImport}
                disabled={
                  selectedPaths.size === 0 ||
                  !destinationPath.trim() ||
                  isImporting
                }
              >
                {isImporting ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Importing...
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4 mr-2" />
                    Import {selectedPaths.size} Items
                  </>
                )}
              </Button>
            </div>
          </MotionDiv>
        </>
      )}
    </AnimatePresence>
  );
}

export default LocalImportDialog;
