/**
 * GitHub Import Panel Component
 *
 * Sidebar panel for GitHub repository import functionality.
 * Provides quick access to GitHub import features with repository search and clone.
 *
 * Features:
 * - Quick repository URL input
 * - Recent repositories
 * - Import history
 * - One-click clone functionality
 *
 * @module @neurectomy/components/shell/GitHubImportPanel
 * @author @APEX @CANVAS
 */

import React, { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  Github,
  Plus,
  History,
  Star,
  GitFork,
  ExternalLink,
  Download,
  RefreshCw,
} from "lucide-react";
import { Button, Input, ScrollArea, Separator } from "@neurectomy/ui";
import { GitHubImportDialog } from "./GitHubImportDialog";

// ============================================================================
// Types
// ============================================================================

interface RecentRepository {
  id: string;
  name: string;
  owner: string;
  fullName: string;
  description?: string;
  stars: number;
  forks: number;
  updatedAt: string;
  url: string;
}

// ============================================================================
// Component
// ============================================================================

interface GitHubImportPanelProps {
  onImport?: (repoUrl: string, targetPath: string) => Promise<void>;
}

export function GitHubImportPanel({ onImport }: GitHubImportPanelProps = {}) {
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [quickUrl, setQuickUrl] = useState("");

  // Mock recent repositories - in real implementation, this would come from local storage or API
  const recentRepos: RecentRepository[] = [
    {
      id: "1",
      name: "neurectomy",
      owner: "neurectomy",
      fullName: "neurectomy/neurectomy",
      description: "AI-powered IDE for agent development",
      stars: 42,
      forks: 8,
      updatedAt: "2024-01-15",
      url: "https://github.com/neurectomy/neurectomy",
    },
    {
      id: "2",
      name: "vscode",
      owner: "microsoft",
      fullName: "microsoft/vscode",
      description: "Visual Studio Code",
      stars: 150000,
      forks: 25000,
      updatedAt: "2024-01-14",
      url: "https://github.com/microsoft/vscode",
    },
  ];

  const handleQuickImport = useCallback(() => {
    if (quickUrl.trim()) {
      // Open the full import dialog with the URL pre-filled
      setImportDialogOpen(true);
    }
  }, [quickUrl]);

  const handleRepoClick = useCallback((repo: RecentRepository) => {
    setQuickUrl(repo.url);
    setImportDialogOpen(true);
  }, []);

  const handleKeyPress = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter") {
        handleQuickImport();
      }
    },
    [handleQuickImport]
  );

  return (
    <>
      <div className="h-full flex flex-col bg-card">
        {/* Header */}
        <div className="p-4 border-b border-border">
          <div className="flex items-center gap-2 mb-3">
            <Github className="w-5 h-5 text-primary" />
            <h3 className="font-semibold text-sm">GitHub Import</h3>
          </div>

          {/* Quick Import */}
          <div className="space-y-2">
            <Input
              placeholder="Paste repository URL..."
              value={quickUrl}
              onChange={(e) => setQuickUrl(e.target.value)}
              onKeyPress={handleKeyPress}
              className="text-xs"
            />
            <Button
              onClick={handleQuickImport}
              disabled={!quickUrl.trim()}
              size="sm"
              className="w-full"
            >
              <Download className="w-4 h-4 mr-2" />
              Quick Import
            </Button>
          </div>
        </div>

        {/* Content */}
        <ScrollArea className="flex-1">
          <div className="p-4 space-y-4">
            {/* Recent Repositories */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <History className="w-4 h-4 text-muted-foreground" />
                <h4 className="font-medium text-sm">Recent</h4>
              </div>

              <div className="space-y-2">
                {recentRepos.map((repo) => (
                  <div
                    key={repo.id}
                    className="p-3 rounded-md border border-border hover:bg-accent cursor-pointer transition-colors"
                    onClick={() => handleRepoClick(repo)}
                  >
                    <div className="flex items-start justify-between mb-1">
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm truncate">
                          {repo.fullName}
                        </div>
                        {repo.description && (
                          <div className="text-xs text-muted-foreground truncate mt-1">
                            {repo.description}
                          </div>
                        )}
                      </div>
                      <ExternalLink className="w-3 h-3 text-muted-foreground flex-shrink-0 ml-2" />
                    </div>

                    <div className="flex items-center gap-3 text-xs text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <Star className="w-3 h-3" />
                        {repo.stars.toLocaleString()}
                      </div>
                      <div className="flex items-center gap-1">
                        <GitFork className="w-3 h-3" />
                        {repo.forks.toLocaleString()}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <Separator />

            {/* Actions */}
            <div>
              <Button
                onClick={() => setImportDialogOpen(true)}
                variant="outline"
                size="sm"
                className="w-full"
              >
                <Plus className="w-4 h-4 mr-2" />
                Browse GitHub
              </Button>
            </div>
          </div>
        </ScrollArea>
      </div>

      {/* Import Dialog */}
      <GitHubImportDialog
        isOpen={importDialogOpen}
        onClose={() => setImportDialogOpen(false)}
        onImport={async (repo, targetPath) => {
          if (onImport) {
            await onImport(repo.html_url, targetPath);
          }
        }}
        initialUrl={quickUrl}
      />
    </>
  );
}
