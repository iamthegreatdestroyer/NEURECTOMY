/**
 * GitHub Import Dialog Component
 *
 * Modal dialog for importing GitHub repositories with:
 * - Repository URL input with validation
 * - Authentication support (OAuth/PAT)
 * - Clone progress with real-time feedback
 * - Error handling and retry mechanisms
 *
 * @module @neurectomy/components
 * @author @CANVAS @APEX
 */

import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Github,
  Download,
  AlertCircle,
  CheckCircle2,
  Loader2,
  X,
  Key,
  Globe,
  Lock,
  FolderOpen,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Button,
  Input,
  Label,
  Textarea,
  Progress,
  Alert,
  AlertDescription,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  Badge,
} from "@neurectomy/ui";
import { invoke } from "@tauri-apps/api/core";

// =============================================================================
// Types
// =============================================================================

export interface GitHubRepository {
  id: number;
  name: string;
  full_name: string;
  description: string;
  html_url: string;
  clone_url: string;
  ssh_url: string;
  language: string;
  stargazers_count: number;
  forks_count: number;
  size: number;
  private: boolean;
  owner: {
    login: string;
    avatar_url: string;
  };
}

export interface CloneProgress {
  phase: "initializing" | "cloning" | "configuring" | "complete";
  progress: number;
  message: string;
  currentFile?: string;
}

export interface GitHubImportDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onImport: (repository: GitHubRepository, localPath: string) => Promise<void>;
  initialUrl?: string;
}

// =============================================================================
// GitHub API Integration
// =============================================================================

class GitHubAPI {
  private baseURL = "https://api.github.com";
  private token?: string;

  constructor(token?: string) {
    this.token = token;
  }

  private async request<T>(endpoint: string): Promise<T> {
    const headers: Record<string, string> = {
      Accept: "application/vnd.github.v3+json",
      "User-Agent": "NEURECTOMY-IDE/1.0",
    };

    if (this.token) {
      headers["Authorization"] = `token ${this.token}`;
    }

    const response = await fetch(`${this.baseURL}${endpoint}`, { headers });

    if (!response.ok) {
      if (response.status === 401) {
        throw new Error(
          "Authentication required. Please provide a GitHub token."
        );
      }
      if (response.status === 403) {
        throw new Error("Access denied. Check your token permissions.");
      }
      if (response.status === 404) {
        throw new Error("Repository not found.");
      }
      throw new Error(`GitHub API error: ${response.statusText}`);
    }

    return response.json();
  }

  async getRepository(owner: string, repo: string): Promise<GitHubRepository> {
    return this.request<GitHubRepository>(`/repos/${owner}/${repo}`);
  }

  async getUserRepositories(username: string): Promise<GitHubRepository[]> {
    return this.request<GitHubRepository[]>(`/users/${username}/repos`);
  }

  async searchRepositories(
    query: string
  ): Promise<{ items: GitHubRepository[] }> {
    return this.request<{ items: GitHubRepository[] }>(
      `/search/repositories?q=${encodeURIComponent(query)}&sort=stars&order=desc`
    );
  }
}

// =============================================================================
// Repository Card Component
// =============================================================================

interface RepositoryCardProps {
  repository: GitHubRepository;
  isSelected: boolean;
  onSelect: () => void;
}

function RepositoryCard({
  repository,
  isSelected,
  onSelect,
}: RepositoryCardProps) {
  return (
    <MotionDiv
      className={cn(
        "p-4 border rounded-lg cursor-pointer transition-all duration-200",
        isSelected
          ? "border-primary bg-primary/5 ring-2 ring-primary/20"
          : "border-border hover:border-primary/50 hover:bg-muted/50"
      )}
      onClick={onSelect}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <Github className="w-4 h-4" />
          <span className="font-medium text-sm">{repository.full_name}</span>
          {repository.private && (
            <Lock className="w-3 h-3 text-muted-foreground" />
          )}
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="text-xs">
            {repository.language || "Unknown"}
          </Badge>
        </div>
      </div>

      {repository.description && (
        <p className="text-sm text-muted-foreground mb-3 line-clamp-2">
          {repository.description}
        </p>
      )}

      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <div className="flex items-center gap-4">
          <span>⭐ {repository.stargazers_count.toLocaleString()}</span>
          <span>🍴 {repository.forks_count.toLocaleString()}</span>
          <span>{(repository.size / 1024).toFixed(1)} MB</span>
        </div>
        <div className="flex items-center gap-1">
          <img
            src={repository.owner.avatar_url}
            alt={repository.owner.login}
            className="w-4 h-4 rounded-full"
          />
          <span>{repository.owner.login}</span>
        </div>
      </div>
    </MotionDiv>
  );
}

// =============================================================================
// Main Dialog Component
// =============================================================================

export function GitHubImportDialog({
  isOpen,
  onClose,
  onImport,
  initialUrl,
}: GitHubImportDialogProps) {
  // --------------------------------
  // State
  // --------------------------------
  const [activeTab, setActiveTab] = useState<"url" | "search" | "browse">(
    "url"
  );
  const [repositoryUrl, setRepositoryUrl] = useState(initialUrl || "");
  const [searchQuery, setSearchQuery] = useState("");
  const [githubToken, setGithubToken] = useState("");
  const [localPath, setLocalPath] = useState("");

  const [repositories, setRepositories] = useState<GitHubRepository[]>([]);
  const [selectedRepository, setSelectedRepository] =
    useState<GitHubRepository | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isCloning, setIsCloning] = useState(false);
  const [cloneProgress, setCloneProgress] = useState<CloneProgress | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);

  const githubAPI = new GitHubAPI(githubToken || undefined);

  // --------------------------------
  // Effects
  // --------------------------------
  useEffect(() => {
    if (isOpen) {
      // Reset state when dialog opens
      setRepositoryUrl("");
      setSearchQuery("");
      setRepositories([]);
      setSelectedRepository(null);
      setError(null);
      setCloneProgress(null);
      setIsCloning(false);
    }
  }, [isOpen]);

  // --------------------------------
  // Handlers
  // --------------------------------
  const handleUrlImport = useCallback(async () => {
    if (!repositoryUrl.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      // Parse GitHub URL
      const urlMatch = repositoryUrl.match(/github\.com\/([^\/]+)\/([^\/\.]+)/);
      if (!urlMatch) {
        throw new Error("Invalid GitHub repository URL");
      }

      const [, owner, repo] = urlMatch;
      const repository = await githubAPI.getRepository(owner, repo);
      setSelectedRepository(repository);
      setActiveTab("browse");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to fetch repository"
      );
    } finally {
      setIsLoading(false);
    }
  }, [repositoryUrl, githubAPI]);

  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const result = await githubAPI.searchRepositories(searchQuery);
      setRepositories(result.items);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to search repositories"
      );
    } finally {
      setIsLoading(false);
    }
  }, [searchQuery, githubAPI]);

  const handleClone = useCallback(async () => {
    if (!selectedRepository || !localPath.trim()) return;

    setIsCloning(true);
    setError(null);
    setCloneProgress({
      phase: "initializing",
      progress: 0,
      message: "Initializing clone...",
    });

    try {
      await onImport(selectedRepository, localPath);
      setCloneProgress({
        phase: "complete",
        progress: 100,
        message: "Repository cloned successfully!",
      });
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to clone repository"
      );
      setCloneProgress(null);
    } finally {
      setIsCloning(false);
    }
  }, [selectedRepository, localPath, onImport]);

  const handleClose = useCallback(() => {
    if (isCloning) return; // Prevent closing during clone
    onClose();
  }, [isCloning, onClose]);

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
                <Github className="w-6 h-6" />
                <h2 className="text-xl font-semibold">
                  Import GitHub Repository
                </h2>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleClose}
                disabled={isCloning}
              >
                <X className="w-4 h-4" />
              </Button>
            </div>

            {/* Content */}
            <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
              {/* Authentication */}
              <div className="mb-6">
                <Label
                  htmlFor="token"
                  className="text-sm font-medium mb-2 block"
                >
                  GitHub Personal Access Token (Optional)
                </Label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <Key className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <Input
                      id="token"
                      type="password"
                      placeholder="ghp_xxxxxxxxxxxxxxxxxxxx"
                      value={githubToken}
                      onChange={(e) => setGithubToken(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                  <Button variant="outline" size="sm">
                    <Globe className="w-4 h-4 mr-2" />
                    Get Token
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Required for private repositories. Create one at{" "}
                  <a
                    href="https://github.com/settings/tokens"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:underline"
                  >
                    GitHub Settings
                  </a>
                </p>
              </div>

              <Tabs
                value={activeTab}
                onValueChange={(value) => setActiveTab(value as any)}
              >
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="url">Import by URL</TabsTrigger>
                  <TabsTrigger value="search">Search Repositories</TabsTrigger>
                  <TabsTrigger value="browse">Browse & Select</TabsTrigger>
                </TabsList>

                {/* URL Import Tab */}
                <TabsContent value="url" className="space-y-4">
                  <div>
                    <Label
                      htmlFor="repo-url"
                      className="text-sm font-medium mb-2 block"
                    >
                      Repository URL
                    </Label>
                    <Input
                      id="repo-url"
                      placeholder="https://github.com/owner/repository"
                      value={repositoryUrl}
                      onChange={(e) => setRepositoryUrl(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && handleUrlImport()}
                    />
                  </div>
                  <Button
                    onClick={handleUrlImport}
                    disabled={isLoading || !repositoryUrl.trim()}
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Fetching Repository...
                      </>
                    ) : (
                      <>
                        <Download className="w-4 h-4 mr-2" />
                        Fetch Repository
                      </>
                    )}
                  </Button>
                </TabsContent>

                {/* Search Tab */}
                <TabsContent value="search" className="space-y-4">
                  <div>
                    <Label
                      htmlFor="search"
                      className="text-sm font-medium mb-2 block"
                    >
                      Search Repositories
                    </Label>
                    <Input
                      id="search"
                      placeholder="react, typescript, machine learning..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                    />
                  </div>
                  <Button
                    onClick={handleSearch}
                    disabled={isLoading || !searchQuery.trim()}
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Searching...
                      </>
                    ) : (
                      <>
                        <Github className="w-4 h-4 mr-2" />
                        Search
                      </>
                    )}
                  </Button>
                </TabsContent>

                {/* Browse & Select Tab */}
                <TabsContent value="browse" className="space-y-4">
                  {repositories.length > 0 && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-96 overflow-y-auto">
                      {repositories.map((repo) => (
                        <RepositoryCard
                          key={repo.id}
                          repository={repo}
                          isSelected={selectedRepository?.id === repo.id}
                          onSelect={() => setSelectedRepository(repo)}
                        />
                      ))}
                    </div>
                  )}

                  {selectedRepository && (
                    <MotionDiv
                      className="p-4 border border-primary/20 rounded-lg bg-primary/5"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                    >
                      <h3 className="font-medium mb-2">Selected Repository</h3>
                      <RepositoryCard
                        repository={selectedRepository}
                        isSelected={true}
                        onSelect={() => {}}
                      />
                    </MotionDiv>
                  )}
                </TabsContent>
              </Tabs>

              {/* Local Path Selection */}
              {selectedRepository && (
                <MotionDiv
                  className="mt-6 space-y-4"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <div>
                    <Label
                      htmlFor="local-path"
                      className="text-sm font-medium mb-2 block"
                    >
                      Local Path
                    </Label>
                    <div className="flex gap-2">
                      <div className="relative flex-1">
                        <FolderOpen className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                        <Input
                          id="local-path"
                          placeholder="C:\Projects\my-repo"
                          value={localPath}
                          onChange={(e) => setLocalPath(e.target.value)}
                          className="pl-10"
                        />
                      </div>
                      <Button variant="outline" size="sm">
                        Browse...
                      </Button>
                    </div>
                  </div>
                </MotionDiv>
              )}

              {/* Clone Progress */}
              {cloneProgress && (
                <MotionDiv
                  className="mt-6 space-y-4"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <Label className="text-sm font-medium">
                        Clone Progress
                      </Label>
                      <Badge variant="outline">{cloneProgress.phase}</Badge>
                    </div>
                    <Progress value={cloneProgress.progress} className="mb-2" />
                    <p className="text-sm text-muted-foreground">
                      {cloneProgress.message}
                    </p>
                    {cloneProgress.currentFile && (
                      <p className="text-xs text-muted-foreground mt-1">
                        {cloneProgress.currentFile}
                      </p>
                    )}
                  </div>
                </MotionDiv>
              )}

              {/* Error Display */}
              {error && (
                <Alert className="mt-6">
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
                disabled={isCloning}
              >
                Cancel
              </Button>
              <Button
                onClick={handleClone}
                disabled={!selectedRepository || !localPath.trim() || isCloning}
              >
                {isCloning ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Cloning...
                  </>
                ) : (
                  <>
                    <Download className="w-4 h-4 mr-2" />
                    Clone Repository
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

export default GitHubImportDialog;
