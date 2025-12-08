/**
 * IDE View - Professional Code Editor Interface
 *
 * VS Code-inspired IDE with NEURECTOMY design system
 * Integrating: Agent Execution Graph, GitOps Overlay, Experiment Sidebar, Enhanced Status Bar
 */

import { useState, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  File,
  Folder,
  FolderOpen,
  GitBranch,
  Terminal as TerminalIcon,
  Search,
  Settings,
  Box,
  Zap,
  ChevronRight,
  ChevronDown,
  Plus,
  X,
  RefreshCw,
  Code2,
  FileText,
  FlaskConical,
  Network,
  PanelBottomClose,
  PanelBottomOpen,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { EditorManager } from "@/components/editors/EditorManager";
import { useEditorStore } from "@/stores/editor-store";
import type { EditorFile } from "@neurectomy/types";
import { useIDEKeyboardShortcuts } from "@/hooks/useIDEKeyboardShortcuts";

// Import new NEURECTOMY components
import {
  AgentExecutionGraph,
  mockExecutionNodes,
} from "@/components/agent-graph";
import { GitOpsOverlay } from "@/components/gitops";
import { ExperimentSidebar } from "@/components/experiments";
import { EnhancedStatusBar } from "@/components/status-bar";

interface FileNode {
  name: string;
  type: "file" | "folder";
  path: string;
  children?: FileNode[];
  expanded?: boolean;
}

// Mock file tree
const mockFileTree: FileNode[] = [
  {
    name: "src",
    type: "folder",
    path: "/src",
    expanded: true,
    children: [
      {
        name: "features",
        type: "folder",
        path: "/src/features",
        children: [
          {
            name: "agent-editor",
            type: "folder",
            path: "/src/features/agent-editor",
            children: [],
          },
          {
            name: "dashboard",
            type: "folder",
            path: "/src/features/dashboard",
            children: [],
          },
        ],
      },
      {
        name: "components",
        type: "folder",
        path: "/src/components",
        children: [
          {
            name: "editors",
            type: "folder",
            path: "/src/components/editors",
            children: [],
          },
        ],
      },
      { name: "App.tsx", type: "file", path: "/src/App.tsx" },
      { name: "main.tsx", type: "file", path: "/src/main.tsx" },
    ],
  },
  {
    name: "package.json",
    type: "file",
    path: "/package.json",
  },
  {
    name: "README.md",
    type: "file",
    path: "/README.md",
  },
];

type ActivityBarItem =
  | "explorer"
  | "search"
  | "git"
  | "agents"
  | "experiments"
  | "extensions"
  | "settings";

export default function IDEView() {
  const [activeActivity, setActiveActivity] =
    useState<ActivityBarItem>("explorer");
  const [fileTree, setFileTree] = useState<FileNode[]>(mockFileTree);
  const [terminalOpen, setTerminalOpen] = useState(true);
  const [agentGraphOpen, setAgentGraphOpen] = useState(false);

  // Simulated metrics for EnhancedStatusBar
  const [metrics] = useState({
    cpu: 45,
    gpu: 62,
    memory: 8.4,
  });
  const [activeAgents] = useState(3);
  const [deploymentStatus] = useState<
    "idle" | "deploying" | "success" | "failed"
  >("success");

  const { openFiles, activeFileId, openFile, closeFile, setActiveFile } =
    useEditorStore();

  // Keyboard shortcuts handlers - memoized to prevent re-binding
  const shortcutHandlers = useMemo(
    () => ({
      toggleTerminal: () => setTerminalOpen((prev) => !prev),
      toggleAgentGraph: () => setAgentGraphOpen((prev) => !prev),
      switchToExplorer: () => setActiveActivity("explorer"),
      switchToSearch: () => setActiveActivity("search"),
      switchToGit: () => setActiveActivity("git"),
      switchToAgents: () => setActiveActivity("agents"),
      switchToExperiments: () => setActiveActivity("experiments"),
      switchToExtensions: () => setActiveActivity("extensions"),
      switchToSettings: () => setActiveActivity("settings"),
      closeActiveFile: () => {
        if (activeFileId) closeFile(activeFileId);
      },
    }),
    [activeFileId, closeFile]
  );

  // Register keyboard shortcuts
  useIDEKeyboardShortcuts(shortcutHandlers);

  const toggleFolder = useCallback(
    (path: string) => {
      const updateTree = (nodes: FileNode[]): FileNode[] => {
        return nodes.map((node) => {
          if (node.path === path && node.type === "folder") {
            return { ...node, expanded: !node.expanded };
          }
          if (node.children) {
            return { ...node, children: updateTree(node.children) };
          }
          return node;
        });
      };
      setFileTree(updateTree(fileTree));
    },
    [fileTree]
  );

  const handleFileClick = useCallback(
    (file: FileNode) => {
      if (file.type === "file") {
        const fileToOpen: Omit<EditorFile, "id" | "isDirty" | "lastModified"> =
          {
            path: file.path,
            name: file.name,
            language: getLanguageFromPath(file.path),
            content: "// File content will be loaded from backend\n",
          };
        openFile(fileToOpen);
        // setActiveFile is handled automatically by openFile
      } else {
        toggleFolder(file.path);
      }
    },
    [openFile, toggleFolder]
  );

  return (
    <div className="flex h-screen w-screen bg-background text-foreground overflow-hidden">
      {/* Activity Bar */}
      <div className="w-12 bg-card flex flex-col items-center py-2 space-y-3 border-r border-border">
        <ActivityBarIcon
          icon={Folder}
          label="Explorer"
          active={activeActivity === "explorer"}
          onClick={() => setActiveActivity("explorer")}
        />
        <ActivityBarIcon
          icon={Search}
          label="Search"
          active={activeActivity === "search"}
          onClick={() => setActiveActivity("search")}
        />
        <ActivityBarIcon
          icon={GitBranch}
          label="Source Control"
          active={activeActivity === "git"}
          onClick={() => setActiveActivity("git")}
        />
        <ActivityBarIcon
          icon={Zap}
          label="Agent Orchestration"
          active={activeActivity === "agents"}
          onClick={() => setActiveActivity("agents")}
        />
        <ActivityBarIcon
          icon={FlaskConical}
          label="Experiments"
          active={activeActivity === "experiments"}
          onClick={() => setActiveActivity("experiments")}
        />
        <ActivityBarIcon
          icon={Box}
          label="Extensions"
          active={activeActivity === "extensions"}
          onClick={() => setActiveActivity("extensions")}
        />
        <div className="flex-1" />
        <ActivityBarIcon
          icon={Settings}
          label="Settings"
          active={activeActivity === "settings"}
          onClick={() => setActiveActivity("settings")}
        />
      </div>

      {/* Sidebar Panel */}
      <div className="w-64 bg-card border-r border-border overflow-y-auto">
        <AnimatePresence mode="wait">
          {activeActivity === "explorer" && (
            <ExplorerPanel
              fileTree={fileTree}
              onFileClick={handleFileClick}
              onToggleFolder={toggleFolder}
            />
          )}
          {activeActivity === "search" && <SearchPanel />}
          {activeActivity === "git" && <GitPanel />}
          {activeActivity === "agents" && <AgentPanel />}
          {activeActivity === "experiments" && <ExperimentsPanel />}
          {activeActivity === "extensions" && <ExtensionsPanel />}
          {activeActivity === "settings" && <SettingsPanel />}
        </AnimatePresence>
      </div>

      {/* Main Editor Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Tab Bar */}
        <div className="h-9 bg-card flex items-center border-b border-border overflow-x-auto">
          {openFiles.map((file) => (
            <EditorTab
              key={file.id}
              file={file}
              active={file.id === activeFileId}
              onClose={() => closeFile(file.id)}
              onClick={() => setActiveFile(file.id)}
            />
          ))}
          {openFiles.length === 0 && (
            <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
              No files open
            </div>
          )}
        </div>

        {/* Editor Content */}
        <div className="flex-1 relative bg-background overflow-hidden">
          {openFiles.length === 0 ? <WelcomeScreen /> : <EditorManager />}
        </div>

        {/* Agent Execution Graph Panel (Collapsible) */}
        <AnimatePresence>
          {agentGraphOpen && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 200, opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              style={{
                overflow: "hidden",
                borderTop: "1px solid hsl(var(--border))",
              }}
            >
              <div className="h-full bg-card">
                <div className="h-8 flex items-center justify-between px-3 border-b border-border">
                  <div className="flex items-center gap-2 text-xs font-medium">
                    <Network size={14} className="text-neural-blue" />
                    <span>Agent Execution Graph</span>
                  </div>
                  <button
                    onClick={() => setAgentGraphOpen(false)}
                    className="p-1 hover:bg-muted rounded transition-colors"
                    title="Close Agent Graph"
                  >
                    <X size={14} />
                  </button>
                </div>
                <div className="h-[calc(100%-2rem)] overflow-auto">
                  <AgentExecutionGraph nodes={mockExecutionNodes} compact />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Terminal Panel */}
        {terminalOpen && (
          <div className="h-64 border-t border-border bg-card">
            <TerminalPanel onClose={() => setTerminalOpen(false)} />
          </div>
        )}
      </div>

      {/* Enhanced Status Bar */}
      <EnhancedStatusBar
        gitBranch="main"
        gitSynced={true}
        metrics={metrics}
        activeAgents={activeAgents}
        deploymentStatus={deploymentStatus}
        encoding="UTF-8"
        lineCol={{ line: 1, col: 1 }}
        onToggleTerminal={() => setTerminalOpen(!terminalOpen)}
        onToggleAgentGraph={() => setAgentGraphOpen(!agentGraphOpen)}
        terminalOpen={terminalOpen}
        agentGraphOpen={agentGraphOpen}
      />
    </div>
  );
}

// Activity Bar Icon
function ActivityBarIcon({
  icon: Icon,
  label,
  active,
  onClick,
}: {
  icon: React.ComponentType<any>;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      title={label}
      className={cn(
        "w-10 h-10 flex items-center justify-center rounded-lg transition-all duration-200 relative group",
        active
          ? "text-primary bg-primary/10"
          : "text-muted-foreground hover:text-foreground hover:bg-muted"
      )}
    >
      <Icon size={20} />
      {active && (
        <div className="absolute left-0 w-1 h-8 bg-primary rounded-r-full" />
      )}
      {/* Tooltip */}
      <div className="absolute left-full ml-2 px-2 py-1 bg-popover text-popover-foreground text-xs rounded-md shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all whitespace-nowrap z-50">
        {label}
      </div>
    </button>
  );
}

// Explorer Panel
function ExplorerPanel({
  fileTree,
  onFileClick,
  onToggleFolder,
}: {
  fileTree: FileNode[];
  onFileClick: (file: FileNode) => void;
  onToggleFolder: (path: string) => void;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      style={{ padding: "0.75rem" }}
    >
      <div className="flex items-center justify-between px-2 py-2 mb-2">
        <span className="text-xs font-semibold uppercase text-muted-foreground tracking-wider">
          Explorer
        </span>
        <div className="flex gap-1">
          <button
            className="p-1 hover:bg-muted rounded transition-colors"
            title="New File"
          >
            <FileText size={14} />
          </button>
          <button
            className="p-1 hover:bg-muted rounded transition-colors"
            title="New Folder"
          >
            <Folder size={14} />
          </button>
          <button
            className="p-1 hover:bg-muted rounded transition-colors"
            title="Refresh"
          >
            <RefreshCw size={14} />
          </button>
        </div>
      </div>
      <FileTreeView
        nodes={fileTree}
        onFileClick={onFileClick}
        onToggleFolder={onToggleFolder}
      />
    </motion.div>
  );
}

// File Tree View
function FileTreeView({
  nodes,
  onFileClick,
  onToggleFolder,
  level = 0,
}: {
  nodes: FileNode[];
  onFileClick: (file: FileNode) => void;
  onToggleFolder: (path: string) => void;
  level?: number;
}) {
  return (
    <div>
      {nodes.map((node) => (
        <div key={node.path}>
          <button
            onClick={() => onFileClick(node)}
            className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-muted rounded text-sm text-left transition-colors"
            style={{ paddingLeft: `${level * 12 + 8}px` }}
          >
            {node.type === "folder" ? (
              <>
                {node.expanded ? (
                  <ChevronDown size={14} className="text-muted-foreground" />
                ) : (
                  <ChevronRight size={14} className="text-muted-foreground" />
                )}
                {node.expanded ? (
                  <FolderOpen size={14} className="text-neural-blue" />
                ) : (
                  <Folder size={14} className="text-neural-blue" />
                )}
              </>
            ) : (
              <>
                <div className="w-3.5" />
                <File size={14} className="text-muted-foreground" />
              </>
            )}
            <span className="truncate">{node.name}</span>
          </button>
          {node.type === "folder" && node.expanded && node.children && (
            <FileTreeView
              nodes={node.children}
              onFileClick={onFileClick}
              onToggleFolder={onToggleFolder}
              level={level + 1}
            />
          )}
        </div>
      ))}
    </div>
  );
}

// Editor Tab
function EditorTab({
  file,
  active,
  onClose,
  onClick,
}: {
  file: any;
  active: boolean;
  onClose: () => void;
  onClick: () => void;
}) {
  return (
    <div
      onClick={onClick}
      className={cn(
        "flex items-center gap-2 px-3 h-full cursor-pointer border-r border-border transition-colors",
        active
          ? "bg-background text-foreground"
          : "bg-card text-muted-foreground hover:bg-muted"
      )}
    >
      <File size={14} />
      <span className="text-sm">{file.name}</span>
      {file.isDirty && <div className="w-1.5 h-1.5 rounded-full bg-primary" />}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onClose();
        }}
        className="ml-2 hover:bg-muted rounded p-0.5 transition-colors"
        title={`Close ${file.name}`}
        aria-label={`Close ${file.name}`}
      >
        <X size={14} />
      </button>
    </div>
  );
}

// Welcome Screen
function WelcomeScreen() {
  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center max-w-2xl px-6">
        <div className="w-16 h-16 bg-gradient-to-br from-primary via-synapse-purple to-cipher-cyan rounded-2xl flex items-center justify-center mx-auto mb-4">
          <Code2 size={32} className="text-white" />
        </div>
        <h1 className="text-4xl font-bold mb-3 bg-gradient-to-r from-neural-blue via-synapse-purple to-cipher-cyan bg-clip-text text-transparent">
          NEURECTOMY IDE
        </h1>
        <p className="text-muted-foreground mb-8">
          Next-generation code editor with embedded AI agent orchestration
        </p>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <button className="p-4 bg-card hover:bg-muted rounded-lg border border-border text-left transition-colors group">
            <div className="font-semibold mb-1 group-hover:text-primary transition-colors">
              Open Folder
            </div>
            <div className="text-muted-foreground text-xs">
              Start editing existing project
            </div>
          </button>
          <button className="p-4 bg-card hover:bg-muted rounded-lg border border-border text-left transition-colors group">
            <div className="font-semibold mb-1 group-hover:text-primary transition-colors">
              New File
            </div>
            <div className="text-muted-foreground text-xs">
              Create a new file
            </div>
          </button>
          <button className="p-4 bg-card hover:bg-muted rounded-lg border border-border text-left transition-colors group">
            <div className="font-semibold mb-1 group-hover:text-primary transition-colors">
              Clone Repository
            </div>
            <div className="text-muted-foreground text-xs">Clone from Git</div>
          </button>
          <button className="p-4 bg-card hover:bg-muted rounded-lg border border-border text-left transition-colors group">
            <div className="font-semibold mb-1 group-hover:text-primary transition-colors">
              Agent Workspace
            </div>
            <div className="text-muted-foreground text-xs">
              Create AI agent project
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}

// Terminal Panel
function TerminalPanel({ onClose }: { onClose: () => void }) {
  return (
    <div className="h-full flex flex-col">
      <div className="h-9 bg-card flex items-center justify-between px-3 border-b border-border">
        <div className="flex items-center gap-2 text-sm">
          <TerminalIcon size={14} />
          <span className="font-medium">Terminal</span>
        </div>
        <div className="flex gap-1">
          <button
            className="p-1 hover:bg-muted rounded transition-colors"
            title="New Terminal"
          >
            <Plus size={14} />
          </button>
          <button
            onClick={onClose}
            className="p-1 hover:bg-muted rounded transition-colors"
            title="Close"
          >
            <X size={14} />
          </button>
        </div>
      </div>
      <div className="flex-1 p-3 font-mono text-sm overflow-auto bg-background">
        <div className="text-matrix-green">
          <span className="text-muted-foreground">
            PS C:\Users\sgbil\NEURECTOMY&gt;
          </span>{" "}
          <span className="animate-pulse">_</span>
        </div>
      </div>
    </div>
  );
}

// Search Panel
function SearchPanel() {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      style={{ padding: "1rem" }}
    >
      <h3 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider mb-3">
        Search
      </h3>
      <input
        type="text"
        placeholder="Search files..."
        className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
      />
    </motion.div>
  );
}

// Git Panel - Enhanced with GitOps Overlay
function GitPanel() {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      style={{ height: "100%", display: "flex", flexDirection: "column" }}
    >
      <div className="p-4 border-b border-border">
        <h3 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider mb-3">
          Source Control
        </h3>
        <div className="p-4 bg-muted/50 rounded-lg border border-border">
          <div className="text-sm text-muted-foreground text-center">
            No changes
          </div>
        </div>
      </div>
      {/* GitOps Deployment Status */}
      <div className="flex-1 overflow-auto">
        <GitOpsOverlay />
      </div>
    </motion.div>
  );
}

// Agent Panel
function AgentPanel() {
  const agents = [
    { name: "@APEX", status: "active" as const, color: "neural-blue" },
    { name: "@CIPHER", status: "idle" as const, color: "cipher-cyan" },
    { name: "@ARCHITECT", status: "active" as const, color: "forge-orange" },
    { name: "@TENSOR", status: "active" as const, color: "synapse-purple" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      style={{ padding: "1rem" }}
    >
      <h3 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider mb-3">
        Agents
      </h3>
      <div className="space-y-2">
        {agents.map((agent) => (
          <div
            key={agent.name}
            className="flex items-center gap-3 p-3 bg-muted/50 rounded-lg border border-border hover:border-primary/50 transition-colors cursor-pointer"
          >
            <div
              className={cn(
                "w-2 h-2 rounded-full",
                agent.status === "active"
                  ? "bg-matrix-green"
                  : "bg-muted-foreground"
              )}
            />
            <span className="text-sm font-medium">{agent.name}</span>
            <span
              className={cn(
                "ml-auto text-xs px-2 py-0.5 rounded-full",
                agent.status === "active"
                  ? "bg-matrix-green/20 text-matrix-green"
                  : "bg-muted text-muted-foreground"
              )}
            >
              {agent.status}
            </span>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

// Extensions Panel
function ExtensionsPanel() {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      style={{ padding: "1rem" }}
    >
      <h3 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider mb-3">
        Extensions
      </h3>
      <div className="p-4 bg-muted/50 rounded-lg border border-border">
        <div className="text-sm text-muted-foreground text-center">
          No extensions installed
        </div>
      </div>
    </motion.div>
  );
}

// Experiments Panel - Using ExperimentSidebar
function ExperimentsPanel() {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      style={{ height: "100%" }}
    >
      <ExperimentSidebar />
    </motion.div>
  );
}

// Settings Panel
function SettingsPanel() {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      style={{ padding: "1rem" }}
    >
      <h3 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider mb-3">
        Settings
      </h3>
      <div className="space-y-3">
        <div>
          <label
            htmlFor="theme-select"
            className="text-xs font-medium mb-1.5 block text-muted-foreground"
          >
            Theme
          </label>
          <select
            id="theme-select"
            title="Select Theme"
            className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
          >
            <option>Dark (default)</option>
            <option>Light</option>
            <option>High Contrast</option>
          </select>
        </div>
      </div>
    </motion.div>
  );
}

function getLanguageFromPath(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase();
  const map: Record<string, string> = {
    ts: "typescript",
    tsx: "typescript",
    js: "javascript",
    jsx: "javascript",
    py: "python",
    rs: "rust",
    go: "go",
    java: "java",
    cpp: "cpp",
    c: "c",
    md: "markdown",
    json: "json",
    yaml: "yaml",
    yml: "yaml",
  };
  return map[ext || ""] || "plaintext";
}
