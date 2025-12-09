/**
 * IDE View v2 - Professional Code Editor Interface
 *
 * Complete rewrite using new shell components based on
 * VS Code, Continue, Zed, Lapce, IntelliJ, and Theia patterns.
 *
 * @module @neurectomy/features/ide
 * @author @APEX @ARCHITECT @CANVAS
 */

import { useState, useCallback, useMemo, ReactNode } from "react";
import {
  Folder,
  Search,
  GitBranch,
  Zap,
  FlaskConical,
  Box,
  Settings,
  FileText,
  File,
  FolderOpen,
  ChevronRight,
  ChevronDown,
  Code2,
  RefreshCw,
  Plus,
  MoreHorizontal,
} from "lucide-react";
import { cn } from "@/lib/utils";

// Import shell components
import {
  ApplicationShell,
  ActivityBar,
  ActivityBarItem,
  SidebarContainer,
  SidebarSection,
  EditorArea,
  EditorPane,
  EditorSplit,
  EditorNode,
  EditorTab,
  AIPanel,
  ChatMessage,
  MessageContext,
  BottomPanel,
  BottomPanelTab,
  TerminalInstance,
  Problem,
  OutputMessage,
  StatusBar,
  StatusBarItem,
} from "@/components/shell";

// Import editor
import { EditorManager } from "@/components/editors/EditorManager";
import { useEditorStore } from "@/stores/editor-store";
import { useWorkspaceStore } from "@/stores/workspace-store";
import type { EditorFile } from "@neurectomy/types";

// ============================================================================
// Types
// ============================================================================

type ActivityId =
  | "explorer"
  | "search"
  | "git"
  | "agents"
  | "experiments"
  | "extensions"
  | "settings";

interface FileNode {
  name: string;
  type: "file" | "folder";
  path: string;
  children?: FileNode[];
  expanded?: boolean;
}

// ============================================================================
// Mock Data
// ============================================================================

const mockFileTree: FileNode[] = [
  {
    name: "src",
    type: "folder",
    path: "/src",
    expanded: true,
    children: [
      {
        name: "components",
        type: "folder",
        path: "/src/components",
        children: [
          {
            name: "shell",
            type: "folder",
            path: "/src/components/shell",
            children: [],
          },
          {
            name: "editors",
            type: "folder",
            path: "/src/components/editors",
            children: [],
          },
        ],
      },
      {
        name: "features",
        type: "folder",
        path: "/src/features",
        children: [
          {
            name: "ide",
            type: "folder",
            path: "/src/features/ide",
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
      { name: "App.tsx", type: "file", path: "/src/App.tsx" },
      { name: "main.tsx", type: "file", path: "/src/main.tsx" },
    ],
  },
  { name: "package.json", type: "file", path: "/package.json" },
  { name: "README.md", type: "file", path: "/README.md" },
  { name: "tsconfig.json", type: "file", path: "/tsconfig.json" },
];

// ============================================================================
// Main Component
// ============================================================================

export default function IDEViewV2() {
  // --------------------------------
  // Activity Bar State
  // --------------------------------
  const [activeActivity, setActiveActivity] = useState<ActivityId | null>(
    "explorer"
  );

  const activityItems: ActivityBarItem[] = [
    { id: "explorer", icon: Folder, label: "Explorer" },
    { id: "search", icon: Search, label: "Search" },
    { id: "git", icon: GitBranch, label: "Source Control" },
    { id: "agents", icon: Zap, label: "Agent Orchestration", badge: 3 },
    { id: "experiments", icon: FlaskConical, label: "Experiments" },
    { id: "extensions", icon: Box, label: "Extensions" },
    { id: "settings", icon: Settings, label: "Settings", position: "bottom" },
  ];

  // --------------------------------
  // Sidebar State
  // --------------------------------
  const [fileTree, setFileTree] = useState<FileNode[]>(mockFileTree);
  const { layout, toggleSidebar, toggleBottomPanel } = useWorkspaceStore();

  // --------------------------------
  // Editor State
  // --------------------------------
  const { openFiles, activeFileId, openFile, closeFile, setActiveFile } =
    useEditorStore();
  const [editorRoot, setEditorRoot] = useState<EditorNode | null>(null);
  const [activePane, setActivePane] = useState<string | null>("main");

  // Convert openFiles to EditorPane format
  const editorPaneRoot = useMemo<EditorNode | null>(() => {
    if (openFiles.length === 0) return null;

    return {
      id: "main",
      type: "pane",
      tabs: openFiles.map((f) => ({
        id: f.id,
        title: f.name,
        dirty: f.isDirty,
        path: f.path,
        icon: <FileIcon filename={f.name} />,
      })),
      activeTabId: activeFileId,
    } as EditorPane;
  }, [openFiles, activeFileId]);

  // --------------------------------
  // AI Panel State
  // --------------------------------
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [isAIStreaming, setIsAIStreaming] = useState(false);

  const handleSendMessage = useCallback(
    (content: string, context?: MessageContext[]) => {
      const userMessage: ChatMessage = {
        id: `msg-${Date.now()}`,
        role: "user",
        content,
        context,
        timestamp: new Date(),
      };
      setChatMessages((prev) => [...prev, userMessage]);

      // Simulate AI response
      setIsAIStreaming(true);
      setTimeout(() => {
        const aiMessage: ChatMessage = {
          id: `msg-${Date.now() + 1}`,
          role: "assistant",
          content: `I can help you with that. Based on your question about "${content.slice(0, 50)}...", here's what I suggest:\n\n\`\`\`typescript\n// Example code\nconst result = await processQuery(input);\nconsole.log(result);\n\`\`\`\n\nWould you like me to explain further?`,
          timestamp: new Date(),
        };
        setChatMessages((prev) => [...prev, aiMessage]);
        setIsAIStreaming(false);
      }, 1500);
    },
    []
  );

  // --------------------------------
  // Bottom Panel State
  // --------------------------------
  const [bottomPanelTab, setBottomPanelTab] =
    useState<BottomPanelTab>("terminal");
  const [terminals, setTerminals] = useState<TerminalInstance[]>([
    { id: "term-1", name: "bash", active: true },
  ]);
  const [activeTerminalId, setActiveTerminalId] = useState<string | null>(
    "term-1"
  );
  const [problems] = useState<Problem[]>([
    {
      id: "p1",
      severity: "error",
      message: "Cannot find module @/types",
      source: "TypeScript",
      file: "src/App.tsx",
      line: 5,
    },
    {
      id: "p2",
      severity: "warning",
      message: 'Unused variable "result"',
      source: "ESLint",
      file: "src/utils.ts",
      line: 23,
    },
  ]);
  const [outputMessages] = useState<OutputMessage[]>([
    {
      id: "o1",
      channel: "TypeScript",
      message: "Starting compilation...",
      timestamp: new Date(),
      level: "info",
    },
    {
      id: "o2",
      channel: "TypeScript",
      message: "Found 0 errors. Watching for file changes.",
      timestamp: new Date(),
      level: "info",
    },
  ]);

  // --------------------------------
  // Status Bar Items
  // --------------------------------
  const statusBarLeftItems: StatusBarItem[] = [
    {
      id: "git-branch",
      content: (
        <span className="flex items-center gap-1">
          <GitBranch size={12} />
          main
        </span>
      ),
      tooltip: "Git branch: main",
      onClick: () => console.log("Show git branches"),
    },
    {
      id: "sync",
      content: (
        <span className="flex items-center gap-1">
          <RefreshCw size={10} />
          Synced
        </span>
      ),
      tooltip: "Git sync status",
    },
  ];

  const statusBarRightItems: StatusBarItem[] = [
    {
      id: "line-col",
      content: "Ln 1, Col 1",
      tooltip: "Go to Line",
      onClick: () => {},
    },
    {
      id: "spaces",
      content: "Spaces: 2",
      tooltip: "Select Indentation",
      onClick: () => {},
    },
    {
      id: "encoding",
      content: "UTF-8",
      tooltip: "Select Encoding",
      onClick: () => {},
    },
    {
      id: "eol",
      content: "LF",
      tooltip: "Select End of Line Sequence",
      onClick: () => {},
    },
    {
      id: "language",
      content: "TypeScript React",
      tooltip: "Select Language Mode",
      onClick: () => {},
    },
  ];

  // --------------------------------
  // File Tree Handlers
  // --------------------------------
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
        openFile({
          path: file.path,
          name: file.name,
          language: getLanguageFromPath(file.path),
          content: `// Content of ${file.name}\n// Loaded from backend...\n\nexport default function Component() {\n  return <div>Hello World</div>;\n}\n`,
        });
      } else {
        toggleFolder(file.path);
      }
    },
    [openFile, toggleFolder]
  );

  // --------------------------------
  // Editor Handlers
  // --------------------------------
  const handleCloseTab = useCallback(
    (paneId: string, tabId: string) => {
      closeFile(tabId);
    },
    [closeFile]
  );

  const handleActivateTab = useCallback(
    (paneId: string, tabId: string) => {
      setActiveFile(tabId);
    },
    [setActiveFile]
  );

  const handleSplitPane = useCallback(
    (paneId: string, direction: "horizontal" | "vertical") => {
      // TODO: Implement pane splitting
      console.log("Split pane:", paneId, direction);
    },
    []
  );

  const handleMoveTab = useCallback(
    (fromPaneId: string, toPaneId: string, tabId: string) => {
      // TODO: Implement tab moving
      console.log("Move tab:", fromPaneId, toPaneId, tabId);
    },
    []
  );

  const handlePinTab = useCallback((paneId: string, tabId: string) => {
    // TODO: Implement tab pinning
    console.log("Pin tab:", paneId, tabId);
  }, []);

  const renderTabContent = useCallback((tab: EditorTab) => {
    return <EditorManager />;
  }, []);

  // --------------------------------
  // Terminal Handlers
  // --------------------------------
  const handleNewTerminal = useCallback(() => {
    const id = `term-${Date.now()}`;
    setTerminals((prev) => [
      ...prev,
      { id, name: `bash ${prev.length + 1}`, active: false },
    ]);
    setActiveTerminalId(id);
  }, []);

  const handleCloseTerminal = useCallback(
    (id: string) => {
      setTerminals((prev) => prev.filter((t) => t.id !== id));
      if (activeTerminalId === id) {
        setActiveTerminalId(terminals[0]?.id || null);
      }
    },
    [activeTerminalId, terminals]
  );

  const renderTerminal = useCallback((terminalId: string) => {
    return (
      <div className="h-full bg-background p-2 font-mono text-sm">
        <div className="text-green-500">$ </div>
        <div className="text-muted-foreground">Terminal: {terminalId}</div>
        <div className="text-muted-foreground mt-2">Type commands here...</div>
      </div>
    );
  }, []);

  // --------------------------------
  // Render
  // --------------------------------
  console.log("[IDEViewV2] Rendering with layout:", layout);

  return (
    <ApplicationShell
      activityBar={
        <ActivityBar
          items={activityItems}
          activeId={activeActivity}
          onItemClick={(id) => {
            if (id === activeActivity) {
              setActiveActivity(null);
              toggleSidebar("left");
            } else {
              setActiveActivity(id as ActivityId);
              if (!layout.leftSidebarCollapsed) {
                // Already open, just switch panel
              } else {
                toggleSidebar("left");
              }
            }
          }}
          className="bg-card"
        />
      }
      primarySidebar={
        activeActivity && !layout.leftSidebarCollapsed ? (
          <SidebarContent
            activity={activeActivity}
            fileTree={fileTree}
            onFileClick={handleFileClick}
          />
        ) : null
      }
      editorArea={
        <EditorArea
          root={editorPaneRoot}
          activePane={activePane}
          onActivePaneChange={setActivePane}
          onCloseTab={handleCloseTab}
          onActivateTab={handleActivateTab}
          onSplitPane={handleSplitPane}
          onMoveTab={handleMoveTab}
          onPinTab={handlePinTab}
          renderTabContent={renderTabContent}
          welcomeScreen={<WelcomeScreen />}
          className="bg-background"
        />
      }
      secondarySidebar={
        <AIPanel
          messages={chatMessages}
          onSendMessage={handleSendMessage}
          onCancelStream={() => setIsAIStreaming(false)}
          onClearChat={() => setChatMessages([])}
          isStreaming={isAIStreaming}
          modelName="Claude Sonnet 4"
          className="bg-card"
        />
      }
      bottomPanel={
        !layout.bottomPanelCollapsed ? (
          <BottomPanel
            activeTab={bottomPanelTab}
            onTabChange={setBottomPanelTab}
            terminals={terminals}
            activeTerminalId={activeTerminalId}
            onTerminalChange={setActiveTerminalId}
            onNewTerminal={handleNewTerminal}
            onCloseTerminal={handleCloseTerminal}
            problems={problems}
            outputMessages={outputMessages}
            renderTerminal={renderTerminal}
            onToggleMaximize={() => {}}
            className="bg-card"
          />
        ) : null
      }
      statusBar={
        <StatusBar
          leftItems={statusBarLeftItems}
          rightItems={statusBarRightItems}
          className="bg-primary text-primary-foreground"
        />
      }
    />
  );
}

// ============================================================================
// Sidebar Content
// ============================================================================

function SidebarContent({
  activity,
  fileTree,
  onFileClick,
}: {
  activity: ActivityId;
  fileTree: FileNode[];
  onFileClick: (file: FileNode) => void;
}) {
  switch (activity) {
    case "explorer":
      return (
        <SidebarContainer
          title="Explorer"
          showSearch
          searchPlaceholder="Search files..."
          actions={[
            {
              id: "new-file",
              icon: FileText,
              label: "New File",
              onClick: () => {},
            },
            {
              id: "new-folder",
              icon: Folder,
              label: "New Folder",
              onClick: () => {},
            },
            {
              id: "refresh",
              icon: RefreshCw,
              label: "Refresh",
              onClick: () => {},
            },
          ]}
          className="bg-card"
        >
          <FileTreeComponent nodes={fileTree} onFileClick={onFileClick} />
        </SidebarContainer>
      );

    case "search":
      return (
        <SidebarContainer
          title="Search"
          showSearch
          searchPlaceholder="Search in files..."
          className="bg-card"
        >
          <div className="p-4 text-sm text-muted-foreground">
            Type to search across all files in the workspace.
          </div>
        </SidebarContainer>
      );

    case "git":
      return (
        <SidebarContainer
          title="Source Control"
          sections={[
            {
              id: "changes",
              title: "Changes",
              content: (
                <div className="p-2 text-sm text-muted-foreground">
                  No changes
                </div>
              ),
              badge: 0,
            },
            {
              id: "staged",
              title: "Staged Changes",
              content: (
                <div className="p-2 text-sm text-muted-foreground">
                  Nothing staged
                </div>
              ),
              badge: 0,
            },
          ]}
          className="bg-card"
        />
      );

    case "agents":
      return (
        <SidebarContainer
          title="Agent Orchestration"
          sections={[
            {
              id: "running",
              title: "Running Agents",
              content: <AgentList />,
              badge: 3,
            },
            {
              id: "available",
              title: "Available Agents",
              content: <AvailableAgents />,
            },
          ]}
          className="bg-card"
        />
      );

    case "experiments":
      return (
        <SidebarContainer title="Experiments" className="bg-card">
          <div className="p-4 text-sm text-muted-foreground">
            A/B tests and feature experiments will appear here.
          </div>
        </SidebarContainer>
      );

    case "extensions":
      return (
        <SidebarContainer
          title="Extensions"
          showSearch
          searchPlaceholder="Search extensions..."
          className="bg-card"
        >
          <div className="p-4 text-sm text-muted-foreground">
            Installed extensions will appear here.
          </div>
        </SidebarContainer>
      );

    case "settings":
      return (
        <SidebarContainer
          title="Settings"
          showSearch
          searchPlaceholder="Search settings..."
          className="bg-card"
        >
          <div className="p-4 text-sm text-muted-foreground">
            Settings editor will appear here.
          </div>
        </SidebarContainer>
      );

    default:
      return null;
  }
}

// ============================================================================
// File Tree Component
// ============================================================================

function FileTreeComponent({
  nodes,
  onFileClick,
  depth = 0,
}: {
  nodes: FileNode[];
  onFileClick: (file: FileNode) => void;
  depth?: number;
}) {
  return (
    <div>
      {nodes.map((node) => (
        <FileTreeNode
          key={node.path}
          node={node}
          onFileClick={onFileClick}
          depth={depth}
        />
      ))}
    </div>
  );
}

function FileTreeNode({
  node,
  onFileClick,
  depth,
}: {
  node: FileNode;
  onFileClick: (file: FileNode) => void;
  depth: number;
}) {
  const isFolder = node.type === "folder";
  const paddingLeft = depth * 12 + 8;

  return (
    <div>
      <button
        onClick={() => onFileClick(node)}
        className={cn(
          "w-full flex items-center gap-1.5 py-1 px-2 text-sm",
          "hover:bg-accent/50 transition-colors text-left"
        )}
        style={{ paddingLeft }}
      >
        {isFolder ? (
          <>
            {node.expanded ? (
              <ChevronDown size={12} />
            ) : (
              <ChevronRight size={12} />
            )}
            {node.expanded ? (
              <FolderOpen size={14} className="text-yellow-500" />
            ) : (
              <Folder size={14} className="text-yellow-500" />
            )}
          </>
        ) : (
          <>
            <span className="w-3" />
            <FileIcon filename={node.name} />
          </>
        )}
        <span className="truncate">{node.name}</span>
      </button>

      {isFolder && node.expanded && node.children && (
        <FileTreeComponent
          nodes={node.children}
          onFileClick={onFileClick}
          depth={depth + 1}
        />
      )}
    </div>
  );
}

// ============================================================================
// File Icon
// ============================================================================

function FileIcon({ filename }: { filename: string }) {
  const ext = filename.split(".").pop()?.toLowerCase();

  const colors: Record<string, string> = {
    tsx: "text-blue-400",
    ts: "text-blue-400",
    jsx: "text-yellow-400",
    js: "text-yellow-400",
    json: "text-yellow-500",
    md: "text-gray-400",
    css: "text-pink-400",
    html: "text-orange-400",
    py: "text-green-400",
    rs: "text-orange-500",
    go: "text-cyan-400",
  };

  return (
    <Code2 size={14} className={colors[ext || ""] || "text-muted-foreground"} />
  );
}

// ============================================================================
// Agent Components
// ============================================================================

function AgentList() {
  const agents = [
    { id: "1", name: "@APEX", status: "running", task: "Code generation" },
    { id: "2", name: "@TENSOR", status: "running", task: "ML optimization" },
    { id: "3", name: "@ARCHITECT", status: "running", task: "System design" },
  ];

  return (
    <div className="space-y-1">
      {agents.map((agent) => (
        <div
          key={agent.id}
          className="flex items-center gap-2 px-2 py-1.5 hover:bg-accent/50 rounded"
        >
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium truncate">{agent.name}</div>
            <div className="text-xs text-muted-foreground truncate">
              {agent.task}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function AvailableAgents() {
  const agents = ["@CIPHER", "@AXIOM", "@VELOCITY", "@QUANTUM", "@FORTRESS"];

  return (
    <div className="space-y-1">
      {agents.map((agent) => (
        <button
          key={agent}
          className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-accent/50 rounded text-left"
        >
          <Plus size={12} className="text-muted-foreground" />
          <span className="text-sm">{agent}</span>
        </button>
      ))}
    </div>
  );
}

// ============================================================================
// Welcome Screen
// ============================================================================

function WelcomeScreen() {
  return (
    <div className="h-full w-full flex flex-col items-center justify-center gap-8 text-muted-foreground">
      <div className="text-6xl font-bold text-primary/20">NEURECTOMY</div>
      <div className="flex flex-col items-center gap-2">
        <p className="text-lg">Welcome to NEURECTOMY IDE</p>
        <p className="text-sm">Open a file or folder to get started</p>
      </div>
      <div className="grid grid-cols-2 gap-4 text-sm">
        <KeyboardShortcut keys={["Ctrl", "P"]} label="Quick Open" />
        <KeyboardShortcut
          keys={["Ctrl", "Shift", "P"]}
          label="Command Palette"
        />
        <KeyboardShortcut keys={["Ctrl", "Shift", "E"]} label="Explorer" />
        <KeyboardShortcut keys={["Ctrl", "`"]} label="Terminal" />
      </div>
    </div>
  );
}

function KeyboardShortcut({ keys, label }: { keys: string[]; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex items-center gap-1">
        {keys.map((key, i) => (
          <span key={i}>
            <kbd className="px-2 py-1 text-xs bg-muted rounded border border-border">
              {key}
            </kbd>
            {i < keys.length - 1 && <span className="mx-0.5">+</span>}
          </span>
        ))}
      </div>
      <span className="text-muted-foreground">{label}</span>
    </div>
  );
}

// ============================================================================
// Helpers
// ============================================================================

function getLanguageFromPath(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase();
  const languageMap: Record<string, string> = {
    ts: "typescript",
    tsx: "typescriptreact",
    js: "javascript",
    jsx: "javascriptreact",
    json: "json",
    md: "markdown",
    css: "css",
    html: "html",
    py: "python",
    rs: "rust",
    go: "go",
  };
  return languageMap[ext || ""] || "plaintext";
}
