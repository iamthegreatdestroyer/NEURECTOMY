/**
 * Bottom Panel Component
 *
 * Multi-tab panel for Terminal, Output, Problems, and Debug Console.
 * Supports multiple terminal instances and filterable output.
 *
 * @module @neurectomy/shell
 * @author @APEX @FLUX
 */

import { useState, useCallback, ReactNode } from "react";
import { cn } from "@/lib/utils";
import {
  Terminal,
  AlertTriangle,
  Info,
  Bug,
  X,
  Plus,
  ChevronDown,
  Maximize2,
  Minimize2,
  Filter,
  Trash2,
  Play,
  Square,
  SplitSquareHorizontal,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export type BottomPanelTab = "terminal" | "output" | "problems" | "debug";

export interface TerminalInstance {
  id: string;
  name: string;
  cwd?: string;
  active: boolean;
}

export interface Problem {
  id: string;
  severity: "error" | "warning" | "info";
  message: string;
  source: string;
  file?: string;
  line?: number;
  column?: number;
}

export interface OutputMessage {
  id: string;
  channel: string;
  message: string;
  timestamp: Date;
  level: "info" | "warn" | "error";
}

export interface BottomPanelProps {
  activeTab: BottomPanelTab;
  onTabChange: (tab: BottomPanelTab) => void;
  terminals: TerminalInstance[];
  activeTerminalId: string | null;
  onTerminalChange: (id: string) => void;
  onNewTerminal: () => void;
  onCloseTerminal: (id: string) => void;
  problems: Problem[];
  outputMessages: OutputMessage[];
  renderTerminal: (terminalId: string) => ReactNode;
  isMaximized?: boolean;
  onToggleMaximize?: () => void;
  className?: string;
}

// ============================================================================
// Main Component
// ============================================================================

export function BottomPanel({
  activeTab,
  onTabChange,
  terminals,
  activeTerminalId,
  onTerminalChange,
  onNewTerminal,
  onCloseTerminal,
  problems,
  outputMessages,
  renderTerminal,
  isMaximized,
  onToggleMaximize,
  className,
}: BottomPanelProps) {
  const [outputFilter, setOutputFilter] = useState<string>("all");
  const [problemFilter, setProblemFilter] = useState<
    Problem["severity"] | "all"
  >("all");

  const errorCount = problems.filter((p) => p.severity === "error").length;
  const warningCount = problems.filter((p) => p.severity === "warning").length;

  const tabs: {
    id: BottomPanelTab;
    label: string;
    icon: typeof Terminal;
    badge?: number | string;
  }[] = [
    { id: "terminal", label: "Terminal", icon: Terminal },
    { id: "output", label: "Output", icon: Info },
    {
      id: "problems",
      label: "Problems",
      icon: AlertTriangle,
      badge:
        errorCount > 0
          ? errorCount
          : warningCount > 0
            ? warningCount
            : undefined,
    },
    { id: "debug", label: "Debug Console", icon: Bug },
  ];

  return (
    <div className={cn("h-full flex flex-col bg-background", className)}>
      {/* Tab Bar */}
      <div className="h-9 flex items-center justify-between border-b border-border shrink-0">
        <div className="flex items-center">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={cn(
                "h-9 px-3 flex items-center gap-2 text-sm border-b-2 transition-colors",
                activeTab === tab.id
                  ? "border-primary text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              )}
            >
              <tab.icon size={14} />
              <span>{tab.label}</span>
              {tab.badge !== undefined && (
                <span
                  className={cn(
                    "px-1.5 py-0.5 text-xs rounded-full",
                    tab.id === "problems" && errorCount > 0
                      ? "bg-destructive text-destructive-foreground"
                      : "bg-muted text-muted-foreground"
                  )}
                >
                  {tab.badge}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Panel Actions */}
        <div className="flex items-center gap-1 px-2">
          {onToggleMaximize && (
            <button
              onClick={onToggleMaximize}
              className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground"
              title={isMaximized ? "Restore" : "Maximize"}
            >
              {isMaximized ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
            </button>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === "terminal" && (
          <TerminalPanel
            terminals={terminals}
            activeId={activeTerminalId}
            onChange={onTerminalChange}
            onNew={onNewTerminal}
            onClose={onCloseTerminal}
            renderTerminal={renderTerminal}
          />
        )}
        {activeTab === "output" && (
          <OutputPanel
            messages={outputMessages}
            filter={outputFilter}
            onFilterChange={setOutputFilter}
          />
        )}
        {activeTab === "problems" && (
          <ProblemsPanel
            problems={problems}
            filter={problemFilter}
            onFilterChange={setProblemFilter}
          />
        )}
        {activeTab === "debug" && <DebugConsolePanel />}
      </div>
    </div>
  );
}

// ============================================================================
// Terminal Panel
// ============================================================================

interface TerminalPanelProps {
  terminals: TerminalInstance[];
  activeId: string | null;
  onChange: (id: string) => void;
  onNew: () => void;
  onClose: (id: string) => void;
  renderTerminal: (id: string) => ReactNode;
}

function TerminalPanel({
  terminals,
  activeId,
  onChange,
  onNew,
  onClose,
  renderTerminal,
}: TerminalPanelProps) {
  return (
    <div className="h-full flex flex-col">
      {/* Terminal Tabs */}
      <div className="h-8 flex items-center border-b border-border bg-muted/30 shrink-0">
        <div className="flex-1 flex items-center overflow-x-auto">
          {terminals.map((terminal) => (
            <button
              key={terminal.id}
              onClick={() => onChange(terminal.id)}
              className={cn(
                "group h-8 px-3 flex items-center gap-2 text-xs border-r border-border",
                terminal.id === activeId
                  ? "bg-background text-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
              )}
            >
              <Terminal size={12} />
              <span>{terminal.name}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onClose(terminal.id);
                }}
                className={cn(
                  "p-0.5 rounded hover:bg-accent",
                  "opacity-0 group-hover:opacity-100"
                )}
              >
                <X size={10} />
              </button>
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1 px-2">
          <button
            onClick={onNew}
            className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground"
            title="New Terminal"
          >
            <Plus size={14} />
          </button>
          <button
            className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground"
            title="Split Terminal"
          >
            <SplitSquareHorizontal size={14} />
          </button>
        </div>
      </div>

      {/* Terminal Content */}
      <div className="flex-1 overflow-hidden">
        {activeId ? (
          renderTerminal(activeId)
        ) : (
          <div className="h-full flex items-center justify-center text-muted-foreground">
            <button
              onClick={onNew}
              className="flex items-center gap-2 px-4 py-2 rounded bg-muted hover:bg-accent transition-colors"
            >
              <Plus size={16} />
              <span>New Terminal</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Output Panel
// ============================================================================

interface OutputPanelProps {
  messages: OutputMessage[];
  filter: string;
  onFilterChange: (filter: string) => void;
}

function OutputPanel({ messages, filter, onFilterChange }: OutputPanelProps) {
  const channels = [
    "all",
    ...Array.from(new Set(messages.map((m) => m.channel))),
  ];
  const filteredMessages =
    filter === "all" ? messages : messages.filter((m) => m.channel === filter);

  return (
    <div className="h-full flex flex-col">
      {/* Filter Bar */}
      <div className="h-8 flex items-center gap-2 px-3 border-b border-border bg-muted/30 shrink-0">
        <Filter size={12} className="text-muted-foreground" />
        <select
          value={filter}
          onChange={(e) => onFilterChange(e.target.value)}
          className="h-6 text-xs bg-background border border-border rounded px-2"
        >
          {channels.map((ch) => (
            <option key={ch} value={ch}>
              {ch === "all" ? "All Channels" : ch}
            </option>
          ))}
        </select>
        <div className="flex-1" />
        <button
          className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground"
          title="Clear Output"
        >
          <Trash2 size={12} />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto font-mono text-xs">
        {filteredMessages.length === 0 ? (
          <div className="h-full flex items-center justify-center text-muted-foreground">
            No output messages
          </div>
        ) : (
          filteredMessages.map((msg) => (
            <div
              key={msg.id}
              className={cn(
                "px-3 py-0.5 hover:bg-accent/50",
                msg.level === "error" && "text-destructive",
                msg.level === "warn" && "text-yellow-500"
              )}
            >
              <span className="text-muted-foreground">
                [{msg.timestamp.toLocaleTimeString()}]
              </span>{" "}
              <span className="text-muted-foreground">[{msg.channel}]</span>{" "}
              <span>{msg.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Problems Panel
// ============================================================================

interface ProblemsPanelProps {
  problems: Problem[];
  filter: Problem["severity"] | "all";
  onFilterChange: (filter: Problem["severity"] | "all") => void;
}

function ProblemsPanel({
  problems,
  filter,
  onFilterChange,
}: ProblemsPanelProps) {
  const filteredProblems =
    filter === "all" ? problems : problems.filter((p) => p.severity === filter);

  const errorCount = problems.filter((p) => p.severity === "error").length;
  const warningCount = problems.filter((p) => p.severity === "warning").length;
  const infoCount = problems.filter((p) => p.severity === "info").length;

  const SeverityIcon = ({ severity }: { severity: Problem["severity"] }) => {
    switch (severity) {
      case "error":
        return <X size={14} className="text-destructive" />;
      case "warning":
        return <AlertTriangle size={14} className="text-yellow-500" />;
      case "info":
        return <Info size={14} className="text-blue-500" />;
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Filter Bar */}
      <div className="h-8 flex items-center gap-3 px-3 border-b border-border bg-muted/30 shrink-0">
        <button
          onClick={() => onFilterChange("all")}
          className={cn(
            "flex items-center gap-1 text-xs",
            filter === "all" ? "text-foreground" : "text-muted-foreground"
          )}
        >
          All ({problems.length})
        </button>
        <button
          onClick={() => onFilterChange("error")}
          className={cn(
            "flex items-center gap-1 text-xs",
            filter === "error" ? "text-destructive" : "text-muted-foreground"
          )}
        >
          <X size={12} /> {errorCount}
        </button>
        <button
          onClick={() => onFilterChange("warning")}
          className={cn(
            "flex items-center gap-1 text-xs",
            filter === "warning" ? "text-yellow-500" : "text-muted-foreground"
          )}
        >
          <AlertTriangle size={12} /> {warningCount}
        </button>
        <button
          onClick={() => onFilterChange("info")}
          className={cn(
            "flex items-center gap-1 text-xs",
            filter === "info" ? "text-blue-500" : "text-muted-foreground"
          )}
        >
          <Info size={12} /> {infoCount}
        </button>
      </div>

      {/* Problems List */}
      <div className="flex-1 overflow-y-auto">
        {filteredProblems.length === 0 ? (
          <div className="h-full flex items-center justify-center text-muted-foreground">
            No problems detected
          </div>
        ) : (
          filteredProblems.map((problem) => (
            <div
              key={problem.id}
              className="flex items-start gap-2 px-3 py-2 hover:bg-accent/50 cursor-pointer border-b border-border"
            >
              <SeverityIcon severity={problem.severity} />
              <div className="flex-1 min-w-0">
                <p className="text-sm truncate">{problem.message}</p>
                <p className="text-xs text-muted-foreground">
                  {problem.source}
                  {problem.file && ` • ${problem.file}`}
                  {problem.line && `:${problem.line}`}
                  {problem.column && `:${problem.column}`}
                </p>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Debug Console Panel
// ============================================================================

function DebugConsolePanel() {
  return (
    <div className="h-full flex flex-col">
      {/* Debug Actions */}
      <div className="h-8 flex items-center gap-2 px-3 border-b border-border bg-muted/30 shrink-0">
        <button
          className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground"
          title="Start Debugging"
        >
          <Play size={14} />
        </button>
        <button
          className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground"
          title="Stop Debugging"
        >
          <Square size={14} />
        </button>
        <div className="flex-1" />
        <button
          className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground"
          title="Clear Console"
        >
          <Trash2 size={12} />
        </button>
      </div>

      {/* Console Content */}
      <div className="flex-1 overflow-y-auto font-mono text-xs">
        <div className="h-full flex items-center justify-center text-muted-foreground">
          Debug console is ready. Start debugging to see output.
        </div>
      </div>

      {/* Input */}
      <div className="h-8 flex items-center px-2 border-t border-border">
        <span className="text-primary mr-2">{">"}</span>
        <input
          type="text"
          placeholder="Evaluate expression..."
          className="flex-1 h-6 bg-transparent text-xs focus:outline-none"
        />
      </div>
    </div>
  );
}

export default BottomPanel;
