/**
 * Agent Execution Graph - Dust.tt-style Visual Agent Collaboration
 *
 * Real-time visualization of multi-agent execution flow with:
 * - Visual node graph showing agent collaboration
 * - Real-time status: idle, running, completed, error
 * - Timing information on each node
 * - Click to expand agent reasoning
 *
 * Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
 */

import { useState, useEffect, useCallback } from "react";
import { AnimatePresence } from "framer-motion";
import { MotionDiv } from "@/lib/motion";
import {
  CheckCircle2,
  Circle,
  Loader2,
  AlertCircle,
  Clock,
  ChevronDown,
  ChevronUp,
  Zap,
  ArrowRight,
  Pause,
  Play,
  RotateCcw,
  Maximize2,
  Minimize2,
} from "lucide-react";
import { cn } from "@/lib/utils";

// =============================================================================
// Types
// =============================================================================

export type AgentStatus =
  | "idle"
  | "queued"
  | "running"
  | "completed"
  | "error"
  | "paused";

export interface AgentNode {
  id: string;
  codename: string;
  displayName: string;
  status: AgentStatus;
  /** Execution time in milliseconds */
  executionTime?: number;
  /** Start timestamp */
  startedAt?: number;
  /** End timestamp */
  completedAt?: number;
  /** Current task description */
  currentTask?: string;
  /** Reasoning/thought process */
  reasoning?: string[];
  /** Output produced */
  output?: string;
  /** Error message if status is error */
  error?: string;
  /** Dependencies - agent IDs that must complete first */
  dependencies?: string[];
  /** Color theme */
  color: string;
}

export interface AgentExecutionGraphProps {
  nodes: AgentNode[];
  onNodeClick?: (node: AgentNode) => void;
  onPause?: () => void;
  onResume?: () => void;
  onReset?: () => void;
  isPaused?: boolean;
  className?: string;
  compact?: boolean;
}

// =============================================================================
// Mock Data for Demo
// =============================================================================

export const mockExecutionNodes: AgentNode[] = [
  {
    id: "apex-1",
    codename: "@APEX",
    displayName: "Code Analysis",
    status: "completed",
    executionTime: 2340,
    startedAt: Date.now() - 10000,
    completedAt: Date.now() - 7660,
    currentTask: "Analyzing code structure and patterns",
    reasoning: [
      "Scanning project structure...",
      "Identified 47 TypeScript files",
      "Found 12 React components",
      "Detected design patterns: Factory, Observer, Strategy",
    ],
    output: "Analysis complete: High-quality codebase with 94% type coverage",
    color: "neural-blue",
  },
  {
    id: "cipher-1",
    codename: "@CIPHER",
    displayName: "Security Audit",
    status: "running",
    executionTime: 5230,
    startedAt: Date.now() - 5230,
    currentTask: "Scanning for vulnerabilities",
    reasoning: [
      "Checking for XSS vulnerabilities...",
      "Validating input sanitization...",
      "Reviewing authentication flow...",
    ],
    dependencies: ["apex-1"],
    color: "cipher-cyan",
  },
  {
    id: "tensor-1",
    codename: "@TENSOR",
    displayName: "ML Optimization",
    status: "queued",
    currentTask: "Pending security audit completion",
    dependencies: ["cipher-1"],
    color: "synapse-purple",
  },
  {
    id: "flux-1",
    codename: "@FLUX",
    displayName: "Deployment Prep",
    status: "idle",
    currentTask: "Waiting for ML optimization",
    dependencies: ["tensor-1"],
    color: "forge-orange",
  },
];

// =============================================================================
// Status Components
// =============================================================================

function StatusIcon({
  status,
  size = 16,
}: {
  status: AgentStatus;
  size?: number;
}) {
  switch (status) {
    case "completed":
      return <CheckCircle2 size={size} className="text-matrix-green" />;
    case "running":
      return <Loader2 size={size} className="text-neural-blue animate-spin" />;
    case "queued":
      return <Clock size={size} className="text-forge-orange" />;
    case "error":
      return <AlertCircle size={size} className="text-red-500" />;
    case "paused":
      return <Pause size={size} className="text-yellow-500" />;
    default:
      return <Circle size={size} className="text-muted-foreground" />;
  }
}

function StatusBadge({ status }: { status: AgentStatus }) {
  const statusConfig: Record<
    AgentStatus,
    { bg: string; text: string; label: string }
  > = {
    idle: { bg: "bg-muted", text: "text-muted-foreground", label: "Idle" },
    queued: {
      bg: "bg-forge-orange/20",
      text: "text-forge-orange",
      label: "Queued",
    },
    running: {
      bg: "bg-neural-blue/20",
      text: "text-neural-blue",
      label: "Running",
    },
    completed: {
      bg: "bg-matrix-green/20",
      text: "text-matrix-green",
      label: "Done",
    },
    error: { bg: "bg-red-500/20", text: "text-red-500", label: "Error" },
    paused: {
      bg: "bg-yellow-500/20",
      text: "text-yellow-500",
      label: "Paused",
    },
  };

  const config = statusConfig[status];

  return (
    <span
      className={cn(
        "text-xs px-2 py-0.5 rounded-full font-medium",
        config.bg,
        config.text
      )}
    >
      {config.label}
    </span>
  );
}

// =============================================================================
// Agent Node Component
// =============================================================================

function AgentNodeCard({
  node,
  onClick,
  isExpanded,
  compact,
}: {
  node: AgentNode;
  onClick?: () => void;
  isExpanded: boolean;
  compact?: boolean;
}) {
  const [elapsedTime, setElapsedTime] = useState(node.executionTime || 0);

  // Update elapsed time for running nodes
  useEffect(() => {
    if (node.status === "running" && node.startedAt) {
      const interval = setInterval(() => {
        setElapsedTime(Date.now() - node.startedAt!);
      }, 100);
      return () => clearInterval(interval);
    }
    setElapsedTime(node.executionTime || 0);
  }, [node.status, node.startedAt, node.executionTime]);

  const formatTime = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  const colorMap: Record<string, string> = {
    "neural-blue": "border-neural-blue/50 hover:border-neural-blue",
    "cipher-cyan": "border-cipher-cyan/50 hover:border-cipher-cyan",
    "synapse-purple": "border-synapse-purple/50 hover:border-synapse-purple",
    "forge-orange": "border-forge-orange/50 hover:border-forge-orange",
    "matrix-green": "border-matrix-green/50 hover:border-matrix-green",
  };

  const glowMap: Record<string, string> = {
    "neural-blue": "shadow-neural-blue/20",
    "cipher-cyan": "shadow-cipher-cyan/20",
    "synapse-purple": "shadow-synapse-purple/20",
    "forge-orange": "shadow-forge-orange/20",
    "matrix-green": "shadow-matrix-green/20",
  };

  return (
    <MotionDiv
      layout
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      className={cn(
        "bg-card rounded-lg border-2 transition-all duration-200 cursor-pointer",
        colorMap[node.color] || "border-border hover:border-primary",
        node.status === "running" && `shadow-lg ${glowMap[node.color]}`,
        compact ? "min-w-[100px]" : "min-w-[140px]"
      )}
      onClick={onClick}
    >
      {/* Header */}
      <div
        className={cn(
          "px-3 py-2 border-b border-border",
          compact && "px-2 py-1.5"
        )}
      >
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <StatusIcon status={node.status} size={compact ? 14 : 16} />
            <span className={cn("font-bold", compact ? "text-xs" : "text-sm")}>
              {node.codename}
            </span>
          </div>
          {(node.status === "running" || node.status === "completed") && (
            <span
              className={cn(
                "font-mono",
                compact ? "text-[10px]" : "text-xs",
                "text-muted-foreground"
              )}
            >
              {formatTime(elapsedTime)}
            </span>
          )}
        </div>
        {!compact && (
          <div className="text-xs text-muted-foreground mt-1 truncate">
            {node.displayName}
          </div>
        )}
      </div>

      {/* Body */}
      <div className={cn("px-3 py-2", compact && "px-2 py-1.5")}>
        <div className="flex items-center justify-between">
          <StatusBadge status={node.status} />
          {onClick && (
            <MotionDiv
              animate={{ rotate: isExpanded ? 180 : 0 }}
              className="text-muted-foreground"
            >
              <ChevronDown size={14} />
            </MotionDiv>
          )}
        </div>

        {/* Current Task */}
        {!compact && node.currentTask && (
          <div className="mt-2 text-xs text-muted-foreground line-clamp-2">
            {node.currentTask}
          </div>
        )}
      </div>

      {/* Expanded Details */}
      <AnimatePresence>
        {isExpanded && !compact && (
          <MotionDiv
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="px-3 py-2 border-t border-border bg-muted/30">
              {/* Reasoning */}
              {node.reasoning && node.reasoning.length > 0 && (
                <div className="mb-2">
                  <div className="text-xs font-medium text-muted-foreground mb-1">
                    Reasoning:
                  </div>
                  <div className="space-y-1">
                    {node.reasoning.map((step, i) => (
                      <div key={i} className="flex items-start gap-2 text-xs">
                        <div className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5 flex-shrink-0" />
                        <span>{step}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Output */}
              {node.output && (
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-1">
                    Output:
                  </div>
                  <div className="text-xs bg-background rounded p-2 font-mono">
                    {node.output}
                  </div>
                </div>
              )}

              {/* Error */}
              {node.error && (
                <div>
                  <div className="text-xs font-medium text-red-500 mb-1">
                    Error:
                  </div>
                  <div className="text-xs bg-red-500/10 text-red-500 rounded p-2 font-mono">
                    {node.error}
                  </div>
                </div>
              )}
            </div>
          </MotionDiv>
        )}
      </AnimatePresence>
    </MotionDiv>
  );
}

// =============================================================================
// Connection Line Component
// =============================================================================

function ConnectionArrow({ isActive }: { isActive?: boolean }) {
  return (
    <div className="flex items-center px-1">
      <div
        className={cn(
          "w-8 h-0.5 rounded transition-colors",
          isActive ? "bg-primary" : "bg-border"
        )}
      />
      <ArrowRight
        size={14}
        className={cn(
          "transition-colors -ml-1",
          isActive ? "text-primary" : "text-border"
        )}
      />
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

export function AgentExecutionGraph({
  nodes = mockExecutionNodes,
  onNodeClick,
  onPause,
  onResume,
  onReset,
  isPaused = false,
  className,
  compact = false,
}: AgentExecutionGraphProps) {
  const [expandedNodeId, setExpandedNodeId] = useState<string | null>(null);
  const [isMinimized, setIsMinimized] = useState(false);

  const handleNodeClick = useCallback(
    (node: AgentNode) => {
      if (expandedNodeId === node.id) {
        setExpandedNodeId(null);
      } else {
        setExpandedNodeId(node.id);
      }
      onNodeClick?.(node);
    },
    [expandedNodeId, onNodeClick]
  );

  const activeCount = nodes.filter((n) => n.status === "running").length;
  const completedCount = nodes.filter((n) => n.status === "completed").length;
  const totalCount = nodes.length;

  // Progress calculation
  const progress = (completedCount / totalCount) * 100;

  if (isMinimized) {
    return (
      <div
        className={cn(
          "flex items-center justify-between px-4 py-2 bg-card border-t border-border",
          className
        )}
      >
        <div className="flex items-center gap-3">
          <Zap size={16} className="text-primary" />
          <span className="text-sm font-medium">Agent Pipeline</span>
          <div className="flex items-center gap-2">
            <div className="w-24 h-1.5 bg-muted rounded-full overflow-hidden">
              <MotionDiv
                className="h-full bg-gradient-to-r from-neural-blue via-synapse-purple to-matrix-green"
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
              />
            </div>
            <span className="text-xs text-muted-foreground">
              {completedCount}/{totalCount}
            </span>
          </div>
        </div>
        <button
          onClick={() => setIsMinimized(false)}
          className="p-1 hover:bg-muted rounded transition-colors"
          title="Expand"
        >
          <Maximize2 size={14} />
        </button>
      </div>
    );
  }

  return (
    <div className={cn("bg-card border-t border-border", className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border">
        <div className="flex items-center gap-3">
          <Zap size={16} className="text-primary" />
          <span className="text-sm font-medium">Agent Execution Pipeline</span>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <Loader2
                size={12}
                className={
                  activeCount > 0 ? "animate-spin text-neural-blue" : ""
                }
              />
              {activeCount} active
            </span>
            <span>•</span>
            <span className="flex items-center gap-1">
              <CheckCircle2 size={12} className="text-matrix-green" />
              {completedCount}/{totalCount}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-1">
          {/* Controls */}
          {isPaused ? (
            <button
              onClick={onResume}
              className="p-1.5 hover:bg-muted rounded transition-colors text-matrix-green"
              title="Resume"
            >
              <Play size={14} />
            </button>
          ) : (
            <button
              onClick={onPause}
              className="p-1.5 hover:bg-muted rounded transition-colors"
              title="Pause"
            >
              <Pause size={14} />
            </button>
          )}
          <button
            onClick={onReset}
            className="p-1.5 hover:bg-muted rounded transition-colors"
            title="Reset"
          >
            <RotateCcw size={14} />
          </button>
          <div className="w-px h-4 bg-border mx-1" />
          <button
            onClick={() => setIsMinimized(true)}
            className="p-1.5 hover:bg-muted rounded transition-colors"
            title="Minimize"
          >
            <Minimize2 size={14} />
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="h-1 bg-muted">
        <MotionDiv
          className="h-full bg-gradient-to-r from-neural-blue via-synapse-purple to-matrix-green"
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.3 }}
        />
      </div>

      {/* Graph Content */}
      <div className={cn("p-4 overflow-x-auto", compact && "p-2")}>
        <div className="flex items-start gap-1 min-w-max">
          {nodes.map((node, index) => (
            <div key={node.id} className="flex items-center">
              <AgentNodeCard
                node={node}
                onClick={() => handleNodeClick(node)}
                isExpanded={expandedNodeId === node.id}
                compact={compact}
              />
              {index < nodes.length - 1 && (
                <ConnectionArrow
                  isActive={
                    node.status === "completed" ||
                    nodes[index + 1]?.status === "running"
                  }
                />
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default AgentExecutionGraph;
