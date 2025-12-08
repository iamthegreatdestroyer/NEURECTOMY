/**
 * MLflow Experiment Sidebar - Experiment Tracking & Comparison
 *
 * MLflow-inspired experiment management with:
 * - Side-by-side run comparison
 * - Inline metrics charts
 * - Parameter diff highlighting
 * - Run history with status indicators
 *
 * Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
 */

import { useState, useMemo } from "react";
import { AnimatePresence } from "framer-motion";
import { MotionDiv } from "@/lib/motion";
import {
  FlaskConical,
  Play,
  CheckCircle2,
  XCircle,
  Clock,
  ChevronDown,
  ChevronRight,
  TrendingUp,
  TrendingDown,
  Minus,
  BarChart3,
  GitCompare,
  Star,
  StarOff,
  Trash2,
  Download,
  Filter,
  Search,
  SortAsc,
  Eye,
  EyeOff,
} from "lucide-react";
import { cn } from "@/lib/utils";

// =============================================================================
// Types
// =============================================================================

export type RunStatus = "running" | "completed" | "failed" | "scheduled";

export interface ExperimentRun {
  id: string;
  name: string;
  status: RunStatus;
  startTime: Date;
  endTime?: Date;
  duration?: number;
  metrics: Record<string, number>;
  params: Record<string, string | number>;
  tags?: Record<string, string>;
  starred?: boolean;
  artifacts?: string[];
}

export interface Experiment {
  id: string;
  name: string;
  description?: string;
  runs: ExperimentRun[];
  createdAt: Date;
  lastActivity: Date;
}

export interface ExperimentSidebarProps {
  experiments?: Experiment[];
  selectedRuns?: string[];
  onSelectRun?: (runId: string) => void;
  onCompareRuns?: (runIds: string[]) => void;
  onStarRun?: (runId: string) => void;
  onDeleteRun?: (runId: string) => void;
  className?: string;
}

// =============================================================================
// Mock Data
// =============================================================================

export const mockExperiments: Experiment[] = [
  {
    id: "exp-1",
    name: "Agent Performance Optimization",
    description: "Optimizing response time and accuracy",
    createdAt: new Date(Date.now() - 604800000),
    lastActivity: new Date(),
    runs: [
      {
        id: "run-1",
        name: "baseline-v1",
        status: "completed",
        startTime: new Date(Date.now() - 3600000),
        endTime: new Date(Date.now() - 3300000),
        duration: 300000,
        metrics: {
          accuracy: 0.892,
          f1_score: 0.876,
          latency_ms: 245,
          throughput: 1250,
        },
        params: {
          model: "gpt-4-turbo",
          temperature: 0.7,
          max_tokens: 2048,
          batch_size: 32,
        },
        starred: true,
      },
      {
        id: "run-2",
        name: "optimized-v1",
        status: "completed",
        startTime: new Date(Date.now() - 1800000),
        endTime: new Date(Date.now() - 1500000),
        duration: 300000,
        metrics: {
          accuracy: 0.923,
          f1_score: 0.912,
          latency_ms: 198,
          throughput: 1580,
        },
        params: {
          model: "gpt-4-turbo",
          temperature: 0.5,
          max_tokens: 1024,
          batch_size: 64,
        },
        starred: false,
      },
      {
        id: "run-3",
        name: "experiment-v2",
        status: "running",
        startTime: new Date(Date.now() - 600000),
        metrics: {
          accuracy: 0.915,
          f1_score: 0.901,
          latency_ms: 210,
          throughput: 1420,
        },
        params: {
          model: "gpt-4o",
          temperature: 0.6,
          max_tokens: 1536,
          batch_size: 48,
        },
      },
      {
        id: "run-4",
        name: "failed-attempt",
        status: "failed",
        startTime: new Date(Date.now() - 7200000),
        endTime: new Date(Date.now() - 7100000),
        duration: 100000,
        metrics: {},
        params: {
          model: "gpt-4-turbo",
          temperature: 1.5,
          max_tokens: 4096,
          batch_size: 128,
        },
      },
    ],
  },
  {
    id: "exp-2",
    name: "Code Generation Quality",
    description: "Measuring code generation accuracy",
    createdAt: new Date(Date.now() - 1209600000),
    lastActivity: new Date(Date.now() - 86400000),
    runs: [
      {
        id: "run-5",
        name: "codegen-baseline",
        status: "completed",
        startTime: new Date(Date.now() - 172800000),
        endTime: new Date(Date.now() - 172500000),
        duration: 300000,
        metrics: {
          pass_rate: 0.78,
          syntax_errors: 12,
          compile_success: 0.92,
        },
        params: {
          language: "typescript",
          context_window: 8000,
        },
      },
    ],
  },
];

// =============================================================================
// Sub-Components
// =============================================================================

function RunStatusIcon({ status }: { status: RunStatus }) {
  switch (status) {
    case "completed":
      return <CheckCircle2 size={12} className="text-matrix-green" />;
    case "running":
      return (
        <div className="relative">
          <div className="w-3 h-3 rounded-full bg-neural-blue/30 animate-ping absolute" />
          <div className="w-3 h-3 rounded-full bg-neural-blue relative" />
        </div>
      );
    case "failed":
      return <XCircle size={12} className="text-red-500" />;
    case "scheduled":
      return <Clock size={12} className="text-muted-foreground" />;
  }
}

function MetricBadge({
  name,
  value,
  comparison,
  format = "number",
}: {
  name: string;
  value: number;
  comparison?: number;
  format?: "number" | "percent" | "ms";
}) {
  const formatValue = (v: number) => {
    if (format === "percent") return `${(v * 100).toFixed(1)}%`;
    if (format === "ms") return `${v.toFixed(0)}ms`;
    return v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(3);
  };

  const diff = comparison !== undefined ? value - comparison : undefined;
  const isImprovement =
    diff !== undefined && (name.includes("latency") ? diff < 0 : diff > 0);

  return (
    <div className="flex items-center justify-between text-xs">
      <span className="text-muted-foreground">{name}</span>
      <div className="flex items-center gap-1">
        <span className="font-mono">{formatValue(value)}</span>
        {diff !== undefined && diff !== 0 && (
          <span
            className={cn(
              "flex items-center text-[10px]",
              isImprovement ? "text-matrix-green" : "text-red-500"
            )}
          >
            {isImprovement ? (
              <TrendingUp size={10} />
            ) : (
              <TrendingDown size={10} />
            )}
            {Math.abs(diff) < 1
              ? (Math.abs(diff) * 100).toFixed(1) + "%"
              : Math.abs(diff).toFixed(1)}
          </span>
        )}
      </div>
    </div>
  );
}

function MiniChart({
  values,
  color = "primary",
}: {
  values: number[];
  color?: string;
}) {
  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = max - min || 1;

  return (
    <div className="flex items-end gap-0.5 h-6">
      {values.map((v, i) => (
        <div
          key={i}
          className={cn("w-1 rounded-t bg-primary/60", `bg-${color}/60`)}
          style={{ height: `${((v - min) / range) * 100}%`, minHeight: "2px" }}
        />
      ))}
    </div>
  );
}

function RunCard({
  run,
  isSelected,
  isExpanded,
  onSelect,
  onToggle,
  onStar,
  onDelete,
  comparisonRun,
}: {
  run: ExperimentRun;
  isSelected: boolean;
  isExpanded: boolean;
  onSelect: () => void;
  onToggle: () => void;
  onStar?: () => void;
  onDelete?: () => void;
  comparisonRun?: ExperimentRun;
}) {
  const formatDuration = (ms?: number) => {
    if (!ms) return "-";
    if (ms < 60000) return `${Math.floor(ms / 1000)}s`;
    return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
  };

  const formatTimeAgo = (date: Date) => {
    const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
    if (seconds < 60) return "just now";
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  };

  return (
    <MotionDiv
      layout
      className={cn(
        "rounded-lg border transition-all",
        isSelected
          ? "border-primary bg-primary/5"
          : "border-border bg-card/50 hover:border-primary/50"
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2">
        <button
          onClick={onSelect}
          className={cn(
            "w-4 h-4 rounded border-2 flex items-center justify-center transition-colors",
            isSelected
              ? "bg-primary border-primary"
              : "border-muted-foreground/50"
          )}
        >
          {isSelected && (
            <CheckCircle2 size={10} className="text-primary-foreground" />
          )}
        </button>

        <RunStatusIcon status={run.status} />

        <button
          onClick={onToggle}
          className="flex-1 flex items-center gap-2 text-left"
        >
          <span className="text-sm font-medium truncate">{run.name}</span>
          <MotionDiv animate={{ rotate: isExpanded ? 90 : 0 }}>
            <ChevronRight size={12} className="text-muted-foreground" />
          </MotionDiv>
        </button>

        <div className="flex items-center gap-1">
          {onStar && (
            <button
              onClick={onStar}
              className="p-1 hover:bg-muted rounded transition-colors"
            >
              {run.starred ? (
                <Star
                  size={12}
                  className="text-forge-orange fill-forge-orange"
                />
              ) : (
                <StarOff size={12} className="text-muted-foreground" />
              )}
            </button>
          )}
        </div>
      </div>

      {/* Meta */}
      <div className="px-3 pb-2 flex items-center gap-3 text-xs text-muted-foreground">
        <span>{formatTimeAgo(run.startTime)}</span>
        {run.duration && (
          <>
            <span>•</span>
            <span>{formatDuration(run.duration)}</span>
          </>
        )}
      </div>

      {/* Expanded Content */}
      <AnimatePresence>
        {isExpanded && (
          <MotionDiv
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="px-3 pb-3 space-y-3 border-t border-border pt-3">
              {/* Metrics */}
              {Object.keys(run.metrics).length > 0 && (
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-2 flex items-center gap-1">
                    <BarChart3 size={12} />
                    Metrics
                  </div>
                  <div className="space-y-1">
                    {Object.entries(run.metrics).map(([key, value]) => (
                      <MetricBadge
                        key={key}
                        name={key.replace(/_/g, " ")}
                        value={value}
                        comparison={comparisonRun?.metrics[key]}
                        format={
                          key.includes("accuracy") ||
                          key.includes("rate") ||
                          key.includes("score")
                            ? "percent"
                            : key.includes("ms") || key.includes("latency")
                              ? "ms"
                              : "number"
                        }
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Parameters */}
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-2">
                  Parameters
                </div>
                <div className="grid grid-cols-2 gap-x-3 gap-y-1">
                  {Object.entries(run.params).map(([key, value]) => (
                    <div
                      key={key}
                      className="flex items-center justify-between text-xs"
                    >
                      <span className="text-muted-foreground truncate">
                        {key}
                      </span>
                      <span
                        className={cn(
                          "font-mono",
                          comparisonRun &&
                            comparisonRun.params[key] !== value &&
                            "text-forge-orange"
                        )}
                      >
                        {String(value)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Actions */}
              <div className="flex gap-2 pt-1">
                <button
                  title="View Run Details"
                  className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs bg-muted hover:bg-muted/80 rounded transition-colors"
                >
                  <Eye size={12} />
                  <span>View</span>
                </button>
                <button
                  title="Download Artifacts"
                  className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs bg-muted hover:bg-muted/80 rounded transition-colors"
                >
                  <Download size={12} />
                  <span>Artifacts</span>
                </button>
                {onDelete && (
                  <button
                    onClick={onDelete}
                    title="Delete Run"
                    aria-label="Delete Run"
                    className="px-2 py-1.5 text-xs bg-muted hover:bg-red-500/20 hover:text-red-500 rounded transition-colors"
                  >
                    <Trash2 size={12} />
                  </button>
                )}
              </div>
            </div>
          </MotionDiv>
        )}
      </AnimatePresence>
    </MotionDiv>
  );
}

// =============================================================================
// Main Component
// =============================================================================

export function ExperimentSidebar({
  experiments = mockExperiments,
  selectedRuns = [],
  onSelectRun,
  onCompareRuns,
  onStarRun,
  onDeleteRun,
  className,
}: ExperimentSidebarProps) {
  const [expandedExperiment, setExpandedExperiment] = useState<string | null>(
    experiments[0]?.id || null
  );
  const [expandedRun, setExpandedRun] = useState<string | null>(null);
  const [localSelectedRuns, setLocalSelectedRuns] =
    useState<string[]>(selectedRuns);
  const [searchQuery, setSearchQuery] = useState("");
  const [showStarredOnly, setShowStarredOnly] = useState(false);

  const handleSelectRun = (runId: string) => {
    setLocalSelectedRuns((prev) =>
      prev.includes(runId)
        ? prev.filter((id) => id !== runId)
        : [...prev, runId]
    );
    onSelectRun?.(runId);
  };

  const comparisonRun = useMemo(() => {
    if (localSelectedRuns.length !== 2) return undefined;
    for (const exp of experiments) {
      const run = exp.runs.find((r) => r.id === localSelectedRuns[0]);
      if (run) return run;
    }
    return undefined;
  }, [localSelectedRuns, experiments]);

  const filteredExperiments = useMemo(() => {
    return experiments.map((exp) => ({
      ...exp,
      runs: exp.runs.filter((run) => {
        if (showStarredOnly && !run.starred) return false;
        if (
          searchQuery &&
          !run.name.toLowerCase().includes(searchQuery.toLowerCase())
        )
          return false;
        return true;
      }),
    }));
  }, [experiments, searchQuery, showStarredOnly]);

  return (
    <MotionDiv
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className={cn("p-4 space-y-4 overflow-y-auto", className)}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider flex items-center gap-2">
          <FlaskConical size={14} />
          Experiments
        </h3>
      </div>

      {/* Search & Filters */}
      <div className="space-y-2">
        <div className="relative">
          <Search
            size={12}
            className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground"
          />
          <input
            type="text"
            placeholder="Search runs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-7 pr-3 py-1.5 text-xs bg-background border border-border rounded focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setShowStarredOnly(!showStarredOnly)}
            className={cn(
              "flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs rounded border transition-colors",
              showStarredOnly
                ? "bg-forge-orange/20 border-forge-orange text-forge-orange"
                : "bg-muted border-border"
            )}
          >
            <Star size={12} />
            Starred
          </button>
          {localSelectedRuns.length >= 2 && (
            <button
              onClick={() => onCompareRuns?.(localSelectedRuns)}
              className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs bg-primary text-primary-foreground rounded transition-colors"
            >
              <GitCompare size={12} />
              Compare ({localSelectedRuns.length})
            </button>
          )}
        </div>
      </div>

      {/* Experiments List */}
      <div className="space-y-3">
        {filteredExperiments.map((experiment) => (
          <div key={experiment.id} className="space-y-2">
            {/* Experiment Header */}
            <button
              onClick={() =>
                setExpandedExperiment(
                  expandedExperiment === experiment.id ? null : experiment.id
                )
              }
              className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-muted rounded transition-colors"
            >
              <MotionDiv
                animate={{
                  rotate: expandedExperiment === experiment.id ? 90 : 0,
                }}
              >
                <ChevronRight size={14} className="text-muted-foreground" />
              </MotionDiv>
              <FlaskConical size={14} className="text-synapse-purple" />
              <span className="text-sm font-medium truncate flex-1 text-left">
                {experiment.name}
              </span>
              <span className="text-xs text-muted-foreground">
                {experiment.runs.length}
              </span>
            </button>

            {/* Runs */}
            <AnimatePresence>
              {expandedExperiment === experiment.id && (
                <MotionDiv
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="overflow-hidden pl-6 space-y-2"
                >
                  {experiment.runs.map((run) => (
                    <RunCard
                      key={run.id}
                      run={run}
                      isSelected={localSelectedRuns.includes(run.id)}
                      isExpanded={expandedRun === run.id}
                      onSelect={() => handleSelectRun(run.id)}
                      onToggle={() =>
                        setExpandedRun(expandedRun === run.id ? null : run.id)
                      }
                      onStar={onStarRun ? () => onStarRun(run.id) : undefined}
                      onDelete={
                        onDeleteRun ? () => onDeleteRun(run.id) : undefined
                      }
                      comparisonRun={
                        localSelectedRuns.length === 2 &&
                        localSelectedRuns.includes(run.id)
                          ? comparisonRun
                          : undefined
                      }
                    />
                  ))}
                  {experiment.runs.length === 0 && (
                    <div className="text-xs text-muted-foreground text-center py-4">
                      No runs match your filters
                    </div>
                  )}
                </MotionDiv>
              )}
            </AnimatePresence>
          </div>
        ))}
      </div>
    </MotionDiv>
  );
}

export default ExperimentSidebar;
