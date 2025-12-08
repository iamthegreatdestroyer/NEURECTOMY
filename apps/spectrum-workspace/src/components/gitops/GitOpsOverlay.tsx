/**
 * GitOps Pipeline Overlay - Flux-style Deployment Status
 *
 * Real-time deployment visualization with:
 * - Progressive delivery status (prod/staging/dev)
 * - Rollback controls accessible
 * - Pipeline visualization with stage progress
 * - Health indicators for each environment
 *
 * Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
 */

import { useState, useEffect } from "react";
import { AnimatePresence } from "framer-motion";
import { MotionDiv } from "@/lib/motion";
import {
  CheckCircle2,
  Circle,
  Loader2,
  AlertCircle,
  Clock,
  GitBranch,
  GitCommit,
  GitPullRequest,
  Rocket,
  RotateCcw,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Server,
  Activity,
  Shield,
  Zap,
  AlertTriangle,
  RefreshCw,
} from "lucide-react";
import { cn } from "@/lib/utils";

// =============================================================================
// Types
// =============================================================================

export type EnvironmentStatus =
  | "healthy"
  | "deploying"
  | "degraded"
  | "down"
  | "pending";
export type PipelineStage =
  | "build"
  | "test"
  | "security"
  | "staging"
  | "production";

export interface DeploymentEnvironment {
  name: string;
  status: EnvironmentStatus;
  version: string;
  lastDeployed: Date;
  podCount: number;
  podReady: number;
  cpu: number;
  memory: number;
  url?: string;
  commitSha?: string;
}

export interface PipelineRun {
  id: string;
  branch: string;
  commit: string;
  commitMessage: string;
  author: string;
  startedAt: Date;
  completedAt?: Date;
  stages: PipelineStageStatus[];
  currentStage?: PipelineStage;
  status: "running" | "success" | "failed" | "pending";
}

export interface PipelineStageStatus {
  name: PipelineStage;
  status: "pending" | "running" | "success" | "failed" | "skipped";
  duration?: number;
  logs?: string[];
}

export interface GitOpsOverlayProps {
  environments?: DeploymentEnvironment[];
  pipeline?: PipelineRun;
  onRollback?: (env: string, version: string) => void;
  onRefresh?: () => void;
  className?: string;
}

// =============================================================================
// Mock Data
// =============================================================================

export const mockEnvironments: DeploymentEnvironment[] = [
  {
    name: "production",
    status: "healthy",
    version: "v2.1.0",
    lastDeployed: new Date(Date.now() - 86400000), // 1 day ago
    podCount: 3,
    podReady: 3,
    cpu: 45,
    memory: 62,
    url: "https://neurectomy.app",
    commitSha: "abc1234",
  },
  {
    name: "staging",
    status: "deploying",
    version: "v2.2.0-beta",
    lastDeployed: new Date(),
    podCount: 2,
    podReady: 1,
    cpu: 78,
    memory: 54,
    url: "https://staging.neurectomy.app",
    commitSha: "def5678",
  },
  {
    name: "development",
    status: "healthy",
    version: "v2.2.0-dev.45",
    lastDeployed: new Date(Date.now() - 3600000), // 1 hour ago
    podCount: 1,
    podReady: 1,
    cpu: 23,
    memory: 38,
    url: "https://dev.neurectomy.app",
    commitSha: "ghi9012",
  },
];

export const mockPipeline: PipelineRun = {
  id: "run-12345",
  branch: "main",
  commit: "def5678",
  commitMessage: "feat(ide): Add agent execution graph",
  author: "developer",
  startedAt: new Date(Date.now() - 300000), // 5 min ago
  stages: [
    { name: "build", status: "success", duration: 45000 },
    { name: "test", status: "success", duration: 120000 },
    { name: "security", status: "success", duration: 30000 },
    { name: "staging", status: "running", duration: 60000 },
    { name: "production", status: "pending" },
  ],
  currentStage: "staging",
  status: "running",
};

// =============================================================================
// Sub-Components
// =============================================================================

function EnvironmentStatusIcon({ status }: { status: EnvironmentStatus }) {
  switch (status) {
    case "healthy":
      return <CheckCircle2 size={14} className="text-matrix-green" />;
    case "deploying":
      return <Loader2 size={14} className="text-neural-blue animate-spin" />;
    case "degraded":
      return <AlertTriangle size={14} className="text-forge-orange" />;
    case "down":
      return <AlertCircle size={14} className="text-red-500" />;
    case "pending":
      return <Clock size={14} className="text-muted-foreground" />;
  }
}

function EnvironmentCard({
  env,
  onRollback,
  isExpanded,
  onToggle,
}: {
  env: DeploymentEnvironment;
  onRollback?: () => void;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const statusColors: Record<EnvironmentStatus, string> = {
    healthy: "border-matrix-green/50",
    deploying: "border-neural-blue/50",
    degraded: "border-forge-orange/50",
    down: "border-red-500/50",
    pending: "border-muted",
  };

  const envIcons: Record<string, React.ReactNode> = {
    production: <Shield size={14} className="text-matrix-green" />,
    staging: <Rocket size={14} className="text-neural-blue" />,
    development: <GitBranch size={14} className="text-synapse-purple" />,
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
        "rounded-lg border-2 bg-card/50 transition-all",
        statusColors[env.status]
      )}
    >
      {/* Header */}
      <button
        onClick={onToggle}
        className="w-full px-3 py-2 flex items-center justify-between hover:bg-muted/30 transition-colors"
      >
        <div className="flex items-center gap-2">
          <EnvironmentStatusIcon status={env.status} />
          {envIcons[env.name] || <Server size={14} />}
          <span className="font-medium text-sm capitalize">{env.name}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-muted-foreground">
            {env.version}
          </span>
          <MotionDiv animate={{ rotate: isExpanded ? 180 : 0 }}>
            <ChevronDown size={14} className="text-muted-foreground" />
          </MotionDiv>
        </div>
      </button>

      {/* Expanded Details */}
      <AnimatePresence>
        {isExpanded && (
          <MotionDiv
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="px-3 pb-3 space-y-3">
              {/* Pod Status */}
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Pods</span>
                <div className="flex items-center gap-1">
                  {Array.from({ length: env.podCount }).map((_, i) => (
                    <div
                      key={i}
                      className={cn(
                        "w-2 h-2 rounded-full",
                        i < env.podReady ? "bg-matrix-green" : "bg-muted"
                      )}
                    />
                  ))}
                  <span className="ml-1 text-muted-foreground">
                    {env.podReady}/{env.podCount}
                  </span>
                </div>
              </div>

              {/* Resource Usage */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">CPU</span>
                  <span>{env.cpu}%</span>
                </div>
                <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                  <MotionDiv
                    className={cn(
                      "h-full rounded-full",
                      env.cpu > 80
                        ? "bg-red-500"
                        : env.cpu > 60
                          ? "bg-forge-orange"
                          : "bg-matrix-green"
                    )}
                    initial={{ width: 0 }}
                    animate={{ width: `${env.cpu}%` }}
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Memory</span>
                  <span>{env.memory}%</span>
                </div>
                <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                  <MotionDiv
                    className={cn(
                      "h-full rounded-full",
                      env.memory > 80
                        ? "bg-red-500"
                        : env.memory > 60
                          ? "bg-forge-orange"
                          : "bg-neural-blue"
                    )}
                    initial={{ width: 0 }}
                    animate={{ width: `${env.memory}%` }}
                  />
                </div>
              </div>

              {/* Meta Info */}
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <div className="flex items-center gap-1">
                  <Clock size={12} />
                  <span>{formatTimeAgo(env.lastDeployed)}</span>
                </div>
                {env.commitSha && (
                  <div className="flex items-center gap-1 font-mono">
                    <GitCommit size={12} />
                    <span>{env.commitSha.slice(0, 7)}</span>
                  </div>
                )}
              </div>

              {/* Actions */}
              <div className="flex gap-2 pt-1">
                {env.url && (
                  <a
                    href={env.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs bg-muted hover:bg-muted/80 rounded transition-colors"
                  >
                    <ExternalLink size={12} />
                    <span>Open</span>
                  </a>
                )}
                {onRollback && env.name !== "development" && (
                  <button
                    onClick={onRollback}
                    className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs bg-muted hover:bg-red-500/20 hover:text-red-500 rounded transition-colors"
                  >
                    <RotateCcw size={12} />
                    <span>Rollback</span>
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

function PipelineStageIcon({
  status,
}: {
  status: PipelineStageStatus["status"];
}) {
  switch (status) {
    case "success":
      return <CheckCircle2 size={12} className="text-matrix-green" />;
    case "running":
      return <Loader2 size={12} className="text-neural-blue animate-spin" />;
    case "failed":
      return <AlertCircle size={12} className="text-red-500" />;
    case "skipped":
      return <Circle size={12} className="text-muted-foreground" />;
    default:
      return <Circle size={12} className="text-muted-foreground" />;
  }
}

function PipelineView({ pipeline }: { pipeline: PipelineRun }) {
  const stageIcons: Record<PipelineStage, React.ReactNode> = {
    build: <Zap size={12} />,
    test: <Activity size={12} />,
    security: <Shield size={12} />,
    staging: <Rocket size={12} />,
    production: <Server size={12} />,
  };

  const formatDuration = (ms?: number) => {
    if (!ms) return "-";
    if (ms < 60000) return `${Math.floor(ms / 1000)}s`;
    return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
  };

  return (
    <div className="space-y-2">
      {/* Pipeline Header */}
      <div className="flex items-center gap-2 text-xs">
        <GitBranch size={12} className="text-primary" />
        <span className="font-medium">{pipeline.branch}</span>
        <span className="text-muted-foreground">•</span>
        <span className="font-mono text-muted-foreground">
          {pipeline.commit.slice(0, 7)}
        </span>
      </div>

      <div className="text-xs text-muted-foreground truncate">
        {pipeline.commitMessage}
      </div>

      {/* Stages */}
      <div className="flex items-center gap-1">
        {pipeline.stages.map((stage, index) => (
          <div key={stage.name} className="flex items-center">
            <div
              className={cn(
                "flex items-center gap-1 px-2 py-1 rounded text-xs",
                stage.status === "success" &&
                  "bg-matrix-green/10 text-matrix-green",
                stage.status === "running" &&
                  "bg-neural-blue/10 text-neural-blue",
                stage.status === "failed" && "bg-red-500/10 text-red-500",
                stage.status === "pending" && "bg-muted text-muted-foreground",
                stage.status === "skipped" &&
                  "bg-muted/50 text-muted-foreground"
              )}
            >
              <PipelineStageIcon status={stage.status} />
              <span className="capitalize">{stage.name}</span>
              {stage.duration && (
                <span className="text-[10px] opacity-70">
                  {formatDuration(stage.duration)}
                </span>
              )}
            </div>
            {index < pipeline.stages.length - 1 && (
              <div
                className={cn(
                  "w-3 h-0.5 mx-0.5",
                  stage.status === "success" ? "bg-matrix-green/50" : "bg-muted"
                )}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

export function GitOpsOverlay({
  environments = mockEnvironments,
  pipeline = mockPipeline,
  onRollback,
  onRefresh,
  className,
}: GitOpsOverlayProps) {
  const [expandedEnv, setExpandedEnv] = useState<string | null>("staging");

  return (
    <MotionDiv
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className={cn("p-4 space-y-4", className)}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider">
          GitOps / Deployments
        </h3>
        <button
          onClick={onRefresh}
          className="p-1 hover:bg-muted rounded transition-colors"
          title="Refresh"
        >
          <RefreshCw size={14} className="text-muted-foreground" />
        </button>
      </div>

      {/* Active Pipeline */}
      {pipeline && (
        <div className="p-3 bg-muted/30 rounded-lg border border-border">
          <div className="flex items-center gap-2 mb-2">
            <Loader2 size={12} className="text-neural-blue animate-spin" />
            <span className="text-xs font-medium">Active Pipeline</span>
          </div>
          <PipelineView pipeline={pipeline} />
        </div>
      )}

      {/* Environments */}
      <div className="space-y-2">
        <div className="text-xs font-medium text-muted-foreground mb-2">
          Environments
        </div>
        {environments.map((env) => (
          <EnvironmentCard
            key={env.name}
            env={env}
            isExpanded={expandedEnv === env.name}
            onToggle={() =>
              setExpandedEnv(expandedEnv === env.name ? null : env.name)
            }
            onRollback={
              onRollback ? () => onRollback(env.name, env.version) : undefined
            }
          />
        ))}
      </div>
    </MotionDiv>
  );
}

export default GitOpsOverlay;
