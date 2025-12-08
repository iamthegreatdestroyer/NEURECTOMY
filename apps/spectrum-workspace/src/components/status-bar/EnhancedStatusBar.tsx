/**
 * Enhanced Status Bar - MLflow-inspired Metrics Display
 *
 * Professional status bar with:
 * - Git branch & status
 * - Real-time metrics (CPU, GPU, Memory)
 * - Agent activity indicators
 * - Deployment status
 * - Live notifications
 *
 * Copyright (c) 2025 NEURECTOMY. All Rights Reserved.
 */

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  GitBranch,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Zap,
  Cpu,
  HardDrive,
  Rocket,
  Terminal,
  Bell,
  Wifi,
  WifiOff,
  Activity,
  Cloud,
  CloudOff,
  Network,
} from "lucide-react";
import { cn } from "@/lib/utils";

// =============================================================================
// Types
// =============================================================================

export interface StatusBarMetrics {
  cpu: number;
  memory: number;
  gpu?: number;
}

export interface AgentActivity {
  activeCount: number;
  totalCount: number;
  currentTask?: string;
}

export interface DeploymentStatus {
  environment: string;
  status: "healthy" | "deploying" | "error";
  version: string;
}

export interface StatusBarProps {
  gitBranch?: string;
  gitSynced?: boolean;
  isDirty?: boolean;
  encoding?: string;
  language?: string;
  lineCol?: { line: number; col: number };
  metrics?: { cpu: number; memory: number; gpu?: number };
  activeAgents?: number;
  agentActivity?: AgentActivity;
  deployment?: DeploymentStatus;
  deploymentStatus?: "idle" | "deploying" | "success" | "failed";
  isConnected?: boolean;
  notifications?: number;
  terminalOpen?: boolean;
  agentGraphOpen?: boolean;
  onToggleTerminal?: () => void;
  onToggleAgentGraph?: () => void;
  onOpenNotifications?: () => void;
  className?: string;
}

// =============================================================================
// Mock Data
// =============================================================================

export const mockMetrics: StatusBarMetrics = {
  cpu: 45,
  memory: 62,
  gpu: 67,
};

export const mockAgentActivity: AgentActivity = {
  activeCount: 3,
  totalCount: 12,
  currentTask: "Analyzing code patterns...",
};

export const mockDeployment: DeploymentStatus = {
  environment: "staging",
  status: "deploying",
  version: "v2.2.0",
};

// =============================================================================
// Sub-Components
// =============================================================================

function StatusBarSection({
  children,
  className,
  onClick,
}: {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
}) {
  const Comp = onClick ? "button" : "div";
  return (
    <Comp
      onClick={onClick}
      className={cn(
        "flex items-center gap-1.5 px-2 py-0.5 text-xs",
        onClick && "hover:bg-primary-foreground/10 rounded transition-colors",
        className
      )}
    >
      {children}
    </Comp>
  );
}

function Divider() {
  return <div className="w-px h-3 bg-primary-foreground/20" />;
}

function MetricBar({
  value,
  color = "bg-primary-foreground",
  className,
}: {
  value: number;
  color?: string;
  className?: string;
}) {
  const barColor =
    value > 80 ? "bg-red-400" : value > 60 ? "bg-yellow-400" : color;

  return (
    <div
      className={cn(
        "w-10 h-1.5 bg-primary-foreground/20 rounded-full overflow-hidden",
        className
      )}
    >
      <div
        className={cn(
          "h-full rounded-full transition-all duration-500",
          barColor
        )}
        style={{ width: `${value}%` }}
      />
    </div>
  );
}

function PulsingDot({ active }: { active: boolean }) {
  return (
    <div className="relative">
      {active && (
        <div className="absolute inset-0 rounded-full bg-matrix-green animate-ping opacity-50" />
      )}
      <div
        className={cn(
          "w-1.5 h-1.5 rounded-full",
          active ? "bg-matrix-green" : "bg-muted-foreground"
        )}
      />
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

export function EnhancedStatusBar({
  gitBranch = "main",
  gitSynced = true,
  isDirty = false,
  encoding = "UTF-8",
  language = "TypeScript",
  lineCol = { line: 1, col: 1 },
  metrics = mockMetrics,
  activeAgents,
  agentActivity = mockAgentActivity,
  deployment = mockDeployment,
  deploymentStatus,
  isConnected = true,
  notifications = 0,
  terminalOpen = false,
  agentGraphOpen = false,
  onToggleTerminal,
  onToggleAgentGraph,
  onOpenNotifications,
  className,
}: StatusBarProps) {
  const [showMetricsTooltip, setShowMetricsTooltip] = useState(false);

  // Merge activeAgents prop with agentActivity if provided
  const effectiveAgentActivity =
    activeAgents !== undefined
      ? { ...agentActivity, activeCount: activeAgents }
      : agentActivity;

  // Map deploymentStatus prop to deployment if provided
  const effectiveDeployment = deploymentStatus
    ? {
        ...deployment,
        status:
          deploymentStatus === "idle"
            ? ("healthy" as const)
            : deploymentStatus === "deploying"
              ? ("deploying" as const)
              : deploymentStatus === "success"
                ? ("healthy" as const)
                : ("error" as const),
      }
    : deployment;

  // Animated metrics for demo
  const [animatedMetrics, setAnimatedMetrics] = useState(metrics);

  useEffect(() => {
    // Simulate real-time metrics updates
    const interval = setInterval(() => {
      setAnimatedMetrics({
        cpu: Math.min(
          100,
          Math.max(0, metrics.cpu + (Math.random() - 0.5) * 10)
        ),
        memory: Math.min(
          100,
          Math.max(0, metrics.memory + (Math.random() - 0.5) * 5)
        ),
        gpu: metrics.gpu
          ? Math.min(100, Math.max(0, metrics.gpu + (Math.random() - 0.5) * 8))
          : undefined,
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [metrics]);

  return (
    <div
      className={cn(
        "h-6 bg-primary flex items-center justify-between text-primary-foreground text-xs",
        className
      )}
    >
      {/* Left Section */}
      <div className="flex items-center">
        {/* Git Branch */}
        <StatusBarSection>
          <GitBranch size={12} />
          <span className="font-medium">{gitBranch}</span>
          {isDirty && <span className="text-forge-orange">*</span>}
        </StatusBarSection>

        <Divider />

        {/* Sync Status */}
        <StatusBarSection>
          {gitSynced ? (
            <>
              <Cloud size={12} className="text-matrix-green" />
              <span>Synced</span>
            </>
          ) : (
            <>
              <CloudOff size={12} className="text-red-400" />
              <span>Offline</span>
            </>
          )}
        </StatusBarSection>

        <Divider />

        {/* Deployment Status */}
        <StatusBarSection>
          {effectiveDeployment.status === "deploying" ? (
            <Loader2 size={12} className="animate-spin text-neural-blue" />
          ) : effectiveDeployment.status === "healthy" ? (
            <CheckCircle2 size={12} className="text-matrix-green" />
          ) : (
            <AlertCircle size={12} className="text-red-400" />
          )}
          <span className="capitalize">{effectiveDeployment.environment}</span>
          <span className="text-primary-foreground/60">
            {effectiveDeployment.version}
          </span>
          {effectiveDeployment.status === "deploying" && (
            <span className="text-neural-blue animate-pulse">deploying...</span>
          )}
        </StatusBarSection>
      </div>

      {/* Center Section - Agent Activity */}
      <div className="flex items-center">
        <StatusBarSection className="bg-primary-foreground/5 rounded-full px-3">
          <PulsingDot active={effectiveAgentActivity.activeCount > 0} />
          <Zap size={12} className="text-synapse-purple" />
          <span className="font-medium">
            {effectiveAgentActivity.activeCount}/
            {effectiveAgentActivity.totalCount} Agents
          </span>
          {effectiveAgentActivity.currentTask && (
            <>
              <span className="text-primary-foreground/40">•</span>
              <span className="text-primary-foreground/70 max-w-48 truncate">
                {effectiveAgentActivity.currentTask}
              </span>
            </>
          )}
        </StatusBarSection>
      </div>

      {/* Right Section */}
      <div className="flex items-center">
        {/* Resource Metrics */}
        <StatusBarSection
          className="relative"
          onClick={() => setShowMetricsTooltip(!showMetricsTooltip)}
        >
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <Cpu size={11} />
              <MetricBar value={animatedMetrics.cpu} color="bg-neural-blue" />
              <span className="w-7 text-right">
                {Math.round(animatedMetrics.cpu)}%
              </span>
            </div>
            <div className="flex items-center gap-1">
              <HardDrive size={11} />
              <MetricBar
                value={animatedMetrics.memory}
                color="bg-synapse-purple"
              />
              <span className="w-7 text-right">
                {Math.round(animatedMetrics.memory)}%
              </span>
            </div>
            {animatedMetrics.gpu !== undefined && (
              <div className="flex items-center gap-1">
                <Activity size={11} />
                <MetricBar
                  value={animatedMetrics.gpu}
                  color="bg-matrix-green"
                />
                <span className="w-7 text-right">
                  {Math.round(animatedMetrics.gpu)}%
                </span>
              </div>
            )}
          </div>
        </StatusBarSection>

        <Divider />

        {/* File Info */}
        <StatusBarSection>{encoding}</StatusBarSection>
        <StatusBarSection>{language}</StatusBarSection>
        <StatusBarSection>
          Ln {lineCol.line}, Col {lineCol.col}
        </StatusBarSection>

        <Divider />

        {/* Notifications */}
        <StatusBarSection onClick={onOpenNotifications}>
          <div className="relative">
            <Bell size={12} />
            {notifications > 0 && (
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full flex items-center justify-center text-[8px] font-bold">
                {notifications > 9 ? "9+" : notifications}
              </div>
            )}
          </div>
        </StatusBarSection>

        {/* Terminal Toggle */}
        <StatusBarSection onClick={onToggleTerminal}>
          <Terminal size={12} />
          <span>{terminalOpen ? "Hide" : "Show"} Terminal</span>
        </StatusBarSection>

        {/* Agent Graph Toggle */}
        <StatusBarSection onClick={onToggleAgentGraph}>
          <Network
            size={12}
            className={agentGraphOpen ? "text-neural-blue" : ""}
          />
          <span>{agentGraphOpen ? "Hide" : "Show"} Graph</span>
        </StatusBarSection>
      </div>
    </div>
  );
}

export default EnhancedStatusBar;
