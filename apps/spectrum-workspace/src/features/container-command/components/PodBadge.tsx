/**
 * PodBadge Component
 *
 * Visual badge for Kubernetes pod status and metadata display.
 */

import {
  CheckCircle,
  Clock,
  XCircle,
  AlertCircle,
  HelpCircle,
  RefreshCw,
  Zap,
} from "lucide-react";
import type { KubernetesPod } from "@/stores/container-store";

interface PodBadgeProps {
  pod: KubernetesPod;
  size?: "sm" | "md" | "lg";
  showDetails?: boolean;
  onClick?: () => void;
}

/**
 * PodBadge - Visual representation of a Kubernetes pod
 *
 * Features:
 * - Status-based color coding
 * - Pod name and namespace display
 * - Container count
 * - Restart count with warning
 * - Resource usage (optional)
 * - Interactive click handler
 *
 * @example
 * ```tsx
 * <PodBadge
 *   pod={pod}
 *   size="md"
 *   showDetails
 *   onClick={() => selectPod(pod.id)}
 * />
 * ```
 */
export function PodBadge({
  pod,
  size = "md",
  showDetails = false,
  onClick,
}: PodBadgeProps) {
  const statusConfig = {
    Running: {
      color: "text-green-500",
      bg: "bg-green-500/10",
      border: "border-green-500",
      icon: CheckCircle,
    },
    Pending: {
      color: "text-yellow-500",
      bg: "bg-yellow-500/10",
      border: "border-yellow-500",
      icon: Clock,
    },
    Succeeded: {
      color: "text-blue-500",
      bg: "bg-blue-500/10",
      border: "border-blue-500",
      icon: CheckCircle,
    },
    Failed: {
      color: "text-red-500",
      bg: "bg-red-500/10",
      border: "border-red-500",
      icon: XCircle,
    },
    Unknown: {
      color: "text-gray-500",
      bg: "bg-gray-500/10",
      border: "border-gray-500",
      icon: HelpCircle,
    },
  };

  const status = statusConfig[pod.status] || statusConfig.Unknown;
  const StatusIcon = status.icon;

  const sizeClasses = {
    sm: "text-xs px-2 py-1",
    md: "text-sm px-3 py-1.5",
    lg: "text-base px-4 py-2",
  };

  const iconSizes = {
    sm: "w-3 h-3",
    md: "w-4 h-4",
    lg: "w-5 h-5",
  };

  const hasHighRestarts = pod.restarts > 5;

  return (
    <div
      className={`
        inline-flex items-center gap-2 rounded-lg border
        ${status.border} ${status.bg} ${sizeClasses[size]}
        ${onClick ? "cursor-pointer hover:opacity-80 transition-opacity" : ""}
      `}
      onClick={onClick}
    >
      <StatusIcon className={`${iconSizes[size]} ${status.color}`} />

      <div className="flex flex-col">
        <div className="flex items-center gap-2">
          <span className={`font-medium ${status.color}`}>{pod.name}</span>
          {showDetails && (
            <span className="text-xs text-muted-foreground">
              {pod.namespace}
            </span>
          )}
        </div>

        {showDetails && (
          <div className="flex items-center gap-3 text-xs text-muted-foreground mt-0.5">
            <span>{pod.containers.length} container(s)</span>

            {hasHighRestarts && (
              <span className="flex items-center gap-1 text-yellow-500">
                <RefreshCw className="w-3 h-3" />
                {pod.restarts} restarts
              </span>
            )}

            {pod.cpu !== undefined && (
              <span className="flex items-center gap-1">
                <Zap className="w-3 h-3" />
                {pod.cpu.toFixed(1)}%
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * PodList - List of pod badges
 */
export function PodList({
  pods,
  selectedPodId,
  onPodClick,
}: {
  pods: KubernetesPod[];
  selectedPodId: string | null;
  onPodClick?: (podId: string) => void;
}) {
  return (
    <div className="space-y-2">
      {pods.map((pod) => (
        <div
          key={pod.id}
          className={`
            p-2 rounded-lg border transition-all
            ${selectedPodId === pod.id ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"}
          `}
        >
          <PodBadge
            pod={pod}
            showDetails
            onClick={() => onPodClick?.(pod.id)}
          />
        </div>
      ))}
    </div>
  );
}
