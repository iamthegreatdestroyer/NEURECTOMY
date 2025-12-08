/**
 * ContainerCard Component
 *
 * Card view component for displaying container information in a grid layout.
 * Alternative to table view for more visual representation.
 */

import {
  Container,
  Play,
  Square,
  Pause,
  Trash2,
  Terminal,
  RefreshCw,
  FileText,
  MoreVertical,
  CheckCircle,
  AlertCircle,
  Clock,
  Cpu,
  HardDrive,
} from "lucide-react";
import type { DockerContainer } from "@/stores/container-store";

interface ContainerCardProps {
  container: DockerContainer;
  isSelected?: boolean;
  onSelect?: (containerId: string) => void;
  onStart?: (containerId: string) => void;
  onStop?: (containerId: string) => void;
  onRestart?: (containerId: string) => void;
  onRemove?: (containerId: string) => void;
  onLogs?: (containerId: string) => void;
  onShell?: (containerId: string) => void;
}

/**
 * ContainerCard - Visual card representation of a Docker container
 *
 * Features:
 * - Status indicator with color coding
 * - Container metadata display
 * - Resource usage metrics
 * - Quick action buttons
 * - Port mappings
 * - Labels display
 *
 * @example
 * ```tsx
 * <ContainerCard
 *   container={container}
 *   isSelected={selectedId === container.id}
 *   onSelect={handleSelect}
 *   onStart={handleStart}
 *   onStop={handleStop}
 * />
 * ```
 */
export function ContainerCard({
  container,
  isSelected = false,
  onSelect,
  onStart,
  onStop,
  onRestart,
  onRemove,
  onLogs,
  onShell,
}: ContainerCardProps) {
  const statusConfig = {
    running: { color: "bg-green-500", icon: CheckCircle, label: "Running" },
    stopped: { color: "bg-gray-500", icon: Square, label: "Stopped" },
    paused: { color: "bg-yellow-500", icon: Pause, label: "Paused" },
    exited: { color: "bg-gray-500", icon: Square, label: "Exited" },
    dead: { color: "bg-red-500", icon: AlertCircle, label: "Dead" },
  };

  const status = statusConfig[container.status] || statusConfig.stopped;
  const StatusIcon = status.icon;

  return (
    <div
      className={`
        relative bg-card border-2 rounded-xl p-4 transition-all cursor-pointer
        hover:shadow-lg hover:border-primary/50
        ${isSelected ? "border-primary shadow-lg" : "border-border"}
      `}
      onClick={() => onSelect?.(container.id)}
    >
      {/* Status Indicator */}
      <div className="absolute top-2 right-2 flex items-center gap-1.5">
        <div className={`w-2 h-2 rounded-full ${status.color} animate-pulse`} />
      </div>

      {/* Header */}
      <div className="flex items-start gap-3 mb-4">
        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
          <Container className="w-5 h-5 text-primary" />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold truncate mb-1">{container.name}</h3>
          <p className="text-xs text-muted-foreground truncate">
            {container.image}
          </p>
        </div>
      </div>

      {/* Status Badge */}
      <div className="mb-4">
        <div
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border ${status.color.replace("bg-", "border-")} ${status.color.replace("bg-", "text-")} bg-opacity-10`}
        >
          <StatusIcon className="w-3 h-3" />
          <span className="text-xs font-medium">{status.label}</span>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-cyan-500" />
          <div>
            <p className="text-xs text-muted-foreground">CPU</p>
            <p className="text-sm font-semibold text-cyan-500">
              {container.cpu?.toFixed(1) ?? "-"}%
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <HardDrive className="w-4 h-4 text-violet-500" />
          <div>
            <p className="text-xs text-muted-foreground">Memory</p>
            <p className="text-sm font-semibold text-violet-500">
              {container.memory?.toFixed(0) ?? "-"} MB
            </p>
          </div>
        </div>
      </div>

      {/* Ports */}
      {container.ports && container.ports.length > 0 && (
        <div className="mb-4">
          <p className="text-xs text-muted-foreground mb-1.5">Ports</p>
          <div className="flex flex-wrap gap-1">
            {container.ports.slice(0, 3).map((port, idx) => (
              <span
                key={idx}
                className="text-xs px-2 py-0.5 bg-muted rounded border border-border"
              >
                {port.host}:{port.container}
              </span>
            ))}
            {container.ports.length > 3 && (
              <span className="text-xs px-2 py-0.5 text-muted-foreground">
                +{container.ports.length - 3} more
              </span>
            )}
          </div>
        </div>
      )}

      {/* Labels */}
      {container.labels && Object.keys(container.labels).length > 0 && (
        <div className="mb-4">
          <p className="text-xs text-muted-foreground mb-1.5">Labels</p>
          <div className="flex flex-wrap gap-1">
            {Object.entries(container.labels)
              .slice(0, 2)
              .map(([key, value]) => (
                <span
                  key={key}
                  className="text-xs px-2 py-0.5 bg-primary/10 text-primary rounded"
                  title={`${key}=${value}`}
                >
                  {key}
                </span>
              ))}
            {Object.keys(container.labels).length > 2 && (
              <span className="text-xs px-2 py-0.5 text-muted-foreground">
                +{Object.keys(container.labels).length - 2}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Created At */}
      <div className="flex items-center gap-1.5 text-xs text-muted-foreground mb-4">
        <Clock className="w-3 h-3" />
        <span>
          Created {new Date(container.createdAt).toLocaleDateString()}
        </span>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2 pt-4 border-t border-border">
        {container.status === "stopped" || container.status === "exited" ? (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onStart?.(container.id);
            }}
            className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 bg-green-500/10 text-green-500 rounded-lg hover:bg-green-500/20 transition-colors text-sm"
            title="Start"
          >
            <Play className="w-4 h-4" />
            Start
          </button>
        ) : (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onStop?.(container.id);
            }}
            className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 bg-red-500/10 text-red-500 rounded-lg hover:bg-red-500/20 transition-colors text-sm"
            title="Stop"
          >
            <Square className="w-4 h-4" />
            Stop
          </button>
        )}

        <button
          onClick={(e) => {
            e.stopPropagation();
            onRestart?.(container.id);
          }}
          className="px-3 py-2 bg-muted rounded-lg hover:bg-muted/80 transition-colors"
          title="Restart"
        >
          <RefreshCw className="w-4 h-4" />
        </button>

        <button
          onClick={(e) => {
            e.stopPropagation();
            onLogs?.(container.id);
          }}
          className="px-3 py-2 bg-muted rounded-lg hover:bg-muted/80 transition-colors"
          title="View Logs"
        >
          <FileText className="w-4 h-4" />
        </button>

        <button
          onClick={(e) => {
            e.stopPropagation();
            onShell?.(container.id);
          }}
          className="px-3 py-2 bg-muted rounded-lg hover:bg-muted/80 transition-colors"
          title="Open Shell"
        >
          <Terminal className="w-4 h-4" />
        </button>

        <button
          onClick={(e) => {
            e.stopPropagation();
            if (confirm(`Remove container "${container.name}"?`)) {
              onRemove?.(container.id);
            }
          }}
          className="px-3 py-2 bg-red-500/10 text-red-500 rounded-lg hover:bg-red-500/20 transition-colors"
          title="Remove"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
