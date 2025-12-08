/**
 * DockerManager Component
 *
 * Comprehensive Docker container management interface.
 * Provides container lifecycle control, real-time monitoring, and detailed inspection.
 */

import { useState, useCallback, useMemo } from "react";
import {
  Container,
  Play,
  Pause,
  Square,
  Trash2,
  Terminal,
  RefreshCw,
  Search,
  Filter,
  MoreVertical,
  FileText,
  AlertCircle,
  CheckCircle,
  Clock,
  Cpu,
  HardDrive,
  Network,
} from "lucide-react";
import { useContainerStore } from "@/stores/container-store";
import type {
  DockerContainer,
  ContainerStatus,
} from "@/stores/container-store";

/**
 * DockerManager - Docker container management interface
 *
 * Features:
 * - Container list with real-time status
 * - Start, stop, restart, remove operations
 * - Container logs and shell access
 * - Resource usage monitoring
 * - Advanced filtering and search
 *
 * @example
 * ```tsx
 * <DockerManager />
 * ```
 */
export function DockerManager() {
  const {
    containers,
    selectedContainerId,
    selectContainer,
    updateContainer,
    removeContainer,
  } = useContainerStore();

  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<ContainerStatus | "all">(
    "all"
  );
  const [showLogs, setShowLogs] = useState<string | null>(null);

  // Filter containers
  const filteredContainers = useMemo(() => {
    return containers.filter((container) => {
      const matchesSearch =
        container.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        container.image.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesStatus =
        statusFilter === "all" || container.status === statusFilter;
      return matchesSearch && matchesStatus;
    });
  }, [containers, searchQuery, statusFilter]);

  // Calculate totals
  const stats = useMemo(() => {
    const running = containers.filter((c) => c.status === "running").length;
    const stopped = containers.filter((c) => c.status === "stopped").length;
    const totalCpu = containers
      .filter((c) => c.status === "running")
      .reduce((sum, c) => sum + c.cpu, 0);
    const totalMemory = containers
      .filter((c) => c.status === "running")
      .reduce((sum, c) => sum + c.memory, 0);

    return { running, stopped, totalCpu, totalMemory };
  }, [containers]);

  // Container actions
  const handleStart = useCallback(
    (containerId: string) => {
      updateContainer(containerId, { status: "running" });
    },
    [updateContainer]
  );

  const handleStop = useCallback(
    (containerId: string) => {
      updateContainer(containerId, { status: "stopped" });
    },
    [updateContainer]
  );

  const handleRestart = useCallback(
    (containerId: string) => {
      updateContainer(containerId, { status: "stopped" });
      setTimeout(() => {
        updateContainer(containerId, { status: "running" });
      }, 1000);
    },
    [updateContainer]
  );

  const handleRemove = useCallback(
    (containerId: string) => {
      if (confirm("Are you sure you want to remove this container?")) {
        removeContainer(containerId);
      }
    },
    [removeContainer]
  );

  const handleLogs = useCallback((containerId: string) => {
    setShowLogs(containerId);
  }, []);

  const handleShell = useCallback((containerId: string) => {
    // TODO: Open terminal shell to container
    console.log("Opening shell for container:", containerId);
  }, []);

  return (
    <div className="h-full flex flex-col gap-4 p-4 overflow-auto">
      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-card border border-border rounded-xl p-4">
          <div className="flex items-center gap-2 text-muted-foreground text-sm mb-2">
            <CheckCircle className="w-4 h-4 text-green-500" />
            Running
          </div>
          <p className="text-2xl font-bold text-green-500">{stats.running}</p>
        </div>
        <div className="bg-card border border-border rounded-xl p-4">
          <div className="flex items-center gap-2 text-muted-foreground text-sm mb-2">
            <Square className="w-4 h-4 text-gray-500" />
            Stopped
          </div>
          <p className="text-2xl font-bold text-gray-500">{stats.stopped}</p>
        </div>
        <div className="bg-card border border-border rounded-xl p-4">
          <div className="flex items-center gap-2 text-muted-foreground text-sm mb-2">
            <Cpu className="w-4 h-4 text-cyan-500" />
            Avg CPU
          </div>
          <p className="text-2xl font-bold text-cyan-500">
            {stats.totalCpu.toFixed(1)}%
          </p>
        </div>
        <div className="bg-card border border-border rounded-xl p-4">
          <div className="flex items-center gap-2 text-muted-foreground text-sm mb-2">
            <HardDrive className="w-4 h-4 text-violet-500" />
            Total Memory
          </div>
          <p className="text-2xl font-bold text-violet-500">
            {(stats.totalMemory / 1024).toFixed(1)} GB
          </p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search containers..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-card border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-muted-foreground" />
          <select
            value={statusFilter}
            onChange={(e) =>
              setStatusFilter(e.target.value as ContainerStatus | "all")
            }
            className="px-3 py-2 bg-card border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
          >
            <option value="all">All Status</option>
            <option value="running">Running</option>
            <option value="stopped">Stopped</option>
            <option value="paused">Paused</option>
            <option value="exited">Exited</option>
            <option value="dead">Dead</option>
          </select>
        </div>
      </div>

      {/* Container Table */}
      <div className="bg-card border border-border rounded-xl overflow-hidden">
        <table className="w-full">
          <thead className="bg-muted/50 border-b border-border">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase">
                Name
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase">
                Image
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase">
                Status
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase">
                CPU
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase">
                Memory
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase">
                Ports
              </th>
              <th className="px-4 py-3 text-right text-xs font-medium text-muted-foreground uppercase">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {filteredContainers.map((container) => (
              <tr
                key={container.id}
                onClick={() => selectContainer(container.id)}
                className={`
                  cursor-pointer transition-colors hover:bg-muted/30
                  ${selectedContainerId === container.id ? "bg-primary/5" : ""}
                `}
              >
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <Container className="w-4 h-4 text-primary" />
                    <span className="font-medium">{container.name}</span>
                  </div>
                </td>
                <td className="px-4 py-3 text-sm text-muted-foreground">
                  {container.image}
                </td>
                <td className="px-4 py-3">
                  <StatusBadge status={container.status} />
                </td>
                <td className="px-4 py-3 text-sm">
                  {container.status === "running"
                    ? `${container.cpu.toFixed(1)}%`
                    : "-"}
                </td>
                <td className="px-4 py-3 text-sm">
                  {container.status === "running"
                    ? `${container.memory.toFixed(0)} MB`
                    : "-"}
                </td>
                <td className="px-4 py-3 text-sm text-muted-foreground">
                  {container.ports.length > 0
                    ? container.ports
                        .map((p) => `${p.host}:${p.container}`)
                        .join(", ")
                    : "-"}
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center justify-end gap-1">
                    {container.status === "running" ? (
                      <>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleStop(container.id);
                          }}
                          className="p-1.5 hover:bg-muted rounded transition-colors"
                          title="Stop"
                        >
                          <Square className="w-4 h-4 text-red-500" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRestart(container.id);
                          }}
                          className="p-1.5 hover:bg-muted rounded transition-colors"
                          title="Restart"
                        >
                          <RefreshCw className="w-4 h-4 text-yellow-500" />
                        </button>
                      </>
                    ) : (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleStart(container.id);
                        }}
                        className="p-1.5 hover:bg-muted rounded transition-colors"
                        title="Start"
                      >
                        <Play className="w-4 h-4 text-green-500" />
                      </button>
                    )}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleLogs(container.id);
                      }}
                      className="p-1.5 hover:bg-muted rounded transition-colors"
                      title="Logs"
                    >
                      <FileText className="w-4 h-4" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleShell(container.id);
                      }}
                      className="p-1.5 hover:bg-muted rounded transition-colors"
                      title="Shell"
                    >
                      <Terminal className="w-4 h-4" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRemove(container.id);
                      }}
                      className="p-1.5 hover:bg-muted rounded transition-colors"
                      title="Remove"
                    >
                      <Trash2 className="w-4 h-4 text-red-500" />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {filteredContainers.length === 0 && (
          <div className="text-center py-12">
            <Container className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">No containers found</h3>
            <p className="text-muted-foreground text-sm">
              {searchQuery
                ? "Try adjusting your search query"
                : "No Docker containers available"}
            </p>
          </div>
        )}
      </div>

      {/* Logs Modal */}
      {showLogs && (
        <LogsModal containerId={showLogs} onClose={() => setShowLogs(null)} />
      )}
    </div>
  );
}

/**
 * StatusBadge - Container status indicator
 */
function StatusBadge({ status }: { status: ContainerStatus }) {
  const config = {
    running: {
      color: "bg-green-500/10 text-green-500 border-green-500/20",
      icon: CheckCircle,
      label: "Running",
    },
    stopped: {
      color: "bg-gray-500/10 text-gray-500 border-gray-500/20",
      icon: Square,
      label: "Stopped",
    },
    paused: {
      color: "bg-yellow-500/10 text-yellow-500 border-yellow-500/20",
      icon: Pause,
      label: "Paused",
    },
    exited: {
      color: "bg-gray-500/10 text-gray-500 border-gray-500/20",
      icon: Square,
      label: "Exited",
    },
    dead: {
      color: "bg-red-500/10 text-red-500 border-red-500/20",
      icon: AlertCircle,
      label: "Dead",
    },
  };

  const { color, icon: Icon, label } = config[status];

  return (
    <div
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-medium ${color}`}
    >
      <Icon className="w-3 h-3" />
      <span>{label}</span>
    </div>
  );
}

/**
 * LogsModal - Container logs viewer
 */
function LogsModal({
  containerId,
  onClose,
}: {
  containerId: string;
  onClose: () => void;
}) {
  const container = useContainerStore((state) =>
    state.containers.find((c) => c.id === containerId)
  );

  // Mock logs - in real implementation, fetch from Docker API
  const logs = `[2025-12-07 10:30:15] Container started
[2025-12-07 10:30:16] Application initializing...
[2025-12-07 10:30:17] Server listening on port 8080
[2025-12-07 10:30:18] Database connected
[2025-12-07 10:30:19] Ready to accept connections`;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-card border border-border rounded-xl w-full max-w-4xl max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-border flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold">Container Logs</h2>
            <p className="text-sm text-muted-foreground">{container?.name}</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-muted rounded-lg transition-colors"
          >
            <Square className="w-4 h-4" />
          </button>
        </div>

        {/* Logs Content */}
        <div className="flex-1 overflow-auto p-6">
          <pre className="text-sm font-mono text-muted-foreground whitespace-pre-wrap">
            {logs}
          </pre>
        </div>
      </div>
    </div>
  );
}
